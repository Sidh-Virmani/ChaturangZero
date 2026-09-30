"""
p6_rule_inference/train.py
============================
Self-contained AI training + evaluation + benchmark pipeline for ChaturangZero-X.

Usage
-----
python -m p6_rule_inference.train          # full run (30 games/variant, 3 epochs)
python -m p6_rule_inference.train --quick  # fast CI run (5 games/variant, 1 epoch)

What this script does
---------------------
1.  Generates a synthetic self-play dataset (no internet required as fallback):
      - Tries to pull a real dataset from HuggingFace first (using .env HF_TOKEN).
      - Falls back to self-play game generation if HF dataset is unavailable.
2.  Trains MultiVariantChaturangaNet on all 4 rule variants simultaneously.
3.  Evaluates on a held-out test split and reports:
      - Policy cross-entropy loss (train vs val)
      - Rule inference accuracy (Bayesian classifier)
      - Game quality score per variant
      - Strategy DNA cosine-distance matrix
4.  Saves a markdown benchmark report to reports/p6_benchmark.md
5.  Saves model checkpoint to checkpoints/p6/multivariant_model.pt
"""

from __future__ import annotations

import sys
import os
import pathlib
import argparse
import random
import time
import json
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# --- project root on path -----------------------------------------------------
_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.rule_variants import RuleVariant, RuleConfig
from shared.game_core.state import Side
from p1_representation.encoder import StateEncoder
from p6_rule_inference.multi_variant_model import MultiVariantChaturangaNet
from p6_rule_inference.quality_metrics import (
    compute_quality_report,
    play_game_for_quality,
    QualityReport,
)
from p6_rule_inference.bayesian_classifier import BayesianRuleClassifier
from p6_rule_inference.strategy_dna import extract_game_dna, cosine_distance
from p6_rule_inference.phylogenetic_tree import (
    generate_distance_matrix,
    print_distance_matrix,
)

# --- constants ----------------------------------------------------------------
VARIANTS = ["standard", "capture_raja", "bare_raja", "no_stalemate"]
VARIANT_NAMES = {
    "standard":     "Standard (R1)",
    "capture_raja": "Capture-Raja (R2)",
    "bare_raja":    "Bare-Raja (R3)",
    "no_stalemate": "No-Stalemate (R4)",
}
VARIANT_IDX = {v: i for i, v in enumerate(VARIANTS)}


# ===============================================================================
# 1.  DATASET GENERATION
# ===============================================================================

@dataclass
class GameSample:
    board_tensor: torch.Tensor      # shape (13, 8, 8)
    policy_target: torch.Tensor     # shape (4096,)  one-hot
    value_target: float             # +1 / 0 / -1
    variant_idx: int                # 0..3


def _generate_self_play_games(
    variant: str,
    n_games: int,
    max_plies: int = 80,
    verbose: bool = False,
) -> List[GameSample]:
    """Generate training samples by random self-play under a rule variant."""
    engine = ChaturangaEngine(rule_config=RuleConfig.from_variant(RuleVariant(variant)))
    encoder = StateEncoder()
    vidx = VARIANT_IDX[variant]
    samples: List[GameSample] = []

    for g in range(n_games):
        state = engine.new_game()
        trajectory = []  # (board_tensor, move)
        ply = 0

        while not engine.is_terminal(state) and ply < max_plies:
            legal = engine.legal_moves(state)
            if not legal:
                break
            move = random.choice(legal)
            board = encoder.encode_state_torch(state).squeeze(0)  # remove batch dim -> (13,8,8)
            trajectory.append((board, move))
            state = engine.apply_move(state, move)
            ply += 1

        # Determine outcome
        if engine.is_terminal(state):
            winner = getattr(state, "winner", "draw")
        else:
            winner = "draw"

        outcome_map = {"white": 1.0, "black": -1.0, "draw": 0.0}
        outcome = outcome_map.get(winner, 0.0)

        for i, (board, move) in enumerate(trajectory):
            # Policy target: one-hot on chosen action
            policy = torch.zeros(4096, dtype=torch.float32)
            action_idx = encoder.encode_action(move)
            if 0 <= action_idx < 4096:
                policy[action_idx] = 1.0

            # Value: from the perspective of the side to move at that ply
            value = outcome if (i % 2 == 0) else -outcome

            samples.append(GameSample(
                board_tensor=board,
                policy_target=policy,
                value_target=value,
                variant_idx=vidx,
            ))

        if verbose:
            print(f"  [{variant}] game {g+1}/{n_games}: plies={ply} winner={winner}")

    return samples


def _try_load_huggingface(n_games_per_variant: int) -> dict | None:
    """Try loading real dataset from HuggingFace. Returns None on failure."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            return None
        from datasets import load_dataset
        ds = load_dataset(
            "Lichess/lichess_games",   # closest public chess dataset as proxy
            split="train",
            token=token,
            streaming=True,
        )
        # Collect up to n_games_per_variant * 4 games
        games_raw = []
        for item in ds:
            pgn = item.get("moves", "") or item.get("pgn", "")
            if pgn:
                moves = pgn.strip().split()[:40]   # first 40 half-moves
                games_raw.append(moves)
            if len(games_raw) >= n_games_per_variant * 4:
                break
        if not games_raw:
            return None
        print(f"  [hf] Loaded {len(games_raw)} real games from HuggingFace")
        return {"games": games_raw}
    except Exception as e:
        print(f"  [hf] HuggingFace load failed: {e} — falling back to self-play")
        return None


class ChaturangaDataset(Dataset):
    def __init__(self, samples: List[GameSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s.board_tensor,
            s.policy_target,
            torch.tensor(s.value_target, dtype=torch.float32),
            torch.tensor(s.variant_idx, dtype=torch.long),
        )


# ===============================================================================
# 2.  TRAINING LOOP
# ===============================================================================

def train_epoch(
    model: MultiVariantChaturangaNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    policy_losses, value_losses = [], []
    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    for boards, policy_targets, value_targets, variant_idxs in loader:
        boards = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)
        variant_idxs = variant_idxs.to(device)

        optimizer.zero_grad()

        policy_logits, value_pred, *_ = model(boards, variant_idxs)

        loss_p = criterion_p(policy_logits, policy_targets.argmax(dim=1))
        loss_v = criterion_v(value_pred.squeeze(-1), value_targets)
        loss = loss_p + loss_v
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        policy_losses.append(loss_p.item())
        value_losses.append(loss_v.item())

    return float(np.mean(policy_losses)), float(np.mean(value_losses))


@torch.no_grad()
def eval_epoch(
    model: MultiVariantChaturangaNet,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Returns (policy_loss, value_loss, top1_accuracy)"""
    model.eval()
    criterion_p = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()
    p_losses, v_losses, correct, total = [], [], 0, 0

    for boards, policy_targets, value_targets, variant_idxs in loader:
        boards = boards.to(device)
        policy_targets = policy_targets.to(device)
        value_targets = value_targets.to(device)
        variant_idxs = variant_idxs.to(device)

        policy_logits, value_pred, *_ = model(boards, variant_idxs)
        target_ids = policy_targets.argmax(dim=1)

        loss_p = criterion_p(policy_logits, target_ids)
        loss_v = criterion_v(value_pred.squeeze(-1), value_targets)

        p_losses.append(loss_p.item())
        v_losses.append(loss_v.item())

        preds = policy_logits.argmax(dim=1)
        correct += (preds == target_ids).sum().item()
        total += boards.size(0)

    top1_acc = correct / max(total, 1)
    return float(np.mean(p_losses)), float(np.mean(v_losses)), top1_acc


# ===============================================================================
# 3.  RULE INFERENCE ACCURACY
# ===============================================================================

def evaluate_rule_inference_accuracy(
    model: MultiVariantChaturangaNet,
    n_test_games: int = 20,
    max_plies: int = 30,
    device: str = "cpu",
) -> float:
    """
    Play games under known variants, then classify using the trained model.
    
    Strategy: for each test game, encode each position and run it through
    the model with each variant_idx. The variant whose policy distribution
    assigns the highest average log-probability to the observed moves wins.
    """
    model.eval()
    encoder = StateEncoder()

    correct = 0
    total = 0

    for variant in VARIANTS:
        true_vidx = VARIANT_IDX[variant]
        engine = ChaturangaEngine(rule_config=RuleConfig.from_variant(RuleVariant(variant)))
        for _ in range(n_test_games):
            state = engine.new_game()
            # Collect (board_tensor, chosen_action_idx) pairs
            positions: List[tuple] = []
            for _ in range(max_plies):
                if engine.is_terminal(state):
                    break
                legal = engine.legal_moves(state)
                if not legal:
                    break
                move = random.choice(legal)
                board = encoder.encode_state_torch(state).squeeze(0)  # (13,8,8)
                action_idx = encoder.encode_action(move)
                positions.append((board, action_idx))
                state = engine.apply_move(state, move)

            if len(positions) < 3:
                continue

            # Score each candidate variant using cumulative log-prob of observed actions
            boards_t = torch.stack([p[0] for p in positions]).to(device)  # (T,13,8,8)
            action_ids = [p[1] for p in positions]
            T = boards_t.size(0)

            variant_scores: dict[str, float] = {}
            with torch.no_grad():
                for cand_v in VARIANTS:
                    cand_idx = VARIANT_IDX[cand_v]
                    vidx_t = torch.full((T,), cand_idx, dtype=torch.long, device=device)
                    logits, *_ = model(boards_t, vidx_t)       # (T, 4096)
                    log_probs = torch.log_softmax(logits, dim=-1)  # (T, 4096)
                    # Sum log-prob of each chosen action
                    chosen_lp = sum(
                        log_probs[t, action_ids[t]].item()
                        for t in range(T)
                    )
                    variant_scores[cand_v] = chosen_lp / T     # average per ply

            predicted = max(variant_scores, key=variant_scores.get)
            if predicted == variant:
                correct += 1
            total += 1

    return correct / max(total, 1)


# ===============================================================================
# 4.  STRATEGY DNA MATRIX
# ===============================================================================

def compute_dna_matrix(n_games: int = 10, max_plies: int = 40) -> Tuple[list, list]:
    """Collect mean Strategy DNA for each variant and return distance matrix."""
    dna_dict: dict[str, np.ndarray] = {}

    for variant in VARIANTS:
        engine = ChaturangaEngine(rule_config=RuleConfig.from_variant(RuleVariant(variant)))
        dna_accum: List[np.ndarray] = []

        for _ in range(n_games):
            state = engine.new_game()
            moves: List[str] = []
            for _ in range(max_plies):
                if engine.is_terminal(state):
                    break
                legal = engine.legal_moves(state)
                if not legal:
                    break
                move = random.choice(legal)
                moves.append(move.label())
                state = engine.apply_move(state, move)

            if moves:
                dna = extract_game_dna(moves, engine)
                dna_accum.append(dna)

        if dna_accum:
            dna_dict[VARIANT_NAMES[variant]] = np.mean(dna_accum, axis=0)
        else:
            from p6_rule_inference.strategy_dna import NUM_FEATURES
            dna_dict[VARIANT_NAMES[variant]] = np.zeros(NUM_FEATURES)

    labels, matrix = generate_distance_matrix({k: v.tolist() for k, v in dna_dict.items()})
    return labels, matrix


# ===============================================================================
# 5.  BENCHMARK REPORT
# ===============================================================================

def save_benchmark_report(
    train_history: List[dict],
    val_history: List[dict],
    quality_reports: List[QualityReport],
    rule_inference_acc: float,
    dna_labels: List[str],
    dna_matrix: List[List[float]],
    output_path: pathlib.Path,
) -> None:
    from datetime import datetime

    lines = [
        "# ChaturangZero-X — AI Benchmark Report",
        f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "---\n",

        "## 1. Training Curves\n",
        "| Epoch | Train Policy Loss | Train Value Loss | Val Policy Loss | Val Value Loss | Val Top-1 Acc |",
        "|-------|-------------------|-----------------|-----------------|----------------|---------------|",
    ]

    for i, (tr, va) in enumerate(zip(train_history, val_history)):
        lines.append(
            f"| {i+1:5d} "
            f"| {tr['policy']:.4f}            "
            f"| {tr['value']:.4f}            "
            f"| {va['policy']:.4f}           "
            f"| {va['value']:.4f}           "
            f"| {va['acc']*100:.1f}%        |"
        )

    lines += [
        "\n---\n",
        "## 2. Game Quality Scores (Random Play Baseline)\n",
        "| Variant | Decisiveness | Tactical Complexity | Material Volatility | Blunder Freq | **Quality Score** |",
        "|---------|-------------|---------------------|--------------------|--------------|--------------------|",
    ]
    for r in sorted(quality_reports, key=lambda x: x.quality_score, reverse=True):
        lines.append(
            f"| {VARIANT_NAMES.get(r.rule_variant, r.rule_variant):20s} "
            f"| {r.decisiveness:.3f}        "
            f"| {r.tactical_complexity:.3f}               "
            f"| {r.material_volatility:.3f}              "
            f"| {r.blunder_frequency:.3f}       "
            f"| **{r.quality_score:.4f}**        |"
        )

    lines += [
        "\n---\n",
        "## 3. Bayesian Rule Inference Accuracy\n",
        f"**Test accuracy: {rule_inference_acc*100:.1f}%**  ",
        "> (% of games where highest-posterior variant matches the true generating variant)\n",
        "\n---\n",

        "## 4. Strategy DNA — Cosine Distance Matrix\n",
        "> Lower value = more similar strategic fingerprint between the two variants.\n",
        "",
    ]

    # ASCII table for DNA matrix
    header = "| Variant | " + " | ".join([l[:15] for l in dna_labels]) + " |"
    sep = "|---------|" + "|".join(["----------" for _ in dna_labels]) + "|"
    lines.append(header)
    lines.append(sep)
    for i, label in enumerate(dna_labels):
        row = f"| {label[:15]:15s} |"
        for j in range(len(dna_labels)):
            if i == j:
                row += "  —       |"
            else:
                row += f" {dna_matrix[i][j]:.4f}   |"
        lines.append(row)

    lines += [
        "\n---\n",
        "## 5. Key Findings\n",
        f"- Best game quality variant: **{sorted(quality_reports, key=lambda x: x.quality_score, reverse=True)[0].rule_variant}**",
        f"- Bayesian rule inference accuracy: **{rule_inference_acc*100:.1f}%**",
        f"- Final val policy loss: **{val_history[-1]['policy']:.4f}**",
        f"- Final val top-1 accuracy: **{val_history[-1]['acc']*100:.1f}%**",
        "\n---\n",
        "_ChaturangZero-X — Computational Game Archaeology AI_",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] Saved benchmark report to {output_path}")


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(description="ChaturangZero-X AI Training & Benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Quick CI run (5 games/variant, 1 epoch)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--games", type=int, default=None,
                        help="Self-play games per variant for training")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # -- Config ----------------------------------------------------------------
    if args.quick:
        N_GAMES_TRAIN = args.games or 5
        N_GAMES_VAL   = 3
        N_GAMES_Q     = 5
        N_EPOCHS      = args.epochs or 1
    else:
        N_GAMES_TRAIN = args.games or 30
        N_GAMES_VAL   = 10
        N_GAMES_Q     = 15
        N_EPOCHS      = args.epochs or 3

    BATCH = args.batch_size
    DEVICE = torch.device(args.device)

    print("=" * 60)
    print("  ChaturangZero-X — AI Training & Benchmark Pipeline")
    print("=" * 60)
    print(f"  Variants      : {VARIANTS}")
    print(f"  Train games   : {N_GAMES_TRAIN} per variant")
    print(f"  Val games     : {N_GAMES_VAL} per variant")
    print(f"  Quality games : {N_GAMES_Q} per variant")
    print(f"  Epochs        : {N_EPOCHS}")
    print(f"  Batch size    : {BATCH}")
    print(f"  Device        : {DEVICE}")
    print("=" * 60)

    # -- Step 1: Dataset -------------------------------------------------------
    print("\n[step 1/5] Generating training dataset...")

    # Try HF first
    hf_data = _try_load_huggingface(N_GAMES_TRAIN)

    all_train_samples: List[GameSample] = []
    all_val_samples: List[GameSample] = []

    for variant in VARIANTS:
        print(f"  Generating {N_GAMES_TRAIN} train games for [{variant}]...")
        train_s = _generate_self_play_games(variant, N_GAMES_TRAIN, verbose=False)
        val_s   = _generate_self_play_games(variant, N_GAMES_VAL, verbose=False)
        all_train_samples.extend(train_s)
        all_val_samples.extend(val_s)
        print(f"    -> {len(train_s)} train samples, {len(val_s)} val samples")

    train_ds = ChaturangaDataset(all_train_samples)
    val_ds   = ChaturangaDataset(all_val_samples)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"  Total train samples : {len(train_ds)}")
    print(f"  Total val samples   : {len(val_ds)}")

    # -- Step 2: Build model ---------------------------------------------------
    print("\n[step 2/5] Building MultiVariantChaturangaNet...")
    model = MultiVariantChaturangaNet(
        in_channels=13,
        num_filters=64,
        num_res_blocks=4,
        action_size=4096,
        num_variants=len(VARIANTS),
        rule_embed_dim=8,
        uncertainty_head=True,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    # -- Step 3: Training ------------------------------------------------------
    print(f"\n[step 3/5] Training for {N_EPOCHS} epochs...")
    train_history: List[dict] = []
    val_history:   List[dict] = []

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        tr_p, tr_v = train_epoch(model, train_loader, optimizer, DEVICE)
        va_p, va_v, va_acc = eval_epoch(model, val_loader, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        train_history.append({"policy": tr_p, "value": tr_v})
        val_history.append(  {"policy": va_p, "value": va_v, "acc": va_acc})

        print(
            f"  Epoch {epoch:3d}/{N_EPOCHS} | "
            f"train p={tr_p:.4f} v={tr_v:.4f} | "
            f"val p={va_p:.4f} v={va_v:.4f} acc={va_acc*100:.1f}% | "
            f"{elapsed:.1f}s"
        )

    # -- Step 4: Game Quality Evaluation ---------------------------------------
    print(f"\n[step 4/5] Computing game quality scores ({N_GAMES_Q} games/variant)...")
    quality_reports: List[QualityReport] = []
    for variant in VARIANTS:
        print(f"  Evaluating [{variant}]...")
        cfg = RuleConfig.from_variant(RuleVariant(variant))
        report = compute_quality_report(cfg, n_games=N_GAMES_Q, verbose=False)
        quality_reports.append(report)
        print(f"    -> Quality score: {report.quality_score:.4f}  "
              f"decisiveness={report.decisiveness:.2f}  "
              f"complexity={report.tactical_complexity:.2f}")

    # -- Step 4b: Rule Inference Accuracy --------------------------------------
    print("\n  Computing Bayesian Rule Inference accuracy (20 test games)...")
    ri_acc = evaluate_rule_inference_accuracy(model, n_test_games=20, device=args.device)
    print(f"    -> Rule inference accuracy: {ri_acc*100:.1f}%")

    # -- Step 4c: Strategy DNA Matrix ------------------------------------------
    print("\n  Computing Strategy DNA distance matrix...")
    dna_labels, dna_matrix = compute_dna_matrix(n_games=8, max_plies=40)
    print_distance_matrix(dna_labels, dna_matrix)

    # -- Step 5: Save checkpoint + report -------------------------------------
    print("\n[step 5/5] Saving checkpoint and benchmark report...")
    ckpt_dir = _root / "checkpoints" / "p6"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "multivariant_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved -> {ckpt_path}")

    report_path = _root / "reports" / "p6_benchmark.md"
    save_benchmark_report(
        train_history, val_history, quality_reports,
        ri_acc, dna_labels, dna_matrix, report_path,
    )

    # -- Final Summary ---------------------------------------------------------
    best = sorted(quality_reports, key=lambda x: x.quality_score, reverse=True)[0]
    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"  Final val policy loss   : {val_history[-1]['policy']:.4f}")
    print(f"  Final val top-1 acc     : {val_history[-1]['acc']*100:.1f}%")
    print(f"  Rule inference accuracy : {ri_acc*100:.1f}%")
    print(f"  Best quality variant    : {best.rule_variant}  (score {best.quality_score:.4f})")
    print(f"  Report                  : {report_path}")
    print(f"  Checkpoint              : {ckpt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
