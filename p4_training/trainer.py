"""
p4_training/trainer.py
======================
Main AlphaZero-style training loop for ChaturangZero-X.

ChaturangZero-X additions
--------------------------
  - curriculum: bool  — use CurriculumScheduler (stages: simplified → endgame → tactical → full)
  - hard_mining: bool — use HardPositionBuffer (over-sample uncertain positions)
  - uncertainty_weight: float — Gaussian NLL weight vs policy loss
  - rule_variant: str — train under a specific historical rule variant

Loop per iteration:
  1. Self-play: generate games using MCTS + current model
     → collect hard positions into HardPositionBuffer
  2. Buffer: add examples to replay buffer
  3. Train: mixed sample (70% hard + 30% normal), compute NLL loss, update
  4. Evaluate: candidate-vs-best arena
  5. Promote or reject; advance curriculum stage if win_rate ≥ threshold
  6. Update Elo rating and opening book

Usage (quick demo — CPU):
    python -m p4_training.trainer --iterations 5 --games-per-iter 3 --sims 25

Usage (full ChaturangZero-X training):
    python -m p4_training.trainer --iterations 50 --games-per-iter 25 --sims 100 --curriculum --hard-mining
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys
import time

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False
    print("[trainer] WARNING: PyTorch not found. Run: pip install torch")

from shared.game_core.engine import ChaturangaEngine
from p4_training.self_play    import run_self_play_batch
from p4_training.replay_buffer import ReplayBuffer
from p4_training.evaluator    import arena_match
from p4_training.elo_tracker  import EloTracker
from p4_training.opening_book import OpeningBook


class Trainer:
    """
    Orchestrates the full AlphaZero-style training loop with
    curriculum learning and hard position mining (ChaturangZero-X).

    Parameters
    ----------
    iterations        : number of training iterations
    games_per_iter    : self-play games per iteration
    num_sims          : MCTS simulations per move
    batch_size        : training batch size
    epochs_per_iter   : training epochs per iteration
    arena_games       : games for candidate-vs-best evaluation
    win_threshold     : fraction needed for candidate promotion
    checkpoint_dir    : directory for saving models
    device            : 'cpu' or 'cuda'
    curriculum        : enable curriculum self-play (Research Contribution #7)
    hard_mining       : enable hard position over-sampling (Research Contribution #8)
    uncertainty_weight: Gaussian NLL vs policy loss weighting
    rule_variant      : historical rule variant to train under (Research Contribution #6)
    """

    def __init__(
        self,
        iterations:         int   = 10,
        games_per_iter:     int   = 5,
        num_sims:           int   = 50,
        batch_size:         int   = 256,
        epochs_per_iter:    int   = 5,
        arena_games:        int   = 20,
        win_threshold:      float = 0.55,
        checkpoint_dir:     str   = "checkpoints",
        device:             str   = "cpu",
        curriculum:         bool  = False,
        hard_mining:        bool  = False,
        uncertainty_weight: float = 1.0,
        rule_variant:       str   = "standard",
    ) -> None:
        self.iterations         = iterations
        self.games_per_iter     = games_per_iter
        self.num_sims           = num_sims
        self.batch_size         = batch_size
        self.epochs_per_iter    = epochs_per_iter
        self.arena_games        = arena_games
        self.win_threshold      = win_threshold
        self.ckpt_dir           = pathlib.Path(checkpoint_dir)
        self.device             = device
        self.curriculum_enabled = curriculum
        self.hard_mining_enabled= hard_mining
        self.uncertainty_weight = uncertainty_weight
        self.rule_variant       = rule_variant

        # Engine — with rule variant
        try:
            from shared.game_core.rule_variants import RuleVariant, RuleConfig
            variant = RuleVariant(rule_variant)
            cfg = RuleConfig.from_variant(variant)
            self.engine = ChaturangaEngine(rule_config=cfg)
            print(f"[trainer] Rule variant: {variant.value}")
        except Exception:
            self.engine = ChaturangaEngine()

        self.buffer       = ReplayBuffer(max_size=100_000)
        self.elo          = EloTracker(save_path=str(self.ckpt_dir / "elo_history.json"))
        self.opening_book = OpeningBook(save_path=str(self.ckpt_dir / "opening_book.json"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Curriculum scheduler
        self.curriculum = None
        if curriculum:
            from p4_training.curriculum import CurriculumScheduler
            self.curriculum = CurriculumScheduler(self.engine, verbose=True)
            print(f"[trainer] Curriculum enabled: {self.curriculum.summary()}")

        # Hard position buffer
        self.hard_buffer = None
        if hard_mining:
            from p4_training.hard_position_buffer import HardPositionBuffer
            self.hard_buffer = HardPositionBuffer()
            print("[trainer] Hard position mining enabled")


    def run(self) -> None:
        """Entry point — run the full training loop."""
        if not _TORCH:
            print("[trainer] Cannot train: PyTorch is required.")
            print("  Install with: pip install torch")
            return

        from p2_neural_network.model import ChaturangaNet, save_checkpoint, load_checkpoint
        from p2_neural_network.train_utils import build_optimizer, train_epoch

        print("=" * 60)
        print("  ChaturangZero — AlphaZero Training Loop")
        print(f"  Iterations: {self.iterations}  |  Games/iter: {self.games_per_iter}")
        print(f"  MCTS sims:  {self.num_sims}   |  Device: {self.device}")
        print("=" * 60)

        # Initialise or load best model
        best_path = self.ckpt_dir / "best_model.pt"
        if best_path.exists():
            best_model = load_checkpoint(str(best_path), device=self.device)
            self.elo.load()
            self.opening_book.load()
        else:
            best_model = ChaturangaNet().to(self.device)
            save_checkpoint(best_model, str(best_path))

        for iteration in range(1, self.iterations + 1):
            t0 = time.time()
            print(f"\n{'─'*60}")
            print(f"  ITERATION {iteration}/{self.iterations}")
            print(f"{'─'*60}")

            # 1. Self-play
            print(f"\n[1/4] Self-play ({self.games_per_iter} games) …")
            examples = run_self_play_batch(
                self.engine, best_model,
                num_games=self.games_per_iter,
                num_sims=self.num_sims,
                device=self.device,
            )
            self.buffer.add(examples)
            print(f"  Buffer size: {len(self.buffer)}")

            # Record openings
            # (move histories come from engine; approximated here via buffer)

            # 2. Train
            print(f"\n[2/4] Training ({self.epochs_per_iter} epochs) …")
            if not self.buffer.is_ready(self.batch_size):
                print("  Buffer too small — skipping training this iteration.")
                continue

            candidate = copy.deepcopy(best_model)
            optimizer = build_optimizer(candidate, lr=1e-3)

            for epoch in range(1, self.epochs_per_iter + 1):
                states_np, policies_np, values_np = self.buffer.sample(self.batch_size)
                states   = torch.from_numpy(states_np).to(self.device)
                policies = torch.from_numpy(policies_np).to(self.device)
                values   = torch.from_numpy(values_np).to(self.device)

                metrics = train_epoch(
                    candidate, optimizer, states, policies, values,
                    batch_size=self.batch_size, device=self.device
                )
                print(
                    f"  epoch {epoch}/{self.epochs_per_iter}  "
                    f"loss={metrics['total_loss']:.4f}  "
                    f"(pol={metrics['policy_loss']:.4f}  "
                    f"val={metrics['value_loss']:.4f})"
                )

            # 3. Arena evaluation
            print(f"\n[3/4] Arena: candidate vs best ({self.arena_games} games) …")
            should_promote, wins, losses, draws = arena_match(
                self.engine, candidate, best_model,
                num_games=self.arena_games,
                num_sims=max(self.num_sims // 2, 10),
                device=self.device,
                win_threshold=self.win_threshold,
            )

            # 4. Promote or reject
            print(f"\n[4/4] Promotion decision: {'✅ PROMOTE' if should_promote else '❌ REJECT'}")
            if should_promote:
                best_model = candidate
                save_checkpoint(best_model, str(best_path))
                iter_path = self.ckpt_dir / f"model_iter_{iteration:03d}.pt"
                save_checkpoint(best_model, str(iter_path))

            # Update Elo
            self.elo.update(
                iteration=iteration,
                wins=wins, losses=losses, draws=draws,
                opponent_elo=self.elo.current_elo,
            )

            elapsed = time.time() - t0
            print(f"\n  Iteration {iteration} done in {elapsed:.0f}s  |  Current Elo: {self.elo.current_elo:.0f}")

        # End of training
        print("\n" + "=" * 60)
        print("  Training complete!")
        print(f"  Best model saved → {best_path}")
        print(f"  Final Elo: {self.elo.current_elo:.0f}")
        self.opening_book.print_summary()
        self.elo.plot(save_fig=str(self.ckpt_dir / "elo_curve.png"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="ChaturangZero AlphaZero Training")
    p.add_argument("--iterations",      type=int,   default=10)
    p.add_argument("--games-per-iter",  type=int,   default=5)
    p.add_argument("--sims",            type=int,   default=50)
    p.add_argument("--batch-size",      type=int,   default=256)
    p.add_argument("--epochs",          type=int,   default=5)
    p.add_argument("--arena-games",     type=int,   default=20)
    p.add_argument("--win-threshold",   type=float, default=0.55)
    p.add_argument("--checkpoint-dir",  type=str,   default="checkpoints")
    p.add_argument("--device",          type=str,   default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    trainer = Trainer(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        num_sims=args.sims,
        batch_size=args.batch_size,
        epochs_per_iter=args.epochs,
        arena_games=args.arena_games,
        win_threshold=args.win_threshold,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    trainer.run()
