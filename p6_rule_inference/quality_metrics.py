"""
p6_rule_inference/quality_metrics.py
=====================================
Game Quality Metrics (Contribution D — foundation layer).

Defines what makes a "good" Chaturanga game, measured automatically
from self-play game records.

Metrics
-------
  decisiveness       : fraction of games that end in a decisive result (not draw)
  game_length        : mean number of plies per game (proxy for depth of play)
  length_variance    : low variance = consistently interesting games
  tactical_complexity: mean number of moves that change material balance ≥ 1 piece
  material_volatility: std of material balance across all positions in a game
  blunder_frequency  : fraction of positions where a blunder (±2+ piece swing) occurred
  raja_safety_tension: mean proximity of Rajas to danger (attack count)
  opening_diversity  : number of unique first-5-ply sequences (exploration breadth)

Combined quality score
----------------------
  Q = w1*decisiveness + w2*length_score + w3*tactical_complexity
      + w4*material_volatility - w5*blunder_frequency

where length_score = 1 - |mean_length - TARGET_LENGTH| / TARGET_LENGTH
(penalizes both too-short and too-long games)
"""

from __future__ import annotations

import sys
import pathlib
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, PieceKind, Side
from shared.game_core.rule_variants import RuleConfig, RuleVariant


# Target game length in plies (balanced game — not too short, not a draw grind)
TARGET_LENGTH = 60

# Default quality weights
DEFAULT_WEIGHTS = {
    "decisiveness":        0.30,
    "length_score":        0.20,
    "tactical_complexity": 0.25,
    "material_volatility": 0.10,
    "blunder_penalty":    -0.15,
}

_PIECE_VALUE = {
    PieceKind.RAJA:   0,
    PieceKind.MANTRI: 2,
    PieceKind.RATHA:  5,
    PieceKind.GAJA:   2,
    PieceKind.ASHVA:  3,
    PieceKind.PADATI: 1,
}


@dataclass
class GameRecord:
    """Compact record of one self-play game."""
    plies:           int
    decisive:        bool                  # True if not draw
    winner:          str                   # "white", "black", "draw"
    material_by_ply: list[float]           # material balance at each ply
    capture_plies:   list[int]             # plies where a capture occurred
    check_plies:     list[int]             # plies that resulted in check
    opening_key:     str                   # first 5 moves as string (diversity)


@dataclass
class QualityReport:
    """Quality metrics for a set of games played under one rule config."""
    rule_variant:        str
    n_games:             int
    decisiveness:        float   # 0–1
    mean_length:         float   # plies
    length_variance:     float
    length_score:        float   # 0–1 (penalty for extreme lengths)
    tactical_complexity: float   # 0–1
    material_volatility: float   # 0–1
    blunder_frequency:   float   # 0–1
    opening_diversity:   float   # 0–1
    quality_score:       float   # composite 0–1
    weights:             dict    = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    def summary(self) -> str:
        return (
            f"\n{'═'*50}\n"
            f"  Rule variant : {self.rule_variant}\n"
            f"  Games played : {self.n_games}\n"
            f"  ─────────────────────────────────────\n"
            f"  Decisiveness      : {self.decisiveness:.3f}\n"
            f"  Mean game length  : {self.mean_length:.1f} plies\n"
            f"  Length score      : {self.length_score:.3f}\n"
            f"  Tactical complexity: {self.tactical_complexity:.3f}\n"
            f"  Material volatility: {self.material_volatility:.3f}\n"
            f"  Blunder frequency : {self.blunder_frequency:.3f}\n"
            f"  Opening diversity : {self.opening_diversity:.3f}\n"
            f"  ─────────────────────────────────────\n"
            f"  ★ QUALITY SCORE   : {self.quality_score:.4f}\n"
            f"{'═'*50}\n"
        )

    def to_dict(self) -> dict:
        return {
            "ruleVariant":       self.rule_variant,
            "nGames":            self.n_games,
            "decisiveness":      round(self.decisiveness, 4),
            "meanLength":        round(self.mean_length, 2),
            "lengthScore":       round(self.length_score, 4),
            "tacticalComplexity":round(self.tactical_complexity, 4),
            "materialVolatility":round(self.material_volatility, 4),
            "blunderFrequency":  round(self.blunder_frequency, 4),
            "openingDiversity":  round(self.opening_diversity, 4),
            "qualityScore":      round(self.quality_score, 4),
        }


def _material_balance(state: ChaturangaState) -> float:
    """White material minus Black material in piece-value units."""
    total = 0.0
    for p in state.pieces:
        val = _PIECE_VALUE.get(p.kind, 0)
        total += val if p.side == Side.WHITE else -val
    return total


def play_game_for_quality(
    engine: ChaturangaEngine,
    model=None,
    max_plies: int = 250,
    num_sims: int = 10,
    device: str = "cpu",
) -> GameRecord:
    """
    Play one game and record quality-relevant statistics.
    Uses MCTS if model is provided, else random moves.
    """
    from shared.game_core.state import Move
    from p1_representation.encoder import StateEncoder

    state  = engine.new_game()
    encoder = StateEncoder()

    material_by_ply: list[float] = []
    capture_plies:   list[int]   = []
    check_plies:     list[int]   = []
    opening_moves:   list[str]   = []

    if model is not None:
        from p3_mcts.search import MCTSSearch
        mcts = MCTSSearch(engine, model=model, num_simulations=num_sims,
                          temperature=1.0, device=device)

    ply = 0
    while not engine.is_terminal(state) and ply < max_plies:
        material_by_ply.append(_material_balance(state))
        legal = engine.legal_moves(state)
        if not legal:
            break

        # Choose move
        if model is not None:
            policy, _ = mcts.search(state)
            legal_idx = {StateEncoder.encode_action(m): m for m in legal}
            best_idx  = max(legal_idx, key=lambda i: policy[i])
            move      = legal_idx[best_idx]
        else:
            move = random.choice(legal)

        # Record capture
        if move.captured is not None:
            capture_plies.append(ply)

        # Record opening (first 5 moves)
        if ply < 10:
            opening_moves.append(move.label())

        state = engine.apply_move(state, move)
        ply  += 1

        # Record check
        if state.status == "check":
            check_plies.append(ply)

    winner   = state.winner if engine.is_terminal(state) else "draw"
    decisive = winner in ("white", "black")

    return GameRecord(
        plies=ply,
        decisive=decisive,
        winner=winner,
        material_by_ply=material_by_ply,
        capture_plies=capture_plies,
        check_plies=check_plies,
        opening_key="|".join(opening_moves),
    )


def compute_quality_report(
    rule_config:   RuleConfig,
    n_games:       int   = 30,
    model          = None,
    num_sims:      int   = 10,
    device:        str   = "cpu",
    weights:       dict  | None = None,
    verbose:       bool  = True,
) -> QualityReport:
    """
    Play `n_games` under `rule_config` and compute the quality report.

    Parameters
    ----------
    rule_config : which historical rule variant to evaluate
    n_games     : number of self-play games
    model       : ChaturangaNet (None = random play)
    weights     : override default quality weights
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    engine  = ChaturangaEngine(rule_config=rule_config)
    records: list[GameRecord] = []

    for i in range(n_games):
        rec = play_game_for_quality(engine, model, num_sims=num_sims, device=device)
        records.append(rec)
        if verbose:
            print(f"  [{rule_config.variant.value}] game {i+1}/{n_games}: "
                  f"plies={rec.plies} winner={rec.winner}")

    # ── Compute metrics ────────────────────────────────────────────────────────
    decisive_count = sum(1 for r in records if r.decisive)
    decisiveness   = decisive_count / max(n_games, 1)

    lengths        = [r.plies for r in records]
    mean_length    = float(np.mean(lengths))
    length_variance= float(np.var(lengths))
    length_score   = max(0.0, 1.0 - abs(mean_length - TARGET_LENGTH) / TARGET_LENGTH)

    # Tactical complexity: captures per ply (normalised)
    capture_rates  = [len(r.capture_plies) / max(r.plies, 1) for r in records]
    tactical_complexity = float(np.mean(capture_rates))
    tactical_complexity = min(1.0, tactical_complexity * 5)   # scale to ~0-1

    # Material volatility: std of material balance within each game, then averaged
    volatilities = []
    for r in records:
        if len(r.material_by_ply) > 1:
            volatilities.append(float(np.std(r.material_by_ply)))
    material_volatility = float(np.mean(volatilities)) / 8.0 if volatilities else 0.0
    material_volatility = min(1.0, material_volatility)

    # Blunder frequency: large material swings (≥ 3 units) in a single ply
    blunder_counts = []
    for r in records:
        m = r.material_by_ply
        blunders = sum(
            1 for i in range(1, len(m)) if abs(m[i] - m[i-1]) >= 3
        )
        blunder_counts.append(blunders / max(r.plies, 1))
    blunder_frequency = min(1.0, float(np.mean(blunder_counts)) * 10)

    # Opening diversity: unique opening keys / n_games
    unique_openings  = len(set(r.opening_key for r in records))
    opening_diversity = unique_openings / max(n_games, 1)

    # ── Composite quality score ────────────────────────────────────────────────
    quality_score = (
          weights.get("decisiveness",        0.30) * decisiveness
        + weights.get("length_score",        0.20) * length_score
        + weights.get("tactical_complexity", 0.25) * tactical_complexity
        + weights.get("material_volatility", 0.10) * material_volatility
        + weights.get("blunder_penalty",    -0.15) * blunder_frequency
    )
    quality_score = float(np.clip(quality_score, 0.0, 1.0))

    report = QualityReport(
        rule_variant=rule_config.variant.value,
        n_games=n_games,
        decisiveness=decisiveness,
        mean_length=mean_length,
        length_variance=length_variance,
        length_score=length_score,
        tactical_complexity=tactical_complexity,
        material_volatility=material_volatility,
        blunder_frequency=blunder_frequency,
        opening_diversity=opening_diversity,
        quality_score=quality_score,
        weights=weights,
    )

    if verbose:
        print(report.summary())

    return report


def rank_variants(
    reports: list[QualityReport],
) -> list[tuple[str, float]]:
    """Rank rule variants by quality score, highest first."""
    ranked = sorted(reports, key=lambda r: r.quality_score, reverse=True)
    return [(r.rule_variant, r.quality_score) for r in ranked]
