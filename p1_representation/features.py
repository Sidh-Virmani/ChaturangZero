"""
p1_representation/features.py
==============================
Strategic feature extraction for Chaturanga positions.

This module extracts ~20 human-interpretable board features from any
ChaturangaState. These features serve three purposes:

  1. Input to the MoveExplainer: tell humans *why* the AI prefers a move
  2. Input to the HardPositionBuffer: detect structurally complex positions
  3. Research: correlate learned values with measurable strategic concepts

Feature categories
------------------
  Material     : raw piece count difference
  Mobility     : total legal moves per side (piece activity)
  King Safety  : squares attacked around own/opponent Raja
  Pawn Structure: passed Padati, advanced Padati
  Piece-specific: Ratha control of open files, Gaja reachability, Ashva outpost
"""

from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, field

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.state  import ChaturangaState, Piece, PieceKind, Side, Square
from shared.game_core.rules  import pseudolegal_moves, is_square_attacked


# ── Piece values (centipawns) for material count ─────────────────────────────
_PIECE_VALUE = {
    PieceKind.RAJA:   0,     # King — not counted in material
    PieceKind.MANTRI: 150,   # Ferz (very weak)
    PieceKind.RATHA:  500,
    PieceKind.GAJA:   200,   # Elephant jump — medium value
    PieceKind.ASHVA:  300,
    PieceKind.PADATI: 100,
}

# Feature names (in order, for documentation / explainability)
FEATURE_NAMES = [
    "material_balance",          # 0
    "own_mobility",              # 1
    "opp_mobility",              # 2
    "mobility_ratio",            # 3
    "raja_safety_own",           # 4  (lower = less safe)
    "raja_safety_opp",           # 5
    "passed_padati_own",         # 6
    "passed_padati_opp",         # 7
    "padati_advancement_own",    # 8
    "padati_advancement_opp",    # 9
    "ratha_open_files_own",      # 10
    "ratha_open_files_opp",      # 11
    "gaja_coverage_own",         # 12
    "gaja_coverage_opp",         # 13
    "ashva_outpost_own",         # 14
    "ashva_outpost_opp",         # 15
    "raja_centrality_own",       # 16
    "raja_centrality_opp",       # 17
    "piece_count_own",           # 18
    "piece_count_opp",           # 19
]

NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class FeatureVector:
    """Named feature vector for one position from one side's perspective."""
    values:  np.ndarray        # shape (NUM_FEATURES,), dtype float32
    side:    Side
    names:   list[str] = field(default_factory=lambda: FEATURE_NAMES)

    def __getitem__(self, name: str) -> float:
        return float(self.values[self.names.index(name)])

    def top_features(self, k: int = 5) -> list[tuple[str, float]]:
        """Return the k most positive features (for explaining why a move is good)."""
        ranked = sorted(zip(self.names, self.values), key=lambda x: abs(x[1]), reverse=True)
        return ranked[:k]


class StrategicFeatureExtractor:
    """
    Extracts a fixed-size float32 feature vector from any ChaturangaState.

    All features are computed from White's perspective (positive = good for White).
    When called for Black, the sign is flipped at the end.
    """

    def extract(self, state: ChaturangaState, side: Side | None = None) -> FeatureVector:
        """
        Compute features for `side` (defaults to state.current_turn).
        Positive values = good for `side`.
        """
        if side is None:
            side = state.current_turn
        opp = side.opponent()

        values = np.zeros(NUM_FEATURES, dtype=np.float32)

        own_pieces = state.pieces_of(side)
        opp_pieces = state.pieces_of(opp)

        # 0 — material_balance (in centipawns, normalised to [-1, 1])
        own_mat = sum(_PIECE_VALUE[p.kind] for p in own_pieces)
        opp_mat = sum(_PIECE_VALUE[p.kind] for p in opp_pieces)
        max_mat  = 2 * (500 + 200 + 300 + 150 + 8 * 100)  # rough max
        values[0] = np.clip((own_mat - opp_mat) / max_mat, -1.0, 1.0)

        # 1, 2, 3 — mobility
        own_mob = sum(len(pseudolegal_moves(p, state)) for p in own_pieces)
        opp_mob = sum(len(pseudolegal_moves(p, state)) for p in opp_pieces)
        values[1] = np.clip(own_mob / 50.0, 0, 1)
        values[2] = np.clip(opp_mob / 50.0, 0, 1)
        values[3] = np.clip((own_mob - opp_mob) / 50.0, -1, 1)

        # 4, 5 — raja safety
        own_raja = next((p.square for p in own_pieces if p.kind == PieceKind.RAJA), None)
        opp_raja = next((p.square for p in opp_pieces if p.kind == PieceKind.RAJA), None)

        values[4] = -self._count_attacks_around(own_raja, opp, state) / 8.0 if own_raja else -1.0
        values[5] =  self._count_attacks_around(opp_raja, side, state) / 8.0 if opp_raja else 1.0

        # 6, 7 — passed Padati
        own_padati_files = {p.square.file for p in own_pieces if p.kind == PieceKind.PADATI}
        opp_padati_files = {p.square.file for p in opp_pieces if p.kind == PieceKind.PADATI}

        own_passed = sum(
            1 for p in own_pieces
            if p.kind == PieceKind.PADATI and p.square.file not in opp_padati_files
        )
        opp_passed = sum(
            1 for p in opp_pieces
            if p.kind == PieceKind.PADATI and p.square.file not in own_padati_files
        )
        values[6] = np.clip(own_passed / 4.0, 0, 1)
        values[7] = np.clip(opp_passed / 4.0, 0, 1)

        # 8, 9 — Padati advancement (avg rank, White advances toward rank 8)
        own_padati = [p for p in own_pieces if p.kind == PieceKind.PADATI]
        opp_padati = [p for p in opp_pieces if p.kind == PieceKind.PADATI]
        if own_padati:
            if side == Side.WHITE:
                values[8] = np.mean([p.square.rank / 7.0 for p in own_padati])
            else:
                values[8] = np.mean([(7 - p.square.rank) / 7.0 for p in own_padati])
        if opp_padati:
            if opp == Side.WHITE:
                values[9] = np.mean([p.square.rank / 7.0 for p in opp_padati])
            else:
                values[9] = np.mean([(7 - p.square.rank) / 7.0 for p in opp_padati])

        # 10, 11 — Ratha on open files (no Padati on same file)
        all_padati_files = own_padati_files | opp_padati_files
        own_rathas = [p for p in own_pieces if p.kind == PieceKind.RATHA]
        opp_rathas = [p for p in opp_pieces if p.kind == PieceKind.RATHA]
        values[10] = np.clip(
            sum(1 for r in own_rathas if r.square.file not in all_padati_files) / 2.0, 0, 1
        )
        values[11] = np.clip(
            sum(1 for r in opp_rathas if r.square.file not in all_padati_files) / 2.0, 0, 1
        )

        # 12, 13 — Gaja coverage (how many jump destinations are available)
        own_gaja = [p for p in own_pieces if p.kind == PieceKind.GAJA]
        opp_gaja = [p for p in opp_pieces if p.kind == PieceKind.GAJA]
        values[12] = np.clip(
            sum(len(pseudolegal_moves(g, state)) for g in own_gaja) / (4 * max(len(own_gaja), 1)), 0, 1
        )
        values[13] = np.clip(
            sum(len(pseudolegal_moves(g, state)) for g in opp_gaja) / (4 * max(len(opp_gaja), 1)), 0, 1
        )

        # 14, 15 — Ashva on outpost (square not attackable by enemy Padati)
        own_ashva = [p for p in own_pieces if p.kind == PieceKind.ASHVA]
        opp_ashva = [p for p in opp_pieces if p.kind == PieceKind.ASHVA]
        values[14] = np.clip(
            sum(1 for a in own_ashva if not self._attacked_by_padati(a.square, opp, state)) / max(len(own_ashva), 1), 0, 1
        )
        values[15] = np.clip(
            sum(1 for a in opp_ashva if not self._attacked_by_padati(a.square, side, state)) / max(len(opp_ashva), 1), 0, 1
        )

        # 16, 17 — Raja centrality (distance from centre d4/e4/d5/e5)
        center = (3.5, 3.5)
        if own_raja:
            dist = abs(own_raja.file - center[0]) + abs(own_raja.rank - center[1])
            values[16] = -np.clip(dist / 7.0, 0, 1)  # negative = far from centre = bad in endgame
        if opp_raja:
            dist = abs(opp_raja.file - center[0]) + abs(opp_raja.rank - center[1])
            values[17] = np.clip(dist / 7.0, 0, 1)   # positive = opponent far from centre = good

        # 18, 19 — piece counts
        values[18] = len(own_pieces) / 16.0
        values[19] = len(opp_pieces) / 16.0

        return FeatureVector(values=values, side=side)

    def extract_delta(
        self,
        state_before: ChaturangaState,
        state_after:  ChaturangaState,
        side:         Side,
    ) -> np.ndarray:
        """
        Compute feature delta between before/after a move.
        Positive values = the move improved that feature for `side`.
        """
        before = self.extract(state_before, side).values
        after  = self.extract(state_after, side).values
        return after - before

    def explain_delta(
        self,
        delta: np.ndarray,
        threshold: float = 0.05,
    ) -> list[str]:
        """
        Convert a feature delta into human-readable reasons.
        Returns list of reason strings (positive deltas = improvements).
        """
        reasons = []
        for i, name in enumerate(FEATURE_NAMES):
            d = float(delta[i])
            if abs(d) < threshold:
                continue
            direction = "improves" if d > 0 else "weakens"
            label = _FEATURE_LABELS.get(name, name.replace("_", " "))
            reasons.append(f"{direction} {label} ({d:+.2f})")
        # Sort: biggest improvements first
        reasons.sort(key=lambda r: -abs(float(r.split("(")[1].rstrip(")"))))
        return reasons[:5]

    # ── Private helpers ────────────────────────────────────────────────────────

    def _count_attacks_around(
        self,
        square: Square | None,
        attacker_side: Side,
        state: ChaturangaState,
    ) -> int:
        """Count how many squares adjacent to `square` are attacked by `attacker_side`."""
        if square is None:
            return 0
        count = 0
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                f, r = square.file + df, square.rank + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    if is_square_attacked(Square(f, r), attacker_side, state):
                        count += 1
        return count

    def _attacked_by_padati(
        self,
        square: Square,
        attacker_side: Side,
        state: ChaturangaState,
    ) -> bool:
        """Check if `square` is attacked by any Padati of `attacker_side`."""
        direction = -1 if attacker_side == Side.WHITE else 1
        for df in (-1, 1):
            f = square.file + df
            r = square.rank + direction
            if 0 <= f < 8 and 0 <= r < 8:
                candidate = Square(f, r)
                piece = next(
                    (p for p in state.pieces
                     if p.square == candidate and p.side == attacker_side and p.kind == PieceKind.PADATI),
                    None
                )
                if piece:
                    return True
        return False


# Human-readable labels for explainability output
_FEATURE_LABELS = {
    "material_balance":       "material balance",
    "own_mobility":           "piece activity",
    "opp_mobility":           "opponent activity",
    "mobility_ratio":         "mobility advantage",
    "raja_safety_own":        "Raja safety",
    "raja_safety_opp":        "opponent Raja exposure",
    "passed_padati_own":      "passed Padati count",
    "passed_padati_opp":      "opponent passed Padati",
    "padati_advancement_own": "Padati advancement",
    "padati_advancement_opp": "opponent Padati advancement",
    "ratha_open_files_own":   "Ratha on open files",
    "ratha_open_files_opp":   "opponent Ratha on open files",
    "gaja_coverage_own":      "Gaja jump coverage",
    "gaja_coverage_opp":      "opponent Gaja coverage",
    "ashva_outpost_own":      "Ashva on outpost",
    "ashva_outpost_opp":      "opponent Ashva outpost",
    "raja_centrality_own":    "Raja centralisation",
    "raja_centrality_opp":    "opponent Raja centralisation",
    "piece_count_own":        "own piece count",
    "piece_count_opp":        "opponent piece count",
}
