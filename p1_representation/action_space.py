"""
p1_representation/action_space.py
===================================
Compact Chaturanga Action Space (Research Contribution #2).

Problem with 4096 flat logits
------------------------------
The baseline uses action_index = from_sq * 64 + to_sq (4096 total).
Most of these 4096 slots are IMPOSSIBLE — e.g. a Ratha can never jump.
Training 4096 independent logits with sparse signal is sample-inefficient.

Compact representation
-----------------------
We define piece-specific movement templates (direction + distance):

  Ratha  : 4 directions × up to 7 squares = 28 templates
  Ashva  : 8 L-shape offsets = 8 templates
  Gaja   : 4 diagonal jumps (exactly 2 squares) = 4 templates
  Raja   : 8 directions × 1 square = 8 templates
  Mantri : 4 diagonals × 1 square = 4 templates
  Padati : 1 forward + 2 diagonal captures = 3 templates

Each template is a (Δfile, Δrank) offset (or list for sliding pieces).
The compact index = piece_kind * MAX_TEMPLATES_PER_PIECE + template_id + from_sq * (sum templates)

However, for compatibility with the existing 4096 encoder, we provide:
  - encode_move_compact(move) → compact int
  - decode_compact(idx) → (from_sq, to_sq) tuple
  - COMPACT_ACTION_SIZE (total valid slots)

The compact space is used as a drop-in replacement for action_size in model.py.
The existing encoder.py remains unchanged (backward-compatible).
"""

from __future__ import annotations

import sys
import pathlib
from typing import Iterator

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.state import Move, PieceKind, Side, Square


# ── Movement template definitions ────────────────────────────────────────────

# Each entry: (piece_kind, list of (delta_file, delta_rank))
# For sliding pieces (Ratha), each direction generates up to 7 steps.

_RATHA_DIRS   = [(1,0),(-1,0),(0,1),(0,-1)]         # 4 dir × 7 dist = 28
_ASHVA_JUMPS  = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]  # 8
_GAJA_JUMPS   = [(2,2),(2,-2),(-2,2),(-2,-2)]         # exactly 4 diagonal jumps
_RAJA_DIRS    = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]     # 8
_MANTRI_DIRS  = [(1,1),(1,-1),(-1,1),(-1,-1)]          # 4 diagonals × 1
_PADATI_WHITE = [(0,1),(1,1),(-1,1)]                   # forward + 2 diagonal captures
_PADATI_BLACK = [(0,-1),(1,-1),(-1,-1)]


def _all_ratha_templates() -> list[tuple[int,int]]:
    templates = []
    for df, dr in _RATHA_DIRS:
        for dist in range(1, 8):
            templates.append((df * dist, dr * dist))
    return templates


# Pre-compute all (df, dr) offsets per piece kind
_TEMPLATES: dict[PieceKind, list[tuple[int,int]]] = {
    PieceKind.RATHA:  _all_ratha_templates(),   # 28
    PieceKind.ASHVA:  _ASHVA_JUMPS,             # 8
    PieceKind.GAJA:   _GAJA_JUMPS,              # 4
    PieceKind.RAJA:   _RAJA_DIRS,               # 8
    PieceKind.MANTRI: _MANTRI_DIRS,             # 4
    PieceKind.PADATI: _PADATI_WHITE,            # 3 (white); black handled separately
}

# Template counts per piece (for offset computation)
_TEMPLATE_COUNT = {k: len(v) for k, v in _TEMPLATES.items()}

# Cumulative offsets per piece kind for flattening
_PIECE_KINDS_ORDER = [
    PieceKind.RATHA, PieceKind.ASHVA, PieceKind.GAJA,
    PieceKind.RAJA, PieceKind.MANTRI, PieceKind.PADATI,
]
_MAX_TEMPLATES = max(_TEMPLATE_COUNT.values())   # 28
# Total compact slots: 64 squares × max_templates × 6 piece kinds
# We flatten as: kind_offset + from_sq * max_templates + template_id
_KIND_OFFSETS: dict[PieceKind, int] = {}
_offset = 0
for _k in _PIECE_KINDS_ORDER:
    _KIND_OFFSETS[_k] = _offset
    _offset += 64 * _MAX_TEMPLATES
COMPACT_ACTION_SIZE: int = _offset  # total compact slots


# ── Encoding / decoding ───────────────────────────────────────────────────────

def encode_move_compact(move: Move) -> int:
    """
    Encode a Move into a compact action index.

    Index = kind_offset + from_sq_idx * MAX_TEMPLATES + template_idx
    """
    kind     = move.piece
    templates = _TEMPLATES[kind] if kind != PieceKind.PADATI else (
        _PADATI_WHITE if move.side == Side.WHITE else _PADATI_BLACK
    )
    from_sq_idx = move.from_sq.file * 8 + move.from_sq.rank
    df = move.to_sq.file - move.from_sq.file
    dr = move.to_sq.rank - move.from_sq.rank

    try:
        t_idx = templates.index((df, dr))
    except ValueError:
        # Fallback: use raw from/to encoding (shouldn't happen with legal moves)
        return move.from_sq.file * 8 * 64 + move.from_sq.rank * 64 + move.to_sq.file * 8 + move.to_sq.rank

    kind_offset = _KIND_OFFSETS.get(kind, 0)
    return kind_offset + from_sq_idx * _MAX_TEMPLATES + t_idx


def decode_compact(idx: int) -> tuple[int, int]:
    """
    Decode a compact action index back to (from_sq_idx, to_sq_idx).
    Returns (-1, -1) if decoding fails.
    """
    for kind in _PIECE_KINDS_ORDER:
        kind_offset = _KIND_OFFSETS[kind]
        kind_slots  = 64 * _MAX_TEMPLATES
        if idx < kind_offset + kind_slots:
            local_idx    = idx - kind_offset
            from_sq_idx  = local_idx // _MAX_TEMPLATES
            t_idx        = local_idx % _MAX_TEMPLATES
            templates    = _TEMPLATES[kind]
            if t_idx >= len(templates):
                return (-1, -1)
            df, dr       = templates[t_idx]
            from_file    = from_sq_idx // 8
            from_rank    = from_sq_idx % 8
            to_file      = from_file + df
            to_rank      = from_rank + dr
            if not (0 <= to_file < 8 and 0 <= to_rank < 8):
                return (-1, -1)
            to_sq_idx = to_file * 8 + to_rank
            return (from_sq_idx, to_sq_idx)
    return (-1, -1)


def legal_action_mask_compact(
    legal_moves: list[Move],
) -> np.ndarray:
    """
    Build a boolean mask of shape (COMPACT_ACTION_SIZE,) for legal moves.
    """
    mask = np.zeros(COMPACT_ACTION_SIZE, dtype=bool)
    for m in legal_moves:
        idx = encode_move_compact(m)
        if 0 <= idx < COMPACT_ACTION_SIZE:
            mask[idx] = True
    return mask


class ChaturangaActionSpace:
    """
    Compact action space for ChaturangZero-X.

    Provides encode / decode / mask methods compatible with the
    ChaturangaNet when action_size=COMPACT_ACTION_SIZE.

    Usage
    -----
        space = ChaturangaActionSpace()
        idx   = space.encode(move)
        mask  = space.legal_mask(legal_moves)   # np.ndarray bool
        model = ChaturangaNet(action_size=space.size)
    """

    size: int = COMPACT_ACTION_SIZE

    def encode(self, move: Move) -> int:
        return encode_move_compact(move)

    def decode(self, idx: int) -> tuple[int, int]:
        return decode_compact(idx)

    def legal_mask(self, legal_moves: list[Move]) -> np.ndarray:
        return legal_action_mask_compact(legal_moves)

    def __repr__(self) -> str:
        return (
            f"ChaturangaActionSpace(size={self.size}, "
            f"vs_flat=4096, reduction={4096 - self.size} slots)"
        )
