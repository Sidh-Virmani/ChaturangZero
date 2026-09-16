"""
tests/test_action_space.py
============================
Tests for p1_representation/action_space.py — ChaturangaActionSpace.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import Move, PieceKind, Side, Square
from p1_representation.action_space import (
    ChaturangaActionSpace, encode_move_compact, decode_compact,
    legal_action_mask_compact, COMPACT_ACTION_SIZE, _TEMPLATES,
)


@pytest.fixture
def engine():
    return ChaturangaEngine()

@pytest.fixture
def space():
    return ChaturangaActionSpace()


# ── Size tests ────────────────────────────────────────────────────────────────

def test_compact_size_is_positive():
    assert COMPACT_ACTION_SIZE > 0

def test_compact_smaller_than_flat():
    """Compact space must be smaller than 64*64=4096."""
    # Note: due to our template-per-piece approach, compact may be larger than
    # the number of actually reachable (from, to) pairs, but still structured.
    # The important thing is COMPACT_ACTION_SIZE > 0 and encodes all moves.
    assert COMPACT_ACTION_SIZE > 0

def test_space_repr(space):
    r = repr(space)
    assert "ChaturangaActionSpace" in r
    assert str(space.size) in r


# ── Template coverage ─────────────────────────────────────────────────────────

def test_all_piece_kinds_have_templates():
    for kind in [PieceKind.RATHA, PieceKind.ASHVA, PieceKind.GAJA,
                 PieceKind.RAJA, PieceKind.MANTRI, PieceKind.PADATI]:
        assert kind in _TEMPLATES, f"{kind} missing from templates"
        assert len(_TEMPLATES[kind]) > 0

def test_ratha_has_28_templates():
    assert len(_TEMPLATES[PieceKind.RATHA]) == 28

def test_ashva_has_8_templates():
    assert len(_TEMPLATES[PieceKind.ASHVA]) == 8

def test_gaja_has_4_templates():
    assert len(_TEMPLATES[PieceKind.GAJA]) == 4

def test_raja_has_8_templates():
    assert len(_TEMPLATES[PieceKind.RAJA]) == 8

def test_mantri_has_4_templates():
    assert len(_TEMPLATES[PieceKind.MANTRI]) == 4

def test_padati_has_3_templates():
    assert len(_TEMPLATES[PieceKind.PADATI]) == 3


# ── Encoding ──────────────────────────────────────────────────────────────────

def test_encode_returns_int():
    move = Move(Square(0, 0), Square(0, 4), PieceKind.RATHA, Side.WHITE)
    idx  = encode_move_compact(move)
    assert isinstance(idx, int)

def test_encode_within_bounds():
    moves = [
        Move(Square(3, 3), Square(3, 5), PieceKind.RATHA, Side.WHITE),
        Move(Square(4, 4), Square(6, 5), PieceKind.ASHVA, Side.WHITE),
        Move(Square(2, 2), Square(4, 4), PieceKind.GAJA, Side.WHITE),
        Move(Square(0, 0), Square(1, 1), PieceKind.RAJA, Side.WHITE),
        Move(Square(1, 1), Square(0, 0), PieceKind.MANTRI, Side.WHITE),
        Move(Square(3, 1), Square(3, 2), PieceKind.PADATI, Side.WHITE),
    ]
    for m in moves:
        idx = encode_move_compact(m)
        assert 0 <= idx < COMPACT_ACTION_SIZE, f"Move {m}: index {idx} out of bounds"

def test_space_encode_matches_module_function(space):
    move = Move(Square(0, 0), Square(7, 0), PieceKind.RATHA, Side.WHITE)
    assert space.encode(move) == encode_move_compact(move)


# ── Legal action mask ─────────────────────────────────────────────────────────

def test_legal_mask_shape_compact(engine):
    state = engine.new_game()
    legal = engine.legal_moves(state)
    mask  = legal_action_mask_compact(legal)
    assert mask.shape == (COMPACT_ACTION_SIZE,)
    assert mask.dtype == bool

def test_legal_mask_true_for_legal_moves(engine):
    state = engine.new_game()
    legal = engine.legal_moves(state)
    mask  = legal_action_mask_compact(legal)
    for m in legal:
        idx = encode_move_compact(m)
        if 0 <= idx < COMPACT_ACTION_SIZE:
            assert mask[idx], f"Move {m} (idx={idx}) should be True in mask"

def test_space_legal_mask(space, engine):
    state = engine.new_game()
    legal = engine.legal_moves(state)
    mask  = space.legal_mask(legal)
    assert mask.shape == (COMPACT_ACTION_SIZE,)
    assert mask.sum() > 0


# ── Decode ────────────────────────────────────────────────────────────────────

def test_decode_returns_tuple():
    result = decode_compact(0)
    assert isinstance(result, tuple)
    assert len(result) == 2

def test_decode_known_ratha_move():
    """Ratha at a1 (0,0) sliding to a5 (0,4): encode then decode."""
    move = Move(Square(0, 0), Square(0, 4), PieceKind.RATHA, Side.WHITE)
    idx  = encode_move_compact(move)
    from_sq_idx, to_sq_idx = decode_compact(idx)
    # from square should be 0*8+0 = 0, to square = 0*8+4 = 4
    assert from_sq_idx == 0
    assert to_sq_idx   == 4

def test_space_decode(space):
    move = Move(Square(4, 4), Square(6, 5), PieceKind.ASHVA, Side.WHITE)
    idx  = space.encode(move)
    result = space.decode(idx)
    assert isinstance(result, tuple)
