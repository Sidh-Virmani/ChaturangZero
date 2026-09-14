"""
tests/test_representation.py
=============================
Tests for the P1 encoder (state/action encoding).
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from shared.game_core.engine  import ChaturangaEngine
from shared.game_core.state   import Piece, PieceKind, Side, Square, ChaturangaState
from p1_representation.encoder import StateEncoder, ACTION_SIZE, NUM_CHANNELS


@pytest.fixture
def encoder():
    return StateEncoder()


@pytest.fixture
def start():
    return ChaturangaEngine().new_game()


# ── Encoding shape / dtype ────────────────────────────────────────────────────

def test_encode_state_shape(encoder, start):
    tensor = encoder.encode_state(start)
    assert tensor.shape == (NUM_CHANNELS, 8, 8)
    assert tensor.dtype == np.float32


def test_encode_state_binary(encoder, start):
    """All values in piece channels should be 0 or 1."""
    tensor = encoder.encode_state(start)
    assert set(tensor[:12].ravel().astype(int)).issubset({0, 1})


def test_turn_channel_white(encoder, start):
    tensor = encoder.encode_state(start)
    assert (tensor[12] == 1.0).all(), "White's turn → turn channel all ones"


def test_turn_channel_black(encoder):
    state = ChaturangaState(current_turn=Side.BLACK)
    tensor = encoder.encode_state(state)
    assert (tensor[12] == 0.0).all(), "Black's turn → turn channel all zeros"


def test_white_pieces_in_correct_channels(encoder, start):
    """White's pieces must be in channels 0-5, not 6-11."""
    tensor = encoder.encode_state(start)
    white_total = tensor[:6].sum()
    assert white_total == 16.0, "16 white pieces"
    assert tensor[6:12].sum() == 16.0, "16 black pieces"


# ── Action encoding ───────────────────────────────────────────────────────────

def test_encode_action_range():
    """All actions must be in [0, 4095]."""
    from shared.game_core.state import Move
    for f in range(8):
        for r in range(8):
            for tf in range(8):
                for tr in range(8):
                    move = Move(Square(f, r), Square(tf, tr), PieceKind.RAJA, Side.WHITE)
                    idx  = StateEncoder.encode_action(move)
                    assert 0 <= idx < ACTION_SIZE


def test_decode_action_roundtrip():
    """encode → decode should be identity for all square pairs."""
    from shared.game_core.state import Move
    for f in range(8):
        for r in range(8):
            for tf in range(8):
                for tr in range(8):
                    move     = Move(Square(f, r), Square(tf, tr), PieceKind.RAJA, Side.WHITE)
                    idx      = StateEncoder.encode_action(move)
                    from_sq, to_sq = StateEncoder.decode_action(idx)
                    assert from_sq == Square(f, r)
                    assert to_sq   == Square(tf, tr)


# ── Legal action mask ─────────────────────────────────────────────────────────

def test_legal_mask_shape(encoder, start):
    engine = ChaturangaEngine()
    legal  = engine.legal_moves(start)
    mask   = encoder.legal_action_mask(start, legal)
    assert mask.shape == (ACTION_SIZE,)
    assert mask.dtype == bool


def test_legal_mask_count(encoder, start):
    engine = ChaturangaEngine()
    legal  = engine.legal_moves(start)
    mask   = encoder.legal_action_mask(start, legal)
    assert mask.sum() == len(legal)


# ── Canonical encoding ────────────────────────────────────────────────────────

def test_canonical_white_same_as_normal(encoder, start):
    normal    = encoder.encode_state(start)
    canonical = encoder.encode_canonical(start)
    np.testing.assert_array_equal(normal, canonical)


def test_canonical_black_flips_ranks(encoder):
    """In canonical black encoding, ranks should be flipped."""
    state = ChaturangaState(
        pieces=[Piece(PieceKind.RAJA, Side.BLACK, Square(4, 7))],
        current_turn=Side.BLACK,
    )
    canonical = encoder.encode_canonical(state)
    # Black Raja at (4, 7) → becomes own piece at (4, 0) after flip
    # Channel 0 is own Raja (black maps to channel 0 when it's black's turn)
    assert canonical[0, 4, 0] == 1.0


# ── Augmentation ──────────────────────────────────────────────────────────────

def test_augment_returns_two_examples(encoder, start):
    enc = encoder.encode_state(start)
    aug = encoder.augment_example(enc, action_index=100, value=1.0)
    assert len(aug) == 2


def test_mirror_changes_file_axis(encoder, start):
    enc      = encoder.encode_state(start)
    mirrored = encoder.mirror_horizontal(enc)
    # The mirror of column 0 should equal original column 7
    np.testing.assert_array_equal(mirrored[:, 0, :], enc[:, 7, :])
