"""
tests/test_features.py
========================
Tests for p1_representation/features.py — StrategicFeatureExtractor.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Piece, PieceKind, Side, Square
from p1_representation.features import (
    StrategicFeatureExtractor, FEATURE_NAMES, NUM_FEATURES, FeatureVector
)


@pytest.fixture
def engine():
    return ChaturangaEngine()

@pytest.fixture
def start(engine):
    return engine.new_game()

@pytest.fixture
def extractor():
    return StrategicFeatureExtractor()


# ── Shape / type tests ────────────────────────────────────────────────────────

def test_feature_vector_shape(extractor, start):
    fv = extractor.extract(start)
    assert fv.values.shape == (NUM_FEATURES,), f"Expected ({NUM_FEATURES},), got {fv.values.shape}"

def test_feature_dtype(extractor, start):
    fv = extractor.extract(start)
    assert fv.values.dtype == np.float32

def test_feature_names_count():
    assert len(FEATURE_NAMES) == NUM_FEATURES

def test_feature_side_assignment(extractor, start):
    fv = extractor.extract(start, side=Side.WHITE)
    assert fv.side == Side.WHITE

def test_feature_getitem(extractor, start):
    fv = extractor.extract(start)
    val = fv["material_balance"]
    assert isinstance(val, float)

def test_top_features_length(extractor, start):
    fv = extractor.extract(start)
    top = fv.top_features(3)
    assert len(top) == 3


# ── Semantic correctness ──────────────────────────────────────────────────────

def test_material_balance_equal_start(extractor, start):
    """Initial position has equal material."""
    fv = extractor.extract(start, side=Side.WHITE)
    # Should be near 0
    assert abs(fv["material_balance"]) < 0.05

def test_mobility_positive(extractor, start):
    """Both sides have some mobility at the start."""
    fv = extractor.extract(start, side=Side.WHITE)
    assert fv["own_mobility"] > 0
    assert fv["opp_mobility"] > 0

def test_piece_count_equal_start(extractor, start):
    fv = extractor.extract(start)
    assert abs(fv["piece_count_own"] - fv["piece_count_opp"]) < 0.01

def test_ratha_not_on_open_file_at_start(extractor, start):
    """Rathas are blocked by Padati at start — not on open files."""
    fv = extractor.extract(start, side=Side.WHITE)
    assert fv["ratha_open_files_own"] == 0.0

def test_feature_delta_capture_improves_material(extractor, engine):
    """Capturing an enemy piece should improve material_balance feature."""
    # White Ratha at a1, Black Padati at a5, White Raja at h1, Black Raja at h8
    w_raja  = Piece(PieceKind.RAJA,   Side.WHITE, Square(7, 0))
    b_raja  = Piece(PieceKind.RAJA,   Side.BLACK, Square(7, 7))
    w_ratha = Piece(PieceKind.RATHA,  Side.WHITE, Square(0, 0))
    b_padati= Piece(PieceKind.PADATI, Side.BLACK, Square(0, 4))
    before  = ChaturangaState(pieces=[w_raja, b_raja, w_ratha, b_padati], current_turn=Side.WHITE)

    legal   = engine.legal_moves(before)
    # Find the capturing move
    captures = [m for m in legal if m.captured is not None]
    if captures:
        move   = captures[0]
        after  = engine.apply_move(before, move)
        delta  = extractor.extract_delta(before, after, Side.WHITE)
        # Material balance should improve for white
        mat_idx = FEATURE_NAMES.index("material_balance")
        assert delta[mat_idx] > 0, "Capture should improve material balance"

def test_explain_delta_returns_strings(extractor, start, engine):
    """explain_delta should return a list of strings."""
    fv_before = extractor.extract(start)
    moves = engine.legal_moves(start)
    if moves:
        after  = engine.apply_move(start, moves[0])
        delta  = extractor.extract_delta(start, after, Side.WHITE)
        reasons = extractor.explain_delta(delta, threshold=0.0)
        for r in reasons:
            assert isinstance(r, str)
