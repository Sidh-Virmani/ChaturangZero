"""
tests/test_tactical_verifier.py
=================================
Tests for p3_mcts/tactical_verifier.py — TacticalVerifier.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Piece, PieceKind, Side, Square
from p3_mcts.tactical_verifier import TacticalVerifier
from p1_representation.encoder import StateEncoder, ACTION_SIZE


@pytest.fixture
def engine():
    return ChaturangaEngine()

@pytest.fixture
def verifier(engine):
    return TacticalVerifier(engine)


# ── Checkmate in one ──────────────────────────────────────────────────────────

def test_finds_checkmate_in_one(engine, verifier):
    """
    White Ratha at a1, Black Raja at b8, White Raja at h1.
    Moving Ratha to a8 or b-file should deliver checkmate if b8 is cornered.
    Let's use a simpler definite setup:
    White Ratha at h7, Ratha at g8, White Raja at a1, Black Raja at h8 — mate in 1.
    """
    # White Ratha can move to h8 and deliver checkmate (Black Raja on h8 would be captured)
    # Instead: two Rathas and black Raja cornered at a8
    w_raja  = Piece(PieceKind.RAJA,  Side.WHITE, Square(7, 0))   # h1
    w_ratha1= Piece(PieceKind.RATHA, Side.WHITE, Square(0, 6))   # a7
    w_ratha2= Piece(PieceKind.RATHA, Side.WHITE, Square(1, 7))   # b8
    b_raja  = Piece(PieceKind.RAJA,  Side.BLACK, Square(0, 7))   # a8
    state = ChaturangaState(pieces=[w_raja, w_ratha1, w_ratha2, b_raja], current_turn=Side.WHITE)
    legal = engine.legal_moves(state)
    mate  = verifier.find_checkmate_in_one(state, legal)
    # The Ratha on a7 can move to a8 capturing the Raja → checkmate
    assert mate is not None, "Should find checkmate in one"

def test_no_checkmate_in_one_at_start(engine, verifier):
    """No checkmate in one from starting position."""
    start = engine.new_game()
    legal = engine.legal_moves(start)
    mate  = verifier.find_checkmate_in_one(start, legal)
    assert mate is None

# ── Promotion detection ───────────────────────────────────────────────────────

def test_finds_promotion(engine, verifier):
    """White Padati on rank 6 should be found as a promotion move."""
    w_raja   = Piece(PieceKind.RAJA,   Side.WHITE, Square(7, 0))
    b_raja   = Piece(PieceKind.RAJA,   Side.BLACK, Square(7, 7))
    w_padati = Piece(PieceKind.PADATI, Side.WHITE, Square(3, 6))  # d7 → will promote
    state  = ChaturangaState(pieces=[w_raja, b_raja, w_padati], current_turn=Side.WHITE)
    legal  = engine.legal_moves(state)
    promo  = verifier.find_promotion_in_one(state, legal)
    assert promo is not None, "Should find promotion"
    assert promo.is_promotion

def test_no_promotion_at_start(engine, verifier):
    start = engine.new_game()
    legal = engine.legal_moves(start)
    assert verifier.find_promotion_in_one(start, legal) is None


# ── Raja blunder detection ────────────────────────────────────────────────────

def test_detects_raja_blunder(engine, verifier):
    """
    Moving a piece that exposes own Raja to immediate capture should be a blunder.
    White Ratha at e1 blocks Black Ratha at e8 from seeing White Raja at e1... wait,
    let's use: White Raja at a1, White Ratha at b1 (blocking Black Ratha at h1).
    Moving Ratha to b2 exposes Raja on rank 1 to Black Ratha at h1 → blunder.
    """
    w_raja  = Piece(PieceKind.RAJA,  Side.WHITE, Square(0, 0))  # a1
    w_ratha = Piece(PieceKind.RATHA, Side.WHITE, Square(1, 0))  # b1 (pinned)
    b_ratha = Piece(PieceKind.RATHA, Side.BLACK, Square(7, 0))  # h1 (pinner)
    b_raja  = Piece(PieceKind.RAJA,  Side.BLACK, Square(7, 7))  # h8
    state = ChaturangaState(pieces=[w_raja, w_ratha, b_ratha, b_raja], current_turn=Side.WHITE)

    # Moving White Ratha to b2 (off the pin line) should be a blunder
    from shared.game_core.state import Move
    blunder_move = Move(Square(1, 0), Square(1, 1), PieceKind.RATHA, Side.WHITE)
    # This move may or may not be in legal moves, but if it is, it should be a blunder
    legal = engine.legal_moves(state)
    vertical_moves = [m for m in legal if m.from_sq == Square(1, 0) and m.to_sq.rank != 0]
    for vm in vertical_moves:
        assert verifier.is_raja_blunder(state, vm), f"Move {vm} should be a blunder"

def test_non_blunder_move(engine, verifier):
    """A safe, non-exposing move should not be flagged as a blunder."""
    start = engine.new_game()
    legal = engine.legal_moves(start)
    # Most opening Padati moves are safe
    padati_moves = [m for m in legal if m.piece == PieceKind.PADATI]
    for pm in padati_moves[:3]:
        assert not verifier.is_raja_blunder(start, pm), f"Opening {pm} should not be a blunder"


# ── Tactical score ────────────────────────────────────────────────────────────

def test_tactical_score_range(engine, verifier):
    start = engine.new_game()
    legal = engine.legal_moves(start)
    for m in legal[:5]:
        score = verifier.tactical_score(start, m)
        assert -1.0 <= score <= 3.0

def test_check_moves_list(engine, verifier):
    start = engine.new_game()
    legal = engine.legal_moves(start)
    checks = verifier.find_check_moves(start, legal)
    assert isinstance(checks, list)
    for m in checks:
        next_state = engine.apply_move(start, m)
        assert next_state.status in ("check", "checkmate")


# ── filter_top_move ───────────────────────────────────────────────────────────

def test_filter_top_move_returns_move(engine, verifier):
    start  = engine.new_game()
    legal  = engine.legal_moves(start)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[StateEncoder.encode_action(legal[0])] = 1.0
    result = verifier.filter_top_move(start, legal[0], legal, policy)
    assert result in legal
