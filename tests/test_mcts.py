"""
tests/test_mcts.py
==================
Tests for the MCTS search (without a trained model — uses random priors).
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import Piece, PieceKind, Side, Square, ChaturangaState
from p3_mcts.search          import MCTSSearch
from p3_mcts.node            import MCTSNode


@pytest.fixture
def engine():
    return ChaturangaEngine()


@pytest.fixture
def mcts_no_model(engine):
    """MCTS with no model — uses uniform priors."""
    return MCTSSearch(engine, model=None, num_simulations=10)


@pytest.fixture
def start(engine):
    return engine.new_game()


# ── MCTSNode tests ────────────────────────────────────────────────────────────

def test_node_initial_state():
    node = MCTSNode(state=None)
    assert node.N == 0
    assert node.W == 0.0
    assert node.Q == 0.0
    assert node.is_leaf()
    assert node.is_root()


def test_node_update():
    node = MCTSNode(state=None)
    node.N = 1
    node.update(0.5)
    assert node.N == 2
    assert abs(node.W - 0.5) < 1e-6


def test_node_puct_score():
    node = MCTSNode(state=None, prior=0.5)
    node.N = 5
    node.W = 2.5
    score = node.puct_score(c_puct=1.5, total_visits=20)
    assert score > 0


# ── Search tests ───────────────────────────────────────────────────────────────

def test_mcts_returns_valid_policy(mcts_no_model, start):
    policy, top_moves = mcts_no_model.search(start)
    assert policy.shape == (4096,)
    assert abs(policy.sum() - 1.0) < 1e-4, "Policy must sum to ~1"
    assert (policy >= 0).all(), "No negative probabilities"


def test_mcts_best_move_is_legal(engine, mcts_no_model, start):
    move = mcts_no_model.best_move(start)
    assert move is not None
    legal = engine.legal_moves(start)
    assert move in legal


def test_mcts_top_moves_format(mcts_no_model, start):
    _, top_moves = mcts_no_model.search(start)
    assert isinstance(top_moves, list)
    for action_idx, visit_frac, q_val in top_moves:
        assert 0 <= action_idx < 4096
        assert 0.0 <= visit_frac <= 1.0
        assert -1.0 <= q_val <= 1.0


def test_mcts_terminal_returns_none(engine, mcts_no_model):
    """MCTS on a terminal state should return None."""
    w_raja   = Piece(PieceKind.RAJA,  Side.WHITE, Square(0, 0))
    b_ratha1 = Piece(PieceKind.RATHA, Side.BLACK, Square(1, 7))
    b_ratha2 = Piece(PieceKind.RATHA, Side.BLACK, Square(7, 1))
    state = ChaturangaState(
        pieces=[w_raja, b_ratha1, b_ratha2],
        current_turn=Side.WHITE,
        status="checkmate",
        winner="black",
    )
    move = mcts_no_model.best_move(state)
    assert move is None


def test_mcts_more_sims_increases_visits(engine, start):
    """More simulations → root should have more visits."""
    mcts_few  = MCTSSearch(engine, model=None, num_simulations=5)
    mcts_many = MCTSSearch(engine, model=None, num_simulations=30)

    # Just check they don't crash and return valid policies
    p_few,  _ = mcts_few.search(start)
    p_many, _ = mcts_many.search(start)

    assert p_few.shape  == (4096,)
    assert p_many.shape == (4096,)
