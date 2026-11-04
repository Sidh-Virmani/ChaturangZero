"""Equivalence and work-avoidance tests for Phase 1A."""

from dataclasses import replace

import numpy as np
import pytest

from benchmarks import fixed_positions
from p1_representation import ACTION_SPACE_SIZE, legal_action_mask, move_to_action
from p2_neural_network import Evaluation
from p3_mcts import MCTS, MCTSConfig, MCTSNode
from p3_mcts.node import PendingChild
from shared.game_core import (
    GameState, IllegalMoveError, Move, Piece, PieceType, Side,
    apply_move, initial_state, legal_moves, terminal_status,
)
from shared.game_core.core import _analyze_for_search, _apply_known_legal_move


class DeterministicEvaluator:
    def evaluate(self, state):
        policy = legal_action_mask(state).astype(np.float32)
        policy *= np.arange(1, ACTION_SPACE_SIZE + 1, dtype=np.float32)
        policy /= policy.sum()
        return Evaluation(policy, 0.25 if state.side_to_move is Side.WHITE else -0.4)


def test_trusted_transitions_equal_public_for_legal_moves():
    states = [item.state for item in fixed_positions()]
    states += [apply_move(state, legal_moves(state)[0]) for state in states.copy()]
    # Forward/capture promotion for both colours, alongside ordinary captures.
    promotion = GameState.from_mapping({
        'a1': Piece(Side.WHITE, PieceType.RAJA),
        'h8': Piece(Side.BLACK, PieceType.RAJA),
        'b7': Piece(Side.WHITE, PieceType.PADATI),
        'c8': Piece(Side.BLACK, PieceType.RATHA),
        'f2': Piece(Side.BLACK, PieceType.PADATI),
        'e1': Piece(Side.WHITE, PieceType.RATHA),
    })
    states += [promotion, replace(promotion, side_to_move=Side.BLACK)]
    for state in states:
        before = replace(state)
        moves, outcome = _analyze_for_search(state)
        assert moves == legal_moves(state)
        assert outcome == terminal_status(state)
        for move in moves:
            assert _apply_known_legal_move(state, move) == apply_move(state, move)
        assert state == before
    with pytest.raises(IllegalMoveError):
        apply_move(initial_state(), ('a2', 'a4'))


def test_expansion_defers_transitions_and_reuses_analysis(monkeypatch):
    import p3_mcts.node as module
    calls = []
    original = module._analyze_for_search
    def analyze(state):
        calls.append(state)
        return original(state)
    monkeypatch.setattr(module, '_analyze_for_search', analyze)
    root = MCTSNode(initial_state())
    root.expand([(move_to_action(m), m) for m in root.legal_moves],
                DeterministicEvaluator().evaluate(root.state).policy)
    assert len(calls) == 1
    assert all(isinstance(child, PendingChild) for child in root.children.values())
    action, child = root.select_best_child(1.5)
    assert action == min(root.children)  # Preserve zero-visit tie breaking.
    assert len(calls) == 2
    assert sum(isinstance(c, MCTSNode) for c in root.children.values()) == 1
    assert root.materialize_child(action) is child
    assert len(calls) == 2


def test_step_materializes_unvisited_action_and_rejects_illegal_action():
    search = MCTS(DeterministicEvaluator(), MCTSConfig(num_simulations=1, temperature=0))
    search.search(initial_state())
    root = search.root
    action = next(a for a, child in root.children.items() if isinstance(child, PendingChild))
    edge = root.children[action]
    search.step(action)
    assert search.root is root.children[action]
    assert search.root.parent is None
    assert search.root.state == apply_move(root.state, edge.move)
    assert search.root.prior == edge.prior
    with pytest.raises(IllegalMoveError):
        search.step(Move.from_names('a1', 'a1'))


@pytest.mark.parametrize('temperature,noise', [(0.0, False), (1.0, False), (1.0, True)])
def test_lazy_search_matches_eager_public_transition_reference(monkeypatch, temperature, noise):
    import p3_mcts.mcts as module

    class EagerReference(MCTSNode):
        def expand(self, actions, priors):
            super().expand(actions, priors)
            for action, move in actions:
                self.children[action] = EagerReference(
                    apply_move(self.state, move), parent=self, action_taken=action,
                    prior=self.children[action].prior,
                )

    def snapshot(node):
        return (node.visit_count, node.value_sum, tuple(
            (a, c.prior, c.visit_count, c.value_sum,
             snapshot(c) if isinstance(c, MCTSNode) and c.visit_count else None)
            for a, c in node.children.items()
        ))

    config = MCTSConfig(num_simulations=12, temperature=temperature, add_noise=noise)
    for position in fixed_positions():
        state = position.state
        np.random.seed(91)
        lazy = MCTS(DeterministicEvaluator(), config)
        result = lazy.search(state)
        expected_tree = snapshot(lazy.root)
        with monkeypatch.context() as patch:
            patch.setattr(module, 'MCTSNode', EagerReference)
            np.random.seed(91)
            eager = MCTS(DeterministicEvaluator(), config)
            reference = eager.search(state)
        np.testing.assert_array_equal(result[0], reference[0])
        assert result[1:] == reference[1:]
        assert result[1] in legal_moves(state)
        assert expected_tree == snapshot(eager.root)


@pytest.mark.parametrize('reason', ['bare_king', 'checkmate', 'stalemate'])
def test_terminal_analysis_does_not_evaluate_network(reason):
    pieces = {
        'a1': Piece(Side.WHITE, PieceType.RAJA),
        'h8': Piece(Side.BLACK, PieceType.RAJA),
        'h3': Piece(Side.BLACK, PieceType.MANTRI),
    }
    if reason != 'bare_king':
        pieces['h2'] = Piece(Side.WHITE, PieceType.PADATI)
        for square in (('a2', 'b2') if reason == 'checkmate' else ('b3', 'c2')):
            pieces[square] = Piece(Side.BLACK, PieceType.RATHA)
    state = GameState.from_mapping(pieces)
    moves, outcome = _analyze_for_search(state)
    assert moves == ()
    assert outcome.reason.value == reason
    class NeverEvaluate:
        def evaluate(self, state):
            pytest.fail('Terminal node must not evaluate the network')
    search = MCTS(NeverEvaluate())
    with pytest.raises(ValueError, match='terminal'):
        search.search(state)
    assert search.root.terminal_value == -1


def test_expansion_rejects_unknown_move_before_trusted_transition():
    root = MCTSNode(initial_state())
    illegal = Move.from_names('a2', 'a4')
    with pytest.raises(ValueError, match='core-legal'):
        root.expand([(move_to_action(illegal), illegal)], np.ones(ACTION_SPACE_SIZE))
