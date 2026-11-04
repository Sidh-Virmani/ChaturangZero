"""Output equivalence and legal-generation counts for Phase 1B."""
import numpy as np
import torch

from benchmarks import fixed_positions
from p1_representation import legal_action_mask
from p2_neural_network import ModelConfig, NetworkEvaluator, PolicyValueNetwork
from p3_mcts import MCTS, MCTSConfig
from shared.game_core import GameState, Piece, PieceType, Side, initial_state, legal_moves


def evaluator():
    torch.manual_seed(17)
    return NetworkEvaluator(PolicyValueNetwork(ModelConfig(
        channels=8, residual_blocks=1, value_hidden=16,
    )))


def test_precomputed_moves_exactly_match_normal_evaluation(monkeypatch):
    import p2_neural_network.inference as module
    network = evaluator()
    states = [p.state for p in fixed_positions()]
    states.append(GameState.from_mapping({
        'a1': Piece(Side.WHITE, PieceType.RAJA),
        'h8': Piece(Side.BLACK, PieceType.RAJA),
        'h7': Piece(Side.BLACK, PieceType.MANTRI),
    }))
    for state in states:
        expected = network.evaluate(state)
        moves = legal_moves(state)
        mask = legal_action_mask(state)
        with monkeypatch.context() as patch:
            def unexpected(*args):
                raise AssertionError('Precomputed path regenerated a legal mask')
            patch.setattr(module, 'legal_action_mask', unexpected)
            actual = network.evaluate_with_legal_moves(state, moves)
        np.testing.assert_array_equal(actual.policy, expected.policy)
        assert actual.value == expected.value
        assert np.all(actual.policy[mask == 0] == 0)
        if moves:
            np.testing.assert_allclose(actual.policy.sum(), 1, atol=1e-6)
        else:
            assert not actual.policy.any()
    assert network.evaluate_batch(()) == ()
    batch = network.evaluate_batch(states)
    for state, result in zip(states, batch):
        np.testing.assert_allclose(result.policy, network.evaluate(state).policy, atol=1e-6)


def test_search_reuses_legal_moves_without_changing_visits(monkeypatch):
    import shared.game_core.core as core
    network = evaluator()
    class OrdinaryEvaluator:
        def evaluate(self, state):
            return network.evaluate(state)

    original = core._legal_moves_unchecked
    count = 0
    def counted(state):
        nonlocal count
        count += 1
        return original(state)
    monkeypatch.setattr(core, '_legal_moves_unchecked', counted)
    config = MCTSConfig(num_simulations=25, temperature=0, add_noise=False)
    old = MCTS(OrdinaryEvaluator(), config)
    expected = old.search(initial_state())
    assert count == 52
    count = 0
    new = MCTS(network, config)
    actual = new.search(initial_state())
    assert count == 26
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    def snapshot(node):
        return (node.visit_count, node.value_sum, tuple(
            (action, child.prior, child.visit_count, child.value_sum,
             snapshot(child) if child.visit_count else None)
            for action, child in node.children.items()
        ))
    assert snapshot(new.root) == snapshot(old.root)
