"""Unit and integration coverage for the P3 Monte Carlo Tree Search."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from p1_representation import ACTION_SPACE_SIZE, legal_action_mask, move_to_action
from p2_neural_network import Evaluation, ModelConfig, NetworkEvaluator, PolicyValueNetwork
from p3_mcts import MCTS, MCTSConfig, MCTSNode
from shared.game_core import (
    GameState,
    Piece,
    PieceType,
    Side,
    apply_move,
    initial_state,
    legal_moves,
)


class UniformEvaluator:
    """Small deterministic evaluator that records every queried state."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.calls: list[GameState] = []

    def evaluate(self, state: GameState) -> Evaluation:
        self.calls.append(state)
        mask = legal_action_mask(state).astype(np.float32)
        if mask.sum():
            mask /= mask.sum()
        return Evaluation(mask, self.value)


def terminal_bare_king_state() -> GameState:
    return GameState.from_mapping(
        {
            "a1": Piece(Side.WHITE, PieceType.RAJA),
            "h8": Piece(Side.BLACK, PieceType.RAJA),
            "h7": Piece(Side.BLACK, PieceType.MANTRI),
        },
        side_to_move=Side.WHITE,
    )


def test_mcts_config_validation() -> None:
    assert MCTSConfig() == MCTSConfig(
        num_simulations=50,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
        add_noise=False,
    )
    for kwargs in (
        {"num_simulations": -1},
        {"num_simulations": 0},
        {"c_puct": 0},
        {"dirichlet_alpha": -0.1},
        {"dirichlet_epsilon": -0.1},
        {"dirichlet_epsilon": 1.1},
        {"temperature": -0.1},
    ):
        with pytest.raises(ValueError):
            MCTSConfig(**kwargs)


def test_mcts_node_puct_selection() -> None:
    root_state = initial_state()
    first_move, second_move = legal_moves(root_state)[:2]
    first = MCTSNode(
        apply_move(root_state, first_move),
        parent=None,
        action_taken=move_to_action(first_move),
        prior=0.2,
        visit_count=2,
        value_sum=-1.0,
    )
    second = MCTSNode(
        apply_move(root_state, second_move),
        parent=None,
        action_taken=move_to_action(second_move),
        prior=0.8,
        visit_count=8,
        value_sum=4.0,
    )
    root = MCTSNode(root_state, visit_count=10)
    first.parent = root
    second.parent = root
    root.children = {
        move_to_action(first_move): first,
        move_to_action(second_move): second,
    }

    expected_first = 0.5 + 1.5 * 0.2 * np.sqrt(10) / 3
    expected_second = -0.5 + 1.5 * 0.8 * np.sqrt(10) / 9
    assert first.puct_score(1.5, 10) == pytest.approx(expected_first)
    assert second.puct_score(1.5, 10) == pytest.approx(expected_second)
    action, selected = root.select_best_child(1.5)
    assert action == move_to_action(first_move)
    assert selected is first


def test_mcts_zero_sum_sign_inversion() -> None:
    root = MCTSNode(initial_state())
    move_1 = legal_moves(root.state)[0]
    child = MCTSNode(
        apply_move(root.state, move_1),
        parent=root,
        action_taken=move_to_action(move_1),
    )
    move_2 = legal_moves(child.state)[0]
    leaf = MCTSNode(
        apply_move(child.state, move_2),
        parent=child,
        action_taken=move_to_action(move_2),
    )

    MCTS._backpropagate(leaf, 0.75)

    assert leaf.visit_count == child.visit_count == root.visit_count == 1
    assert leaf.value_sum == pytest.approx(0.75)
    assert child.value_sum == pytest.approx(-0.75)
    assert root.value_sum == pytest.approx(0.75)


def test_terminal_leaf_not_queried() -> None:
    evaluator = UniformEvaluator()
    mcts = MCTS(evaluator, MCTSConfig(num_simulations=1))

    with pytest.raises(ValueError, match="terminal"):
        mcts.search(terminal_bare_king_state())

    assert evaluator.calls == []
    assert mcts.root is not None
    assert mcts.root.is_terminal
    assert mcts.root.terminal_value == -1.0


def test_dirichlet_noise_changes_priors() -> None:
    state = initial_state()
    plain = MCTS(UniformEvaluator(), MCTSConfig(num_simulations=1, add_noise=False))
    noisy = MCTS(UniformEvaluator(), MCTSConfig(num_simulations=1, add_noise=True))
    np.random.seed(19)
    plain.search(state)
    np.random.seed(19)
    noisy.search(state)

    assert plain.root is not None and noisy.root is not None
    plain_priors = np.array([child.prior for child in plain.root.children.values()])
    noisy_priors = np.array([child.prior for child in noisy.root.children.values()])
    assert not np.allclose(plain_priors, noisy_priors)
    assert noisy_priors.sum() == pytest.approx(1.0)


def test_temperature_sampling_vs_argmax() -> None:
    root = MCTSNode(initial_state())
    moves = legal_moves(root.state)[:3]
    for visits, move in zip((1, 3, 6), moves, strict=True):
        action = move_to_action(move)
        root.children[action] = MCTSNode(
            apply_move(root.state, move),
            parent=root,
            action_taken=action,
            visit_count=visits,
        )

    greedy = MCTS._policy_from_visits(root, temperature=0.0)
    expected_action = move_to_action(moves[2])
    assert np.count_nonzero(greedy) == 1
    assert greedy[expected_action] == 1.0
    legal_actions = [move_to_action(move) for move in moves]
    assert MCTS._select_action(greedy, temperature=0.0, legal_actions=legal_actions) == expected_action

    sampling = MCTS._policy_from_visits(root, temperature=1.0)
    actual = sampling[[move_to_action(move) for move in moves]]
    np.testing.assert_allclose(actual, np.array([0.1, 0.3, 0.6]), atol=1e-7)
    assert sampling.sum() == pytest.approx(1.0)


def test_action_sampling_repairs_float32_normalization_drift(monkeypatch) -> None:
    legal = np.array([7, 203, 3112], dtype=np.int64)
    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    policy[legal] = np.array([0.1, 0.2, 0.70000005], dtype=np.float32)
    policy[99] = 0.5  # Even a positive illegal entry must remain unavailable.
    captured = {}

    def capture_choice(size, p):
        captured["size"] = size
        captured["probabilities"] = p.copy()
        return int(legal[-1])

    monkeypatch.setattr(np.random, "choice", capture_choice)
    selected = MCTS._select_action(policy, temperature=1.0, legal_actions=legal)
    probabilities = captured["probabilities"]

    assert selected == int(legal[-1])
    assert captured["size"] == ACTION_SPACE_SIZE
    assert probabilities.dtype == np.float64
    assert probabilities.sum(dtype=np.float64) == 1.0
    assert probabilities[99] == 0.0
    assert np.all(probabilities[np.setdiff1d(np.arange(ACTION_SPACE_SIZE), legal)] == 0.0)
    expected = policy[legal].astype(np.float64)
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities[legal], expected, rtol=0, atol=1e-15)


def test_action_sampling_replaces_invalid_weights_with_legal_fallback(monkeypatch) -> None:
    legal = np.array([4, 18, 507, 4090], dtype=np.int64)
    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
    policy[legal] = [np.nan, np.inf, -0.25, 0.0]
    policy[100] = 1.0  # Illegal mass must not be used as a fallback.
    captured = {}

    def capture_choice(size, p):
        captured["probabilities"] = p.copy()
        return int(legal[1])

    monkeypatch.setattr(np.random, "choice", capture_choice)
    selected = MCTS._select_action(policy, temperature=1.0, legal_actions=legal)
    probabilities = captured["probabilities"]

    assert selected == int(legal[1])
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0.0)
    assert probabilities.sum(dtype=np.float64) == 1.0
    np.testing.assert_allclose(probabilities[legal], np.full(4, 0.25))
    assert np.count_nonzero(probabilities) == len(legal)


def test_policy_output_shape_and_legality() -> None:
    state = initial_state()
    before = replace(state)
    np.random.seed(5)
    policy, move, action = MCTS(
        UniformEvaluator(),
        MCTSConfig(num_simulations=8, temperature=1.0),
    ).search(state)

    mask = legal_action_mask(state).astype(bool)
    assert state == before
    assert policy.shape == (ACTION_SPACE_SIZE,)
    assert policy.dtype == np.float32
    assert policy.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(policy[~mask] == 0.0)
    assert move in legal_moves(state)
    assert action == move_to_action(move)


def test_step_reuses_searched_child() -> None:
    mcts = MCTS(UniformEvaluator(), MCTSConfig(num_simulations=2, temperature=0))
    _, move, action = mcts.search(initial_state())
    assert mcts.root is not None
    expected_child = mcts.root.children[action]

    mcts.step(move)

    assert mcts.root is expected_child
    assert mcts.root.parent is None
    assert mcts.root.state == apply_move(initial_state(), move)

    visits_before = mcts.root.visit_count
    next_state = mcts.root.state
    mcts.search(next_state)
    assert mcts.root is expected_child
    assert mcts.root.visit_count == visits_before + 2


def test_full_mcts_integration_with_untrained_network() -> None:
    torch.manual_seed(23)
    np.random.seed(23)
    model = PolicyValueNetwork(
        ModelConfig(channels=16, residual_blocks=2, policy_channels=2, value_hidden=32)
    )
    evaluator = NetworkEvaluator(model, device="cpu")
    state = initial_state()

    policy, move, action = MCTS(
        evaluator,
        MCTSConfig(num_simulations=20, temperature=1.0),
    ).search(state)

    assert move in legal_moves(state)
    assert action == move_to_action(move)
    assert policy.shape == (ACTION_SPACE_SIZE,)
    assert policy.dtype == np.float32
    assert np.isfinite(policy).all()
    assert np.all(policy >= 0)
    assert policy.sum() == pytest.approx(1.0, abs=1e-6)
