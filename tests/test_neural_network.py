from dataclasses import replace

import numpy as np
import pytest
import torch

from p1_representation import ACTION_SPACE_SIZE, STATE_SHAPE, encode_state, legal_action_mask
from p2_neural_network import (
    ModelConfig,
    NetworkEvaluator,
    PolicyValueNetwork,
    load_checkpoint,
    policy_value_loss,
    save_checkpoint,
)
from shared.game_core import GameState, Piece, PieceType, Side, apply_move, initial_state


@pytest.fixture
def small_model() -> PolicyValueNetwork:
    torch.manual_seed(7)
    return PolicyValueNetwork(
        ModelConfig(channels=16, residual_blocks=2, policy_channels=2, value_hidden=32)
    )


def encoded_batch(*states: GameState) -> torch.Tensor:
    return torch.from_numpy(np.stack([encode_state(state) for state in states]))


def midgame_state() -> GameState:
    state = initial_state()
    for move in (("a2", "a3"), ("a7", "a6"), ("b1", "c3")):
        state = apply_move(state, move)
    return state


def test_default_architecture_has_expected_size_and_p1_contract():
    model = PolicyValueNetwork()
    assert model.config == ModelConfig()
    assert model.parameter_count == 840_583
    assert STATE_SHAPE == (13, 8, 8)
    assert ACTION_SPACE_SIZE == 4096


def test_forward_accepts_p1_state_and_returns_policy_and_bounded_value(small_model):
    inputs = encoded_batch(initial_state())
    before = inputs.clone()
    policy_logits, values = small_model(inputs)

    assert policy_logits.shape == (1, ACTION_SPACE_SIZE)
    assert values.shape == (1,)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(values).all()
    assert torch.all(values >= -1) and torch.all(values <= 1)
    assert torch.equal(inputs, before)


def test_batch_forward_preserves_batch_dimension(small_model):
    inputs = encoded_batch(initial_state(), midgame_state())
    policy_logits, values = small_model(inputs)
    assert policy_logits.shape == (2, ACTION_SPACE_SIZE)
    assert values.shape == (2,)


def test_eval_mode_is_deterministic(small_model):
    small_model.eval()
    inputs = encoded_batch(midgame_state())
    with torch.inference_mode():
        first = small_model(inputs)
        second = small_model(inputs)
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)


def test_training_loss_backpropagates_into_all_trainable_branches(small_model):
    small_model.train()
    logits, values = small_model(encoded_batch(initial_state(), midgame_state()))
    targets = torch.zeros_like(logits)
    targets[0, 528] = 1.0
    targets[1, 82] = 1.0
    target_values = torch.tensor([1.0, -1.0])

    losses = policy_value_loss(logits, values, targets, target_values)
    losses.total.backward()

    assert losses.total.item() == pytest.approx(losses.policy.item() + losses.value.item())
    parameters = [parameter for parameter in small_model.parameters() if parameter.requires_grad]
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert any(torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)


def test_evaluator_returns_only_core_legal_normalized_probabilities(small_model):
    state = initial_state()
    before = replace(state)
    small_model.train()
    evaluator = NetworkEvaluator(small_model, device="cpu")
    evaluation = evaluator.evaluate(state)

    assert small_model.training  # the evaluator restores the caller's mode
    assert state == before
    assert evaluation.policy.shape == (ACTION_SPACE_SIZE,)
    assert evaluation.policy.dtype == np.float32
    assert np.isfinite(evaluation.policy).all()
    assert np.isfinite(evaluation.value)
    assert -1.0 <= evaluation.value <= 1.0
    mask = legal_action_mask(state).astype(bool)
    assert np.all(evaluation.policy[~mask] == 0)
    assert np.all(evaluation.policy[mask] > 0)
    assert evaluation.policy.sum() == pytest.approx(1.0, abs=1e-6)


def test_batch_evaluator_matches_single_evaluation_and_keeps_order(small_model):
    states = (initial_state(), midgame_state())
    evaluator = NetworkEvaluator(small_model)
    batch = evaluator.evaluate_batch(states)
    singles = tuple(evaluator.evaluate(state) for state in states)

    assert len(batch) == 2
    for batched, single in zip(batch, singles, strict=True):
        np.testing.assert_allclose(batched.policy, single.policy, rtol=1e-5, atol=1e-7)
        assert batched.value == pytest.approx(single.value, abs=1e-6)
    assert evaluator.evaluate_batch(()) == ()


def test_terminal_state_receives_zero_policy(small_model):
    terminal = GameState.from_mapping({
        "a1": Piece(Side.WHITE, PieceType.RAJA),
        "h8": Piece(Side.BLACK, PieceType.RAJA),
        "h7": Piece(Side.BLACK, PieceType.MANTRI),
    })
    evaluation = NetworkEvaluator(small_model).evaluate(terminal)
    assert np.count_nonzero(evaluation.policy) == 0
    assert np.isfinite(evaluation.value)


def test_checkpoint_round_trip_reproduces_predictions(tmp_path, small_model):
    small_model.eval()
    inputs = encoded_batch(initial_state(), midgame_state())
    with torch.inference_mode():
        expected_logits, expected_values = small_model(inputs)

    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(small_model, path)
    restored = load_checkpoint(path, device="cpu")
    with torch.inference_mode():
        actual_logits, actual_values = restored(inputs)

    assert restored.config == small_model.config
    assert not restored.training
    assert next(restored.parameters()).device.type == "cpu"
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)


def test_forward_rejects_missing_batch_dimension_and_integer_input(small_model):
    with pytest.raises(ValueError, match="Expected encoded states"):
        small_model(torch.from_numpy(encode_state(initial_state())))
    with pytest.raises(TypeError, match="floating-point"):
        small_model(torch.zeros((1, *STATE_SHAPE), dtype=torch.int64))
