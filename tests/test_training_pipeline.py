from dataclasses import replace
from random import Random

import numpy as np
import pytest
import torch

from p1_representation import ACTION_SPACE_SIZE, STATE_SHAPE, encode_state, legal_action_mask, move_to_action
from p2_neural_network import ModelConfig, NetworkEvaluator, load_checkpoint
from p4_training import (
    ReplayBuffer, Trainer, TrainingConfig, TrainingExample, compare_models,
    play_game, smoke_position,
)
from shared.game_core import (
    GameState, Move, Piece, PieceType, Side, apply_move, initial_state,
    legal_moves, terminal_status,
)


def quick_config(tmp_path, **changes) -> TrainingConfig:
    values = dict(
        iterations=1,
        games_per_iteration=1,
        simulations_per_move=1,
        max_game_plies=2,
        opening_temperature_plies=2,
        training_steps=1,
        batch_size=2,
        replay_capacity=3,
        seed=11,
        checkpoint_dir=tmp_path,
        model=ModelConfig(channels=8, residual_blocks=1, policy_channels=2, value_hidden=16),
    )
    values.update(changes)
    return TrainingConfig(**values)


def test_smoke_position_ends_by_core_rules_after_every_legal_reply():
    state = smoke_position()
    moves = legal_moves(state)
    assert moves
    for move in moves:
        ended = apply_move(state, move)
        assert terminal_status(ended).winner is Side.WHITE


def test_real_mcts_self_play_records_p1_input_policy_and_final_reward(tmp_path):
    config = quick_config(tmp_path)
    trainer = Trainer(config)
    start = smoke_position()
    before = replace(start)
    game = play_game(trainer.evaluator, config, start)

    assert start == before
    assert not game.truncated
    assert game.plies == 1
    assert game.outcome.winner is Side.WHITE
    assert len(game.examples) == 1
    example = game.examples[0]
    assert example.state.shape == STATE_SHAPE
    assert example.state.dtype == np.uint8
    np.testing.assert_array_equal(example.state, encode_state(start))
    assert example.value == 1.0
    assert example.actions.dtype == np.int16
    assert example.probabilities.dtype == np.float32
    assert example.policy().shape == (ACTION_SPACE_SIZE,)
    assert example.policy().sum() == pytest.approx(1.0)
    assert set(example.actions).issubset(set(np.flatnonzero(legal_action_mask(start))))


def test_ply_cap_discards_unfinished_games_without_inventing_a_result(tmp_path):
    config = quick_config(tmp_path, max_game_plies=1)
    trainer = Trainer(config)
    game = play_game(trainer.evaluator, config, initial_state())
    assert game.truncated
    assert game.outcome is None
    assert game.plies == 1
    assert game.examples == ()


def test_outcome_targets_follow_each_recorded_player_to_move(tmp_path, monkeypatch):
    start = GameState.from_mapping({
        "a1": Piece(Side.WHITE, PieceType.RAJA),
        "b2": Piece(Side.WHITE, PieceType.RATHA),
        "h8": Piece(Side.BLACK, PieceType.RAJA),
        "b7": Piece(Side.BLACK, PieceType.RATHA),
    })
    white_move = Move.from_names("a1", "a2")
    black_move = Move.from_names("b7", "b2")
    assert white_move in legal_moves(start)
    after_white = apply_move(start, white_move)
    assert black_move in legal_moves(after_white)
    assert terminal_status(apply_move(after_white, black_move)).winner is Side.BLACK

    class ScriptedSearch:
        def __init__(self, evaluator, config):
            pass

        def search(self, state):
            move = white_move if state.side_to_move is Side.WHITE else black_move
            action = move_to_action(move)
            policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            policy[action] = 1.0
            return policy, move, action

        def step(self, move):
            pass

    monkeypatch.setattr("p4_training.self_play.MCTS", ScriptedSearch)
    config = quick_config(tmp_path)
    game = play_game(Trainer(config).evaluator, config, start)
    assert game.plies == 2
    assert game.outcome.winner is Side.BLACK
    assert [example.value for example in game.examples] == [-1.0, 1.0]
    np.testing.assert_array_equal(game.examples[0].state, encode_state(start))
    np.testing.assert_array_equal(game.examples[1].state, encode_state(after_white))


def test_replay_keeps_newest_examples_and_reconstructs_policy(tmp_path):
    state = encode_state(smoke_position()).astype(np.uint8)
    buffer = ReplayBuffer(capacity=2)
    examples = tuple(
        TrainingExample(
            state.copy(), np.array([i], dtype=np.int16),
            np.array([1.0], dtype=np.float32), float(i - 1),
        )
        for i in range(3)
    )
    buffer.add_game(examples)
    assert len(buffer) == 2
    assert {example.value for example in buffer.sample(20, Random(3))} == {0.0, 1.0}
    restored = ReplayBuffer.from_checkpoint(2, buffer.to_checkpoint())
    assert len(restored) == 2
    assert set(np.flatnonzero(restored.sample(1, Random(0))[0].policy())) <= {1, 2}


def test_training_updates_parameters_and_saves_loadable_model_and_resume_state(tmp_path):
    config = quick_config(tmp_path)
    trainer = Trainer(config)
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    metrics = trainer.run(smoke_position())

    assert len(metrics) == 1
    assert metrics[0].games_completed == 1
    assert metrics[0].games_truncated == 0
    assert metrics[0].examples_added == 1
    assert metrics[0].updates == 1
    assert np.isfinite(metrics[0].total_loss)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.model.parameters(), strict=True)
    )

    model_path = tmp_path / "latest_model.pt"
    state_path = tmp_path / "training_state.pt"
    assert model_path.is_file() and state_path.is_file()
    loaded = load_checkpoint(model_path)
    state = smoke_position()
    expected = trainer.evaluator.evaluate(state)
    actual = NetworkEvaluator(loaded).evaluate(state)
    np.testing.assert_array_equal(actual.policy, expected.policy)
    assert actual.value == expected.value

    resumed = Trainer.resume(tmp_path, iterations=1)
    assert resumed.completed_iterations == 1
    assert resumed.config.model == config.model
    assert resumed.history == list(metrics)
    assert len(resumed.replay) == 1
    next_metrics = resumed.run(smoke_position())
    assert next_metrics[0].iteration == 2
    assert next_metrics[0].games_completed == 1
    assert next_metrics[0].replay_size == 2


def test_resume_restores_random_streams_and_optimizer_for_identical_continuation(tmp_path):
    first = Trainer(quick_config(tmp_path / "first", iterations=2))
    first.run(smoke_position())
    first_weights = [parameter.detach().clone() for parameter in first.model.parameters()]

    second = Trainer(quick_config(tmp_path / "second", iterations=1))
    second.run(smoke_position())
    resumed = Trainer.resume(tmp_path / "second", iterations=1)
    resumed.run(smoke_position())
    assert resumed.history == first.history
    for expected, actual in zip(first_weights, resumed.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_optional_arena_alternates_colors_without_noise(tmp_path):
    trainer = Trainer(quick_config(tmp_path))
    result = compare_models(
        trainer.model, trainer.model,
        games=2, simulations_per_move=1, max_game_plies=2,
        start_state=smoke_position(),
    )
    assert result.games == 2
    assert result.candidate_wins == 1
    assert result.reference_wins == 1
    assert result.truncated == 0
