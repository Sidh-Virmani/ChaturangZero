from dataclasses import replace
from random import Random

import numpy as np
import pytest
import torch

from benchmarks import fixed_positions
from p1_representation import (
    encode_state, legal_action_mask, rotate_action, rotate_encoded_state, rotate_state,
)
from p4_training import Trainer, TrainingConfig, TrainingExample, smoke_position
from p4_training.augmentation import augment_example, rotate_example
from shared.game_core import Move, apply_move, initial_state, terminal_status


def example_for(state, value=1.0):
    actions = np.flatnonzero(legal_action_mask(state)).astype(np.int16)
    probabilities = np.arange(1, len(actions) + 1, dtype=np.float32)
    probabilities /= probabilities.sum()
    return TrainingExample(encode_state(state).astype(np.uint8), actions, probabilities, value)


@pytest.mark.parametrize('value', [-1.0, 0.0, 1.0])
def test_example_rotation_matches_core_and_preserves_targets(value):
    for position in fixed_positions():
        state = position.state
        example = example_for(state, value)
        transformed = rotate_example(example)
        np.testing.assert_array_equal(transformed.state, encode_state(rotate_state(state)))
        assert transformed.state.dtype == example.state.dtype
        np.testing.assert_array_equal(transformed.actions,
                                     [rotate_action(int(a)) for a in example.actions])
        np.testing.assert_array_equal(transformed.probabilities, example.probabilities)
        assert transformed.probabilities.sum() == example.probabilities.sum()
        assert transformed.value == value
        assert np.all(legal_action_mask(rotate_state(state))[transformed.actions])
        assert np.count_nonzero(transformed.policy()) == np.count_nonzero(example.policy())
        restored = rotate_example(transformed)
        for name in ('state', 'actions', 'probabilities'):
            np.testing.assert_array_equal(getattr(restored, name), getattr(example, name))
            assert not np.shares_memory(getattr(transformed, name), getattr(example, name))


def test_history_rotation_and_encoding_are_consistent():
    state = initial_state()
    for uci in ('b1a3', 'b8a6', 'a3b1', 'a6b8') * 2:
        state = apply_move(state, Move.from_names(uci[:2], uci[2:]))
    transformed = rotate_state(state)
    assert terminal_status(state).draw and terminal_status(transformed).draw
    assert len(transformed.position_history) == len(state.position_history) == 8
    assert rotate_state(transformed) == state
    for dtype in (np.uint8, np.float32):
        np.testing.assert_array_equal(
            rotate_encoded_state(encode_state(state).astype(dtype)),
            encode_state(transformed).astype(dtype),
        )


def test_seeded_augmentation_and_disabled_rng_behavior():
    example = example_for(initial_state())
    first, second = Random(19), Random(19)
    decisions = []
    for _ in range(32):
        a = augment_example(example, first, 0.5)
        b = augment_example(example, second, 0.5)
        np.testing.assert_array_equal(a.state, b.state)
        decisions.append(a is example)
    assert any(decisions) and not all(decisions)
    before = first.getstate()
    assert augment_example(example, first, 0) is example
    assert first.getstate() == before
    assert augment_example(example, first, 1) is not example


@pytest.mark.parametrize('probability', [-1, 1.1, float('nan'), float('inf'), True])
def test_augmentation_probability_is_validated(probability):
    with pytest.raises(ValueError, match='symmetry_probability'):
        TrainingConfig(symmetry_probability=probability)


def test_trainer_consumes_rotated_batch_without_modifying_replay(tmp_path):
    config = replace(TrainingConfig.smoke(tmp_path), symmetry_probability=1)
    trainer = Trainer(config)
    example = example_for(smoke_position())
    originals = tuple(array.copy() for array in (example.state, example.actions, example.probabilities))
    trainer.replay.add_game((example,))
    inputs = []
    handle = trainer.model.register_forward_pre_hook(lambda model, args: inputs.append(args[0].clone()))
    try:
        losses = trainer._train_steps(1)
    finally:
        handle.remove()
    np.testing.assert_array_equal(inputs[0][0].numpy(), rotate_example(example).state)
    assert np.isfinite(losses).all()
    for actual, original in zip((example.state, example.actions, example.probabilities), originals):
        np.testing.assert_array_equal(actual, original)
    assert len(trainer.replay) == 1


def test_augmented_resume_matches_uninterrupted_updates_and_rng(tmp_path):
    config = replace(TrainingConfig.smoke(tmp_path / 'continuous'), symmetry_probability=0.5)
    continuous = Trainer(replace(config, iterations=2))
    continuous.run(smoke_position())
    expected_rng = continuous.sample_rng.getstate()
    split = Trainer(replace(config, checkpoint_dir=tmp_path / 'split'))
    split.run(smoke_position())
    resumed = Trainer.resume(tmp_path / 'split', iterations=1)
    assert resumed.config.symmetry_probability == 0.5
    resumed.run(smoke_position())
    assert resumed.sample_rng.getstate() == expected_rng
    assert resumed.history == continuous.history
    for name, tensor in continuous.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], tensor, rtol=0, atol=0)


def test_legacy_checkpoint_keeps_augmentation_disabled(tmp_path):
    trainer = Trainer(replace(TrainingConfig.smoke(tmp_path), symmetry_probability=0))
    trainer.save()
    path = tmp_path / 'training_state.pt'
    saved = torch.load(path, weights_only=True)
    saved['config'].pop('symmetry_probability')
    torch.save(saved, path)
    resumed = Trainer.resume(tmp_path, iterations=1)
    assert resumed.config.symmetry_probability == 0
    assert resumed.sample_rng.getstate() == trainer.sample_rng.getstate()
