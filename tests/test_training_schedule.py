"""Data-scaled update budgets and explicit fixed-schedule compatibility."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from p4_training import Trainer, TrainingConfig, smoke_position
from shared.game_core import initial_state


def populated_trainer(tmp_path, **settings):
    config = replace(TrainingConfig.smoke(tmp_path), **settings)
    trainer = Trainer(config)
    trainer.run(smoke_position())
    return trainer


@pytest.mark.parametrize('new_examples,batch,passes,expected', [
    (0, 64, 1, 0), (1, 64, 1, 1), (63, 64, 1, 1),
    (64, 64, 1, 1), (65, 64, 1, 2), (128, 64, 1, 2),
    (640, 64, 1, 10), (128, 32, 1, 4), (128, 64, 2, 4),
    (65, 64, 0.5, 1),
])
def test_budget_scales_with_examples_passes_and_batch(tmp_path, new_examples, batch, passes, expected):
    # Seed replay without optimizer work; the normal schedule still depends on
    # new_examples, not the number already in the buffer.
    from p4_training import play_game
    trainer = Trainer(replace(TrainingConfig.smoke(tmp_path), batch_size=batch, training_passes=passes))
    trainer.replay.add_game(play_game(trainer.evaluator, trainer.config, smoke_position()).examples)
    assert trainer.update_count(new_examples) == expected


def test_zero_new_data_skips_optimizer_and_preserves_model_rng_and_replay(tmp_path):
    trainer = populated_trainer(tmp_path, max_game_plies=1, symmetry_probability=1)
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    sample_rng = trainer.sample_rng.getstate()
    metrics = trainer.run(initial_state())[0]
    assert metrics.examples_added == metrics.updates == 0
    assert metrics.replay_size == 1 and metrics.games_truncated == 1
    assert metrics.policy_loss is metrics.value_loss is metrics.total_loss is None
    assert metrics.training_passes == 1 and metrics.training_steps is None
    assert metrics.batch_size == 2
    assert trainer.sample_rng.getstate() == sample_rng
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_positive_data_metrics_and_saved_schedule(tmp_path):
    trainer = populated_trainer(tmp_path, training_passes=3, batch_size=2, symmetry_probability=1)
    metrics = trainer.history[0]
    assert metrics.examples_added == 1
    assert metrics.updates == 2
    assert metrics.training_passes == 3 and metrics.batch_size == 2
    assert np.isfinite([metrics.policy_loss, metrics.value_loss, metrics.total_loss]).all()
    resumed = Trainer.resume(tmp_path, iterations=1)
    assert resumed.config.training_passes == 3
    assert resumed.config.training_steps is None
    assert resumed.history == trainer.history


def test_explicit_fixed_mode_can_train_existing_replay_without_new_data(tmp_path):
    trainer = populated_trainer(tmp_path, training_passes=None, training_steps=2, max_game_plies=1)
    resumed = Trainer.resume(tmp_path, iterations=1)
    metrics = resumed.run(initial_state())[0]
    assert metrics.examples_added == 0 and metrics.updates == 2
    assert metrics.training_steps == 2 and metrics.training_passes is None
    assert np.isfinite(metrics.total_loss)


def test_old_checkpoint_migrates_to_scaled_schedule(tmp_path):
    trainer = populated_trainer(tmp_path, max_game_plies=1)
    path = tmp_path / 'training_state.pt'
    saved = torch.load(path, weights_only=True)
    saved['config'].pop('training_passes')
    saved['config']['training_steps'] = 50
    for item in saved['history']:
        for key in ('training_passes', 'training_steps', 'batch_size'):
            item.pop(key)
    torch.save(saved, path)
    resumed = Trainer.resume(tmp_path, iterations=1)
    assert resumed.config.training_steps is None
    assert resumed.config.training_passes == 1
    assert resumed.sample_rng.getstate() == trainer.sample_rng.getstate()
    assert resumed.run(initial_state())[0].updates == 0


@pytest.mark.parametrize('passes', [0, -1, float('nan'), float('inf'), True])
def test_invalid_passes_rejected(passes):
    with pytest.raises(ValueError, match='training_passes'):
        TrainingConfig(training_passes=passes)


def test_config_rejects_ambiguous_schedule_and_empty_replay_has_no_updates(tmp_path):
    with pytest.raises(ValueError, match='Choose'):
        TrainingConfig(training_steps=2, training_passes=1)
    assert Trainer(TrainingConfig.smoke(tmp_path)).update_count(0) == 0


def test_cli_schedule_selection(tmp_path, monkeypatch):
    import p4_training.__main__ as cli
    import sys
    recorded = []
    class CaptureTrainer:
        def __init__(self, config):
            self.config = config
            recorded.append(config)
        def run(self, **kwargs):
            return ()
    monkeypatch.setattr(cli, 'Trainer', CaptureTrainer)
    for options in ([], ['--training-passes', '2'], ['--training-steps', '7']):
        monkeypatch.setattr(sys, 'argv', ['p4_training', 'train', '--output', str(tmp_path), *options])
        cli.main()
    assert recorded[0].training_passes == 1
    assert recorded[1].training_passes == 2
    assert recorded[2].training_steps == 7 and recorded[2].training_passes is None
    monkeypatch.setattr(sys, 'argv', ['p4_training', 'train', '--training-passes', '2', '--training-steps', '7'])
    with pytest.raises(SystemExit):
        cli.main()
