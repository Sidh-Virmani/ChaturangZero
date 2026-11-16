"""Terminal safeguards without expensive self-play."""
from dataclasses import replace
import subprocess
import sys

import pytest
import torch

from p4_training import Trainer, TrainingConfig, smoke_position
from p4_training.operations import training_lock, run_with_progress


def test_lock_rejects_second_process_and_releases_after_exception(tmp_path):
    script = ('from pathlib import Path; from p4_training.operations import training_lock; '
              'ctx=training_lock(Path(__import__("sys").argv[1])); ctx.__enter__()')
    with pytest.raises(KeyboardInterrupt):
        with training_lock(tmp_path):
            result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True)
            assert result.returncode != 0 and b'writer lock' in result.stderr
            raise KeyboardInterrupt
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], capture_output=True)
    assert result.returncode == 0


def test_first_game_interrupt_has_resumable_initial_model(tmp_path, monkeypatch):
    trainer = Trainer(replace(TrainingConfig.smoke(tmp_path), snapshot_interval=1))
    expected = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr('p4_training.trainer.generate_games', interrupt)
    with pytest.raises(KeyboardInterrupt):
        trainer.run()
    resumed = Trainer.resume(tmp_path, 1)
    assert resumed.completed_iterations == 0 and not len(resumed.replay)
    assert not list(tmp_path.glob('iteration_*.pt'))
    for key, value in expected.items():
        assert torch.equal(resumed.model.state_dict()[key], value)


def test_until_resume_stops_at_target_and_repeated_command_does_no_work(tmp_path, monkeypatch, capsys):
    import p4_training.__main__ as cli
    trainer = Trainer(TrainingConfig.smoke(tmp_path))
    trainer.run(smoke_position())
    real_run = Trainer.run
    monkeypatch.setattr(Trainer, 'run', lambda self, **kwargs: real_run(self, smoke_position(), **kwargs))
    monkeypatch.setattr(sys, 'argv', ['p4_training', 'resume', '--output', str(tmp_path), '--until', '2'])
    cli.main()
    assert Trainer.resume(tmp_path, 1).completed_iterations == 2
    before = (tmp_path / 'training_state.pt').read_bytes()
    cli.main()
    assert (tmp_path / 'training_state.pt').read_bytes() == before
    assert 'requires no updates' in capsys.readouterr().out


def test_resume_missing_checkpoint_never_initializes_model(tmp_path, monkeypatch):
    import p4_training.__main__ as cli
    monkeypatch.setattr(sys, 'argv', ['p4_training', 'resume', '--output', str(tmp_path), '--until', '200'])
    with pytest.raises(FileNotFoundError):
        cli.main()
    assert not list(tmp_path.glob('*.pt'))


def test_cli_interrupt_exits_cleanly_and_unlocks(tmp_path, monkeypatch, capsys):
    import p4_training.__main__ as cli
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, '_run_training', interrupt)
    monkeypatch.setattr(sys, 'argv', ['p4_training', 'train', '--output', str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 130 and 'Resume from' in capsys.readouterr().err
    with training_lock(tmp_path):
        pass


def test_progress_reports_absolute_target_and_wall_time(tmp_path, capsys):
    trainer = Trainer(TrainingConfig.smoke(tmp_path))
    class TinyRun:
        config = trainer.config
        completed_iterations = 0
        def run(self, **kwargs):
            trainer.run(smoke_position(), **kwargs)
    run_with_progress(TinyRun(), lambda _: None, lambda *_: None, lambda: None)
    output = capsys.readouterr().out
    assert 'iteration=1/1' in output and 'wall_seconds=' in output and 'eta_minutes=0.0' in output
