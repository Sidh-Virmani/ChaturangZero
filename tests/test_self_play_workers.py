"""Process isolation, deterministic game ordering and restart safety."""
from dataclasses import replace
import multiprocessing as mp

import numpy as np
import pytest
import torch

from p4_training import Trainer, TrainingConfig, smoke_position
from p4_training.workers import generate_games, game_seed


def config(path, **kwargs):
    return replace(TrainingConfig.smoke(path), games_per_iteration=3, **kwargs)


def test_seed_assignment_and_parallel_games_match_serial(tmp_path):
    settings = config(tmp_path)
    trainer = Trainer(settings)
    weights = {name: value.clone() for name, value in trainer.model.state_dict().items()}
    rng = np.random.get_state()
    serial = generate_games(trainer.model, settings, 2)
    parallel = generate_games(trainer.model, replace(settings, self_play_workers=2), 2)
    assert [game.moves for game in serial] == [game.moves for game in parallel]
    assert len(set(game_seed(7, 2, i) for i in range(100))) == 100
    assert game_seed(7, 2, 0) != game_seed(7, 3, 0)
    np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
    for name, value in weights.items():
        assert torch.equal(value, trainer.model.state_dict()[name])
    # Finished games carry the same board/policy/value examples, not just moves.
    a = generate_games(trainer.model, settings, 1, smoke_position())
    b = generate_games(trainer.model, replace(settings, self_play_workers=2), 1, smoke_position())
    for first, second in zip(a, b, strict=True):
        assert first.outcome == second.outcome and first.plies == second.plies == 1
        for x, y in zip(first.examples, second.examples, strict=True):
            np.testing.assert_array_equal(x.state, y.state)
            np.testing.assert_array_equal(x.policy(), y.policy())
            assert x.value == y.value


def test_parallel_resume_reproduces_updates_and_saves_snapshots(tmp_path):
    a = Trainer(config(tmp_path / 'a', self_play_workers=2, iterations=2, snapshot_interval=1))
    a.run(smoke_position())
    b = Trainer(config(tmp_path / 'b', self_play_workers=2, snapshot_interval=1))
    b.run(smoke_position())
    first = (tmp_path / 'b' / 'iteration_000001.pt').read_bytes()
    resumed = Trainer.resume(tmp_path / 'b', 1)
    resumed.run(smoke_position())
    assert a.history == resumed.history
    for name, value in a.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert (tmp_path / 'b' / 'iteration_000001.pt').read_bytes() == first
    assert (tmp_path / 'b' / 'iteration_000002.pt').exists()
    assert resumed.history[-1].audit['self_play_seed_protocol'] == 'iteration-game-v1'


def test_worker_failure_does_not_commit_partial_iteration(tmp_path):
    trainer = Trainer(config(tmp_path, self_play_workers=2))
    trainer.run(smoke_position())
    checkpoint = (tmp_path / 'training_state.pt').read_bytes()
    replay = len(trainer.replay)
    children = {process.pid for process in mp.active_children()}
    with pytest.raises(AttributeError):
        trainer.run('invalid-state')
    assert (tmp_path / 'training_state.pt').read_bytes() == checkpoint
    assert len(trainer.replay) == replay and trainer.completed_iterations == 1
    assert {process.pid for process in mp.active_children()} == children


def test_worker_timeout_cleans_up_children(tmp_path):
    trainer = Trainer(config(tmp_path, self_play_workers=2, worker_timeout_seconds=1,
                             max_game_plies=512, simulations_per_move=100))
    children = {process.pid for process in mp.active_children()}
    with pytest.raises(TimeoutError, match='not committed'):
        trainer.run()
    assert not (tmp_path / 'training_state.pt').exists()
    assert {process.pid for process in mp.active_children()} == children


def test_old_checkpoint_keeps_serial_stream(tmp_path):
    trainer = Trainer(config(tmp_path))
    trainer.save()
    path = tmp_path / 'training_state.pt'
    payload = torch.load(path, weights_only=True)
    for name in ('seeded_games', 'self_play_device', 'self_play_workers', 'cpu_threads'):
        payload['config'].pop(name)
    torch.save(payload, path)
    resumed = Trainer.resume(tmp_path, 1)
    assert not resumed.config.seeded_games and resumed.config.self_play_workers == 1
    assert resumed.run(smoke_position())[0].games_completed == 3


def test_throughput_command_records_seeded_workload_without_touching_checkpoint(tmp_path, monkeypatch):
    from p2_neural_network import save_checkpoint
    from benchmarks.throughput import main
    import json
    trainer = Trainer(config(tmp_path / 'unused'))
    path, output = tmp_path / 'model.pt', tmp_path / 'result.json'
    save_checkpoint(trainer.model, path)
    before = path.read_bytes()
    monkeypatch.setattr('sys.argv', ['throughput', 'self-play', '--checkpoint', str(path),
        '--output', str(output), '--games', '2', '--simulations', '1', '--max-plies', '1'])
    main()
    data = json.loads(output.read_text())
    assert data['completed_games'] == data['examples'] == 0
    assert data['config']['workers'] == 1 and len(data['games']) == 2
    assert all(game['reason'] == 'truncation' for game in data['games'])
    assert data['simulations_per_second'] > 0
    assert path.read_bytes() == before
    assert not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('settings', [dict(self_play_workers=0), dict(worker_threads=0),
    dict(cpu_threads=0), dict(self_play_workers=2, self_play_device='cuda'),
    dict(self_play_workers=2, seeded_games=False), dict(snapshot_interval=-1)])
def test_invalid_worker_settings(settings):
    with pytest.raises(ValueError):
        TrainingConfig(**settings)


def test_new_level_run_cannot_overwrite_orphan_model(tmp_path):
    from p4_training import train_levels
    model = tmp_path / 'latest_model.pt'
    model.write_bytes(b'keep this existing checkpoint')
    with pytest.raises(FileExistsError):
        train_levels(config(tmp_path), easy=1, medium=2, hard=3)
    assert model.read_bytes() == b'keep this existing checkpoint'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware/build unavailable')
def test_cuda_training_cpu_selfplay_and_portable_resume(tmp_path):
    from p2_neural_network import load_checkpoint, NetworkEvaluator
    trainer = Trainer(config(tmp_path, device='cuda', self_play_workers=2))
    metrics = trainer.run(smoke_position())[0]
    assert np.isfinite(metrics.total_loss) and metrics.updates > 0
    cpu = load_checkpoint(tmp_path / 'latest_model.pt', 'cpu')
    actual = NetworkEvaluator(cpu).evaluate(smoke_position())
    expected = trainer.evaluator.evaluate(smoke_position())
    np.testing.assert_allclose(actual.policy, expected.policy, atol=1e-5)
    assert actual.value == pytest.approx(expected.value, abs=1e-5)
    resumed = Trainer.resume(tmp_path, 1, 'cuda')
    assert np.isfinite(resumed.run(smoke_position())[0].total_loss)
    resumed_cpu = Trainer.resume(tmp_path, 1, 'cpu')
    assert np.isfinite(resumed_cpu.run(smoke_position())[0].total_loss)
