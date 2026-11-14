"""Durable records match committed resume history and survive resumed runs."""
from dataclasses import asdict
import json

import torch

from p4_training import Trainer, TrainingConfig, smoke_position
from p4_training.recording import file_sha256


def test_training_records_config_model_identity_timing_and_resume(tmp_path):
    trainer = Trainer(TrainingConfig.smoke(tmp_path))
    first = trainer.run(smoke_position())[0]
    rows = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert rows == [asdict(first)]
    audit = rows[0]['audit']
    assert audit['checkpoint']['sha256'] == file_sha256(tmp_path / 'latest_model.pt')
    assert audit['config']['symmetry_probability'] == .5
    assert audit['learning_rates'] == [.001]
    assert rows[0]['examples_added'] == rows[0]['updates'] == 1
    assert rows[0]['self_play_seconds'] >= 0 and rows[0]['training_seconds'] >= 0
    assert rows[0]['elapsed_seconds'] >= rows[0]['training_seconds']
    assert audit['environment']['git_commit']
    assert audit['start_state']['board']
    # Simulate stale or partial external logs. Checkpoint history remains authoritative.
    (tmp_path / 'metrics.jsonl').write_text('broken\n')
    resumed = Trainer.resume(tmp_path, 1)
    resumed.run(smoke_position())
    rows = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [row['iteration'] for row in rows] == [1, 2]
    assert rows[0] == asdict(first)
    assert rows == [asdict(item) for item in resumed.history]
    metadata = json.loads((tmp_path / 'run.json').read_text())
    assert metadata['completed_iterations'] == 2
    assert metadata['training_state_sha256'] == file_sha256(tmp_path / 'training_state.pt')
    assert metadata['latest_model']['sha256'] == file_sha256(tmp_path / 'latest_model.pt')


def test_older_history_exports_unknown_provenance_without_inventing_it(tmp_path):
    trainer = Trainer(TrainingConfig.smoke(tmp_path))
    trainer.run(smoke_position())
    path = tmp_path / 'training_state.pt'
    payload = torch.load(path, weights_only=True)
    for field in ('audit', 'elapsed_seconds', 'self_play_seconds', 'training_seconds'):
        payload['history'][0].pop(field)
    torch.save(payload, path)
    resumed = Trainer.resume(tmp_path, 1)
    resumed.run(smoke_position())
    rows = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert rows[0]['audit'] is None and rows[0]['elapsed_seconds'] is None
    assert rows[1]['audit']['checkpoint']['sha256'] == file_sha256(tmp_path / 'latest_model.pt')
