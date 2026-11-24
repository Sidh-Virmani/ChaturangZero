"""Checkpoint provenance is optional, read-only and tied to exact model bytes."""
import hashlib
import json

from integration.model_info import checkpoint_identity
from shared.game_core import Move, apply_move, initial_state
from shared.game_core.server import GameSession


def test_metadata_uses_hash_and_snapshot_iteration_not_filename_alone(tmp_path):
    path = tmp_path / 'iteration_000020.pt'
    path.write_bytes(b'frozen checkpoint bytes')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    def row(iteration, sha):
        return {'iteration': iteration, 'audit': {'checkpoint': {'sha256': sha},
            'config': {'simulations_per_move': 50, 'self_play_workers': 6,
                       'device': 'cuda', 'replay_capacity': 20000, 'symmetry_probability': .5}}}
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text('\n'.join(map(json.dumps, [row(20, 'wrong hash'), row(10, digest)])))
    assert checkpoint_identity(path) == (digest, None)
    metrics.write_text('\n'.join(map(json.dumps, [row(20, digest), row(30, digest)])))
    sha, data = checkpoint_identity(path)
    assert sha == digest and data['iteration'] == 20
    assert data['workers'] == 6 and data['device'] == 'cuda'
    assert data['symmetryProbability'] == .5
    assert path.read_bytes() == b'frozen checkpoint bytes'


def test_missing_corrupt_and_run_metadata_fallback(tmp_path):
    path = tmp_path / 'model.pt'
    path.write_bytes(b'checkpoint')
    sha, data = checkpoint_identity(path)
    assert data is None
    (tmp_path / 'metrics.jsonl').write_text('bad json\nnull\n{}\n')
    (tmp_path / 'run.json').write_text(json.dumps({'latest_model': {'sha256': sha},
        'completed_iterations': 200, 'config': {'device': 'cuda'}}))
    assert checkpoint_identity(path)[1]['iteration'] == 200
    (tmp_path / 'run.json').write_text('{corrupt')
    assert checkpoint_identity(path) == (sha, None)


def test_live_repetition_count_uses_core_position_history():
    session = GameSession()
    session.mode = 'machine'
    state = initial_state()
    cycle = ('b1a3', 'b8a6', 'a3b1', 'a6b8')
    for uci in cycle:
        state = apply_move(state, Move.from_names(uci[:2], uci[2:]))
    session.game.state = state
    assert session.state_dict()['repetition'] == {'occurrences': 2, 'drawAt': 3}
    for uci in cycle:
        state = apply_move(state, Move.from_names(uci[:2], uci[2:]))
    session.game.state = state
    result = session.state_dict()
    assert result['repetition']['occurrences'] == 3
    assert result['status']['result'] == 'Draw'
