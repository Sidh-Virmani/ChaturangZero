"""Read-only, hash-matched training provenance for the demo; never loads replay."""
import hashlib
import json
import re
from pathlib import Path


def checkpoint_identity(path: Path) -> tuple[str, dict | None]:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    training = None
    snapshot = re.fullmatch(r'iteration_(\d+)\.pt', path.name)
    iteration = int(snapshot[1]) if snapshot else None
    try:
        with (path.parent / 'metrics.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    audit = row.get('audit') or {}
                    if audit.get('checkpoint', {}).get('sha256') != digest:
                        continue
                    if iteration is not None and row.get('iteration') != iteration:
                        continue
                    config = audit['config']
                    training = {
                        'iteration': row['iteration'],
                        'selfPlaySimulations': config.get('simulations_per_move'),
                        'workers': config.get('self_play_workers'),
                        'replayCapacity': config.get('replay_capacity'),
                        'symmetryProbability': config.get('symmetry_probability'),
                        'device': config.get('device'),
                        'gitCommit': audit.get('environment', {}).get('git_commit'),
                        'source': 'metrics.jsonl · checkpoint SHA-256 matched',
                    }
                except (ValueError, TypeError, KeyError, AttributeError):
                    continue
    except (OSError, UnicodeError):
        pass
    # Older runs can still identify their exact latest model with run.json.
    if training is None:
        try:
            run = json.loads((path.parent / 'run.json').read_text(encoding='utf-8'))
            if run['latest_model']['sha256'] == digest and (iteration is None or iteration == run['completed_iterations']):
                config = run['config']
                training = {'iteration': run['completed_iterations'],
                            'selfPlaySimulations': config.get('simulations_per_move'),
                            'workers': config.get('self_play_workers'),
                            'replayCapacity': config.get('replay_capacity'),
                            'symmetryProbability': config.get('symmetry_probability'),
                            'device': config.get('device'),
                            'gitCommit': run.get('environment', {}).get('git_commit'),
                            'source': 'run.json · checkpoint SHA-256 matched'}
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError):
            pass
    return digest, training
