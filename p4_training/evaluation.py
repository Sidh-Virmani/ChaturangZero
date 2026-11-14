"""Reproducible paired checkpoint evaluation and inspectable experiment files."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch

from p2_neural_network import ModelConfig, NetworkEvaluator, PolicyValueNetwork, load_checkpoint, save_checkpoint
from p3_mcts import MCTSConfig
from shared.game_core import Move, Side, apply_move, terminal_status

from .arena import play_match
from .openings import DEFAULT_SUITE, load_openings
from .recording import atomic_text, file_sha256, json_text, provenance


def freeze_baseline(output: Path, seed: int = 0, config: ModelConfig | None = None) -> Path:
    """Save an untrained P2 checkpoint once, without disturbing caller RNG state."""
    if seed < 0:
        raise ValueError('seed must be non-negative')
    output.mkdir(parents=True, exist_ok=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PolicyValueNetwork(config or ModelConfig())
        path = output / 'model.pt'
        save_checkpoint(model, path)
    atomic_text(output / 'baseline.json', json_text({
        'version': 1, 'kind': 'untrained', 'seed': seed,
        'model_config': asdict(model.config), 'sha256': file_sha256(path),
        'environment': provenance('cpu'),
    }))
    return path


def summarize(records: list[dict]) -> dict:
    """Scores describe this suite; unresolved games are never counted as draws."""
    counts = Counter(item['result'] for item in records)
    games = len(records)
    completed = games - counts['truncation']
    points = counts['win'] + 0.5 * counts['draw']
    by_opening: dict[str, list[dict]] = {}
    for record in records:
        by_opening.setdefault(record['opening_id'], []).append(record)
    pairs = []
    for opening_id, pair in by_opening.items():
        complete = len(pair) == 2 and all(item['result'] != 'truncation' for item in pair)
        pairs.append({'opening_id': opening_id, 'games': len(pair),
                      'resolved': complete,
                      'candidate_score': sum({'win': 1, 'draw': .5, 'loss': 0}[item['result']]
                                             for item in pair) / 2 if complete else None})
    return {
        'games': games, 'wins': counts['win'], 'losses': counts['loss'],
        'draws': counts['draw'], 'truncations': counts['truncation'],
        'completed_games': completed,
        'completed_game_score': points / completed if completed else None,
        'all_game_score_bounds': [points / games, (points + counts['truncation']) / games] if games else None,
        'average_plies': sum(item['plies'] for item in records) / games if games else None,
        'terminal_reasons': dict(Counter(item['terminal_reason'] for item in records if item['terminal_reason'])),
        'pairs': pairs,
        'statistics_note': 'Fixed-suite descriptive results. Bounds account for unresolved truncations, not statistical uncertainty. Paired games are not independent samples; no IID confidence interval or Elo is claimed.',
    }


def evaluate_checkpoints(
    candidate: Path, reference: Path, output: Path, *,
    openings_path: Path = DEFAULT_SUITE, pairs: int | None = None,
    simulations: int = 25, max_plies: int = 512, c_puct: float = 1.5,
    device: str = 'cpu', on_game: Callable[[dict], None] | None = None,
) -> dict:
    """Write a new experiment directory. Never write to either checkpoint."""
    search = MCTSConfig(num_simulations=simulations, c_puct=c_puct,
                        temperature=0.0, add_noise=False)
    if not isinstance(max_plies, int) or isinstance(max_plies, bool) or max_plies <= 0:
        raise ValueError('max_plies must be a positive integer')
    suite = load_openings(openings_path)
    if pairs is not None:
        if not isinstance(pairs, int) or isinstance(pairs, bool) or not 1 <= pairs <= len(suite):
            raise ValueError('pairs must be between 1 and the suite size; no repeated openings')
        suite = suite[:pairs]
    identities, models = {}, {}
    with torch.random.fork_rng(devices=[]):
        for role, path in (('candidate', candidate), ('reference', reference)):
            before = file_sha256(path)
            models[role] = load_checkpoint(path, device)
            if file_sha256(path) != before:
                raise RuntimeError('Checkpoint changed while loading; evaluate frozen copies')
            identities[role] = {'path': str(path.resolve()), 'sha256': before,
                                'model_config': asdict(models[role].config)}
    # Refuse reused output directories; aborted experiments remain inspectable.
    output.mkdir(parents=True, exist_ok=False)
    selected_suite = {'version': 1, 'openings': [asdict(item) for item in suite]}
    atomic_text(output / 'openings.json', json_text(selected_suite))
    manifest = {
        'version': 1, 'protocol': 'paired-fixed-openings-v1', 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'models': identities, 'environment': provenance(device),
        'openings_sha256': file_sha256(output / 'openings.json'),
        'config': {'pairs': len(suite), 'games': 2 * len(suite), 'search': asdict(search),
                   'max_plies_after_opening': max_plies, 'device': device,
                   'tree_policy': 'fresh MCTS per move; no cross-model tree reuse',
                   'tie_break': 'P3 deterministic action ordering'},
    }
    atomic_text(output / 'manifest.json', json_text(manifest))
    evaluators = {role: NetworkEvaluator(model, device) for role, model in models.items()}
    records = []
    started = perf_counter()
    with (output / 'games.jsonl').open('x', encoding='utf-8') as stream:
        for pair_index, opening in enumerate(suite):
            state = opening.state()  # Rebuild complete core history from validated moves.
            for color in (Side.WHITE, Side.BLACK):
                record = play_match(evaluators['candidate'], evaluators['reference'],
                                    color, state, search, max_plies)
                record.update({
                    'game': len(records) + 1, 'pair': pair_index + 1,
                    'opening_id': opening.id, 'opening_moves': list(opening.moves),
                    'candidate_sha256': identities['candidate']['sha256'],
                    'reference_sha256': identities['reference']['sha256'],
                    'search': asdict(search), 'max_plies_after_opening': max_plies,
                })
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                records.append(record)
                if on_game:
                    on_game(record)
    summary = summarize(records)
    summary['elapsed_seconds'] = perf_counter() - started
    atomic_text(output / 'summary.json', json_text(summary))
    with (output / 'games.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in record.items()})
    manifest['status'] = 'complete'
    manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_text(output / 'manifest.json', json_text(manifest))
    return summary


def verify_experiment(directory: Path) -> dict:
    """Replay exported moves through core and check completed report consistency."""
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['version'] != 1 or manifest['status'] != 'complete':
        raise ValueError('Experiment is incomplete or has an unsupported version')
    if file_sha256(directory / 'openings.json') != manifest['openings_sha256']:
        raise ValueError('Opening suite hash mismatch')
    openings = load_openings(directory / 'openings.json')
    records = [json.loads(line) for line in (directory / 'games.jsonl').read_text(encoding='utf-8').splitlines()]
    config = manifest['config']
    if len(records) != len(openings) * 2 or len(records) != config['games'] or len(openings) != config['pairs']:
        raise ValueError('Game/pair counts do not match the manifest')
    for index, record in enumerate(records):
        opening = openings[index // 2]
        side = Side.WHITE if index % 2 == 0 else Side.BLACK
        expected = {
            'game': index + 1, 'pair': index // 2 + 1, 'opening_id': opening.id,
            'opening_moves': list(opening.moves), 'candidate_color': side.value,
            'candidate_sha256': manifest['models']['candidate']['sha256'],
            'reference_sha256': manifest['models']['reference']['sha256'],
            'search': config['search'], 'max_plies_after_opening': config['max_plies_after_opening'],
        }
        state = opening.state()
        for uci in record['moves']:
            if not isinstance(uci, str) or len(uci) != 4:
                raise ValueError('Invalid recorded move')
            state = apply_move(state, Move.from_names(uci[:2], uci[2:]))
        outcome = terminal_status(state)
        count = len(record['moves'])
        if count > config['max_plies_after_opening'] or (not outcome.terminal and count != config['max_plies_after_opening']):
            raise ValueError('Game stopped outside the recorded ply cap')
        expected.update({
            'plies': count, 'final_ply_count': state.ply_count,
            'winner': outcome.winner.value if outcome.winner else None,
            'terminal_reason': outcome.reason.value if outcome.reason else None,
            'stop_reason': outcome.reason.value if outcome.reason else 'max_game_plies',
            'result': ('truncation' if not outcome.terminal else 'draw' if outcome.draw else
                       'win' if outcome.winner is side else 'loss'),
        })
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Game {index + 1} record does not match replay/manifest')
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    expected_summary = summarize(records)
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError('Summary does not match game records')
    with (directory / 'games.csv').open(newline='', encoding='utf-8') as stream:
        csv_rows = list(csv.DictReader(stream))
    expected_csv = [{key: json.dumps(value) if isinstance(value, (dict, list)) else
                     '' if value is None else str(value) for key, value in record.items()}
                    for record in records]
    if csv_rows != expected_csv:
        raise ValueError('CSV does not match JSONL')
    return {'verified_games': len(records), 'verified_pairs': len(openings)}
