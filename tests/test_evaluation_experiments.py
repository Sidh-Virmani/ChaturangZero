"""Paired protocol, real moves, artifact integrity and conservative scoring."""
from dataclasses import asdict
import csv
import json

import pytest
import torch

from p2_neural_network import ModelConfig, load_checkpoint
from p3_mcts import MCTSConfig
from p4_training.arena import play_match
from p4_training.evaluation import evaluate_checkpoints, freeze_baseline, summarize, verify_experiment
from p4_training.openings import DEFAULT_SUITE, Opening, generate_openings, load_openings, validate_openings
from p4_training.recording import file_sha256
from p4_training.self_play import smoke_position
from shared.game_core import Move, Side, apply_move, initial_state, terminal_status


@pytest.fixture
def baseline(tmp_path):
    return freeze_baseline(tmp_path / 'baseline', seed=3,
                           config=ModelConfig(channels=8, residual_blocks=1, value_hidden=16))


def read_rows(directory):
    return [json.loads(line) for line in (directory / 'games.jsonl').read_text().splitlines()]


def test_openings_fixed_distinct_legal_history_and_generator_reproducible():
    openings = load_openings()
    assert len(openings) == 12
    states = [opening.state() for opening in openings]
    assert len({(state.board, state.side_to_move) for state in states}) == 12
    assert all(not terminal_status(state).terminal for state in states)
    assert all(len(state.position_history) == len(opening.moves)
               for state, opening in zip(states, openings))
    assert {state.side_to_move for state in states} == set(Side)
    assert generate_openings(12, 20261114) == json.loads(DEFAULT_SUITE.read_text())
    assert generate_openings(3, 5) == generate_openings(3, 5)
    assert generate_openings(3, 6) != generate_openings(3, 5)


@pytest.mark.parametrize('openings', [(), (Opening('a', ()), Opening('b', ())),
                                     (Opening('a', ()), Opening('a', ('b1a3',)))])
def test_reject_empty_or_duplicate_suites(openings):
    with pytest.raises(ValueError):
        validate_openings(openings)


def test_reject_illegal_and_terminal_openings():
    with pytest.raises(ValueError):
        validate_openings((Opening('illegal', ('a1h8',)),))
    with pytest.raises(ValueError, match='nonterminal'):
        validate_openings((Opening('draw', ('b1a3', 'b8a6', 'a3b1', 'a6b8') * 2),))


def test_baseline_frozen_seeded_rng_safe_and_no_overwrite(tmp_path, baseline):
    before = torch.get_rng_state().clone()
    other = freeze_baseline(tmp_path / 'other', 3, ModelConfig(channels=8, residual_blocks=1, value_hidden=16))
    assert torch.equal(before, torch.get_rng_state())
    a, b = load_checkpoint(baseline), load_checkpoint(other)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    original = baseline.read_bytes()
    with pytest.raises(FileExistsError):
        freeze_baseline(baseline.parent)
    assert baseline.read_bytes() == original


def test_real_paired_experiment_reproducible_exports_and_checkpoint_unchanged(tmp_path, baseline):
    before = baseline.read_bytes()
    results = []
    for name in ('first', 'rerun'):
        path = tmp_path / name
        summary = evaluate_checkpoints(baseline, baseline, path, pairs=3,
                                       simulations=1, max_plies=2)
        rows = read_rows(path)
        assert summary['games'] == summary['truncations'] == 6
        assert summary['completed_game_score'] is None
        assert summary['all_game_score_bounds'] == [0, 1]
        for index in range(0, 6, 2):
            assert rows[index]['opening_id'] == rows[index + 1]['opening_id']
            assert [rows[index]['candidate_color'], rows[index + 1]['candidate_color']] == ['white', 'black']
            assert rows[index]['moves'] == rows[index + 1]['moves']  # identical models
        for row in rows:
            state = initial_state()
            for uci in row['opening_moves'] + row['moves']:
                state = apply_move(state, Move.from_names(uci[:2], uci[2:]))
            assert state.ply_count == row['final_ply_count']
            assert not terminal_status(state).terminal
            assert row['terminal_reason'] is None and row['stop_reason'] == 'max_game_plies'
        manifest = json.loads((path / 'manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['models']['candidate']['sha256'] == file_sha256(baseline)
        assert manifest['openings_sha256'] == file_sha256(path / 'openings.json')
        assert manifest['config']['search'] == asdict(MCTSConfig(num_simulations=1, temperature=0))
        assert len(manifest['environment']['ai_source_sha256']) == 64
        assert len(load_openings(path / 'openings.json')) == 3
        assert verify_experiment(path) == {'verified_games': 6, 'verified_pairs': 3}
        with (path / 'games.csv').open(newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        assert [json.loads(row['moves']) for row in csv_rows] == [row['moves'] for row in rows]
        results.append([{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in rows])
    assert results[0] == results[1]
    assert baseline.read_bytes() == before
    with pytest.raises(FileExistsError):
        evaluate_checkpoints(baseline, baseline, tmp_path / 'first', simulations=1, max_plies=1)


def test_game_records_wins_and_real_repetition_separate_from_caps(monkeypatch):
    class Cycle:
        def __init__(self, *args):
            pass
        def search(self, state):
            uci = ('b1a3', 'b8a6', 'a3b1', 'a6b8')[state.ply_count % 4]
            return None, Move.from_names(uci[:2], uci[2:]), None
    monkeypatch.setattr('p4_training.arena.MCTS', Cycle)
    for cap, result in ((7, 'truncation'), (8, 'draw')):
        row = play_match(None, None, Side.BLACK, initial_state(), MCTSConfig(), cap)
        assert row['result'] == result
        assert row['winner'] is None
        assert row['terminal_reason'] == ('repetition' if cap == 8 else None)
    from shared.game_core import legal_moves
    ended = apply_move(smoke_position(), legal_moves(smoke_position())[0])
    for side, result in ((Side.WHITE, 'win'), (Side.BLACK, 'loss')):
        row = play_match(None, None, side, ended, MCTSConfig(), 2)
        assert row['result'] == result and row['plies'] == 0


def test_scores_do_not_hide_unresolved_pairs():
    rows = [{'result': result, 'opening_id': str(i // 2), 'plies': i + 1,
             'terminal_reason': 'repetition' if result == 'draw' else None}
            for i, result in enumerate(('win', 'loss', 'draw', 'truncation'))]
    result = summarize(rows)
    assert result['completed_game_score'] == .5
    assert result['all_game_score_bounds'] == [.375, .625]
    assert result['pairs'][0]['candidate_score'] == .5
    assert result['pairs'][1]['candidate_score'] is None
    assert result['average_plies'] == 2.5


def test_interruption_keeps_completed_records_and_running_manifest(tmp_path, baseline):
    def interrupt(_):
        raise KeyboardInterrupt
    output = tmp_path / 'interrupted'
    with pytest.raises(KeyboardInterrupt):
        evaluate_checkpoints(baseline, baseline, output, pairs=2,
                             simulations=1, max_plies=1, on_game=interrupt)
    assert len(read_rows(output)) == 1
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'running'
    assert not (output / 'summary.json').exists()


def test_invalid_pair_count_rejected_before_creating_output(tmp_path, baseline):
    with pytest.raises(ValueError, match='pairs'):
        evaluate_checkpoints(baseline, baseline, tmp_path / 'invalid', pairs=99)
    assert not (tmp_path / 'invalid').exists()


def test_draw_exports_verify_and_tampered_score_is_rejected(tmp_path, baseline, monkeypatch):
    class Cycle:
        def __init__(self, *args):
            pass
        def search(self, state):
            uci = ('b1a3', 'b8a6', 'a3b1', 'a6b8')[state.ply_count % 4]
            return None, Move.from_names(uci[:2], uci[2:]), None
    monkeypatch.setattr('p4_training.arena.MCTS', Cycle)
    output = tmp_path / 'draws'
    summary = evaluate_checkpoints(baseline, baseline, output, pairs=1, simulations=1, max_plies=8)
    assert summary['draws'] == 2 and summary['truncations'] == 0
    assert summary['completed_game_score'] == .5
    assert verify_experiment(output)['verified_games'] == 2
    summary['wins'] = 1
    (output / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Summary'):
        verify_experiment(output)


def test_cli_evaluate_and_verify(tmp_path, baseline, monkeypatch, capsys):
    from p4_training.__main__ import main
    output = tmp_path / 'cli'
    monkeypatch.setattr('sys.argv', ['p4_training', 'evaluate', '--candidate', str(baseline),
        '--reference', str(baseline), '--output', str(output), '--pairs', '1',
        '--simulations', '1', '--max-plies', '1'])
    main()
    assert 'W/L/D/T: 0/0/0/2' in capsys.readouterr().out
    monkeypatch.setattr('sys.argv', ['p4_training', 'verify-evaluation', '--output', str(output)])
    main()
    assert json.loads(capsys.readouterr().out)['verified_games'] == 2
