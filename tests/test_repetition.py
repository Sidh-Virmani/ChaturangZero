"""Central three-occurrence rule, including search and data consumers."""
from dataclasses import replace

import numpy as np
import pytest

from p1_representation import ACTION_SPACE_SIZE, legal_action_mask, move_to_action, rotate_state
from p3_mcts import MCTS, MCTSNode
from p4_training import TrainingConfig, compare_models, play_game
from shared.game_core import (
    GameOverError, Move, Side, TerminalReason, apply_move, clone_state,
    initial_state, legal_moves, terminal_status,
)
from shared.game_core.core import _apply_known_legal_move


CYCLE = ('b1a3', 'b8a6', 'a3b1', 'a6b8')


def cycle_state(plies):
    state = initial_state()
    for index in range(plies):
        uci = CYCLE[index % 4]
        move = Move.from_names(uci[:2], uci[2:])
        assert _apply_known_legal_move(state, move) == apply_move(state, move)
        state = apply_move(state, move)
    return state


def test_third_occurrence_is_draw_and_history_is_immutable():
    first, second, third = initial_state(), cycle_state(4), cycle_state(8)
    assert first.board == second.board == third.board
    assert not terminal_status(second).terminal
    outcome = terminal_status(third)
    assert outcome.terminal and outcome.draw and outcome.winner is None
    assert outcome.reason is TerminalReason.REPETITION
    assert outcome.reward(Side.WHITE) == outcome.reward(Side.BLACK) == 0
    assert len(third.position_history) == 8
    assert first.position_history == ()
    assert clone_state(third) == third
    assert legal_moves(third) == ()
    assert not legal_action_mask(third).any()
    with pytest.raises(GameOverError):
        apply_move(third, ('b1', 'a3'))
    assert not terminal_status(replace(third, side_to_move=Side.BLACK)).terminal
    assert not terminal_status(replace(third, position_history=())).terminal
    assert terminal_status(rotate_state(third)).draw
    assert rotate_state(rotate_state(third)) == third


def test_draw_search_leaf_backpropagates_zero_and_root_rejects_search():
    before = cycle_state(7)
    parent = MCTSNode(before)
    move = Move.from_names('a6', 'b8')
    action = move_to_action(move)
    parent.expand([(move_to_action(m), m) for m in parent.legal_moves],
                  np.ones(ACTION_SPACE_SIZE))
    leaf = parent.materialize_child(action)
    assert leaf.is_terminal and leaf.terminal_value == 0
    assert leaf.children == {} and leaf.legal_moves == ()
    MCTS._backpropagate(leaf, leaf.terminal_value)
    assert leaf.visit_count == parent.visit_count == 1
    assert leaf.value_sum == parent.value_sum == 0
    class NeverEvaluate:
        def evaluate(self, state):
            pytest.fail('draw must not query network')
    with pytest.raises(ValueError, match='terminal'):
        MCTS(NeverEvaluate()).search(leaf.state)


class CycleSearch:
    """Fixed legal knight cycle; termination is still decided by the core."""
    def __init__(self, *args):
        pass

    def search(self, state):
        uci = CYCLE[state.ply_count % 4]
        move = Move.from_names(uci[:2], uci[2:])
        assert move in legal_moves(state)
        action = move_to_action(move)
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        policy[action] = 1
        return policy, move, action

    def step(self, move):
        pass


@pytest.mark.parametrize('cap,draw', [(7, False), (8, True), (20, True)])
def test_self_play_retains_zero_targets_only_for_recognized_draw(tmp_path, monkeypatch, cap, draw):
    monkeypatch.setattr('p4_training.self_play.MCTS', CycleSearch)
    config = TrainingConfig(checkpoint_dir=tmp_path, max_game_plies=cap)
    result = play_game(None, config)
    assert result.truncated is not draw
    assert result.plies == (8 if draw else 7)
    if draw:
        assert result.outcome.draw
        assert len(result.examples) == 8
        assert all(example.value == 0 for example in result.examples)
    else:
        assert result.outcome is None and result.examples == ()


@pytest.mark.parametrize('cap,draws,truncated', [(7, 0, 2), (8, 2, 0)])
def test_arena_separates_draws_from_truncations(monkeypatch, cap, draws, truncated):
    monkeypatch.setattr('p4_training.arena.MCTS', CycleSearch)
    monkeypatch.setattr('p4_training.arena.NetworkEvaluator', lambda *args: None)
    result = compare_models(None, None, games=2, max_game_plies=cap)
    assert result.draws == draws and result.truncated == truncated
    assert result.candidate_wins == result.reference_wins == 0
    assert result.games == result.draws + result.truncated


def test_server_reports_draw_and_reset_clears_engine_history():
    from shared.game_core.server import GameSession
    session = GameSession()
    for uci in CYCLE * 2:
        result = session.apply(uci[:2], uci[2:])
    assert result['status']['code'] == 'terminal'
    assert result['status']['result'] == 'Draw'
    assert result['selectableSquares'] == []
    assert 'Draw by repetition' in result['status']['detail']
    session.reset()
    assert session.game.state.position_history == ()


def test_mcts_subtree_keeps_history_but_unrelated_root_does_not():
    from p2_neural_network import Evaluation
    from p3_mcts import MCTSConfig
    class Evaluator:
        def evaluate(self, state):
            policy = legal_action_mask(state).astype(np.float32)
            return Evaluation(policy / policy.sum(), 0.0)
    state = cycle_state(7)
    search = MCTS(Evaluator(), MCTSConfig(num_simulations=1, temperature=0))
    search.search(state)
    search.step(Move.from_names('a6', 'b8'))
    assert search.root.is_terminal and search.root.terminal_value == 0
    assert len(search.root.state.position_history) == 8
    # Same visible board is non-terminal when supplied as a fresh game.
    fresh = initial_state()
    search.search(fresh)
    assert search.root.state == fresh and not search.root.is_terminal


def test_training_metrics_count_completed_draws_without_running_updates(tmp_path, monkeypatch):
    from p2_neural_network import ModelConfig
    from p4_training import Trainer
    monkeypatch.setattr('p4_training.self_play.MCTS', CycleSearch)
    config = TrainingConfig(
        checkpoint_dir=tmp_path, games_per_iteration=1, max_game_plies=8,
        model=ModelConfig(channels=8, residual_blocks=1, value_hidden=16),
    )
    trainer = Trainer(config)
    monkeypatch.setattr(trainer, '_train_steps', lambda updates: [])
    metrics = trainer.run()[0]
    assert metrics.games_completed == metrics.draws == 1
    assert metrics.games_truncated == metrics.white_wins == metrics.black_wins == 0
    assert metrics.examples_added == 8
    assert all(e.value == 0 for e in trainer.replay.sample(8, trainer.sample_rng))
    resumed = Trainer.resume(tmp_path, iterations=1)
    assert resumed.history[0].draws == 1
