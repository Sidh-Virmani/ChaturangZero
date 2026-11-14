"""Optional fixed-search comparison of two saved or in-memory models."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from p2_neural_network import NetworkEvaluator, PolicyValueNetwork
from p3_mcts import MCTS, MCTSConfig
from shared.game_core import GameState, Side, apply_move, initial_state, terminal_status


@dataclass(frozen=True, slots=True)
class ArenaResult:
    candidate_wins: int
    reference_wins: int
    truncated: int
    games: int
    draws: int = 0


def play_match(candidate_evaluator, reference_evaluator, candidate_side: Side,
               state: GameState, search_config: MCTSConfig, max_game_plies: int) -> dict:
    """One game; fresh search per move gives both models the same search budget."""
    started = perf_counter()
    moves = []
    for _ in range(max_game_plies):
        if terminal_status(state).terminal:
            break
        evaluator = candidate_evaluator if state.side_to_move is candidate_side else reference_evaluator
        _, move, _ = MCTS(evaluator, search_config).search(state)
        state = apply_move(state, move)
        moves.append(move.uci)
    outcome = terminal_status(state)
    result = ('truncation' if not outcome.terminal else 'draw' if outcome.draw else
              'win' if outcome.winner is candidate_side else 'loss')
    return {'candidate_color': candidate_side.value, 'result': result,
            'winner': outcome.winner.value if outcome.winner else None,
            'terminal_reason': outcome.reason.value if outcome.reason else None,
            'stop_reason': outcome.reason.value if outcome.reason else 'max_game_plies',
            'plies': len(moves), 'final_ply_count': state.ply_count,
            'moves': moves, 'elapsed_seconds': perf_counter() - started}


def compare_models(
    candidate: PolicyValueNetwork,
    reference: PolicyValueNetwork,
    *,
    games: int = 2,
    simulations_per_move: int = 25,
    max_game_plies: int = 512,
    device: str = "cpu",
    start_state: GameState | None = None,
) -> ArenaResult:
    """Play deterministic-search games, alternating which color uses candidate."""
    if games <= 0 or simulations_per_move <= 0 or max_game_plies <= 0:
        raise ValueError("games, simulations_per_move, and max_game_plies must be positive")
    candidate_evaluator = NetworkEvaluator(candidate, device)
    reference_evaluator = NetworkEvaluator(reference, device)
    search_config = MCTSConfig(
        num_simulations=simulations_per_move, temperature=0.0, add_noise=False
    )
    candidate_wins = reference_wins = truncated = draws = 0

    for game_index in range(games):
        candidate_side = Side.WHITE if game_index % 2 == 0 else Side.BLACK
        record = play_match(candidate_evaluator, reference_evaluator, candidate_side,
                            initial_state() if start_state is None else start_state,
                            search_config, max_game_plies)
        if record['result'] == 'truncation':
            truncated += 1
        elif record['result'] == 'draw':
            draws += 1
        elif record['result'] == 'win':
            candidate_wins += 1
        else:
            reference_wins += 1

    return ArenaResult(candidate_wins, reference_wins, truncated, games, draws)
