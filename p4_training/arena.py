"""Optional fixed-search comparison of two saved or in-memory models."""

from __future__ import annotations

from dataclasses import dataclass

from p2_neural_network import NetworkEvaluator, PolicyValueNetwork
from p3_mcts import MCTS, MCTSConfig
from shared.game_core import GameState, Side, apply_move, initial_state, terminal_status


@dataclass(frozen=True, slots=True)
class ArenaResult:
    candidate_wins: int
    reference_wins: int
    truncated: int
    games: int


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
    candidate_wins = reference_wins = truncated = 0

    for game_index in range(games):
        candidate_side = Side.WHITE if game_index % 2 == 0 else Side.BLACK
        state = initial_state() if start_state is None else start_state
        for _ in range(max_game_plies):
            outcome = terminal_status(state)
            if outcome.terminal:
                break
            evaluator = (
                candidate_evaluator if state.side_to_move is candidate_side
                else reference_evaluator
            )
            _, move, _ = MCTS(evaluator, search_config).search(state)
            state = apply_move(state, move)

        outcome = terminal_status(state)
        if not outcome.terminal:
            truncated += 1
        elif outcome.winner is candidate_side:
            candidate_wins += 1
        else:
            reference_wins += 1

    return ArenaResult(candidate_wins, reference_wins, truncated, games)
