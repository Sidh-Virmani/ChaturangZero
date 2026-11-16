"""Generate completed-game examples using P3 search and game-core outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter, process_time

import numpy as np

from p1_representation import action_to_move, encode_state
from p2_neural_network import NetworkEvaluator
from p3_mcts import MCTS, MCTSConfig
from shared.game_core import (
    GameState, Outcome, Piece, PieceType, Side, apply_move, initial_state,
    terminal_status,
)

from .config import TrainingConfig
from .data import TrainingExample


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    examples: tuple[TrainingExample, ...]
    outcome: Outcome | None
    plies: int
    moves: tuple[str, ...] = ()
    mean_policy_entropy: float = 0.0
    mean_visited_actions: float = 0.0
    elapsed_seconds: float = field(default=0.0, compare=False)
    cpu_seconds: float = field(default=0.0, compare=False)

    @property
    def truncated(self) -> bool:
        return self.outcome is None


def smoke_position() -> GameState:
    """Small valid core position whose legal replies all capture the last Ratha."""
    return GameState.from_mapping({
        "h1": Piece(Side.WHITE, PieceType.RAJA),
        "g1": Piece(Side.WHITE, PieceType.MANTRI),
        "g2": Piece(Side.WHITE, PieceType.RATHA),
        "h2": Piece(Side.BLACK, PieceType.RATHA),
        "a8": Piece(Side.BLACK, PieceType.RAJA),
    })


def play_game(
    evaluator: NetworkEvaluator,
    config: TrainingConfig,
    start_state: GameState | None = None,
) -> SelfPlayResult:
    """Play one game; omit examples if the core has no result by the ply cap."""
    state = initial_state() if start_state is None else start_state
    started, cpu_started = perf_counter(), process_time()
    played_moves: list[str] = []
    entropy_sum = visited_sum = 0.0
    search = MCTS(
        evaluator,
        MCTSConfig(
            num_simulations=config.simulations_per_move,
            temperature=1.0,
            add_noise=True,
        ),
    )
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray, Side]] = []
    plies = 0

    def finish(outcome):
        examples = tuple(
            TrainingExample(board, actions, probabilities, float(outcome.reward(side)))
            for board, actions, probabilities, side in pending
        ) if outcome is not None else ()
        return SelfPlayResult(examples, outcome, plies, tuple(played_moves),
            entropy_sum / plies if plies else 0., visited_sum / plies if plies else 0.,
            perf_counter() - started, process_time() - cpu_started)

    while plies < config.max_game_plies:
        outcome = terminal_status(state)
        if outcome.terminal:
            return finish(outcome)

        policy, sampled_move, _ = search.search(state)
        actions = np.flatnonzero(policy).astype(np.int16)
        probabilities = policy[actions].astype(np.float32, copy=True)
        entropy_sum += float(-np.sum(probabilities * np.log(probabilities)))
        visited_sum += len(actions)
        pending.append((encode_state(state).astype(np.uint8), actions, probabilities, state.side_to_move))

        move = (
            sampled_move if plies < config.opening_temperature_plies
            else action_to_move(int(np.argmax(policy)))
        )
        state = apply_move(state, move)
        search.step(move)
        played_moves.append(move.uci)
        plies += 1

    outcome = terminal_status(state)
    if outcome.terminal:
        return finish(outcome)
    return finish(None)
