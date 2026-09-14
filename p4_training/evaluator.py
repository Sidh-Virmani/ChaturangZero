"""
p4_training/evaluator.py
========================
Candidate-vs-best model arena evaluation.

After training a new candidate model, play `num_games` games between
the candidate and the current best model.  Promote the candidate if it
wins >= `win_threshold` fraction of games.

This prevents a bad training step from permanently degrading performance.
"""

from __future__ import annotations

import random
import sys
import pathlib

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from p1_representation.encoder import StateEncoder
from p3_mcts.search import MCTSSearch


def arena_match(
    engine:        ChaturangaEngine,
    candidate,                           # ChaturangaNet
    best_model,                          # ChaturangaNet
    num_games:     int   = 20,
    num_sims:      int   = 25,
    device:        str   = "cpu",
    win_threshold: float = 0.55,
    verbose:       bool  = True,
) -> tuple[bool, int, int, int]:
    """
    Play `num_games` between candidate and best model.
    Half the games, candidate plays White; the other half, Black.

    Returns
    -------
    should_promote : True if candidate wins >= win_threshold
    wins           : candidate wins
    losses         : candidate losses
    draws          : drawn games
    """
    candidate_wins = 0
    best_wins      = 0
    draws          = 0

    half = num_games // 2

    for game_i in range(num_games):
        # Alternate who plays White
        if game_i < half:
            white_model, black_model = candidate, best_model
            candidate_side = "white"
        else:
            white_model, black_model = best_model, candidate
            candidate_side = "black"

        winner = _play_one_game(
            engine, white_model, black_model, num_sims, device
        )

        if winner == "draw":
            draws += 1
        elif winner == candidate_side:
            candidate_wins += 1
        else:
            best_wins += 1

        if verbose:
            print(
                f"  [arena] game {game_i+1}/{num_games}  "
                f"winner={winner}  "
                f"cand={candidate_wins} best={best_wins} draw={draws}"
            )

    total      = num_games
    win_rate   = (candidate_wins + 0.5 * draws) / total
    should_promote = win_rate >= win_threshold

    print(
        f"\n[arena] Result: candidate W/L/D = {candidate_wins}/{best_wins}/{draws}  "
        f"win_rate={win_rate:.1%}  promote={should_promote}\n"
    )
    return should_promote, candidate_wins, best_wins, draws


def _play_one_game(
    engine,
    white_model,
    black_model,
    num_sims: int,
    device:   str,
    max_moves: int = 150,
) -> str:
    """Return 'white', 'black', or 'draw'."""
    state = engine.new_game()

    from shared.game_core.state import Side
    mcts_white = MCTSSearch(
        engine, model=white_model, num_simulations=num_sims,
        temperature=0.0, device=device
    )
    mcts_black = MCTSSearch(
        engine, model=black_model, num_simulations=num_sims,
        temperature=0.0, device=device
    )

    for _ in range(max_moves):
        if engine.is_terminal(state):
            break
        if state.current_turn == Side.WHITE:
            move = mcts_white.best_move(state)
        else:
            move = mcts_black.best_move(state)

        if move is None:
            break
        state = engine.apply_move(state, move)

    if not engine.is_terminal(state):
        return "draw"
    return state.winner or "draw"
