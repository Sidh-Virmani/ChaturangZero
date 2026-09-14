"""
p4_training/self_play.py
========================
Generates self-play training data using MCTS + the current model.

Each game produces a list of training examples:
    (encoded_state, mcts_policy, outcome)

where `outcome` is the final game result from the perspective of the
player who was to move at that state:  +1 win, -1 loss, 0 draw.
"""

from __future__ import annotations

import sys
import pathlib
import random
from dataclasses import dataclass

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Side
from p1_representation.encoder import StateEncoder, ACTION_SIZE
from p3_mcts.search import MCTSSearch


@dataclass
class TrainingExample:
    encoded_state: np.ndarray   # (13, 8, 8)
    mcts_policy:   np.ndarray   # (4096,) — normalised visit counts
    value:         float        # game result from this player's POV


def play_game(
    engine:          ChaturangaEngine,
    mcts:            MCTSSearch,
    use_canonical:   bool = True,
    augment:         bool = True,
    max_moves:       int  = 200,
    temp_threshold:  int  = 15,   # Use temperature=1 for first N plies, then 0
) -> list[TrainingExample]:
    """
    Play one complete self-play game.

    Parameters
    ----------
    engine        : game engine
    mcts          : configured MCTSSearch (model may be None for random play)
    use_canonical : encode from current player's POV
    augment       : add horizontally mirrored examples
    max_moves     : safety cap — draw if game doesn't end by this ply
    temp_threshold: switch from high-temp to greedy selection after this ply

    Returns
    -------
    list of TrainingExample (one per ply, optionally doubled by augmentation)
    """
    state   = engine.new_game()
    encoder = StateEncoder()
    buffer: list[tuple[np.ndarray, np.ndarray, Side]] = []
    # buffer items: (encoded_state, mcts_policy, turn_side)
    # values are filled in after the game ends

    ply = 0
    while not engine.is_terminal(state) and ply < max_moves:
        # Set MCTS temperature based on ply
        mcts.temperature = 1.0 if ply < temp_threshold else 0.0

        # Run MCTS
        policy, _ = mcts.search(state)

        # Encode state
        if use_canonical:
            enc = encoder.encode_canonical(state)
        else:
            enc = encoder.encode_state(state)

        buffer.append((enc, policy.copy(), state.current_turn))

        # Sample move from MCTS policy
        legal     = engine.legal_moves(state)
        legal_idx = [StateEncoder.encode_action(m) for m in legal]
        legal_probs = np.array([policy[i] for i in legal_idx], dtype=np.float32)
        s = legal_probs.sum()
        if s <= 0:
            legal_probs = np.ones(len(legal), dtype=np.float32) / len(legal)
        else:
            legal_probs /= s

        chosen_idx = np.random.choice(len(legal), p=legal_probs)
        move = legal[chosen_idx]
        state = engine.apply_move(state, move)
        ply += 1

    # Determine game result
    if engine.is_terminal(state):
        winner = state.winner   # 'white', 'black', or 'draw'
    else:
        winner = "draw"   # max_moves reached

    # Build training examples
    examples: list[TrainingExample] = []
    for enc, pol, turn in buffer:
        if winner == "draw":
            v = 0.0
        elif winner == turn.value:
            v = 1.0
        else:
            v = -1.0

        if augment:
            aug = encoder.augment_example(enc, int(np.argmax(pol)), v)
            for aug_enc, aug_idx, aug_v in aug:
                aug_pol = np.zeros(ACTION_SIZE, dtype=np.float32)
                aug_pol[aug_idx] = 1.0
                examples.append(TrainingExample(aug_enc, aug_pol, aug_v))
        else:
            examples.append(TrainingExample(enc, pol, v))

    return examples


def run_self_play_batch(
    engine:      ChaturangaEngine,
    model,
    num_games:   int = 25,
    num_sims:    int = 50,
    device:      str = "cpu",
    verbose:     bool = True,
) -> list[TrainingExample]:
    """
    Play `num_games` self-play games and collect all training examples.
    """
    mcts   = MCTSSearch(engine, model=model, num_simulations=num_sims, device=device)
    all_ex: list[TrainingExample] = []

    for i in range(num_games):
        ex = play_game(engine, mcts)
        all_ex.extend(ex)
        if verbose:
            print(f"  [self-play] game {i+1}/{num_games}: {len(ex)} examples generated")

    return all_ex
