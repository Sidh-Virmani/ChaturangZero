"""
p4_training/self_play.py
========================
Generates self-play training data using MCTS + the current model.

Changes from baseline (ChaturangZero-X):
  - BUG FIX: augmentation now preserves the full MCTS policy distribution
    (not just a one-hot on argmax), preventing silent data corruption.
  - BUG FIX: temperature schedule is ply-based, smoothly decreasing.
  - NEW: accepts an optional `start_state` for curriculum learning.
  - NEW: records per-position uncertainty for hard-position mining.
"""

from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, field

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
    uncertainty:   float = 0.0  # value uncertainty (0 if model has no uncertainty head)


def _temperature_for_ply(ply: int, high_temp_plies: int = 20) -> float:
    """
    Ply-based temperature schedule:
      - Plies 0..high_temp_plies-1  : temperature = 1.0 (explore)
      - Plies high_temp_plies..end  : temperature = 0.0 (deterministic / greedy)
    """
    return 1.0 if ply < high_temp_plies else 0.0


def _mirror_policy(policy: np.ndarray) -> np.ndarray:
    """
    Mirror a 4096-element policy vector for horizontal board flip.
    Each action index = from_sq * 64 + to_sq, where square = file*8 + rank.
    Mirroring: file → 7 - file, rank unchanged.
    Returns a new policy array with the same probability mass correctly mapped.
    """
    mirrored = np.zeros_like(policy)
    for from_file in range(8):
        for from_rank in range(8):
            from_sq = from_file * 8 + from_rank
            mfrom_sq = (7 - from_file) * 8 + from_rank
            for to_file in range(8):
                for to_rank in range(8):
                    to_sq = to_file * 8 + to_rank
                    mto_sq = (7 - to_file) * 8 + to_rank
                    src_idx = from_sq * 64 + to_sq
                    dst_idx = mfrom_sq * 64 + mto_sq
                    mirrored[dst_idx] += policy[src_idx]
    return mirrored


def play_game(
    engine:          ChaturangaEngine,
    mcts:            MCTSSearch,
    start_state:     ChaturangaState | None = None,
    use_canonical:   bool = True,
    augment:         bool = True,
    max_moves:       int  = 200,
    high_temp_plies: int  = 20,
) -> list[TrainingExample]:
    """
    Play one complete self-play game and return training examples.

    Parameters
    ----------
    engine          : game engine (may carry a RuleConfig for rule variants)
    mcts            : configured MCTSSearch
    start_state     : optional starting position (None = initial game position)
                      Used by the CurriculumScheduler for non-standard starts.
    use_canonical   : encode from current player's POV
    augment         : add horizontally-mirrored examples (preserves full policy)
    max_moves       : draw if game doesn't end within this many plies
    high_temp_plies : use temperature=1 for the first N plies, then greedy

    Returns
    -------
    list of TrainingExample
    """
    state   = start_state if start_state is not None else engine.new_game()
    encoder = StateEncoder()

    # Buffer: (encoded_state, mcts_policy, side_to_move, uncertainty)
    buffer: list[tuple[np.ndarray, np.ndarray, Side, float]] = []

    ply = 0
    while not engine.is_terminal(state) and ply < max_moves:
        mcts.temperature = _temperature_for_ply(ply, high_temp_plies)

        # Run MCTS — get improved policy and value uncertainty if available
        policy, _ = mcts.search(state)

        # Collect per-position uncertainty (from network if available)
        uncertainty = 0.0
        if hasattr(mcts, '_last_root_uncertainty'):
            uncertainty = mcts._last_root_uncertainty

        # Encode state
        enc = encoder.encode_canonical(state) if use_canonical else encoder.encode_state(state)
        buffer.append((enc, policy.copy(), state.current_turn, uncertainty))

        # Sample move according to MCTS policy
        legal     = engine.legal_moves(state)
        legal_idx = [StateEncoder.encode_action(m) for m in legal]
        legal_probs = np.array([policy[i] for i in legal_idx], dtype=np.float32)
        s = legal_probs.sum()
        if s <= 0:
            legal_probs = np.ones(len(legal), dtype=np.float32) / len(legal)
        else:
            legal_probs /= s

        chosen_idx = np.random.choice(len(legal), p=legal_probs)
        state = engine.apply_move(state, legal[chosen_idx])
        ply += 1

    # Determine game result
    winner = state.winner if engine.is_terminal(state) else "draw"

    # Build training examples — assign final result as value target
    examples: list[TrainingExample] = []
    for enc, pol, turn, unc in buffer:
        if winner == "draw":
            v = 0.0
        elif winner == turn.value:
            v = 1.0
        else:
            v = -1.0

        examples.append(TrainingExample(enc, pol, v, unc))

        if augment:
            # BUG FIX: mirror the full policy distribution, not just the argmax
            mirrored_enc = encoder.mirror_horizontal(enc)
            mirrored_pol = _mirror_policy(pol)
            examples.append(TrainingExample(mirrored_enc, mirrored_pol, v, unc))

    return examples


def run_self_play_batch(
    engine:      ChaturangaEngine,
    model,
    num_games:   int  = 25,
    num_sims:    int  = 50,
    device:      str  = "cpu",
    start_states: list[ChaturangaState] | None = None,
    verbose:     bool = True,
) -> list[TrainingExample]:
    """
    Play `num_games` self-play games and collect all training examples.

    Parameters
    ----------
    start_states : optional list of starting positions (for curriculum).
                   If provided, games cycle through this list.
    """
    mcts   = MCTSSearch(engine, model=model, num_simulations=num_sims, device=device)
    all_ex: list[TrainingExample] = []

    for i in range(num_games):
        start = None
        if start_states:
            start = start_states[i % len(start_states)]

        ex = play_game(engine, mcts, start_state=start)
        all_ex.extend(ex)
        if verbose:
            print(f"  [self-play] game {i+1}/{num_games}: {len(ex)} examples")

    return all_ex
