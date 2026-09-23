"""
p6_rule_inference/likelihood_estimator.py
==========================================
Likelihood Estimator (Contribution A).

Calculates the log-likelihood of a sequence of observed moves given a 
specific model (which was trained under a specific rule variant).

Math:
  L(Moves | Model_Ri) = sum( log( P(move_t | state_t; Model_Ri) ) )
"""

from __future__ import annotations

import sys
import pathlib
import math

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state import Move
from p1_representation.encoder import StateEncoder


def compute_sequence_log_likelihood(
    move_notations: list[str],
    engine: ChaturangaEngine,
    model,
    device: str = "cpu"
) -> float:
    """
    Computes the log-likelihood of a game sequence under the given model.
    
    Parameters
    ----------
    move_notations : list of move labels, e.g., ["e2e4", "e7e5", ...]
    engine         : engine instance configured for the rules to validate legality
    model          : ChaturangaNet instance
    
    Returns
    -------
    float : sum of log probabilities of the moves.
    """
    state = engine.new_game()
    encoder = StateEncoder()
    
    log_likelihood = 0.0
    
    for notation in move_notations:
        if engine.is_terminal(state):
            break
            
        legal_moves = engine.legal_moves(state)
        # Find the move matching the notation
        move_obj = next((m for m in legal_moves if m.label() == notation), None)
        
        if move_obj is None:
            # Illegal move in this sequence for this engine's rules!
            # The likelihood of an illegal move is 0 (log likelihood -inf)
            return float('-inf')

        # Get policy from model
        try:
            import torch
            enc = encoder.encode_state_torch(state).to(device)
            mask = encoder.legal_action_mask_torch(state, legal_moves).to(device)
            
            with torch.no_grad():
                out = model.predict(enc, mask)
                probs = out[0].cpu().numpy()
        except ImportError:
            # Fallback for tests if torch not present
            probs = np.zeros(encoder.action_size)
            for m in legal_moves:
                probs[encoder.encode_action(m)] = 1.0 / len(legal_moves)
        
        move_idx = encoder.encode_action(move_obj)
        prob = float(probs[move_idx])
        
        # Add epsilon to prevent log(0)
        prob = max(prob, 1e-9)
        log_likelihood += math.log(prob)
        
        # Advance state
        state = engine.apply_move(state, move_obj)
        
    return log_likelihood
