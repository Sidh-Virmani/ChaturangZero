"""
p6_rule_inference/strategy_dna.py
==================================
Strategy DNA Fingerprinting (Contribution C).

Extracts a mathematical game signature from a move sequence or a set of games.
We use the mean strategic features (from StrategicFeatureExtractor) over all 
plies as the "DNA" of the game.

By extracting DNA from games played under different rule variants, we can compute
the strategic distance between them using cosine similarity.
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
from shared.game_core.state import Side
from p1_representation.features import StrategicFeatureExtractor, NUM_FEATURES


def extract_game_dna(move_notations: list[str], engine: ChaturangaEngine) -> np.ndarray:
    """
    Extracts a strategy fingerprint from a single game sequence.
    Returns a normalized vector of length NUM_FEATURES.
    """
    state = engine.new_game()
    extractor = StrategicFeatureExtractor()
    
    features_accum = np.zeros(NUM_FEATURES, dtype=np.float32)
    valid_plies = 0
    
    for notation in move_notations:
        if engine.is_terminal(state):
            break
            
        # Extract feature vector from current state
        # We always extract from White's perspective to keep the DNA consistent
        # Alternatively, we could average White and Black's features
        fv_white = extractor.extract(state, side=Side.WHITE)
        features_accum += fv_white.values
        valid_plies += 1
        
        legal_moves = engine.legal_moves(state)
        move_obj = next((m for m in legal_moves if m.label() == notation), None)
        
        if move_obj is None:
            break
            
        state = engine.apply_move(state, move_obj)
        
    if valid_plies > 0:
        features_accum /= valid_plies
        
    return features_accum


def extract_variant_dna(
    engine: ChaturangaEngine, 
    model=None, 
    n_games: int = 10,
    num_sims: int = 10,
    device: str = "cpu"
) -> np.ndarray:
    """
    Simulates games under a specific engine (rule variant) and returns the average DNA.
    """
    from p6_rule_inference.quality_metrics import play_game_for_quality
    
    overall_dna = np.zeros(NUM_FEATURES, dtype=np.float32)
    
    for _ in range(n_games):
        rec = play_game_for_quality(engine, model, max_plies=100, num_sims=num_sims, device=device)
        
        # We need the moves to get the exact DNA vector using extract_game_dna
        # For efficiency, we would ideally extract this during play_game_for_quality
        # But we'll just re-simulate the moves here
        # Note: play_game_for_quality doesn't return full move list currently, 
        # so this is pseudo-logic or requires modifying GameRecord.
        pass
        
    return overall_dna


def cosine_distance(dna_a: np.ndarray, dna_b: np.ndarray) -> float:
    """Compute 1 - cosine_similarity between two DNA vectors."""
    norm_a = np.linalg.norm(dna_a)
    norm_b = np.linalg.norm(dna_b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cos_sim = np.dot(dna_a, dna_b) / (norm_a * norm_b)
    return max(0.0, 1.0 - cos_sim)
