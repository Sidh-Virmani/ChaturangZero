"""
p6_rule_inference/bayesian_classifier.py
=========================================
Bayesian Rule Inference (Contribution A).

"Given a sequence of moves, which historical rule set was being used?"

Computes the posterior probability of each rule variant using Bayes' theorem:
P(Variant_i | Moves) = exp(LL_i) / sum(exp(LL_j))
(Assuming uniform prior over variants).
"""

from __future__ import annotations

import sys
import pathlib
import math
from typing import Dict

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.rule_variants import RuleVariant, RuleConfig
from p6_rule_inference.likelihood_estimator import compute_sequence_log_likelihood


class BayesianRuleClassifier:
    """
    Infers the most likely rule variant from a sequence of moves.
    Requires one trained model per variant.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.models: dict[RuleVariant, object] = {}
        self.engines: dict[RuleVariant, ChaturangaEngine] = {}

    def add_model(self, variant: RuleVariant, model) -> None:
        """Register a model trained under a specific rule variant."""
        self.models[variant] = model
        self.engines[variant] = ChaturangaEngine(
            rule_config=RuleConfig.from_variant(variant)
        )

    def infer_rules(self, move_notations: list[str]) -> dict[str, float]:
        """
        Compute posterior probabilities for all registered variants.
        
        Returns
        -------
        dict mapping RuleVariant value (str) to probability [0.0, 1.0].
        """
        if not self.models:
            raise ValueError("No models registered. Call add_model first.")

        log_likelihoods = {}
        
        for variant, model in self.models.items():
            engine = self.engines[variant]
            ll = compute_sequence_log_likelihood(
                move_notations=move_notations,
                engine=engine,
                model=model,
                device=self.device
            )
            log_likelihoods[variant.value] = ll
            
        # Convert log-likelihoods to probabilities (Softmax)
        # Subtract max to prevent numerical overflow
        valid_lls = [ll for ll in log_likelihoods.values() if ll != float('-inf')]
        
        if not valid_lls:
            # If all are -inf (illegal moves in all variants), return uniform 0
            return {v: 0.0 for v in log_likelihoods.keys()}
            
        max_ll = max(valid_lls)
        
        probs = {}
        denom = 0.0
        
        for v, ll in log_likelihoods.items():
            if ll == float('-inf'):
                probs[v] = 0.0
            else:
                prob_unnorm = math.exp(ll - max_ll)
                probs[v] = prob_unnorm
                denom += prob_unnorm
                
        # Normalize
        for v in probs:
            probs[v] = round(probs[v] / denom, 4) if denom > 0 else 0.0
            
        return probs

    def print_inference_report(self, move_notations: list[str]) -> None:
        probs = self.infer_rules(move_notations)
        print(f"\n[bayesian] Rule Inference for {len(move_notations)}-move sequence")
        print("--------------------------------------------------------")
        for v, p in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {v:20s} : {p * 100:>5.1f}%")
        print("--------------------------------------------------------")
