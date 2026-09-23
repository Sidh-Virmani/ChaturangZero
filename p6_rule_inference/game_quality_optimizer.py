"""
p6_rule_inference/game_quality_optimizer.py
=============================================
Neural Game Quality Optimizer (Contribution D).

Searches the space of possible rule configurations to find the variant
that produces the highest quality games, as defined by `quality_metrics.py`.

In a full production run with massive compute, this would use Bayesian
Optimization (Gaussian Processes) to search continuous/discrete mixed spaces.
Here, we implement a Grid/Random search over combinations of historical
and theoretical rules.
"""

from __future__ import annotations

import sys
import pathlib
import itertools
from dataclasses import dataclass
from typing import Optional

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.rule_variants import RuleConfig, RuleVariant
from p6_rule_inference.quality_metrics import compute_quality_report, QualityReport, rank_variants


class GameQualityOptimizer:
    """
    Evaluates different rule configurations and finds the one that
    maximizes game quality.
    """

    def __init__(self, model=None, num_sims: int = 25, device: str = "cpu"):
        self.model = model
        self.num_sims = num_sims
        self.device = device
        self.history: list[QualityReport] = []

    def _generate_candidate_configs(self) -> list[RuleConfig]:
        """Generate a mix of historical and novel rule configurations."""
        configs = []
        
        # 1. Historical baselines
        configs.append(RuleConfig.standard())
        configs.append(RuleConfig.capture_raja())
        configs.append(RuleConfig.bare_raja())
        configs.append(RuleConfig.no_stalemate())

        # 2. Novel/Hybrid combinations (AI proposing new variants)
        # What if both bare_raja loses AND stalemate is a loss?
        cfg_aggro = RuleConfig(
            variant=RuleVariant("hybrid_aggressive"),
            checkmate_wins=True,
            capture_raja_wins=True,
            bare_raja_loses=True,
            stalemate_is_loss=True,
        )
        configs.append(cfg_aggro)

        # What if stalemate is a loss, but bare raja is fine?
        cfg_stalemate_loss = RuleConfig(
            variant=RuleVariant("hybrid_anti_draw"),
            checkmate_wins=True,
            stalemate_is_loss=True,
        )
        configs.append(cfg_stalemate_loss)

        return configs

    def optimize(
        self,
        n_games_per_config: int = 20,
        verbose: bool = True
    ) -> QualityReport:
        """
        Run the optimization process. Evaluates candidate configurations
        and returns the one with the highest quality score.
        """
        candidates = self._generate_candidate_configs()
        
        if verbose:
            print(f"[optimizer] Starting optimization over {len(candidates)} configurations.")
            
        for idx, config in enumerate(candidates):
            if verbose:
                print(f"[optimizer] Evaluating {idx+1}/{len(candidates)}: {config.variant.value}")
            
            report = compute_quality_report(
                rule_config=config,
                n_games=n_games_per_config,
                model=self.model,
                num_sims=self.num_sims,
                device=self.device,
                verbose=False  # We'll print the summary ourselves
            )
            self.history.append(report)
            
            if verbose:
                print(f"  -> Quality Score: {report.quality_score:.4f}")

        # Find best
        best_report = max(self.history, key=lambda r: r.quality_score)
        
        if verbose:
            print("\n=== OPTIMIZATION COMPLETE ===")
            print(f"Best Rule Configuration: {best_report.rule_variant}")
            print(best_report.summary())
            
        return best_report

    def get_leaderboard(self) -> list[tuple[str, float]]:
        """Return a sorted list of (variant_name, quality_score)."""
        return rank_variants(self.history)
