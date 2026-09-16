"""
p4_training/rule_sensitivity.py
=================================
Rule Sensitivity Experiment (Research Contribution #10).

Research question
-----------------
How much does learned Chaturanga strategy depend on which historical
rule variant is assumed?

Method
------
  1. Train (or load) a model under each RuleVariant
  2. Cross-evaluate: model trained under Ri plays arena matches under Rj
  3. Compute sensitivity matrix:
       Sensitivity(Ri, Rj) = |win_rate(Mi vs Mj, under Rj) − 0.5|
     Where a higher value means the models are MORE different.
  4. Aggregate: mean_sensitivity = average across all (i≠j) pairs

A low mean_sensitivity means strategy transfers well across rule variants
(rules don't matter much). High = rule choice critically affects strategy.

This is a novel research result — no existing Chaturanga study has done this.

Usage
-----
    experiment = RuleSensitivityExperiment(engine_factory)
    # Train a model per variant (or load checkpoints)
    experiment.add_model(RuleVariant.STANDARD,     model_r1)
    experiment.add_model(RuleVariant.CAPTURE_RAJA, model_r2)
    experiment.add_model(RuleVariant.BARE_RAJA,    model_r3)
    experiment.add_model(RuleVariant.NO_STALEMATE, model_r4)
    matrix = experiment.run(num_arena_games=20)
    experiment.print_matrix(matrix)
    experiment.save_latex(matrix, "results/sensitivity_table.tex")
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
from shared.game_core.rule_variants import RuleVariant, RuleConfig
from p4_training.evaluator import _play_one_game


@dataclass
class SensitivityMatrix:
    variants:   list[RuleVariant]
    win_rates:  np.ndarray    # shape (n, n) — win_rates[i,j] = Mi vs Mj under Rj
    sensitivity: np.ndarray  # shape (n, n) — |win_rate - 0.5|

    @property
    def mean_sensitivity(self) -> float:
        # Exclude diagonal (self vs self)
        n = len(self.variants)
        off_diag = [
            self.sensitivity[i, j]
            for i in range(n) for j in range(n) if i != j
        ]
        return float(np.mean(off_diag)) if off_diag else 0.0

    def summary(self) -> str:
        labels = [v.value for v in self.variants]
        lines  = [f"Rule Sensitivity Matrix (mean={self.mean_sensitivity:.3f})"]
        header = "         " + "  ".join(f"{l:12s}" for l in labels)
        lines.append(header)
        for i, label_i in enumerate(labels):
            row = f"{label_i:9s}"
            for j in range(len(labels)):
                val = self.sensitivity[i, j]
                row += f"  {val:+.3f}      "
            lines.append(row)
        return "\n".join(lines)

    def to_latex(self) -> str:
        """Generate a LaTeX tabular for inclusion in a paper."""
        labels = [v.value.replace("_", r"\_") for v in self.variants]
        n = len(labels)
        col_spec = "l" + "c" * n
        lines = [
            r"\begin{tabular}{" + col_spec + "}",
            r"\hline",
            r"Model $\backslash$ Rules & " + " & ".join(labels) + r" \\",
            r"\hline",
        ]
        for i, label_i in enumerate(labels):
            row_vals = " & ".join(f"{self.sensitivity[i,j]:.3f}" for j in range(n))
            lines.append(f"{label_i} & {row_vals} \\\\")
        lines += [r"\hline", r"\end{tabular}"]
        return "\n".join(lines)


class RuleSensitivityExperiment:
    """
    Runs cross-rule-variant arena matches to measure strategy sensitivity.
    """

    def __init__(self) -> None:
        self._models:   dict[RuleVariant, object]   = {}
        self._configs:  dict[RuleVariant, RuleConfig] = {}

    def add_model(self, variant: RuleVariant, model) -> None:
        """Register a model trained under `variant`."""
        self._models[variant]  = model
        self._configs[variant] = RuleConfig.from_variant(variant)

    def run(
        self,
        num_arena_games: int  = 20,
        num_sims:        int  = 25,
        device:          str  = "cpu",
        verbose:         bool = True,
    ) -> SensitivityMatrix:
        """
        Cross-evaluate all registered models against all rule sets.

        For each (model_i, rules_j) pair:
          - Play `num_arena_games` between model_i and model_j under rules_j
          - Record win_rate[i, j]
        """
        variants = list(self._models.keys())
        n = len(variants)

        win_rates   = np.full((n, n), 0.5)
        sensitivity = np.zeros((n, n))

        for i, vi in enumerate(variants):
            for j, vj in enumerate(variants):
                if i == j:
                    win_rates[i, j] = 0.5
                    sensitivity[i, j] = 0.0
                    continue

                engine_j = ChaturangaEngine(rule_config=self._configs[vj])
                model_i  = self._models[vi]
                model_j  = self._models[vj]

                if verbose:
                    print(f"[sensitivity] {vi.value} vs {vj.value} under {vj.value} rules...")

                wins = 0
                for g in range(num_arena_games):
                    if g < num_arena_games // 2:
                        white, black = model_i, model_j
                        cand_side = "white"
                    else:
                        white, black = model_j, model_i
                        cand_side = "black"
                    winner = _play_one_game(engine_j, white, black, num_sims, device)
                    if winner == cand_side:
                        wins += 1
                    elif winner == "draw":
                        wins += 0.5

                wr = wins / num_arena_games
                win_rates[i, j]   = wr
                sensitivity[i, j] = abs(wr - 0.5)

                if verbose:
                    print(f"  win_rate={wr:.3f}  sensitivity={sensitivity[i,j]:.3f}")

        matrix = SensitivityMatrix(
            variants=variants,
            win_rates=win_rates,
            sensitivity=sensitivity,
        )
        if verbose:
            print("\n" + matrix.summary())
        return matrix

    def save_latex(self, matrix: SensitivityMatrix, path: str) -> None:
        """Save the sensitivity matrix as a LaTeX table."""
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(matrix.to_latex())
        print(f"[sensitivity] LaTeX table saved → {path}")
