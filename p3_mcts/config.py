"""Configuration for Monte Carlo Tree Search."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    """Hyperparameters controlling one MCTS search.

    Values are deliberately validated at construction time so invalid search
    settings fail before any evaluator work is performed.
    """

    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    add_noise: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_simulations, Integral)
            or isinstance(self.num_simulations, bool)
            or self.num_simulations <= 0
        ):
            raise ValueError("num_simulations must be a positive integer")

        self._validate_positive("c_puct", self.c_puct)
        self._validate_positive("dirichlet_alpha", self.dirichlet_alpha)
        self._validate_probability("dirichlet_epsilon", self.dirichlet_epsilon)

        if (
            not isinstance(self.temperature, Real)
            or isinstance(self.temperature, bool)
            or not math.isfinite(float(self.temperature))
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a finite non-negative number")
        if not isinstance(self.add_noise, bool):
            raise ValueError("add_noise must be a boolean")

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be a finite number in [0, 1]")
