"""Validated settings for self-play and network updates."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path

from p2_neural_network import ModelConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """One invocation runs ``iterations`` additional self-play/train cycles."""

    iterations: int = 1
    games_per_iteration: int = 2
    simulations_per_move: int = 25
    max_game_plies: int = 512
    opening_temperature_plies: int = 12
    training_steps: int | None = None
    batch_size: int = 64
    replay_capacity: int = 20_000
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: Path = Path("checkpoints/p4_training")
    model: ModelConfig = field(default_factory=ModelConfig)
    symmetry_probability: float = 0.5
    training_passes: float | None = None
    self_play_workers: int = 1
    worker_threads: int = 1
    cpu_threads: int = 8
    self_play_device: str = 'cpu'
    seeded_games: bool = True
    worker_timeout_seconds: int = 3600
    snapshot_interval: int = 0

    def __post_init__(self) -> None:
        if self.training_steps is not None and self.training_passes is not None:
            raise ValueError("Choose training_passes or fixed training_steps")
        if self.training_steps is None and self.training_passes is None:
            object.__setattr__(self, "training_passes", 1.0)
        if self.training_steps is not None and (
            not isinstance(self.training_steps, int)
            or isinstance(self.training_steps, bool) or self.training_steps <= 0
        ):
            raise ValueError("training_steps must be a positive integer")
        if self.training_passes is not None and (
            not isinstance(self.training_passes, Real)
            or isinstance(self.training_passes, bool)
            or not math.isfinite(float(self.training_passes)) or self.training_passes <= 0
        ):
            raise ValueError("training_passes must be finite and positive")
        if (not isinstance(self.symmetry_probability, Real)
                or isinstance(self.symmetry_probability, bool)
                or not math.isfinite(float(self.symmetry_probability))
                or not 0 <= self.symmetry_probability <= 1):
            raise ValueError("symmetry_probability must be between 0 and 1")
        for name in (
            "iterations", "games_per_iteration", "simulations_per_move",
            "max_game_plies", "batch_size", "replay_capacity", 'self_play_workers',
            'worker_threads', 'cpu_threads', 'worker_timeout_seconds',
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seeded_games, bool):
            raise ValueError('seeded_games must be boolean')
        if self.self_play_workers > 1 and (not self.seeded_games or self.self_play_device != 'cpu'):
            raise ValueError('Parallel self-play requires seeded_games and CPU workers')
        if not isinstance(self.snapshot_interval, int) or isinstance(self.snapshot_interval, bool) or self.snapshot_interval < 0:
            raise ValueError('snapshot_interval must be a non-negative integer')
        if (
            not isinstance(self.opening_temperature_plies, int)
            or isinstance(self.opening_temperature_plies, bool)
            or self.opening_temperature_plies < 0
        ):
            raise ValueError("opening_temperature_plies must be a non-negative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(self.learning_rate, Real)
            or not math.isfinite(float(self.learning_rate))
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if (
            not isinstance(self.weight_decay, Real)
            or not math.isfinite(float(self.weight_decay))
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        if not isinstance(self.checkpoint_dir, Path):
            raise TypeError("checkpoint_dir must be a Path")
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")

    @classmethod
    def smoke(cls, checkpoint_dir: Path) -> "TrainingConfig":
        """A fast real-network run from a short, forced terminal position."""
        return cls(
            iterations=1,
            games_per_iteration=1,
            simulations_per_move=1,
            max_game_plies=4,
            opening_temperature_plies=4,
            batch_size=2,
            replay_capacity=16,
            seed=7,
            checkpoint_dir=checkpoint_dir,
            model=ModelConfig(channels=16, residual_blocks=1, policy_channels=2, value_hidden=32),
        )
