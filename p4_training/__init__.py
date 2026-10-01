"""Self-play and replay-driven training using the existing P1/P2/P3 stack."""

from .arena import ArenaResult, compare_models
from .config import TrainingConfig
from .data import ReplayBuffer, TrainingExample
from .levels import LevelProgress, train_levels
from .self_play import SelfPlayResult, play_game, smoke_position
from .trainer import IterationMetrics, Trainer

__all__ = [
    "ArenaResult",
    "IterationMetrics",
    "LevelProgress",
    "ReplayBuffer",
    "SelfPlayResult",
    "Trainer",
    "TrainingConfig",
    "TrainingExample",
    "compare_models",
    "play_game",
    "smoke_position",
    "train_levels",
]
