"""Compact training examples and bounded replay memory."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random

import numpy as np
import torch

from p1_representation import ACTION_SPACE_SIZE, STATE_SHAPE


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """Encoded board, sparse MCTS policy, and final current-player reward."""

    state: np.ndarray  # uint8, STATE_SHAPE
    actions: np.ndarray  # int16 IDs of positive policy entries
    probabilities: np.ndarray  # float32 weights at those IDs
    value: float

    def policy(self) -> np.ndarray:
        """Return the full P1 action vector required by P2's loss helper."""
        dense = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        dense[self.actions] = self.probabilities
        return dense


class ReplayBuffer:
    """Keep the newest completed-game examples up to a fixed capacity."""

    def __init__(self, capacity: int) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._examples: deque[TrainingExample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._examples)

    def add_game(self, examples: tuple[TrainingExample, ...]) -> None:
        self._examples.extend(examples)

    def sample(self, batch_size: int, rng: Random) -> tuple[TrainingExample, ...]:
        if not self._examples:
            raise ValueError("Cannot sample an empty replay buffer")
        return tuple(rng.choices(tuple(self._examples), k=batch_size))

    def to_checkpoint(self) -> list[dict[str, object]]:
        return [
            {
                "state": torch.from_numpy(example.state.copy()),
                "actions": torch.from_numpy(example.actions.copy()),
                "probabilities": torch.from_numpy(example.probabilities.copy()),
                "value": example.value,
            }
            for example in self._examples
        ]

    @classmethod
    def from_checkpoint(
        cls, capacity: int, saved: list[dict[str, object]]
    ) -> "ReplayBuffer":
        buffer = cls(capacity)
        for item in saved:
            state = item["state"].numpy().copy()
            actions = item["actions"].numpy().copy()
            probabilities = item["probabilities"].numpy().copy()
            if state.shape != STATE_SHAPE or actions.shape != probabilities.shape:
                raise ValueError("Checkpoint contains a malformed training example")
            buffer._examples.append(
                TrainingExample(state, actions, probabilities, float(item["value"]))
            )
        return buffer
