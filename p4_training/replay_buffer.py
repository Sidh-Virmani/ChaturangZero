"""
p4_training/replay_buffer.py
============================
Circular replay buffer that stores self-play training examples.

Instead of training only on the latest game, the AlphaZero approach
samples from a pool of recent examples.  This stabilises training.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


class ReplayBuffer:
    """
    Fixed-size circular buffer.

    Parameters
    ----------
    max_size : maximum number of training examples to store
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self._buffer: deque = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, examples) -> None:
        """Add a list (or single) TrainingExample objects."""
        if hasattr(examples, "__iter__"):
            self._buffer.extend(examples)
        else:
            self._buffer.append(examples)

    def sample(self, batch_size: int):
        """
        Randomly sample `batch_size` examples.
        Returns three numpy arrays: states, policies, values.
        """
        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))
        states   = np.stack([e.encoded_state for e in batch])  # (B, 13, 8, 8)
        policies = np.stack([e.mcts_policy   for e in batch])  # (B, 4096)
        values   = np.array([[e.value]        for e in batch], dtype=np.float32)  # (B, 1)
        return states, policies, values

    def __len__(self) -> int:
        return len(self._buffer)

    def is_ready(self, min_size: int = 256) -> bool:
        """Return True if the buffer has enough examples to train."""
        return len(self._buffer) >= min_size
