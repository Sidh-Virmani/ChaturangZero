"""
p4_training/hard_position_buffer.py
=====================================
Hard Position Mining (Research Contribution #8).

Motivation
----------
Uniform replay (sample any position equally) is wasteful. The network
already understands easy positions well — replaying them adds little signal.

We instead identify "hard" positions during self-play:
  - High value uncertainty (|value_std| > threshold)
  - High policy entropy (the MCTS distribution is very flat — no clear move)
  - Near-zero value mean (position is genuinely ambiguous)

These positions are saved in a HardPositionBuffer and deliberately
over-sampled during training (e.g. 70% hard, 30% normal).

Relationship to ReplayBuffer
-----------------------------
  HardPositionBuffer is a *secondary* buffer that works alongside the
  standard ReplayBuffer. The Trainer mixes samples from both.
"""

from __future__ import annotations

import sys
import pathlib
import random
from collections import deque
from dataclasses import dataclass

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from p4_training.self_play import TrainingExample


@dataclass
class HardPositionMetrics:
    """Metrics recorded when a position is added to the hard buffer."""
    uncertainty:    float   # value_std from network
    policy_entropy: float   # Shannon entropy of MCTS policy
    value_mean:     float   # |value| (how close to 0 = ambiguous)


class HardPositionBuffer:
    """
    Secondary replay buffer that stores difficult training examples.

    A position is "hard" if ANY of these conditions hold:
      1. uncertainty  ≥ uncertainty_threshold
      2. policy_entropy ≥ entropy_threshold   (flat MCTS distribution)
      3. |value_mean| ≤ ambiguity_threshold   (position is near-zero eval)

    Parameters
    ----------
    max_size             : maximum number of examples to store
    uncertainty_threshold: min value_std to classify as hard
    entropy_threshold    : min policy entropy to classify as hard
    ambiguity_threshold  : max |value| to classify as ambiguous
    oversample_ratio     : fraction of hard examples in mixed batches
    """

    def __init__(
        self,
        max_size:              int   = 50_000,
        uncertainty_threshold: float = 0.25,
        entropy_threshold:     float = 3.5,
        ambiguity_threshold:   float = 0.15,
        oversample_ratio:      float = 0.70,
    ) -> None:
        self.max_size              = max_size
        self.uncertainty_threshold = uncertainty_threshold
        self.entropy_threshold     = entropy_threshold
        self.ambiguity_threshold   = ambiguity_threshold
        self.oversample_ratio      = oversample_ratio

        self._buffer: deque[TrainingExample] = deque(maxlen=max_size)
        self._total_added = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def is_hard(self, example: TrainingExample) -> bool:
        """Return True if this example meets the hard-position criteria."""
        # Criterion 1: high network uncertainty
        if example.uncertainty >= self.uncertainty_threshold:
            return True

        # Criterion 2: flat MCTS policy (high entropy)
        pol = example.mcts_policy
        pol = pol[pol > 0]  # only non-zero entries
        if len(pol) > 0:
            entropy = -float(np.sum(pol * np.log(pol + 1e-9)))
            if entropy >= self.entropy_threshold:
                return True

        # Criterion 3: ambiguous value (near zero)
        if abs(example.value) <= self.ambiguity_threshold:
            return True

        return False

    def add_if_hard(self, example: TrainingExample) -> bool:
        """
        Add `example` to the buffer if it qualifies as hard.
        Returns True if added.
        """
        if self.is_hard(example):
            self._buffer.append(example)
            self._total_added += 1
            return True
        return False

    def add_batch(self, examples: list[TrainingExample]) -> int:
        """Filter and add all hard examples from `examples`. Returns count added."""
        return sum(1 for ex in examples if self.add_if_hard(ex))

    def sample(self, n: int) -> list[TrainingExample]:
        """Sample `n` examples from the hard buffer (with replacement if needed)."""
        if len(self._buffer) == 0:
            return []
        n = min(n, len(self._buffer))
        return random.choices(list(self._buffer), k=n)

    def mixed_sample(
        self,
        normal_buffer,     # ReplayBuffer
        total_n: int,
    ) -> list[TrainingExample]:
        """
        Return a mixed batch of `total_n` examples:
          - oversample_ratio  fraction from hard buffer
          - (1 - oversample_ratio) fraction from normal buffer

        Falls back to pure normal buffer if hard buffer is empty.
        """
        if len(self._buffer) == 0:
            return normal_buffer.sample(total_n)

        n_hard   = int(total_n * self.oversample_ratio)
        n_normal = total_n - n_hard

        hard_ex   = self.sample(n_hard)
        normal_ex = normal_buffer.sample(n_normal)
        combined  = hard_ex + normal_ex
        random.shuffle(combined)
        return combined

    def stats(self) -> dict:
        return {
            "buffer_size":  len(self._buffer),
            "total_added":  self._total_added,
            "capacity":     self.max_size,
        }
