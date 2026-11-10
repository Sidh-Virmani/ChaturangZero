"""Ephemeral symmetry replacement for replay samples, using P1 transforms."""

from random import Random

import numpy as np

from p1_representation import rotate_action, rotate_encoded_state

from .data import TrainingExample


def rotate_example(example: TrainingExample) -> TrainingExample:
    """Transform board and sparse policy together; current-player value is unchanged."""
    return TrainingExample(
        rotate_encoded_state(example.state),
        np.array([rotate_action(int(action)) for action in example.actions],
                 dtype=example.actions.dtype),
        example.probabilities.copy(),
        example.value,
    )


def augment_example(
    example: TrainingExample, rng: Random, probability: float
) -> TrainingExample:
    """Choose a temporary equivalent example without modifying replay memory."""
    if probability > 0 and rng.random() < probability:
        return rotate_example(example)
    return example
