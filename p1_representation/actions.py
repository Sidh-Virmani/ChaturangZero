"""Stable action IDs and masks for moves supplied by the game core."""

from numbers import Integral

import numpy as np

from shared.game_core import BOARD_SIZE, GameState, Move, legal_moves


_SQUARE_COUNT = BOARD_SIZE * BOARD_SIZE
ACTION_SPACE_SIZE = _SQUARE_COUNT * _SQUARE_COUNT
"""Number of possible source/destination square pairs, including illegal ones."""


def _check_square(square: int) -> None:
    if not isinstance(square, int) or isinstance(square, bool) or not 0 <= square < _SQUARE_COUNT:
        raise ValueError(f"Square must be an integer from 0 to {_SQUARE_COUNT - 1}: {square!r}")


def move_to_action(move: Move) -> int:
    """Give a core Move a stable ID: ``from_square * 64 + to_square``.

    This translates square pairs only. It does not decide whether the move
    is legal in any position; use ``legal_action_mask`` for that.
    """
    if not isinstance(move, Move):
        raise TypeError("move must be a shared.game_core.Move")
    _check_square(move.from_square)
    _check_square(move.to_square)
    return move.from_square * _SQUARE_COUNT + move.to_square


def action_to_move(action: int) -> Move:
    """Recover the source and destination of an action ID as a core Move."""
    if not isinstance(action, Integral) or isinstance(action, bool) or not 0 <= action < ACTION_SPACE_SIZE:
        raise ValueError(f"Action must be an integer from 0 to {ACTION_SPACE_SIZE - 1}: {action!r}")
    source, destination = divmod(int(action), _SQUARE_COUNT)
    return Move(source, destination)


def legal_action_mask(state: GameState) -> np.ndarray:
    """Return a fresh 4096-entry uint8 array: 1 for each core-legal move."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.uint8)
    for move in legal_moves(state):
        mask[move_to_action(move)] = 1
    return mask
