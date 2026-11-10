"""The board's valid 180-degree, color-swapping training symmetry."""

import numpy as np

from shared.game_core import BOARD_SIZE, GameState, Move, Piece

from .state import STATE_SHAPE

from .actions import action_to_move, move_to_action


_LAST_SQUARE = BOARD_SIZE * BOARD_SIZE - 1


def rotate_encoded_state(encoded: np.ndarray) -> np.ndarray:
    """Rotate P1 planes and swap colours; return an independent same-dtype array.

    Equivalent to encode_state(rotate_state(state)). Encoded states do not
    contain history; use rotate_state for full GameState transformations.
    """
    if encoded.shape != STATE_SHAPE:
        raise ValueError(f"Expected encoded state shape {STATE_SHAPE}")
    rotated = np.empty_like(encoded)
    piece_planes_per_side = (STATE_SHAPE[0] - 1) // 2
    rotated[:piece_planes_per_side] = encoded[piece_planes_per_side:-1, ::-1, ::-1]
    rotated[piece_planes_per_side:-1] = encoded[:piece_planes_per_side, ::-1, ::-1]
    rotated[-1] = 1 - encoded[-1, ::-1, ::-1]
    return rotated


def _rotate_move(move: Move) -> Move:
    return Move(_LAST_SQUARE - move.from_square, _LAST_SQUARE - move.to_square)


def rotate_state(state: GameState) -> GameState:
    """Rotate the board 180 degrees and exchange White/Black labels.

    The core's initial setup, forward Padati direction, and every movement
    rule are preserved by this combined transformation. Return a new state;
    never change the input.
    """
    board = tuple(
        None if piece is None else Piece(piece.side.opponent, piece.kind)
        for piece in reversed(state.board)
    )
    return GameState(
        board,
        side_to_move=state.side_to_move.opponent,
        ply_count=state.ply_count,
        last_move=None if state.last_move is None else _rotate_move(state.last_move),
        position_history=tuple(
            (tuple(None if piece is None else Piece(piece.side.opponent, piece.kind)
                   for piece in reversed(past_board)), past_side.opponent)
            for past_board, past_side in state.position_history
        ),
    )


def rotate_action(action: int) -> int:
    """Return the action ID after the same 180-degree board rotation."""
    move = action_to_move(action)
    return move_to_action(_rotate_move(move))
