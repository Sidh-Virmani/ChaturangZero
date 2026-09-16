"""Fixed-orientation board planes for a :class:`GameState`."""

import numpy as np

from shared.game_core import BOARD_SIZE, GameState, PieceType, Side


_PIECE_TYPES = (
    PieceType.RAJA,
    PieceType.MANTRI,
    PieceType.GAJA,
    PieceType.ASHVA,
    PieceType.RATHA,
    PieceType.PADATI,
)
_PIECE_PLANE = {
    (side, kind): side_offset + kind_offset
    for side_offset, side in ((0, Side.WHITE), (len(_PIECE_TYPES), Side.BLACK))
    for kind_offset, kind in enumerate(_PIECE_TYPES)
}

PLANE_NAMES = tuple(
    f"{side.value}_{kind.value}"
    for side in (Side.WHITE, Side.BLACK)
    for kind in _PIECE_TYPES
) + ("white_to_move",)
"""Plane order: six White pieces, six Black pieces, then White to move."""

STATE_SHAPE = (len(PLANE_NAMES), BOARD_SIZE, BOARD_SIZE)
"""Channel-first shape of every encoded state."""


def encode_state(state: GameState) -> np.ndarray:
    """Return a new float32 array of shape ``(13, 8, 8)``.

    Rows are ranks 1 through 8; columns are files a through h. A piece
    occupies one cell in its side/kind plane. The final plane is all ones
    when White moves and all zeros when Black moves. Only the board and
    side to move are encoded; move history does not affect the result.
    """
    planes = np.zeros(STATE_SHAPE, dtype=np.float32)
    for square, piece in enumerate(state.board):
        if piece is not None:
            row, column = divmod(square, BOARD_SIZE)
            planes[_PIECE_PLANE[piece.side, piece.kind], row, column] = 1.0

    if state.side_to_move is Side.WHITE:
        planes[-1, :, :] = 1.0
    return planes
