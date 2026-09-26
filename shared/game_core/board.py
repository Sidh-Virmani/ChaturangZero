"""Builds the starting position and offers a couple of small board helpers.

This is the only place that knows what the board looks like at move 0.
Everything else (movement rules, check detection) just reads a GameState
and does not care how it was constructed.
"""

from __future__ import annotations

from .types import GameState, Piece, PieceKind, Side, Square

# File order a-h for the frozen CZ v1 starting position.
_BACK_RANK = [
    PieceKind.RATHA,
    PieceKind.ASHVA,
    PieceKind.GAJA,
    PieceKind.MANTRI,
    PieceKind.RAJA,
    PieceKind.GAJA,
    PieceKind.ASHVA,
    PieceKind.RATHA,
]


def initial_state() -> GameState:
    pieces: list[Piece] = []

    for side in (Side.WHITE, Side.BLACK):
        back_rank = 0 if side is Side.WHITE else 7
        pawn_rank = 1 if side is Side.WHITE else 6

        for file_, kind in enumerate(_BACK_RANK):
            square: Square = (file_, back_rank)
            pieces.append(
                Piece(id=f"{side.value}-{kind.value}-{file_}", kind=kind, side=side, square=square)
            )
            pawn_square: Square = (file_, pawn_rank)
            pieces.append(
                Piece(id=f"{side.value}-padati-{file_}", kind=PieceKind.PADATI, side=side, square=pawn_square)
            )

    return GameState(pieces=tuple(pieces), side_to_move=Side.WHITE)


def render_ascii(state: GameState) -> str:
    """Small debugging helper: prints the board as text, rank 8 on top."""
    symbols = {
        PieceKind.RAJA: "K",
        PieceKind.MANTRI: "M",
        PieceKind.GAJA: "G",
        PieceKind.ASHVA: "A",
        PieceKind.RATHA: "R",
        PieceKind.PADATI: "P",
    }

    rows = []
    for rank in range(7, -1, -1):
        row = []
        for file_ in range(8):
            piece = state.piece_at((file_, rank))
            if piece is None:
                row.append(".")
            else:
                letter = symbols[piece.kind]
                row.append(letter if piece.side is Side.WHITE else letter.lower())
        rows.append(f"{rank + 1} " + " ".join(row))
    rows.append("  " + " ".join("abcdefgh"))
    return "\n".join(rows)
