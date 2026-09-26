"""How each piece kind is allowed to move, in isolation.

Everything in this file is "pseudo-legal": it only looks at where pieces
are on the board and whether squares are occupied by a friend or a foe.
It deliberately does NOT check whether a move would leave your own raja
in check — that check-safety filtering happens once, in rules.py, so it
isn't duplicated six times (once per piece kind).

Piece movement in the frozen CZ v1 project ruleset (not a claim about
every historical Chaturanga tradition):

- raja (king):    one square in any of the 8 directions.
- mantri:         one square diagonally only.
- ratha (rook):   any distance orthogonally, blocked by pieces in the way.
- ashva (knight): the standard L-shaped knight jump; can jump over pieces.
- gaja (elephant): exactly two squares diagonally, jumping over whatever
                    sits on the square in between (that square's occupant
                    is irrelevant and is never captured).
- padati (pawn):  one square straight forward (no two-square opening move,
                    no en passant); captures one square diagonally forward;
                    promotes to mantri on reaching the far rank.
"""

from __future__ import annotations

from .types import GameState, Move, Piece, PieceKind, Side, Square, in_bounds

_ORTHOGONAL = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_DIAGONAL = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
_KNIGHT_OFFSETS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]


def _add(square: Square, offset: tuple[int, int]) -> Square:
    return (square[0] + offset[0], square[1] + offset[1])


def _step_moves(state: GameState, piece: Piece, offsets: list[tuple[int, int]]) -> list[Move]:
    """One-square (non-sliding) moves in the given directions."""
    moves = []
    for offset in offsets:
        target = _add(piece.square, offset)
        if not in_bounds(target):
            continue
        occupant = state.piece_at(target)
        if occupant is not None and occupant.side is piece.side:
            continue
        moves.append(
            Move(
                from_sq=piece.square,
                to_sq=target,
                piece_kind=piece.kind,
                side=piece.side,
                captured_kind=occupant.kind if occupant else None,
            )
        )
    return moves


def _sliding_moves(state: GameState, piece: Piece, directions: list[tuple[int, int]]) -> list[Move]:
    moves = []
    for direction in directions:
        target = _add(piece.square, direction)
        while in_bounds(target):
            occupant = state.piece_at(target)
            if occupant is None:
                moves.append(
                    Move(from_sq=piece.square, to_sq=target, piece_kind=piece.kind, side=piece.side)
                )
            else:
                if occupant.side is not piece.side:
                    moves.append(
                        Move(
                            from_sq=piece.square,
                            to_sq=target,
                            piece_kind=piece.kind,
                            side=piece.side,
                            captured_kind=occupant.kind,
                        )
                    )
                break
            target = _add(target, direction)
    return moves


def _gaja_moves(state: GameState, piece: Piece) -> list[Move]:
    moves = []
    for dx, dy in _DIAGONAL:
        target = _add(piece.square, (2 * dx, 2 * dy))
        if not in_bounds(target):
            continue
        occupant = state.piece_at(target)
        if occupant is not None and occupant.side is piece.side:
            continue
        moves.append(
            Move(
                from_sq=piece.square,
                to_sq=target,
                piece_kind=piece.kind,
                side=piece.side,
                captured_kind=occupant.kind if occupant else None,
            )
        )
    return moves


def _padati_moves(state: GameState, piece: Piece) -> list[Move]:
    moves = []
    forward = 1 if piece.side is Side.WHITE else -1
    promotion_rank = 7 if piece.side is Side.WHITE else 0

    straight = _add(piece.square, (0, forward))
    if in_bounds(straight) and state.piece_at(straight) is None:
        promotion = PieceKind.MANTRI if straight[1] == promotion_rank else None
        moves.append(
            Move(from_sq=piece.square, to_sq=straight, piece_kind=piece.kind, side=piece.side, promotion=promotion)
        )

    for dx in (-1, 1):
        target = _add(piece.square, (dx, forward))
        if not in_bounds(target):
            continue
        occupant = state.piece_at(target)
        if occupant is not None and occupant.side is not piece.side:
            promotion = PieceKind.MANTRI if target[1] == promotion_rank else None
            moves.append(
                Move(
                    from_sq=piece.square,
                    to_sq=target,
                    piece_kind=piece.kind,
                    side=piece.side,
                    captured_kind=occupant.kind,
                    promotion=promotion,
                )
            )
    return moves


def pseudo_legal_moves(state: GameState, piece: Piece) -> list[Move]:
    """All moves for a single piece, ignoring whether they leave its own
    raja in check."""
    if piece.kind is PieceKind.RAJA:
        return _step_moves(state, piece, _ORTHOGONAL + _DIAGONAL)
    if piece.kind is PieceKind.MANTRI:
        return _step_moves(state, piece, _DIAGONAL)
    if piece.kind is PieceKind.RATHA:
        return _sliding_moves(state, piece, _ORTHOGONAL)
    if piece.kind is PieceKind.ASHVA:
        return _step_moves(state, piece, _KNIGHT_OFFSETS)
    if piece.kind is PieceKind.GAJA:
        return _gaja_moves(state, piece)
    if piece.kind is PieceKind.PADATI:
        return _padati_moves(state, piece)
    raise ValueError(f"Unknown piece kind: {piece.kind}")


def all_pseudo_legal_moves(state: GameState, side: Side) -> list[Move]:
    moves = []
    for piece in state.pieces:
        if piece.side is side:
            moves.extend(pseudo_legal_moves(state, piece))
    return moves


def attacked_squares(state: GameState, piece: Piece) -> list[Square]:
    """Squares this piece currently *attacks* (i.e. could capture on).

    For every piece except the padati this is the same set as its move
    destinations. The padati is the exception and the reason this
    function exists separately: it moves straight forward but captures
    only diagonally, so its forward square is NOT attacked. Deriving
    "is this square attacked?" from move destinations would wrongly
    treat the square in front of a padati as covered.
    """
    if piece.kind is PieceKind.PADATI:
        forward = 1 if piece.side is Side.WHITE else -1
        squares = []
        for dx in (-1, 1):
            target = _add(piece.square, (dx, forward))
            if in_bounds(target):
                squares.append(target)
        return squares

    return [move.to_sq for move in pseudo_legal_moves(state, piece)]
