"""
shared/game_core/rules.py
=========================
Chaturanga piece movement rules.

This module is the ONLY place that knows how each piece moves.
It is deliberately separated from the engine so that the movement
logic can be read and verified independently.

Chaturanga piece movements (ancient Indian rules):
  Raja   (King)       — 1 step in any of the 8 directions
  Mantri (Minister)   — 1 step diagonally (4 directions) [NOT queen!]
  Ratha  (Chariot)    — slides any number of squares orthogonally
  Gaja   (Elephant)   — jumps EXACTLY 2 squares diagonally
                        (does NOT slide; jumps over any piece)
  Ashva  (Horse)      — L-shape: 2+1 or 1+2 (like modern Knight)
  Padati (Soldier)    — 1 step forward; captures 1 step diagonally forward
                        no double-step from start; no en-passant
"""

from __future__ import annotations

from .state import ChaturangaState, Move, Piece, PieceKind, Side, Square


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sq(f: int, r: int) -> Square | None:
    """Return Square if on board, else None."""
    if 0 <= f <= 7 and 0 <= r <= 7:
        return Square(f, r)
    return None


def _slider_moves(
    piece: Piece,
    state: ChaturangaState,
    directions: list[tuple[int, int]],
) -> list[Move]:
    """Generate sliding moves (Ratha) along given directions."""
    moves: list[Move] = []
    for df, dr in directions:
        f, r = piece.square.file + df, piece.square.rank + dr
        while 0 <= f <= 7 and 0 <= r <= 7:
            sq = Square(f, r)
            occupant = state.piece_at(sq)
            if occupant is None:
                moves.append(Move(piece.square, sq, piece.kind, piece.side))
            elif occupant.side != piece.side:
                moves.append(Move(piece.square, sq, piece.kind, piece.side,
                                  captured=occupant.kind))
                break  # can't slide further after capture
            else:
                break  # blocked by friendly piece
            f += df
            r += dr
    return moves


# ---------------------------------------------------------------------------
# Per-piece pseudolegal move generators
# (pseudolegal = not yet filtered for leaving own king in check)
# ---------------------------------------------------------------------------

def _raja_moves(piece: Piece, state: ChaturangaState) -> list[Move]:
    moves = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            sq = _sq(piece.square.file + df, piece.square.rank + dr)
            if sq is None:
                continue
            occupant = state.piece_at(sq)
            if occupant is None:
                moves.append(Move(piece.square, sq, piece.kind, piece.side))
            elif occupant.side != piece.side:
                moves.append(Move(piece.square, sq, piece.kind, piece.side,
                                  captured=occupant.kind))
    return moves


def _mantri_moves(piece: Piece, state: ChaturangaState) -> list[Move]:
    """Mantri moves 1 step diagonally — like a Ferz/Fers."""
    moves = []
    for df, dr in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        sq = _sq(piece.square.file + df, piece.square.rank + dr)
        if sq is None:
            continue
        occupant = state.piece_at(sq)
        if occupant is None:
            moves.append(Move(piece.square, sq, piece.kind, piece.side))
        elif occupant.side != piece.side:
            moves.append(Move(piece.square, sq, piece.kind, piece.side,
                              captured=occupant.kind))
    return moves


def _ratha_moves(piece: Piece, state: ChaturangaState) -> list[Move]:
    """Ratha slides orthogonally (like Rook)."""
    return _slider_moves(piece, state, [(1, 0), (-1, 0), (0, 1), (0, -1)])


def _gaja_moves(piece: Piece, state: ChaturangaState) -> list[Move]:
    """
    Gaja JUMPS exactly 2 squares diagonally.
    It ignores any piece on the intermediate square.
    It can capture on the destination square.
    """
    moves = []
    for df, dr in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
        sq = _sq(piece.square.file + df, piece.square.rank + dr)
        if sq is None:
            continue
        occupant = state.piece_at(sq)
        if occupant is None:
            moves.append(Move(piece.square, sq, piece.kind, piece.side))
        elif occupant.side != piece.side:
            moves.append(Move(piece.square, sq, piece.kind, piece.side,
                              captured=occupant.kind))
        # if friendly piece is on destination, cannot move there
    return moves


def _ashva_moves(piece: Piece, state: ChaturangaState) -> list[Move]:
    """Ashva moves in an L-shape (like modern Knight)."""
    moves = []
    deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
              (1, -2),  (1, 2),  (2, -1),  (2, 1)]
    for df, dr in deltas:
        sq = _sq(piece.square.file + df, piece.square.rank + dr)
        if sq is None:
            continue
        occupant = state.piece_at(sq)
        if occupant is None:
            moves.append(Move(piece.square, sq, piece.kind, piece.side))
        elif occupant.side != piece.side:
            moves.append(Move(piece.square, sq, piece.kind, piece.side,
                              captured=occupant.kind))
    return moves


def _padati_moves(piece: Piece, state: ChaturangaState) -> list[Move]:
    """
    Padati (Soldier/Pawn):
      - Moves 1 step forward (toward opponent's back rank)
      - Captures 1 step diagonally forward
      - No double-step from starting rank
      - No en passant
      - Promotion: reaches the back rank → becomes Mantri
    """
    moves = []
    forward = 1 if piece.side == Side.WHITE else -1
    f, r = piece.square.file, piece.square.rank

    # Step forward
    step_sq = _sq(f, r + forward)
    if step_sq and state.piece_at(step_sq) is None:
        promotion = (step_sq.rank == 7 and piece.side == Side.WHITE) or \
                    (step_sq.rank == 0 and piece.side == Side.BLACK)
        moves.append(Move(piece.square, step_sq, piece.kind, piece.side,
                          is_promotion=promotion))

    # Diagonal captures
    for df in (-1, 1):
        cap_sq = _sq(f + df, r + forward)
        if cap_sq is None:
            continue
        occupant = state.piece_at(cap_sq)
        if occupant and occupant.side != piece.side:
            promotion = (cap_sq.rank == 7 and piece.side == Side.WHITE) or \
                        (cap_sq.rank == 0 and piece.side == Side.BLACK)
            moves.append(Move(piece.square, cap_sq, piece.kind, piece.side,
                              captured=occupant.kind, is_promotion=promotion))
    return moves


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_GENERATORS = {
    PieceKind.RAJA:   _raja_moves,
    PieceKind.MANTRI: _mantri_moves,
    PieceKind.RATHA:  _ratha_moves,
    PieceKind.GAJA:   _gaja_moves,
    PieceKind.ASHVA:  _ashva_moves,
    PieceKind.PADATI: _padati_moves,
}


def pseudolegal_moves(piece: Piece, state: ChaturangaState) -> list[Move]:
    """All moves for `piece` ignoring whether they leave own king in check."""
    return _GENERATORS[piece.kind](piece, state)


def is_square_attacked(sq: Square, by_side: Side, state: ChaturangaState) -> bool:
    """Return True if `by_side` has any piece that can capture on `sq`."""
    for p in state.pieces_of(by_side):
        for move in pseudolegal_moves(p, state):
            if move.to_sq == sq:
                return True
    return False


def is_in_check(side: Side, state: ChaturangaState) -> bool:
    """Return True if the Raja of `side` is under attack."""
    raja_sq = state.raja_square(side)
    if raja_sq is None:
        return False  # raja already captured (shouldn't happen in normal play)
    return is_square_attacked(raja_sq, side.opponent(), state)
