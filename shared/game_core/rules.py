"""The public API of the game core — this is the only module the rest of
the project (representation, MCTS, training, and eventually a web API for
the React UI) should import from.

It matches the interface the architecture doc asks every module to agree
on:

    legal_moves(state)
    apply_move(state, move)
    is_terminal(state)
    result(state)
    clone_state(state)

movement.py answers "how could this one piece move if we ignore checks?".
This file adds the one rule that ties the whole game together: a move is
only legal if it does not leave your own raja under attack afterwards.
That single check is also what makes "check", "checkmate" and "stalemate"
fall out for free, instead of needing separate special-case code for each.

Draw rules: stalemate, threefold repetition, and the no-progress limit
(HALFMOVE_DRAW_LIMIT). Checkmate takes precedence over draw thresholds.
"""

from __future__ import annotations

from .movement import all_pseudo_legal_moves, attacked_squares, pseudo_legal_moves
from .types import GameOutcome, GameState, Move, Piece, PieceKind, Side, Square

HALFMOVE_DRAW_LIMIT = 100
"""Half-moves (100 = 50 full moves) without a capture or a padati move
before the game is declared a draw.

This is not decoration — without it, games are not guaranteed to end. Two
lone rajas can shuffle forever, and self-play training would hang on such
a game instead of producing a finished result to learn from. Adjust the
number if the team prefers a different cutoff, but do not remove the rule.
"""


def clone_state(state: GameState) -> GameState:
    """Returns an independent copy. GameState is built entirely from
    immutable tuples/dataclasses, so this is cheap and safe to call
    freely (e.g. once per MCTS node)."""
    return GameState(
        pieces=state.pieces,
        side_to_move=state.side_to_move,
        move_history=state.move_history,
        halfmove_clock=state.halfmove_clock,
        prior_positions=state.prior_positions,
    )


def _raja_square(state: GameState, side: Side) -> Square:
    for piece in state.pieces:
        if piece.side is side and piece.kind is PieceKind.RAJA:
            return piece.square
    raise ValueError(f"No raja on the board for {side.value} — invalid position")


def is_square_attacked(state: GameState, square: Square, by_side: Side) -> bool:
    """Would any `by_side` piece be able to capture on `square`?

    Correct for empty squares too, which is why this goes through
    movement.attacked_squares rather than through move destinations —
    a padati moves forward but does not attack forward.
    """
    for piece in state.pieces:
        if piece.side is by_side and square in attacked_squares(state, piece):
            return True
    return False


def is_in_check(state: GameState, side: Side) -> bool:
    return is_square_attacked(state, _raja_square(state, side), side.opponent)


def _leaves_own_raja_safe(state: GameState, move: Move) -> bool:
    return not is_in_check(apply_move(state, move), move.side)


def legal_moves(state: GameState) -> list[Move]:
    """All legal moves for the side whose turn it is."""
    candidates = all_pseudo_legal_moves(state, state.side_to_move)
    return [move for move in candidates if move.captured_kind is not PieceKind.RAJA and _leaves_own_raja_safe(state, move)]


def legal_moves_from(state: GameState, square: Square) -> list[Move]:
    """Legal moves for a single square — this is what a UI calls after a
    player clicks a piece, e.g. gameAdapter.getLegalMoves(square)."""
    return [move for move in legal_moves(state) if move.from_sq == square]


def apply_move(state: GameState, move: Move) -> GameState:
    """Returns a NEW GameState with the move applied. Does not mutate the
    state passed in, so callers (MCTS in particular) can keep exploring
    from the original state afterwards."""
    moved_piece = state.piece_at(move.from_sq)
    if moved_piece is None:
        raise ValueError(f"No piece on {move.from_sq} to move")

    next_kind = move.promotion if move.promotion else moved_piece.kind
    updated_piece = Piece(id=moved_piece.id, kind=next_kind, side=moved_piece.side, square=move.to_sq)

    next_pieces = tuple(
        piece for piece in state.pieces if piece.square != move.from_sq and piece.square != move.to_sq
    ) + (updated_piece,)

    is_progress = move.is_capture or moved_piece.kind is PieceKind.PADATI
    next_halfmove_clock = 0 if is_progress else state.halfmove_clock + 1

    return GameState(
        pieces=next_pieces,
        side_to_move=state.side_to_move.opponent,
        move_history=state.move_history + (move,),
        halfmove_clock=next_halfmove_clock,
        prior_positions=() if is_progress else state.prior_positions + (state.position_key(),),
    )


def is_terminal(state: GameState) -> bool:
    """True if the game is over: no legal moves (checkmate or stalemate),
    or the no-progress limit reached."""
    if not legal_moves(state):
        return True
    return state.halfmove_clock >= HALFMOVE_DRAW_LIMIT or state.repetition_count >= 3


def result(state: GameState) -> GameOutcome:
    """Who won, or is it a draw. Returns ONGOING if the game is not over,
    so this is always safe to call."""
    if not legal_moves(state):
        side_to_move = state.side_to_move
        if is_in_check(state, side_to_move):
            # Checkmate: the side that just delivered it wins.
            return GameOutcome.BLACK_WINS if side_to_move is Side.WHITE else GameOutcome.WHITE_WINS
        # Stalemate.
        return GameOutcome.DRAW

    # Checkmate is checked first above, so a mate delivered on the very
    # move that hits the limit still counts as a win rather than a draw.
    if state.halfmove_clock >= HALFMOVE_DRAW_LIMIT or state.repetition_count >= 3:
        return GameOutcome.DRAW

    return GameOutcome.ONGOING
