"""Strict classical Chaturanga state, movement, and terminal rules.

Squares are integer indices in ``a1 .. h8`` order: a1 is 0 and h8 is 63.
The state is immutable, making it safe to share between MCTS branches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Mapping, Optional

BOARD_SIZE = 8
SQUARE_COUNT = BOARD_SIZE * BOARD_SIZE


class Side(str, Enum):
    WHITE = "white"
    BLACK = "black"

    @property
    def opponent(self) -> "Side":
        return Side.BLACK if self is Side.WHITE else Side.WHITE


class PieceType(str, Enum):
    RAJA = "raja"
    MANTRI = "mantri"
    GAJA = "gaja"
    ASHVA = "ashva"
    RATHA = "ratha"
    PADATI = "padati"


@dataclass(frozen=True, slots=True)
class Piece:
    side: Side
    kind: PieceType


@dataclass(frozen=True, slots=True, order=True)
class Move:
    from_square: int
    to_square: int

    @classmethod
    def from_names(cls, from_square: str, to_square: str) -> "Move":
        return cls(parse_square(from_square), parse_square(to_square))

    @property
    def uci(self) -> str:
        # Promotion is compulsory and therefore needs no action suffix.
        return square_name(self.from_square) + square_name(self.to_square)


class TerminalReason(str, Enum):
    CHECKMATE = "checkmate"
    BARE_KING = "bare_king"
    STALEMATE = "stalemate"
    REPETITION = "repetition"


@dataclass(frozen=True, slots=True)
class Outcome:
    winner: Optional[Side] = None
    reason: Optional[TerminalReason] = None

    @property
    def terminal(self) -> bool:
        return self.winner is not None or self.reason is TerminalReason.REPETITION

    @property
    def draw(self) -> bool:
        return self.terminal and self.winner is None

    def reward(self, perspective: Side) -> int:
        if not self.terminal:
            raise ValueError("A non-terminal position has no game result")
        if self.draw:
            return 0
        return 1 if perspective is self.winner else -1


@dataclass(frozen=True, slots=True)
class GameState:
    board: tuple[Optional[Piece], ...]
    side_to_move: Side = Side.WHITE
    ply_count: int = 0
    last_move: Optional[Move] = None
    # Prior positions only; current position counts once implicitly. Excludes
    # ply_count/last_move because neither changes available Chaturanga moves.
    position_history: tuple[tuple[tuple[Optional[Piece], ...], Side], ...] = ()

    def __post_init__(self) -> None:
        if len(self.board) != SQUARE_COUNT:
            raise ValueError("A Chaturanga board must contain exactly 64 squares")
        if self.ply_count < 0:
            raise ValueError("ply_count cannot be negative")
        for side in Side:
            raja_count = sum(
                piece == Piece(side, PieceType.RAJA) for piece in self.board
            )
            if raja_count != 1:
                raise ValueError(
                    f"A valid position must contain exactly one {side.value} Raja"
                )
        for square, piece in enumerate(self.board):
            if piece is None:
                continue
            if not isinstance(piece, Piece):
                raise TypeError(f"Invalid board entry at {square_name(square)}")
            _, rank_index = _coordinates(square)
            if piece.kind is PieceType.PADATI and rank_index in (0, 7):
                raise ValueError("A Padati on the farthest rank must already be a Mantri")

    @classmethod
    def from_mapping(
        cls,
        pieces: Mapping[str, Piece],
        side_to_move: Side = Side.WHITE,
        *,
        ply_count: int = 0,
    ) -> "GameState":
        board: list[Optional[Piece]] = [None] * SQUARE_COUNT
        for name, piece in pieces.items():
            index = parse_square(name)
            if board[index] is not None:
                raise ValueError(f"Duplicate piece on {name}")
            board[index] = piece
        return cls(tuple(board), side_to_move, ply_count)

    def piece_at(self, square: int | str) -> Optional[Piece]:
        index = parse_square(square) if isinstance(square, str) else square
        _validate_square(index)
        return self.board[index]


class IllegalMoveError(ValueError):
    """Raised when a caller attempts a move outside the legal move set."""


class GameOverError(IllegalMoveError):
    """Raised when a caller attempts to move after a terminal position."""


def parse_square(name: str) -> int:
    if len(name) != 2 or name[0] not in "abcdefgh" or name[1] not in "12345678":
        raise ValueError(f"Invalid square: {name!r}")
    return (int(name[1]) - 1) * BOARD_SIZE + (ord(name[0]) - ord("a"))


def square_name(square: int) -> str:
    _validate_square(square)
    file_index, rank_index = _coordinates(square)
    return chr(ord("a") + file_index) + str(rank_index + 1)


def _validate_square(square: int) -> None:
    if not isinstance(square, int) or isinstance(square, bool) or not 0 <= square < SQUARE_COUNT:
        raise ValueError(f"Square index must be an integer from 0 to 63: {square!r}")


def _coordinates(square: int) -> tuple[int, int]:
    return square % BOARD_SIZE, square // BOARD_SIZE


def _square(file_index: int, rank_index: int) -> Optional[int]:
    if 0 <= file_index < BOARD_SIZE and 0 <= rank_index < BOARD_SIZE:
        return rank_index * BOARD_SIZE + file_index
    return None


def initial_state() -> GameState:
    board: list[Optional[Piece]] = [None] * SQUARE_COUNT
    white_back_rank = (
        PieceType.RATHA,
        PieceType.ASHVA,
        PieceType.GAJA,
        PieceType.MANTRI,
        PieceType.RAJA,
        PieceType.GAJA,
        PieceType.ASHVA,
        PieceType.RATHA,
    )
    # Raja and Mantri are reversed for Black, so the Rajas do not share a file.
    black_back_rank = (
        PieceType.RATHA,
        PieceType.ASHVA,
        PieceType.GAJA,
        PieceType.RAJA,
        PieceType.MANTRI,
        PieceType.GAJA,
        PieceType.ASHVA,
        PieceType.RATHA,
    )
    for file_index, kind in enumerate(white_back_rank):
        board[_square(file_index, 0)] = Piece(Side.WHITE, kind)  # type: ignore[index]
        board[_square(file_index, 1)] = Piece(Side.WHITE, PieceType.PADATI)  # type: ignore[index]
    for file_index, kind in enumerate(black_back_rank):
        board[_square(file_index, 7)] = Piece(Side.BLACK, kind)  # type: ignore[index]
        board[_square(file_index, 6)] = Piece(Side.BLACK, PieceType.PADATI)  # type: ignore[index]
    return GameState(tuple(board))


_KING_STEPS = tuple(
    (df, dr)
    for df in (-1, 0, 1)
    for dr in (-1, 0, 1)
    if (df, dr) != (0, 0)
)
_DIAGONAL_STEPS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_ORTHOGONAL_RAYS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_KNIGHT_JUMPS = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1),
)
_GAJA_JUMPS = ((-2, -2), (-2, 2), (2, -2), (2, 2))


def is_valid_gaja_jump(from_square: int | str, to_square: int | str) -> bool:
    """Validate the Gaja's exact two-diagonal leap.

    Deliberately does not inspect the intervening square: the Gaja jumps it.
    """
    if isinstance(from_square, str):
        from_square = parse_square(from_square)
    if isinstance(to_square, str):
        to_square = parse_square(to_square)
    _validate_square(from_square)
    _validate_square(to_square)
    from_file, from_rank = _coordinates(from_square)
    to_file, to_rank = _coordinates(to_square)
    return abs(to_file - from_file) == 2 and abs(to_rank - from_rank) == 2


def is_valid_padati_move(
    state: GameState, move: Move | tuple[str, str], side: Side
) -> bool:
    """Validate the Padati's single step/capture; a double-step can never pass."""
    if isinstance(move, tuple):
        move = Move.from_names(*move)
    from_square, to_square = move.from_square, move.to_square
    _validate_square(from_square)
    _validate_square(to_square)
    from_file, from_rank = _coordinates(from_square)
    to_file, to_rank = _coordinates(to_square)
    direction = 1 if side is Side.WHITE else -1
    rank_delta = to_rank - from_rank
    file_delta = to_file - from_file
    target = state.board[to_square]
    if rank_delta != direction:  # This explicitly excludes every double-step.
        return False
    if file_delta == 0:
        return target is None
    return abs(file_delta) == 1 and target is not None and target.side is side.opponent


def _piece_attacks_square(
    state: GameState, from_square: int, piece: Piece, target_square: int
) -> bool:
    from_file, from_rank = _coordinates(from_square)
    target_file, target_rank = _coordinates(target_square)
    df, dr = target_file - from_file, target_rank - from_rank

    if piece.kind is PieceType.RAJA:
        return (df, dr) in _KING_STEPS
    if piece.kind is PieceType.MANTRI:
        return (df, dr) in _DIAGONAL_STEPS
    if piece.kind is PieceType.ASHVA:
        return (df, dr) in _KNIGHT_JUMPS
    if piece.kind is PieceType.GAJA:
        return is_valid_gaja_jump(from_square, target_square)
    if piece.kind is PieceType.PADATI:
        direction = 1 if piece.side is Side.WHITE else -1
        return dr == direction and abs(df) == 1
    if piece.kind is PieceType.RATHA:
        if df != 0 and dr != 0:
            return False
        if df == 0 and dr == 0:
            return False
        step_file = 0 if df == 0 else (1 if df > 0 else -1)
        step_rank = 0 if dr == 0 else (1 if dr > 0 else -1)
        file_index, rank_index = from_file + step_file, from_rank + step_rank
        while (file_index, rank_index) != (target_file, target_rank):
            between = _square(file_index, rank_index)
            if between is None or state.board[between] is not None:
                return False
            file_index += step_file
            rank_index += step_rank
        return True
    raise AssertionError(f"Unhandled piece kind: {piece.kind}")


def is_square_attacked(state: GameState, square: int, by_side: Side) -> bool:
    _validate_square(square)
    return any(
        piece is not None
        and piece.side is by_side
        and _piece_attacks_square(state, source, piece, square)
        for source, piece in enumerate(state.board)
    )


def _raja_square(state: GameState, side: Side) -> int:
    squares = [
        square
        for square, piece in enumerate(state.board)
        if piece == Piece(side, PieceType.RAJA)
    ]
    if len(squares) != 1:
        raise ValueError(f"Position must contain exactly one {side.value} Raja")
    return squares[0]


def is_in_check(state: GameState, side: Optional[Side] = None) -> bool:
    checked_side = state.side_to_move if side is None else side
    return is_square_attacked(state, _raja_square(state, checked_side), checked_side.opponent)


def _candidate_destinations(state: GameState, source: int, piece: Piece) -> Iterable[int]:
    file_index, rank_index = _coordinates(source)
    offsets: tuple[tuple[int, int], ...]
    if piece.kind is PieceType.RAJA:
        offsets = _KING_STEPS
    elif piece.kind is PieceType.MANTRI:
        offsets = _DIAGONAL_STEPS
    elif piece.kind is PieceType.ASHVA:
        offsets = _KNIGHT_JUMPS
    elif piece.kind is PieceType.GAJA:
        offsets = _GAJA_JUMPS
    elif piece.kind is PieceType.PADATI:
        direction = 1 if piece.side is Side.WHITE else -1
        offsets = ((0, direction), (-1, direction), (1, direction))
    elif piece.kind is PieceType.RATHA:
        for step_file, step_rank in _ORTHOGONAL_RAYS:
            next_file, next_rank = file_index + step_file, rank_index + step_rank
            while (destination := _square(next_file, next_rank)) is not None:
                yield destination
                if state.board[destination] is not None:
                    break
                next_file += step_file
                next_rank += step_rank
        return
    else:
        raise AssertionError(f"Unhandled piece kind: {piece.kind}")

    for df, dr in offsets:
        destination = _square(file_index + df, rank_index + dr)
        if destination is not None:
            yield destination


def _is_pseudo_legal(state: GameState, move: Move, piece: Piece) -> bool:
    target = state.board[move.to_square]
    if target is not None and target.side is piece.side:
        return False
    # Chaturanga ends at checkmate/bare king; the Raja is never captured.
    if target is not None and target.kind is PieceType.RAJA:
        return False
    if piece.kind is PieceType.PADATI:
        return is_valid_padati_move(state, move, piece.side)
    if piece.kind is PieceType.RATHA:
        return _piece_attacks_square(state, move.from_square, piece, move.to_square)
    if piece.kind is PieceType.GAJA:
        return is_valid_gaja_jump(move.from_square, move.to_square)
    return _piece_attacks_square(state, move.from_square, piece, move.to_square)


def _move_without_validation(
    state: GameState, move: Move, *, record_history: bool = False
) -> GameState:
    board = list(state.board)
    piece = board[move.from_square]
    if piece is None:
        raise AssertionError("Internal move has no source piece")
    board[move.from_square] = None
    _, destination_rank = _coordinates(move.to_square)
    if piece.kind is PieceType.PADATI and destination_rank in (0, 7):
        piece = Piece(piece.side, PieceType.MANTRI)
    board[move.to_square] = piece
    return GameState(
        tuple(board),
        side_to_move=state.side_to_move.opponent,
        ply_count=state.ply_count + 1,
        last_move=move,
        position_history=(
            state.position_history + ((state.board, state.side_to_move),)
            if record_history else ()
        ),
    )


def _legal_moves_unchecked(state: GameState) -> tuple[Move, ...]:
    moves: list[Move] = []
    for source, piece in enumerate(state.board):
        if piece is None or piece.side is not state.side_to_move:
            continue
        for destination in _candidate_destinations(state, source, piece):
            move = Move(source, destination)
            if not _is_pseudo_legal(state, move, piece):
                continue
            next_state = _move_without_validation(state, move)
            if not is_in_check(next_state, piece.side):
                moves.append(move)
    return tuple(sorted(moves))


def _bare_king_winner(state: GameState) -> Optional[Side]:
    white = [piece for piece in state.board if piece is not None and piece.side is Side.WHITE]
    black = [piece for piece in state.board if piece is not None and piece.side is Side.BLACK]
    white_bare = white == [Piece(Side.WHITE, PieceType.RAJA)]
    black_bare = black == [Piece(Side.BLACK, PieceType.RAJA)]
    # A synthetic two-bare-Raja position has no victor under the specified rule.
    if white_bare and len(black) > 1:
        return Side.BLACK
    if black_bare and len(white) > 1:
        return Side.WHITE
    return None


def _apply_known_legal_move(state: GameState, move: Move) -> GameState:
    """Internal search transition: move MUST come from legal_moves(state).

    No legality or terminal recheck is performed. External callers must use
    apply_move. The existing core transition still owns capture and promotion.
    """
    return _move_without_validation(state, move, record_history=True)


def _is_repetition(state: GameState) -> bool:
    """Project rule: third occurrence of board + player to move is a draw."""
    return state.position_history.count((state.board, state.side_to_move)) >= 2


def _analyze_for_search(state: GameState) -> tuple[tuple[Move, ...], Outcome]:
    """Compute legal moves and terminal knowledge together for one immutable state."""
    bare_king_winner = _bare_king_winner(state)
    if bare_king_winner is not None:
        return (), Outcome(bare_king_winner, TerminalReason.BARE_KING)

    moves = _legal_moves_unchecked(state)
    if moves:
        if _is_repetition(state):
            return (), Outcome(reason=TerminalReason.REPETITION)
        return moves, Outcome()
    if is_in_check(state):
        return moves, Outcome(state.side_to_move.opponent, TerminalReason.CHECKMATE)
    return moves, Outcome(state.side_to_move.opponent, TerminalReason.STALEMATE)


def terminal_status(state: GameState) -> Outcome:
    return _analyze_for_search(state)[1]


def legal_moves(state: GameState) -> tuple[Move, ...]:
    """Return only legal actions, in deterministic from/to index order."""
    if _bare_king_winner(state) is not None or _is_repetition(state):
        return ()
    return _legal_moves_unchecked(state)


def apply_move(state: GameState, move: Move | tuple[str, str]) -> GameState:
    """Validate and apply a move, including compulsory Mantri promotion."""
    if isinstance(move, tuple):
        if len(move) != 2:
            raise IllegalMoveError("A move tuple must contain from and to squares")
        move = Move.from_names(*move)
    if not isinstance(move, Move):
        raise TypeError("move must be a Move or a (from, to) square-name tuple")
    _validate_square(move.from_square)
    _validate_square(move.to_square)
    outcome = terminal_status(state)
    if outcome.terminal:
        raise GameOverError(f"Game already ended by {outcome.reason.value}")
    if move not in _legal_moves_unchecked(state):
        raise IllegalMoveError(f"Illegal Chaturanga move: {move.uci}")
    return _apply_known_legal_move(state, move)


def is_terminal(state: GameState) -> bool:
    return terminal_status(state).terminal


def result(state: GameState, perspective: Side) -> int:
    """Return +1 for the victor, -1 for the loser, or 0 for a draw."""
    return terminal_status(state).reward(perspective)


def clone_state(state: GameState) -> GameState:
    """Return an independent immutable state value (useful as a stable API hook)."""
    return replace(state, board=tuple(state.board))


class Game:
    """Small mutable state manager for UI/game-loop clients."""

    def __init__(self, state: Optional[GameState] = None) -> None:
        self.state = initial_state() if state is None else state

    def legal_moves(self) -> tuple[Move, ...]:
        return legal_moves(self.state)

    def apply(self, move: Move | tuple[str, str]) -> GameState:
        self.state = apply_move(self.state, move)
        return self.state

    def reset(self) -> GameState:
        self.state = initial_state()
        return self.state

    @property
    def outcome(self) -> Outcome:
        return terminal_status(self.state)
