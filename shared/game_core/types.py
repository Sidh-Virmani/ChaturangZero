"""Data structures shared by every other file in game_core.

Nothing in here knows how pieces move or whether a position is legal —
it only defines the shapes of the data (what a Piece looks like, what a
Move looks like, what a GameState looks like) so the rest of the engine
can agree on a common vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    WHITE = "white"
    BLACK = "black"

    @property
    def opponent(self) -> "Side":
        return Side.BLACK if self is Side.WHITE else Side.WHITE


class PieceKind(str, Enum):
    RAJA = "raja"      # King
    MANTRI = "mantri"  # Counselor (1 square diagonally)
    GAJA = "gaja"       # Elephant (2 squares diagonally, jumps)
    ASHVA = "ashva"     # Horse (knight move)
    RATHA = "ratha"     # Chariot (rook move)
    PADATI = "padati"   # Foot soldier (pawn)


Square = tuple[int, int]
"""A board square as (file, rank), both 0-7. (0, 0) is a1, (7, 7) is h8."""


def square_to_id(square: Square) -> str:
    file_, rank = square
    return f"{'abcdefgh'[file_]}{rank + 1}"


def square_id_to_square(square_id: str) -> Square:
    """Parse 'e2' into (4, 1). Raises ValueError on anything off-board.

    The validation is not optional politeness: this is the shared parser
    the API and (later) the action-encoding layer both use. Returning an
    off-board coordinate for 'a9' instead of raising would surface as a
    silently empty move list or corrupted training data, which is far
    harder to trace than an exception here.
    """
    if not isinstance(square_id, str) or len(square_id) != 2:
        raise ValueError(f"'{square_id}' is not a square id (expected two characters like 'e2')")

    file_char, rank_char = square_id[0], square_id[1]
    if file_char not in "abcdefgh":
        raise ValueError(f"'{square_id}' has no file between a and h")
    if rank_char not in "12345678":
        raise ValueError(f"'{square_id}' has no rank between 1 and 8")

    return ("abcdefgh".index(file_char), int(rank_char) - 1)


def in_bounds(square: Square) -> bool:
    file_, rank = square
    return 0 <= file_ <= 7 and 0 <= rank <= 7


@dataclass(frozen=True)
class Piece:
    id: str
    kind: PieceKind
    side: Side
    square: Square


@dataclass(frozen=True)
class Move:
    from_sq: Square
    to_sq: Square
    piece_kind: PieceKind
    side: Side
    captured_kind: Optional[PieceKind] = None
    promotion: Optional[PieceKind] = None

    @property
    def is_capture(self) -> bool:
        return self.captured_kind is not None

    def to_notation(self) -> str:
        base = f"{square_to_id(self.from_sq)}{'x' if self.is_capture else '-'}{square_to_id(self.to_sq)}"
        return f"{base}={self.promotion.value}" if self.promotion else base


class GameOutcome(str, Enum):
    ONGOING = "ongoing"
    WHITE_WINS = "white_wins"
    BLACK_WINS = "black_wins"
    DRAW = "draw"


@dataclass(frozen=True)
class GameState:
    pieces: tuple[Piece, ...]
    side_to_move: Side
    move_history: tuple[Move, ...] = field(default_factory=tuple)
    halfmove_clock: int = 0
    """Half-moves since the last capture or padati move. Drives the
    no-progress draw rule in rules.py."""

    prior_positions: tuple[tuple, ...] = field(default_factory=tuple)

    @property
    def repetition_count(self) -> int:
        return 1 + self.prior_positions.count(self.position_key())

    # Lookup table square -> piece, built once per state. Excluded from
    # comparison/hashing because it is derived data, not part of the
    # position's identity. Without it, piece_at() was a linear scan over
    # all 32 pieces and got called millions of times during search.
    _index: Optional[dict] = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {piece.square: piece for piece in self.pieces})

    def piece_at(self, square: Square) -> Optional[Piece]:
        return self._index.get(square)

    def position_key(self) -> tuple:
        """Identity of the *position* alone, ignoring how we got here.

        Two states with the same key are the same position for repetition
        purposes. Piece IDs and move-history length do not affect this identity.
        """
        return (
            tuple(sorted((p.square, p.kind, p.side) for p in self.pieces)),
            self.side_to_move,
        )
