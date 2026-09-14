"""
shared/game_core/state.py
=========================
Pure data classes for the Chaturanga board state.
No movement logic lives here — that is in rules.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    WHITE = "white"
    BLACK = "black"

    def opponent(self) -> "Side":
        return Side.BLACK if self == Side.WHITE else Side.WHITE


class PieceKind(str, Enum):
    RAJA   = "raja"    # King        — 1 step any direction
    MANTRI = "mantri"  # Minister    — 1 step diagonal
    RATHA  = "ratha"   # Chariot     — slides orthogonally (like Rook)
    GAJA   = "gaja"    # Elephant    — jumps exactly 2 squares diagonally
    ASHVA  = "ashva"   # Horse       — L-shape (like Knight)
    PADATI = "padati"  # Soldier     — 1 forward, captures diagonally


# Square is an (file, rank) pair.  file: 0-7 (a=0 … h=7), rank: 0-7 (1=0 … 8=7)
@dataclass(frozen=True)
class Square:
    file: int  # 0..7
    rank: int  # 0..7

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def is_valid(self) -> bool:
        return 0 <= self.file <= 7 and 0 <= self.rank <= 7

    def label(self) -> str:
        """Human-readable label such as 'e4'."""
        return f"{'abcdefgh'[self.file]}{self.rank + 1}"

    @staticmethod
    def from_label(label: str) -> "Square":
        """Parse a label like 'e4' into a Square."""
        file = "abcdefgh".index(label[0])
        rank = int(label[1]) - 1
        return Square(file, rank)

    def __str__(self) -> str:
        return self.label()


@dataclass
class Piece:
    kind:   PieceKind
    side:   Side
    square: Square

    def __repr__(self) -> str:
        return f"{self.side.value[0].upper()}{self.kind.value[:2].upper()}@{self.square}"


@dataclass
class Move:
    from_sq:  Square
    to_sq:    Square
    piece:    PieceKind
    side:     Side
    captured: Optional[PieceKind] = None
    is_promotion: bool = False

    def label(self) -> str:
        cap = f"x{self.captured.value}" if self.captured else ""
        return f"{self.from_sq}{cap}{self.to_sq}"

    def __repr__(self) -> str:
        return self.label()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return NotImplemented
        return (self.from_sq, self.to_sq) == (other.from_sq, other.to_sq)

    def __hash__(self) -> int:
        return hash((self.from_sq, self.to_sq))


@dataclass
class ChaturangaState:
    """
    Complete, self-contained description of one Chaturanga game position.

    Attributes
    ----------
    pieces          : list of all living pieces
    current_turn    : whose turn it is
    move_history    : ordered list of moves played so far
    halfmove_clock  : moves since last capture or pawn move (for 50-move rule)
    status          : 'playing' | 'check' | 'checkmate' | 'draw'
    winner          : None if not terminal, else the winning Side (or 'draw')
    """
    pieces:         list[Piece]           = field(default_factory=list)
    current_turn:   Side                  = Side.WHITE
    move_history:   list[Move]            = field(default_factory=list)
    halfmove_clock: int                   = 0
    status:         str                   = "playing"   # playing|check|checkmate|draw
    winner:         Optional[str]         = None        # 'white'|'black'|'draw'|None

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    def piece_at(self, sq: Square) -> Optional[Piece]:
        for p in self.pieces:
            if p.square == sq:
                return p
        return None

    def pieces_of(self, side: Side) -> list[Piece]:
        return [p for p in self.pieces if p.side == side]

    def raja_square(self, side: Side) -> Optional[Square]:
        for p in self.pieces:
            if p.kind == PieceKind.RAJA and p.side == side:
                return p.square
        return None

    def is_terminal(self) -> bool:
        return self.status in ("checkmate", "draw")

    def result(self) -> Optional[float]:
        """
        Returns the result from the current player's perspective.
        +1 = win, -1 = loss, 0 = draw, None = not terminal.
        """
        if not self.is_terminal():
            return None
        if self.winner == "draw":
            return 0.0
        winning_side = Side(self.winner) if self.winner else None
        return 1.0 if winning_side == self.current_turn else -1.0

    def clone(self) -> "ChaturangaState":
        """Return a deep copy of this state."""
        import copy
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        """Serialise for the API / frontend."""
        return {
            "pieces": [
                {
                    "kind":   p.kind.value,
                    "side":   p.side.value,
                    "square": p.square.label(),
                }
                for p in self.pieces
            ],
            "currentTurn":  self.current_turn.value,
            "status":       self.status,
            "winner":       self.winner,
            "moveCount":    len(self.move_history),
            "halfmoveClock": self.halfmove_clock,
        }


# ---------------------------------------------------------------------------
# Initial board factory
# ---------------------------------------------------------------------------

BACK_RANK: list[PieceKind] = [
    PieceKind.RATHA,
    PieceKind.ASHVA,
    PieceKind.GAJA,
    PieceKind.MANTRI,
    PieceKind.RAJA,
    PieceKind.GAJA,
    PieceKind.ASHVA,
    PieceKind.RATHA,
]


def initial_state() -> ChaturangaState:
    """Return the standard Chaturanga starting position."""
    pieces: list[Piece] = []

    for file_idx in range(8):
        # White back rank (rank 0 = rank 1 on board)
        pieces.append(Piece(BACK_RANK[file_idx], Side.WHITE, Square(file_idx, 0)))
        # White pawns (rank 1 = rank 2 on board)
        pieces.append(Piece(PieceKind.PADATI, Side.WHITE, Square(file_idx, 1)))
        # Black pawns (rank 6 = rank 7 on board)
        pieces.append(Piece(PieceKind.PADATI, Side.BLACK, Square(file_idx, 6)))
        # Black back rank (rank 7 = rank 8 on board)
        pieces.append(Piece(BACK_RANK[file_idx], Side.BLACK, Square(file_idx, 7)))

    return ChaturangaState(pieces=pieces)
