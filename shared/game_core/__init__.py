"""Chaturanga game core — the single source of truth for the rules.

Every other module in this project (representation, neural network, MCTS,
training, and any future UI backend) should call into this package rather
than re-implementing rule logic. Import the public API from here:

    from shared.game_core import (
        initial_state, legal_moves, legal_moves_from, apply_move,
        is_terminal, result, clone_state, is_in_check,
    )
"""

from .board import initial_state, render_ascii
from .rules import (
    apply_move,
    clone_state,
    is_in_check,
    is_terminal,
    legal_moves,
    legal_moves_from,
    result,
)
from .types import GameOutcome, GameState, Move, Piece, PieceKind, Side, Square

__all__ = [
    "initial_state",
    "render_ascii",
    "apply_move",
    "clone_state",
    "is_in_check",
    "is_terminal",
    "legal_moves",
    "legal_moves_from",
    "result",
    "GameOutcome",
    "GameState",
    "Move",
    "Piece",
    "PieceKind",
    "Side",
    "Square",
]
