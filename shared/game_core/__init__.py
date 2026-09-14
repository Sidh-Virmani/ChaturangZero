"""
shared/game_core
================
The authoritative Chaturanga rules engine.

All modules in the project (UI adapter, MCTS, training) must call
this engine for any game-logic decision. No other module may implement
piece-movement rules independently.

Public API (import from here):
    from shared.game_core import ChaturangaEngine, ChaturangaState, Move
"""

from .state import ChaturangaState, Piece, PieceKind, Side, Square, Move  # noqa: F401
from .engine import ChaturangaEngine  # noqa: F401

__all__ = [
    "ChaturangaEngine",
    "ChaturangaState",
    "Piece",
    "PieceKind",
    "Side",
    "Square",
    "Move",
]
