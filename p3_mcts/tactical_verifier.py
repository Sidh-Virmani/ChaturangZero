"""
p3_mcts/tactical_verifier.py
=============================
Lightweight tactical verification layer (Research Contribution #5).

Purpose
-------
The neural network may recommend moves that lose the Raja immediately,
miss a one-move checkmate, or overlook promotion. This module provides
fast (alpha-beta-free) tactical checks applied AFTER MCTS selects the
best move, before it is returned to the caller.

Architecture
------------
  Network recommendation
         ↓
  TacticalVerifier.filter_top_move()
    ├── find_checkmate_in_one() → play it immediately if found
    ├── find_promotion_in_one() → play promotion if found and safe
    ├── is_raja_blunder()       → reject moves that lose Raja in 1
    └── Return safest legal move
         ↓
  Final move returned to UI / self-play

This is intentionally a lightweight pass — it only looks 1 ply ahead.
For deeper tactical analysis see p5_endgame/retrograde.py.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Move, PieceKind, Side
from shared.game_core.rules  import is_in_check
from p1_representation.encoder import StateEncoder


class TacticalVerifier:
    """
    Fast 1-ply tactical filter applied to MCTS move selection.

    All methods are stateless; the engine is injected at construction.
    """

    def __init__(self, engine: ChaturangaEngine) -> None:
        self.engine = engine

    # ── Public API ────────────────────────────────────────────────────────────

    def filter_top_move(
        self,
        state:   ChaturangaState,
        top_move: Move,
        legal:    list[Move],
        policy:   np.ndarray,
    ) -> Move:
        """
        Apply tactical checks and return the best safe move.

        Priority order:
          1. If any move delivers immediate checkmate → play it (overrides network)
          2. If top_move is a Raja blunder → downgrade to next-best safe move
          3. If promotion is available and safe → prefer it
          4. Otherwise return top_move unchanged

        Parameters
        ----------
        state    : current position
        top_move : move recommended by MCTS (highest policy mass)
        legal    : all legal moves from state
        policy   : 4096-element policy array from MCTS
        """
        if not legal:
            return top_move

        # 1. Find checkmate in one — always override
        mate = self.find_checkmate_in_one(state, legal)
        if mate is not None:
            return mate

        # 2. Filter out Raja blunders
        safe_moves = [m for m in legal if not self.is_raja_blunder(state, m)]
        if not safe_moves:
            # All moves lose Raja — return best regardless (game is lost)
            return top_move

        # 3. Check if top_move is safe
        if top_move in safe_moves:
            # Bonus: prefer safe promotion if available
            promo = self.find_promotion_in_one(state, safe_moves)
            if promo is not None and not self.is_raja_blunder(state, promo):
                return promo
            return top_move

        # 4. top_move is a blunder — return safest alternative (highest policy mass)
        legal_idx = {StateEncoder.encode_action(m): m for m in safe_moves}
        best_idx  = max(legal_idx, key=lambda i: policy[i] if i < len(policy) else 0)
        return legal_idx[best_idx]

    def is_raja_blunder(self, state: ChaturangaState, move: Move) -> bool:
        """
        Return True if playing `move` would immediately expose own Raja
        to capture on the next move (i.e., the move is illegal OR leaves
        us in check that isn't resolved).

        Note: legal moves already filter self-checks, so this additionally
        checks: does the opponent have a *legal* Raja-capturing move after us?
        """
        # If move is already illegal, it's a blunder by definition
        if not any(m == move for m in self.engine.legal_moves(state)):
            return True

        next_state = self.engine.apply_move(state, move)
        # Check if opponent can capture our Raja in one (Capture-Raja variant logic)
        our_raja_sq = next(
            (p.square for p in next_state.pieces
             if p.kind == PieceKind.RAJA and p.side == state.current_turn),
            None
        )
        if our_raja_sq is None:
            return True  # Raja disappeared — definitely a blunder

        from shared.game_core.rules import is_square_attacked
        return is_square_attacked(our_raja_sq, state.current_turn.opponent(), next_state)

    def find_checkmate_in_one(
        self, state: ChaturangaState, legal: list[Move]
    ) -> Move | None:
        """
        Search for a move that immediately checkmates the opponent.
        Returns the first such move found, or None.
        """
        for move in legal:
            try:
                next_state = self.engine.apply_move(state, move)
                if next_state.status == "checkmate":
                    return move
            except ValueError:
                continue
        return None

    def find_promotion_in_one(
        self, state: ChaturangaState, legal: list[Move]
    ) -> Move | None:
        """
        Return the first promotion move available, if any.
        Promotions are high-value tactical opportunities that the
        network may underestimate early in training.
        """
        for move in legal:
            if move.is_promotion:
                return move
        return None

    def find_check_moves(
        self, state: ChaturangaState, legal: list[Move]
    ) -> list[Move]:
        """Return all moves that put the opponent in check (useful for ordering)."""
        checks = []
        for move in legal:
            try:
                next_state = self.engine.apply_move(state, move)
                if next_state.status == "check":
                    checks.append(move)
            except ValueError:
                continue
        return checks

    def tactical_score(self, state: ChaturangaState, move: Move) -> float:
        """
        Return a quick 0/1/2/3 tactical priority score for a move.
        Used to boost tactical moves in the AI Insights panel.

          3 = checkmate in one
          2 = promotion
          1 = gives check
          0 = normal move
         -1 = Raja blunder
        """
        try:
            next_state = self.engine.apply_move(state, move)
        except ValueError:
            return -1.0

        if next_state.status == "checkmate":
            return 3.0
        if move.is_promotion:
            return 2.0
        if next_state.status == "check":
            return 1.0
        if self.is_raja_blunder(state, move):
            return -1.0
        return 0.0
