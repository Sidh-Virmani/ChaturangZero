"""
shared/game_core/engine.py
==========================
The ChaturangaEngine — the authoritative public API for game logic.

Design note: recursion avoidance
---------------------------------
`_update_status` must NOT call `legal_moves`, because `legal_moves` calls
`_apply_move_unchecked` which calls `_update_status` — infinite recursion.

Instead, `_update_status` calls `_has_legal_move`, which uses the shared
`_make_move_pieces` primitive to check legality without triggering another
status update cycle.

  legal_moves        — calls _make_move_pieces + is_in_check (no recursion)
  _has_legal_move    — same: _make_move_pieces + is_in_check (no recursion)
  _apply_move_unchecked → _update_status → _has_legal_move  (terminates)

ChaturangZero-X addition
-------------------------
  rule_config: RuleConfig — plug in different historical rule variants
  (R1 standard, R2 capture-Raja, R3 bare-Raja, R4 no-stalemate).
  See shared/game_core/rule_variants.py.
"""

from __future__ import annotations

import copy
from typing import Optional

from .state import (
    ChaturangaState,
    Move,
    Piece,
    PieceKind,
    Side,
    Square,
    initial_state,
)
from .rules import is_in_check, pseudolegal_moves

try:
    from .rule_variants import RuleConfig
except ImportError:
    RuleConfig = None   # type: ignore


_HALFMOVE_DRAW_LIMIT = 100  # 50-move rule (each side counts as 1 half-move)


class ChaturangaEngine:
    """
    Stateless Chaturanga game engine.

    Parameters
    ----------
    rule_config : RuleConfig or None
        If None, uses standard rules (checkmate wins, stalemate draws).
        Pass a RuleConfig to activate one of the historical rule variants.
    """

    def __init__(self, rule_config=None) -> None:
        if rule_config is None and RuleConfig is not None:
            self.rule_config = RuleConfig.standard()
        else:
            self.rule_config = rule_config

    def new_game(self) -> ChaturangaState:
        return initial_state()

    def clone(self, state: ChaturangaState) -> ChaturangaState:
        return copy.deepcopy(state)


    # ------------------------------------------------------------------ #
    # Shared piece-move primitive (no ChaturangaState creation)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_move_pieces(pieces: list[Piece], move: Move) -> list[Piece]:
        """
        Apply `move` to a piece list and return a NEW list.
        Does NOT create a ChaturangaState, does NOT call _update_status.
        This is the fundamental primitive used by both legal_moves and
        _apply_move_unchecked so neither triggers a recursive cycle.
        """
        result: list[Piece] = []
        for p in pieces:
            if p.square == move.to_sq and p.side != move.side:
                continue  # captured piece — omit
            if p.square == move.from_sq and p.side == move.side:
                new_kind = PieceKind.MANTRI if move.is_promotion else p.kind
                result.append(Piece(new_kind, p.side, move.to_sq))
            else:
                result.append(Piece(p.kind, p.side, p.square))
        return result

    # ------------------------------------------------------------------ #
    # Legal move generation  (non-recursive)
    # ------------------------------------------------------------------ #

    def legal_moves(self, state: ChaturangaState) -> list[Move]:
        """
        All fully-legal moves for the current player.
        Pseudolegal moves that leave the own king in check are filtered out.

        Does NOT call _apply_move_unchecked / _update_status to stay
        recursion-free.
        """
        if state.is_terminal():
            return []

        legal: list[Move] = []
        for piece in state.pieces_of(state.current_turn):
            for move in pseudolegal_moves(piece, state):
                next_pieces = self._make_move_pieces(state.pieces, move)
                # Minimal temporary state — no status computation needed
                temp = ChaturangaState(
                    pieces=next_pieces,
                    current_turn=state.current_turn.opponent(),
                    status="playing",
                )
                if not is_in_check(state.current_turn, temp):
                    legal.append(move)
        return legal

    def legal_moves_from(self, state: ChaturangaState, sq: Square) -> list[Move]:
        """Legal moves for the piece on `sq`."""
        return [m for m in self.legal_moves(state) if m.from_sq == sq]

    # ------------------------------------------------------------------ #
    # Apply a move  (public — validates legality)
    # ------------------------------------------------------------------ #

    def apply_move(self, state: ChaturangaState, move: Move) -> ChaturangaState:
        """Apply `move` and return the new state. Raises ValueError if illegal."""
        if not any(m == move for m in self.legal_moves(state)):
            raise ValueError(f"Illegal move: {move}")
        return self._apply_move_unchecked(state, move)

    # ------------------------------------------------------------------ #
    # Apply a move  (internal — no validation)
    # ------------------------------------------------------------------ #

    def _apply_move_unchecked(self, state: ChaturangaState, move: Move) -> ChaturangaState:
        """Apply `move` without legality check; update game status."""
        new_pieces = self._make_move_pieces(state.pieces, move)
        next_turn  = state.current_turn.opponent()

        new_halfmove = (
            0
            if (move.captured is not None or move.piece == PieceKind.PADATI)
            else state.halfmove_clock + 1
        )
        new_history = state.move_history + [move]

        next_state = ChaturangaState(
            pieces=new_pieces,
            current_turn=next_turn,
            move_history=new_history,
            halfmove_clock=new_halfmove,
            status="playing",
            winner=None,
        )
        # _update_status calls _has_legal_move — does NOT recurse into legal_moves
        return self._update_status(next_state)

    # ------------------------------------------------------------------ #
    # Non-recursive helpers for _update_status
    # ------------------------------------------------------------------ #

    def _has_legal_move(self, state: ChaturangaState) -> bool:
        """
        Return True iff the current player has at least one legal move.
        Uses _make_move_pieces (no status update) — safe to call from _update_status.
        """
        for piece in state.pieces_of(state.current_turn):
            for move in pseudolegal_moves(piece, state):
                next_pieces = self._make_move_pieces(state.pieces, move)
                temp = ChaturangaState(
                    pieces=next_pieces,
                    current_turn=state.current_turn.opponent(),
                    status="playing",
                )
                if not is_in_check(state.current_turn, temp):
                    return True
        return False

    def _update_status(self, state: ChaturangaState) -> ChaturangaState:
        """
        Determine the game status after a move has been applied.

        Respects self.rule_config for historical rule variants:
          R3 (bare_raja_loses)  : only-Raja left = loss
          R4 (stalemate_is_loss): stalemate = loss (not draw)
        """
        current   = state.current_turn
        in_check  = is_in_check(current, state)
        has_moves = self._has_legal_move(state)
        cfg       = self.rule_config

        # ── Bare-Raja condition (R3) ──────────────────────────────────────────
        if cfg is not None and cfg.bare_raja_loses:
            own_pieces = state.pieces_of(current)
            if len(own_pieces) == 1 and own_pieces[0].kind.value == "raja":
                # Only Raja left — count as loss unless opponent also bare
                opp_pieces = state.pieces_of(current.opponent())
                if len(opp_pieces) > 1:
                    return ChaturangaState(
                        pieces=state.pieces,
                        current_turn=current,
                        move_history=state.move_history,
                        halfmove_clock=state.halfmove_clock,
                        status="checkmate",        # reuse terminal status
                        winner=current.opponent().value,
                    )

        # ── No legal moves ────────────────────────────────────────────────────
        if not has_moves:
            if in_check:
                # Standard checkmate
                return ChaturangaState(
                    pieces=state.pieces,
                    current_turn=current,
                    move_history=state.move_history,
                    halfmove_clock=state.halfmove_clock,
                    status="checkmate",
                    winner=current.opponent().value,
                )
            else:
                # Stalemate — R4: counts as loss; else draw
                stalemate_is_loss = (cfg is not None and cfg.stalemate_is_loss)
                return ChaturangaState(
                    pieces=state.pieces,
                    current_turn=current,
                    move_history=state.move_history,
                    halfmove_clock=state.halfmove_clock,
                    status="checkmate" if stalemate_is_loss else "draw",
                    winner=current.opponent().value if stalemate_is_loss else "draw",
                )

        # ── 50-move (halfmove) draw ───────────────────────────────────────────
        limit = cfg.halfmove_limit if cfg is not None else _HALFMOVE_DRAW_LIMIT
        if state.halfmove_clock >= limit:
            return ChaturangaState(
                pieces=state.pieces,
                current_turn=current,
                move_history=state.move_history,
                halfmove_clock=state.halfmove_clock,
                status="draw",
                winner="draw",
            )

        status_code = "check" if in_check else "playing"
        return ChaturangaState(
            pieces=state.pieces,
            current_turn=current,
            move_history=state.move_history,
            halfmove_clock=state.halfmove_clock,
            status=status_code,
            winner=None,
        )


    # ------------------------------------------------------------------ #
    # Terminal queries
    # ------------------------------------------------------------------ #

    def is_terminal(self, state: ChaturangaState) -> bool:
        return state.is_terminal()

    def result(self, state: ChaturangaState) -> Optional[float]:
        """+1 win, -1 loss, 0 draw, None = not finished."""
        return state.result()

    def selectable_squares(self, state: ChaturangaState) -> list[Square]:
        return list({m.from_sq for m in self.legal_moves(state)})

    # ------------------------------------------------------------------ #
    # API serialisation
    # ------------------------------------------------------------------ #

    def state_to_api_dict(self, state: ChaturangaState) -> dict:
        selectable = [sq.label() for sq in self.selectable_squares(state)]

        status_label_map = {
            "playing":   "Game in progress",
            "check":     "Check!",
            "checkmate": "Checkmate",
            "draw":      "Draw",
        }
        status_detail_map = {
            "playing":   f"{'Ivory' if state.current_turn == Side.WHITE else 'Onyx'} to move.",
            "check":     f"{'Ivory' if state.current_turn == Side.WHITE else 'Onyx'} is in check!",
            "checkmate": f"{'Ivory' if state.winner == 'white' else 'Onyx'} wins by checkmate!",
            "draw":      "The game is drawn.",
        }

        return {
            "pieces": [
                {
                    "id":     f"{p.side.value}-{p.kind.value}-{p.square.label()}",
                    "kind":   p.kind.value,
                    "side":   p.side.value,
                    "square": p.square.label(),
                }
                for p in state.pieces
            ],
            "currentTurn":       state.current_turn.value,
            "selectableSquares": selectable,
            "lastMove": (
                {
                    "id":       f"move-{len(state.move_history)}",
                    "ply":      len(state.move_history),
                    "side":     state.move_history[-1].side.value,
                    "piece":    state.move_history[-1].piece.value,
                    "from":     state.move_history[-1].from_sq.label(),
                    "to":       state.move_history[-1].to_sq.label(),
                    "captured": (state.move_history[-1].captured.value
                                 if state.move_history[-1].captured else None),
                    "notation": state.move_history[-1].label(),
                }
                if state.move_history else None
            ),
            "moveHistory": [
                {
                    "id":       f"move-{i+1}",
                    "ply":      i + 1,
                    "side":     m.side.value,
                    "piece":    m.piece.value,
                    "from":     m.from_sq.label(),
                    "to":       m.to_sq.label(),
                    "captured": m.captured.value if m.captured else None,
                    "notation": m.label(),
                }
                for i, m in enumerate(state.move_history)
            ],
            "status": {
                "code":   state.status,
                "label":  status_label_map.get(state.status, state.status),
                "detail": status_detail_map.get(state.status, ""),
                "result": state.winner,
            },
            "isAiThinking": False,
        }
