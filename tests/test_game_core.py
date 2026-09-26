"""Tests for shared/game_core.

Run with:
    python -m unittest discover -s tests -v

Covers, roughly in order:
  - the starting position and basic move application
  - each piece's individual movement pattern (blocking, jumping, capturing)
  - promotion
  - check / checkmate / stalemate detection
  - a random-vs-random smoke test (architecture doc milestone: many
    legal games should complete without crashing or reaching an
    impossible state)
"""

from __future__ import annotations

import random
import unittest

from shared.game_core import (
    GameOutcome,
    GameState,
    Piece,
    PieceKind,
    Side,
    apply_move,
    initial_state,
    is_in_check,
    is_terminal,
    legal_moves,
    legal_moves_from,
    result,
)
from shared.game_core.types import square_id_to_square as sq


def state_with(pieces, side_to_move=Side.WHITE) -> GameState:
    """Small helper for building custom positions in tests."""
    return GameState(pieces=tuple(pieces), side_to_move=side_to_move)


class InitialPositionTests(unittest.TestCase):
    def test_32_pieces_white_to_move(self):
        state = initial_state()
        self.assertEqual(len(state.pieces), 32)
        self.assertIs(state.side_to_move, Side.WHITE)

    def test_16_legal_opening_moves(self):
        # 8 padati pushes + 2 ashva x 2 jumps each + 2 gaja x 2 jumps each.
        state = initial_state()
        self.assertEqual(len(legal_moves(state)), 16)

    def test_apply_move_flips_turn_and_moves_piece(self):
        state = initial_state()
        move = next(m for m in legal_moves(state) if m.to_notation() == "e2-e3")
        next_state = apply_move(state, move)

        self.assertIs(next_state.side_to_move, Side.BLACK)
        self.assertIsNone(next_state.piece_at(sq("e2")))
        self.assertEqual(next_state.piece_at(sq("e3")).kind, PieceKind.PADATI)
        # original state must be untouched
        self.assertIsNotNone(state.piece_at(sq("e2")))


class RathaTests(unittest.TestCase):
    def test_blocked_by_own_piece(self):
        state = state_with([
            Piece("wR", PieceKind.RATHA, Side.WHITE, sq("a1")),
            Piece("wP", PieceKind.PADATI, Side.WHITE, sq("a3")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        destinations = {m.to_notation().split("-")[-1] for m in legal_moves_from(state, sq("a1"))}
        self.assertIn("a1-a2", [m.to_notation() for m in legal_moves_from(state, sq("a1"))])
        self.assertNotIn("a3", destinations)  # can't land on own piece
        self.assertNotIn("a4", destinations)  # or slide past it

    def test_captures_and_stops_at_enemy_piece(self):
        state = state_with([
            Piece("wR", PieceKind.RATHA, Side.WHITE, sq("a1")),
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("a4")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        moves = legal_moves_from(state, sq("a1"))
        notations = [m.to_notation() for m in moves]
        self.assertIn("a1xa4", notations)
        self.assertNotIn("a1-a5", notations)  # blocked beyond the capture


class AshvaGajaTests(unittest.TestCase):
    def test_ashva_jumps_over_pieces(self):
        # ashva boxed in on all four adjacent squares must still be able
        # to jump, because knight-style movement ignores blockers.
        state = state_with([
            Piece("wA", PieceKind.ASHVA, Side.WHITE, sq("d4")),
            Piece("wP1", PieceKind.PADATI, Side.WHITE, sq("d5")),
            Piece("wP2", PieceKind.PADATI, Side.WHITE, sq("d3")),
            Piece("wP3", PieceKind.PADATI, Side.WHITE, sq("c4")),
            Piece("wP4", PieceKind.PADATI, Side.WHITE, sq("e4")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        moves = legal_moves_from(state, sq("d4"))
        self.assertEqual(len(moves), 8)  # all 8 knight destinations open

    def test_gaja_jumps_and_ignores_intervening_piece(self):
        state = state_with([
            Piece("wG", PieceKind.GAJA, Side.WHITE, sq("d4")),
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("e5")),  # sits between d4 and f6
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        notations = [m.to_notation() for m in legal_moves_from(state, sq("d4"))]
        self.assertIn("d4-f6", notations)  # jump succeeds despite e5 piece
        self.assertNotIn("d4-e5", notations)  # can't land one square short
        self.assertNotIn("d4xe5", notations)  # and never captures the jumped piece

    def test_gaja_can_capture_on_landing_square(self):
        state = state_with([
            Piece("wG", PieceKind.GAJA, Side.WHITE, sq("d4")),
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("f6")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        notations = [m.to_notation() for m in legal_moves_from(state, sq("d4"))]
        self.assertIn("d4xf6", notations)


class PadatiTests(unittest.TestCase):
    def test_no_double_step_and_blocked_straight_move(self):
        state = state_with([
            Piece("wP", PieceKind.PADATI, Side.WHITE, sq("e2")),
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("e3")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        self.assertEqual(legal_moves_from(state, sq("e2")), [])

    def test_diagonal_capture_only_when_enemy_present(self):
        state = state_with([
            Piece("wP", PieceKind.PADATI, Side.WHITE, sq("e2")),
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("d3")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        notations = [m.to_notation() for m in legal_moves_from(state, sq("e2"))]
        self.assertIn("e2xd3", notations)
        self.assertIn("e2-e3", notations)
        self.assertNotIn("e2-f3", notations)  # nothing to capture there

    def test_promotes_to_mantri_on_far_rank(self):
        state = state_with([
            Piece("wP", PieceKind.PADATI, Side.WHITE, sq("a7")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        moves = legal_moves_from(state, sq("a7"))
        self.assertEqual(len(moves), 1)
        promo_move = moves[0]
        self.assertEqual(promo_move.promotion, PieceKind.MANTRI)

        next_state = apply_move(state, promo_move)
        self.assertEqual(next_state.piece_at(sq("a8")).kind, PieceKind.MANTRI)


class CheckAndTerminalTests(unittest.TestCase):
    def test_in_check_detection(self):
        state = state_with([
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("a8")),
            Piece("wR", PieceKind.RATHA, Side.WHITE, sq("a1")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
        ], side_to_move=Side.BLACK)
        self.assertTrue(is_in_check(state, Side.BLACK))

    def test_checkmate(self):
        # Black raja on a8, mated by two coordinated white rathas.
        state = state_with([
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("a8")),
            Piece("wR1", PieceKind.RATHA, Side.WHITE, sq("a1")),   # covers a-file
            Piece("wR2", PieceKind.RATHA, Side.WHITE, sq("b8")),   # gives check, covers b-file
            Piece("wR3", PieceKind.RATHA, Side.WHITE, sq("c8")),   # defends wR2 against capture
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
        ], side_to_move=Side.BLACK)

        self.assertTrue(is_in_check(state, Side.BLACK))
        self.assertEqual(legal_moves(state), [])
        self.assertTrue(is_terminal(state))
        self.assertEqual(result(state), GameOutcome.WHITE_WINS)

    def test_stalemate(self):
        state = state_with([
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("a8")),
            Piece("wR1", PieceKind.RATHA, Side.WHITE, sq("b1")),   # covers b7, b8
            Piece("wR2", PieceKind.RATHA, Side.WHITE, sq("c7")),   # covers a7 (and b7) via rank 7
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
        ], side_to_move=Side.BLACK)

        self.assertFalse(is_in_check(state, Side.BLACK))
        self.assertEqual(legal_moves(state), [])
        self.assertTrue(is_terminal(state))
        self.assertEqual(result(state), GameOutcome.DRAW)

    def test_ongoing_position_reports_ongoing(self):
        state = initial_state()
        self.assertFalse(is_terminal(state))
        self.assertEqual(result(state), GameOutcome.ONGOING)


class RandomVsRandomSmokeTest(unittest.TestCase):
    def test_many_random_games_complete_without_crashing(self):
        # Mirrors the architecture doc's "random-vs-random full games"
        # milestone: the rules engine should never produce a crash or an
        # impossible state, regardless of which legal moves are chosen.
        rng = random.Random(1234)
        MAX_PLIES = 300

        for game_index in range(15):
            state = initial_state()
            plies = 0
            while not is_terminal(state) and plies < MAX_PLIES:
                moves = legal_moves(state)
                state = apply_move(state, rng.choice(moves))
                plies += 1

            # Every position along the way must have kept exactly one
            # raja per side alive (a rules bug could otherwise let a
            # raja be "captured" like a normal piece).
            white_kings = sum(1 for p in state.pieces if p.side is Side.WHITE and p.kind is PieceKind.RAJA)
            black_kings = sum(1 for p in state.pieces if p.side is Side.BLACK and p.kind is PieceKind.RAJA)
            self.assertEqual(white_kings, 1, f"game {game_index} lost the white raja")
            self.assertEqual(black_kings, 1, f"game {game_index} lost the black raja")


if __name__ == "__main__":
    unittest.main()
