"""Edge-case, invariant and regression tests for shared/game_core.

test_game_core.py checks "does each rule do what we intended?".
This file checks the harder question: "can we break it?"

It does that three ways:
  1. Explicit tests for the tricky corners (padati attack asymmetry,
     pinned pieces, the no-progress draw, immutability).
  2. Invariant fuzzing — play thousands of random legal moves and assert
     that a set of things that must ALWAYS be true never break.
  3. Perft — a standard chess-engine technique that counts every leaf
     node in the move tree to a fixed depth. The absolute numbers are
     specific to our frozen ruleset, so they are locked in here as a
     regression baseline: if someone changes movement.py and a perft
     number moves, they changed the rules (possibly by accident).

Run with:
    python -m unittest discover -s tests -v
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
from shared.game_core.movement import all_pseudo_legal_moves
from shared.game_core.rules import HALFMOVE_DRAW_LIMIT, is_square_attacked
from shared.game_core.types import square_id_to_square as sq


def state_with(pieces, side_to_move=Side.WHITE, halfmove_clock=0) -> GameState:
    return GameState(pieces=tuple(pieces), side_to_move=side_to_move, halfmove_clock=halfmove_clock)


def perft(state: GameState, depth: int) -> int:
    """Count leaf nodes at exactly `depth` plies."""
    if depth == 0:
        return 1
    moves = legal_moves(state)
    if depth == 1:
        return len(moves)
    return sum(perft(apply_move(state, move), depth - 1) for move in moves)


class PadatiAttackAsymmetryTests(unittest.TestCase):
    """A padati MOVES straight but CAPTURES diagonally.

    This asymmetry is the single easiest thing to get wrong in a rules
    engine, because the naive implementation derives "attacked squares"
    from "move destinations" and ends up thinking a padati covers the
    square straight ahead of it.
    """

    def test_square_in_front_of_padati_is_not_attacked(self):
        state = state_with([
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("e7")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("a1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        # Black padati on e7 moves toward rank 1, so e6 is straight ahead.
        self.assertFalse(is_square_attacked(state, sq("e6"), Side.BLACK))
        # ...but it does attack both diagonals.
        self.assertTrue(is_square_attacked(state, sq("d6"), Side.BLACK))
        self.assertTrue(is_square_attacked(state, sq("f6"), Side.BLACK))

    def test_raja_may_step_directly_in_front_of_enemy_padati(self):
        state = state_with([
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("e5")),
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("e7")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("a8")),
        ])
        destinations = {m.to_sq for m in legal_moves_from(state, sq("e5"))}
        self.assertIn(sq("e6"), destinations, "e6 is safe: a padati cannot capture straight ahead")
        self.assertNotIn(sq("d6"), destinations, "d6 is attacked diagonally by the e7 padati")
        self.assertNotIn(sq("f6"), destinations, "f6 is attacked diagonally by the e7 padati")

    def test_padati_cannot_capture_straight_ahead(self):
        state = state_with([
            Piece("wP", PieceKind.PADATI, Side.WHITE, sq("e4")),
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("e5")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("a1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        self.assertEqual(legal_moves_from(state, sq("e4")), [], "blocked head-on, no capture")


class PinAndCheckEvasionTests(unittest.TestCase):
    def test_absolutely_pinned_piece_cannot_move(self):
        # White mantri on e2 is pinned to its raja on e1 by the ratha on e8.
        state = state_with([
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("e1")),
            Piece("wM", PieceKind.MANTRI, Side.WHITE, sq("e2")),
            Piece("bR", PieceKind.RATHA, Side.BLACK, sq("e8")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("a8")),
        ])
        self.assertEqual(legal_moves_from(state, sq("e2")), [],
                         "moving the mantri would expose its own raja")

    def test_only_check_evasions_are_legal(self):
        # White raja on e1 in check from the ratha on e8. The only legal
        # replies are raja steps off the e-file, or blocking on e2.
        state = state_with([
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("e1")),
            Piece("wM", PieceKind.MANTRI, Side.WHITE, sq("d3")),
            Piece("bR", PieceKind.RATHA, Side.BLACK, sq("e8")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("a8")),
        ])
        self.assertTrue(is_in_check(state, Side.WHITE))

        for move in legal_moves(state):
            after = apply_move(state, move)
            self.assertFalse(is_in_check(after, Side.WHITE),
                             f"{move.to_notation()} leaves the raja in check")

        # The mantri on d3 can block by going to e2.
        self.assertIn("d3-e2", [m.to_notation() for m in legal_moves(state)])

    def test_capturing_the_checking_piece_is_legal(self):
        state = state_with([
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("e1")),
            Piece("wR", PieceKind.RATHA, Side.WHITE, sq("a8")),
            Piece("bR", PieceKind.RATHA, Side.BLACK, sq("e8")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h5")),
        ])
        self.assertTrue(is_in_check(state, Side.WHITE))
        self.assertIn("a8xe8", [m.to_notation() for m in legal_moves(state)])


class NoProgressDrawTests(unittest.TestCase):
    def test_halfmove_clock_resets_on_capture_and_padati_move(self):
        state = state_with([
            Piece("wP", PieceKind.PADATI, Side.WHITE, sq("e4")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("a1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ], halfmove_clock=40)

        padati_move = next(m for m in legal_moves(state) if m.piece_kind is PieceKind.PADATI)
        self.assertEqual(apply_move(state, padati_move).halfmove_clock, 0)

        raja_move = next(m for m in legal_moves(state) if m.piece_kind is PieceKind.RAJA)
        self.assertEqual(apply_move(state, raja_move).halfmove_clock, 41)

    def test_game_is_drawn_at_the_no_progress_limit(self):
        state = state_with([
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("a1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ], halfmove_clock=HALFMOVE_DRAW_LIMIT)

        self.assertTrue(is_terminal(state))
        self.assertEqual(result(state), GameOutcome.DRAW)

    def test_checkmate_beats_the_no_progress_limit(self):
        # Position is mate AND at the draw limit; mate must take priority.
        state = state_with([
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("a8")),
            Piece("wR1", PieceKind.RATHA, Side.WHITE, sq("a1")),
            Piece("wR2", PieceKind.RATHA, Side.WHITE, sq("b8")),
            Piece("wR3", PieceKind.RATHA, Side.WHITE, sq("c8")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
        ], side_to_move=Side.BLACK, halfmove_clock=HALFMOVE_DRAW_LIMIT)

        self.assertEqual(result(state), GameOutcome.WHITE_WINS)


class ImmutabilityTests(unittest.TestCase):
    def test_apply_move_never_mutates_the_input_state(self):
        # MCTS depends on this: it explores many futures from one node.
        state = initial_state()
        before_pieces = state.pieces
        before_turn = state.side_to_move
        before_history = state.move_history

        for move in legal_moves(state):
            apply_move(state, move)

        self.assertIs(state.pieces, before_pieces)
        self.assertIs(state.side_to_move, before_turn)
        self.assertIs(state.move_history, before_history)

    def test_sibling_states_do_not_share_mutable_structure(self):
        state = initial_state()
        moves = legal_moves(state)
        a = apply_move(state, moves[0])
        b = apply_move(state, moves[1])
        self.assertNotEqual(a.pieces, b.pieces)
        self.assertEqual(len(a.pieces), len(b.pieces))


class PromotionEdgeTests(unittest.TestCase):
    def test_promotion_by_capture_on_the_far_rank(self):
        state = state_with([
            Piece("wP", PieceKind.PADATI, Side.WHITE, sq("b7")),
            Piece("bR", PieceKind.RATHA, Side.BLACK, sq("c8")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h6")),
        ])
        move = next(m for m in legal_moves_from(state, sq("b7")) if m.to_sq == sq("c8"))
        self.assertEqual(move.promotion, PieceKind.MANTRI)
        self.assertTrue(move.is_capture)

        after = apply_move(state, move)
        self.assertEqual(after.piece_at(sq("c8")).kind, PieceKind.MANTRI)
        self.assertEqual(len(after.pieces), 3, "the captured ratha is gone")

    def test_black_padati_promotes_on_rank_1(self):
        state = state_with([
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("a2")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ], side_to_move=Side.BLACK)
        move = next(m for m in legal_moves_from(state, sq("a2")))
        self.assertEqual(move.promotion, PieceKind.MANTRI)
        self.assertEqual(apply_move(state, move).piece_at(sq("a1")).kind, PieceKind.MANTRI)


class GajaJumpEdgeTests(unittest.TestCase):
    def test_gaja_never_removes_the_piece_it_jumps_over(self):
        state = state_with([
            Piece("wG", PieceKind.GAJA, Side.WHITE, sq("c1")),
            Piece("bP", PieceKind.PADATI, Side.BLACK, sq("d2")),  # jumped over
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        move = next(m for m in legal_moves_from(state, sq("c1")) if m.to_sq == sq("e3"))
        after = apply_move(state, move)
        self.assertIsNotNone(after.piece_at(sq("d2")), "the jumped padati must survive")
        self.assertEqual(len(after.pieces), 4)

    def test_gaja_stays_on_one_colour_complex(self):
        # A 2-square diagonal mover can only ever reach a fixed subset of
        # squares. Verify it never leaves the colour it started on.
        start = sq("c1")
        state = state_with([
            Piece("wG", PieceKind.GAJA, Side.WHITE, start),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ])
        start_colour = (start[0] + start[1]) % 2
        for move in legal_moves_from(state, start):
            self.assertEqual((move.to_sq[0] + move.to_sq[1]) % 2, start_colour)


class InvariantFuzzTests(unittest.TestCase):
    """Play a lot of random legal games and assert things that must never break."""

    def test_invariants_hold_across_random_games(self):
        rng = random.Random(20260901)

        for game_index in range(40):
            state = initial_state()
            ply = 0

            while not is_terminal(state):
                moves = legal_moves(state)
                self.assertTrue(moves, "is_terminal said the game continues but there are no moves")

                # --- invariants on the current position ---
                squares = [p.square for p in state.pieces]
                self.assertEqual(len(squares), len(set(squares)),
                                 f"game {game_index} ply {ply}: two pieces on one square")
                self.assertLessEqual(len(state.pieces), 32, "piece count grew")
                for side in (Side.WHITE, Side.BLACK):
                    kings = [p for p in state.pieces
                             if p.side is side and p.kind is PieceKind.RAJA]
                    self.assertEqual(len(kings), 1,
                                     f"game {game_index} ply {ply}: {side.value} has {len(kings)} rajas")

                # every legal move must be a subset of pseudo-legal moves
                pseudo = set(all_pseudo_legal_moves(state, state.side_to_move))
                self.assertTrue(set(moves).issubset(pseudo))

                # no legal move may leave our own raja in check
                chosen = rng.choice(moves)
                after = apply_move(state, chosen)
                self.assertFalse(is_in_check(after, state.side_to_move),
                                 f"game {game_index} ply {ply}: {chosen.to_notation()} self-checks")

                # a move never captures a raja
                self.assertIsNot(chosen.captured_kind, PieceKind.RAJA,
                                 "a raja was captured — check detection is broken")

                # piece count only ever drops, and only on a capture
                expected = len(state.pieces) - (1 if chosen.is_capture else 0)
                self.assertEqual(len(after.pieces), expected)

                # turn must alternate
                self.assertIs(after.side_to_move, state.side_to_move.opponent)

                state = after
                ply += 1

            self.assertNotEqual(result(state), GameOutcome.ONGOING,
                                "terminal position must have a concrete result")

    def test_every_random_game_terminates(self):
        """The no-progress rule must guarantee games actually end.

        Without it, self-play training would hang on a shuffling endgame.
        """
        rng = random.Random(7)
        HARD_CAP = 5000

        for game_index in range(25):
            state = initial_state()
            plies = 0
            while not is_terminal(state) and plies < HARD_CAP:
                state = apply_move(state, rng.choice(legal_moves(state)))
                plies += 1
            self.assertTrue(is_terminal(state),
                            f"game {game_index} did not finish within {HARD_CAP} plies")


class PerftRegressionTests(unittest.TestCase):
    """Locked-in move-tree node counts for the current frozen ruleset.

    These numbers are NOT taken from any external Chaturanga authority —
    they are generated by this engine. Their value is as a tripwire: if a
    change to movement.py shifts one of them, the rules changed. If that
    was intentional, update the number in the same commit and say so.
    """

    def test_perft_from_initial_position(self):
        state = initial_state()
        self.assertEqual(perft(state, 0), 1)
        self.assertEqual(perft(state, 1), 16)
        # 256 == 16 x 16: the opening position is symmetric and no first
        # move can interfere with the opponent's options, so every one of
        # White's 16 moves must leave Black exactly 16 replies. That this
        # falls out exactly is a good independent correctness signal.
        self.assertEqual(perft(state, 2), 256)
        self.assertEqual(perft(state, 3), 4176)

    def test_every_opening_move_leaves_sixteen_replies(self):
        state = initial_state()
        for move in legal_moves(state):
            self.assertEqual(len(legal_moves(apply_move(state, move))), 16,
                             f"after {move.to_notation()}")

    def test_perft_is_internally_consistent(self):
        # perft(n) must equal the sum of perft(n-1) over all children.
        state = initial_state()
        total = sum(perft(apply_move(state, m), 2) for m in legal_moves(state))
        self.assertEqual(total, perft(state, 3))


if __name__ == "__main__":
    unittest.main()
