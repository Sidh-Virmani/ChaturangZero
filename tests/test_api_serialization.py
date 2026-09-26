"""Tests for the integration layer in output/server/api_server.py.

The API's job is to translate a Python GameState into the exact JSON
shape the React app already expects (BoardState in
shared/ui/src/types/game.ts). A mismatch here shows up as a blank board
or a silently wrong UI rather than a crash, so it is worth pinning down.

These tests call the serializer and GameSession directly rather than
over HTTP, so they need no running server.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "output" / "server"))

from api_server import (  # noqa: E402
    ApiError,
    GameSession,
    serialize_state,
)

from shared.game_core import (  # noqa: E402
    GameState,
    Piece,
    PieceKind,
    Side,
    legal_moves,
)
from shared.game_core.rules import HALFMOVE_DRAW_LIMIT  # noqa: E402
from shared.game_core.types import square_id_to_square as sq  # noqa: E402


def snapshot(pieces, side_to_move=Side.WHITE, halfmove_clock=0) -> dict:
    state = GameState(
        pieces=tuple(pieces), side_to_move=side_to_move, halfmove_clock=halfmove_clock
    )
    return serialize_state(state, legal_moves(state))


class BoardStateShapeTests(unittest.TestCase):
    def test_initial_snapshot_matches_the_ui_contract(self):
        payload = GameSession().snapshot()

        # Every key the TypeScript BoardState interface declares.
        self.assertEqual(
            set(payload),
            {
                "pieces",
                "currentTurn",
                "selectableSquares",
                "lastMove",
                "moveHistory",
                "status",
                "isAiThinking",
                "revision",
                "scenario",
            },
        )
        self.assertEqual(len(payload["pieces"]), 32)
        self.assertEqual(payload["currentTurn"], "white")
        self.assertIsNone(payload["lastMove"])
        self.assertEqual(payload["moveHistory"], [])
        self.assertFalse(payload["isAiThinking"])

    def test_piece_entries_use_string_square_ids(self):
        piece = GameSession().snapshot()["pieces"][0]
        self.assertEqual(set(piece), {"id", "kind", "side", "square"})
        self.assertRegex(piece["square"], r"^[a-h][1-8]$")
        self.assertIn(piece["side"], ("white", "black"))

    def test_selectable_squares_are_exactly_the_movable_squares(self):
        payload = GameSession().snapshot()
        # 8 padati + 2 ashva + 2 gaja can move at the start = 12 squares.
        self.assertEqual(len(payload["selectableSquares"]), 12)
        self.assertNotIn("e1", payload["selectableSquares"], "the raja is boxed in")
        self.assertNotIn("a1", payload["selectableSquares"], "the ratha is boxed in")


class StatusSerializationTests(unittest.TestCase):
    def test_checkmate_reports_terminal_and_names_the_winner(self):
        payload = snapshot([
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("a8")),
            Piece("wR1", PieceKind.RATHA, Side.WHITE, sq("a1")),
            Piece("wR2", PieceKind.RATHA, Side.WHITE, sq("b8")),
            Piece("wR3", PieceKind.RATHA, Side.WHITE, sq("c8")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
        ], side_to_move=Side.BLACK)

        self.assertEqual(payload["status"]["code"], "terminal")
        self.assertEqual(payload["status"]["result"], "Ivory wins by checkmate")
        self.assertEqual(payload["selectableSquares"], [], "board must be unclickable")

    def test_stalemate_reports_a_draw(self):
        payload = snapshot([
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("a8")),
            Piece("wR1", PieceKind.RATHA, Side.WHITE, sq("b1")),
            Piece("wR2", PieceKind.RATHA, Side.WHITE, sq("c7")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("h1")),
        ], side_to_move=Side.BLACK)

        self.assertEqual(payload["status"]["code"], "terminal")
        self.assertEqual(payload["status"]["result"], "Draw")
        self.assertIn("Stalemate", payload["status"]["detail"])

    def test_check_is_reported_without_ending_the_game(self):
        payload = snapshot([
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("e8")),
            Piece("wR", PieceKind.RATHA, Side.WHITE, sq("e1")),
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("a1")),
        ], side_to_move=Side.BLACK)

        self.assertEqual(payload["status"]["code"], "check")
        self.assertIn("check", payload["status"]["label"].lower())
        self.assertTrue(payload["selectableSquares"], "the player must still be able to respond")

    def test_no_progress_limit_reports_terminal_even_with_moves_available(self):
        payload = snapshot([
            Piece("wK", PieceKind.RAJA, Side.WHITE, sq("a1")),
            Piece("bK", PieceKind.RAJA, Side.BLACK, sq("h8")),
        ], halfmove_clock=HALFMOVE_DRAW_LIMIT)

        self.assertEqual(payload["status"]["code"], "terminal")
        self.assertEqual(payload["status"]["result"], "Draw by no progress")


class GameSessionTests(unittest.TestCase):
    def test_playing_a_move_advances_the_game(self):
        session = GameSession()
        payload = session.play("e2", "e3")

        self.assertEqual(payload["currentTurn"], "black")
        self.assertEqual(len(payload["moveHistory"]), 1)
        self.assertEqual(payload["lastMove"]["notation"], "e2-e3")
        self.assertEqual(payload["lastMove"]["ply"], 1)

    def test_illegal_move_is_rejected_and_leaves_the_game_untouched(self):
        session = GameSession()
        with self.assertRaises(ApiError) as caught:
            session.play("e2", "e5")
        self.assertEqual(caught.exception.status, 409)
        # The rejected request must not have advanced the turn.
        self.assertEqual(session.snapshot()["currentTurn"], "white")
        self.assertEqual(session.snapshot()["moveHistory"], [])

    def test_move_from_an_empty_square_is_rejected(self):
        session = GameSession()
        with self.assertRaises(ApiError):
            session.play("e4", "e5")

    def test_off_board_squares_are_rejected_not_crashed_on(self):
        session = GameSession()
        for bad in ("z9", "a9", "", "e", "11"):
            with self.assertRaises(ApiError, msg=f"{bad!r} should be rejected cleanly"):
                session.legal_moves_from(bad)

    def test_reset_restores_the_opening_position(self):
        session = GameSession()
        session.play("e2", "e3")
        session.play("d7", "d6")
        payload = session.reset()

        self.assertEqual(len(payload["pieces"]), 32)
        self.assertEqual(payload["currentTurn"], "white")
        self.assertEqual(payload["moveHistory"], [])
        self.assertIsNone(payload["lastMove"])

    def test_legal_moves_endpoint_matches_what_the_move_endpoint_accepts(self):
        """Anything the UI is told it may play must actually be playable.

        A mismatch here is the classic integration bug: the board
        highlights a destination, the user clicks it, and the engine
        rejects it.
        """
        session = GameSession()
        for square in session.snapshot()["selectableSquares"]:
            for move in session.legal_moves_from(square):
                probe = GameSession()
                probe.play(move["from"], move["to"])  # must not raise


if __name__ == "__main__":
    unittest.main()
