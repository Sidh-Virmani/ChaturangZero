"""Small HTTP boundary between the authoritative Python core and the React UI."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from .core import (
    Game,
    GameOverError,
    IllegalMoveError,
    Move,
    PieceType,
    Side,
    is_in_check,
    legal_moves,
    parse_square,
    square_name,
    terminal_status,
)


@dataclass(slots=True)
class MoveRecord:
    ply: int
    side: Side
    piece: PieceType
    move: Move
    captured: Optional[PieceType]

    def as_dict(self) -> dict[str, Any]:
        capture_marker = "x" if self.captured is not None else "–"
        data: dict[str, Any] = {
            "id": f"move-{self.ply}",
            "ply": self.ply,
            "side": self.side.value,
            "piece": self.piece.value,
            "from": square_name(self.move.from_square),
            "to": square_name(self.move.to_square),
            "notation": (
                f"{square_name(self.move.from_square)}"
                f"{capture_marker}{square_name(self.move.to_square)}"
            ),
        }
        if self.captured is not None:
            data["captured"] = self.captured.value
        return data


@dataclass(slots=True)
class GameSession:
    """Thread-safe state manager used by every API endpoint."""

    game: Game = field(default_factory=Game)
    history: list[MoveRecord] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.game.reset()
            self.history.clear()
            return self._state_dict()

    def state_dict(self) -> dict[str, Any]:
        with self.lock:
            return self._state_dict()

    def legal_moves_from(self, source_name: str) -> list[dict[str, Any]]:
        source = parse_square(source_name)
        with self.lock:
            state = self.game.state
            moves = [move for move in legal_moves(state) if move.from_square == source]
            return [
                {
                    "from": square_name(move.from_square),
                    "to": square_name(move.to_square),
                    "isCapture": state.board[move.to_square] is not None,
                }
                for move in moves
            ]

    def apply(self, from_name: str, to_name: str) -> dict[str, Any]:
        move = Move.from_names(from_name, to_name)
        with self.lock:
            before = self.game.state
            moving_piece = before.board[move.from_square]
            captured_piece = before.board[move.to_square]
            if moving_piece is None:
                raise IllegalMoveError(f"There is no piece on {from_name}")
            self.game.apply(move)
            self.history.append(
                MoveRecord(
                    ply=self.game.state.ply_count,
                    side=moving_piece.side,
                    piece=moving_piece.kind,
                    move=move,
                    captured=captured_piece.kind if captured_piece else None,
                )
            )
            return self._state_dict()

    def _state_dict(self) -> dict[str, Any]:
        state = self.game.state
        moves = legal_moves(state)
        selectable = sorted({square_name(move.from_square) for move in moves})
        outcome = terminal_status(state)

        if outcome.terminal:
            winner_name = "Ivory" if outcome.winner is Side.WHITE else "Onyx"
            reason = outcome.reason.value.replace("_", " ").title()
            status = {
                "code": "terminal",
                "label": reason,
                "detail": f"{winner_name} wins by {reason.lower()}.",
                "result": f"{winner_name} wins",
            }
        elif is_in_check(state):
            checked_name = "Ivory" if state.side_to_move is Side.WHITE else "Onyx"
            status = {
                "code": "check",
                "label": "Raja in check",
                "detail": f"{checked_name} must answer the attack on the Raja.",
            }
        else:
            turn_name = "Ivory" if state.side_to_move is Side.WHITE else "Onyx"
            status = {
                "code": "playing",
                "label": "Game in progress",
                "detail": f"{turn_name} to move.",
            }

        pieces = []
        for square, piece in enumerate(state.board):
            if piece is None:
                continue
            name = square_name(square)
            pieces.append(
                {
                    "id": f"{piece.side.value}-{piece.kind.value}-{name}",
                    "kind": piece.kind.value,
                    "side": piece.side.value,
                    "square": name,
                }
            )

        return {
            "pieces": pieces,
            "currentTurn": state.side_to_move.value,
            "selectableSquares": selectable,
            "lastMove": self.history[-1].as_dict() if self.history else None,
            "moveHistory": [record.as_dict() for record in self.history],
            "status": status,
            "isAiThinking": False,
        }


class GameApiHandler(BaseHTTPRequestHandler):
    session = GameSession()
    server_version = "ChaturangZeroCore/1.0"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/game":
                self._send_json(HTTPStatus.OK, self.session.state_dict())
                return
            if parsed.path == "/api/game/legal-moves":
                source = parse_qs(parsed.query).get("from", [""])[0]
                if not source:
                    raise ValueError("The 'from' query parameter is required")
                self._send_json(HTTPStatus.OK, self.session.legal_moves_from(source))
                return
            self._send_error(HTTPStatus.NOT_FOUND, "API route not found")
        except (ValueError, IllegalMoveError, GameOverError) as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/game/reset":
                self._read_json(allow_empty=True)
                self._send_json(HTTPStatus.OK, self.session.reset())
                return
            if parsed.path == "/api/game/move":
                payload = self._read_json()
                from_name = payload.get("from")
                to_name = payload.get("to")
                if not isinstance(from_name, str) or not isinstance(to_name, str):
                    raise ValueError("Move payload requires string 'from' and 'to' squares")
                self._send_json(
                    HTTPStatus.OK, self.session.apply(from_name, to_name)
                )
                return
            self._send_error(HTTPStatus.NOT_FOUND, "API route not found")
        except (ValueError, TypeError, IllegalMoveError, GameOverError) as error:
            self._send_error(HTTPStatus.BAD_REQUEST, str(error))

    def do_OPTIONS(self) -> None:  # noqa: N802 - browser preflight
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_common_headers()
        self.end_headers()

    def _read_json(self, *, allow_empty: bool = False) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0 and allow_empty:
            return {}
        if length <= 0:
            raise ValueError("A JSON request body is required")
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError("Malformed JSON request body") from error
        if not isinstance(payload, dict):
            raise ValueError("JSON request body must be an object")
        return payload

    def _send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_common_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[game-core] {self.address_string()} - {format % args}")


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), GameApiHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChaturangZero game-core API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    server = create_server(args.host, args.port)
    print(f"ChaturangZero game core listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

