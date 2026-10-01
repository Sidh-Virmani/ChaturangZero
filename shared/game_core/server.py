"""Small HTTP boundary between the authoritative Python core and the React UI."""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Optional
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

if TYPE_CHECKING:
    from integration.ai_player import MachinePlayer


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
    mode: str = "human"
    machine: MachinePlayer | None = None
    analysis: dict[str, Any] | None = None
    search_generation: int = 0
    search_status_data: dict[str, Any] | None = None

    def reset(self, mode: str = "human") -> dict[str, Any]:
        if mode not in ("human", "machine"):
            raise ValueError("mode must be 'human' or 'machine'")
        with self.lock:
            if mode == "machine" and self.machine is None:
                from integration.ai_player import MachinePlayer, configured_checkpoint

                checkpoint = configured_checkpoint()
                if checkpoint is not None:
                    try:
                        simulations = int(os.environ.get("CHATURANGZERO_SIMULATIONS", "32"))
                        self.machine = MachinePlayer(checkpoint, simulations)
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        raise RuntimeError(f"Could not load machine checkpoint: {error}") from error
            self.game.reset()
            self.search_generation += 1
            self.search_status_data = None
            self.history.clear()
            self.mode = mode
            self.analysis = None
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
            if self.mode == "machine" and self.machine is None:
                raise IllegalMoveError("No machine checkpoint is available")
            if self.mode == "machine" and before.side_to_move is Side.BLACK:
                raise IllegalMoveError("It is the machine's turn")
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
            self.analysis = None
            return self._state_dict()

    def apply_machine(self) -> dict[str, Any]:
        with self.lock:
            if self.mode != "machine":
                raise ValueError("Machine move requires Human vs Machine mode")
            if self.machine is None:
                raise RuntimeError("No machine checkpoint is available")
            before = self.game.state
            if before.side_to_move is not Side.BLACK:
                raise IllegalMoveError("It is not the machine's turn")
            if terminal_status(before).terminal:
                raise GameOverError("The game is already over")
            if self.search_status_data is not None and not self.search_status_data["done"]:
                raise RuntimeError("Machine search is already running")
            self.search_generation += 1
            generation = self.search_generation
            machine = self.machine
            self.search_status_data = {
                "phase": "starting", "events": [], "legalActionCount": None,
                "simulationsCompleted": 0, "totalSimulations": machine.config.num_simulations,
                "done": False, "error": None,
            }

        try:
            move, analysis = machine.choose(
                before,
                on_progress=lambda stage, data: self._publish_search_progress(generation, stage, data),
            )
        except Exception as error:
            with self.lock:
                if generation == self.search_generation and self.search_status_data is not None:
                    self.search_status_data.update(phase="error", done=True, error=str(error))
            raise

        with self.lock:
            if generation != self.search_generation or self.game.state != before or self.mode != "machine":
                raise RuntimeError("Machine search was superseded by a new game")
            moving_piece = before.board[move.from_square]
            captured_piece = before.board[move.to_square]
            if moving_piece is None:  # pragma: no cover - core/search invariant
                raise RuntimeError("Machine selected an empty source square")
            self.game.apply(move)
            self.history.append(MoveRecord(
                ply=self.game.state.ply_count,
                side=moving_piece.side,
                piece=moving_piece.kind,
                move=move,
                captured=captured_piece.kind if captured_piece else None,
            ))
            self.analysis = analysis
            if self.search_status_data is not None:
                self.search_status_data.update(phase="complete", done=True)
            return self._state_dict()

    def search_status(self) -> dict[str, Any]:
        """A short snapshot that can be read while MCTS runs in another request."""
        with self.lock:
            if self.search_status_data is None:
                return {"phase": "idle", "events": [], "legalActionCount": None,
                        "simulationsCompleted": 0, "totalSimulations": 0,
                        "done": True, "error": None}
            return {**self.search_status_data, "events": list(self.search_status_data["events"])}

    def _publish_search_progress(self, generation: int, stage: str, data: dict[str, Any]) -> None:
        with self.lock:
            if generation != self.search_generation or self.search_status_data is None:
                return
            progress = self.search_status_data
            progress["phase"] = stage
            if "legalActionCount" in data:
                progress["legalActionCount"] = data["legalActionCount"]
            if "totalSimulations" in data:
                progress["totalSimulations"] = data["totalSimulations"]
            if "simulationsCompleted" in data:
                progress["simulationsCompleted"] = data["simulationsCompleted"]
            messages = {
                "state": "Reading the current board State...",
                "legal": f"Found {progress['legalActionCount']} Legal Actions.",
                "network": "Encoding the board State for the Neural Network...",
                "network_ready": "Policy Network scored Actions; Value Network estimated the State.",
                "search_complete": f"{progress['simulationsCompleted']} MCTS simulations completed.",
                "selected": "Comparing the most visited Candidate Actions...",
            }
            if stage == "mcts" and "simulationsCompleted" not in data:
                messages[stage] = f"Running {progress['totalSimulations']} MCTS simulations with PUCT..."
            if stage == "selected" and "selectedAction" in data:
                action = data["selectedAction"]
                progress["events"].append("Comparing the most visited Candidate Actions...")
                messages[stage] = f"Machine selected {action['from']} → {action['to']}."
            if stage in messages:
                progress["events"].append(messages[stage])

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
            "mode": self.mode,
            "machineReady": self.machine is not None if self.mode == "machine" else False,
            "checkpointLabel": self.machine.label if self.mode == "machine" and self.machine else None,
            "smokeCheckpoint": self.machine.is_smoke if self.mode == "machine" and self.machine else False,
            "analysis": self.analysis,
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
            if parsed.path == "/api/game/ai/status":
                self._send_json(HTTPStatus.OK, self.session.search_status())
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
                payload = self._read_json(allow_empty=True)
                self._send_json(HTTPStatus.OK, self.session.reset(payload.get("mode", "human")))
                return
            if parsed.path == "/api/game/ai/move":
                self._read_json(allow_empty=True)
                self._send_json(HTTPStatus.OK, self.session.apply_machine())
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
        except RuntimeError as error:
            self._send_error(HTTPStatus.SERVICE_UNAVAILABLE, str(error))

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
        if self.path == "/api/game/ai/status":
            return
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

