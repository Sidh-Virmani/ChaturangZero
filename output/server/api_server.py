"""Local session API and built-UI server for ChaturangZero.

Rules come from shared.game_core; neural search comes from integration.engine.
Use python run.py to serve the complete app, or run this file alongside Vite.
See output/README.md for the API contract.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import re
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Make the project root importable so `shared.game_core` resolves whether
# this is run from the repo root or from inside output/server/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.game_core import (  # noqa: E402
    GameOutcome,
    GameState,
    apply_move,
    initial_state,
    is_in_check,
    legal_moves,
    result,
)
from shared.game_core.types import (  # noqa: E402
    Move,
    Side,
    square_id_to_square,
    square_to_id,
)


# --------------------------------------------------------------------------
# Game session: one in-memory game, guarded by a lock.
# --------------------------------------------------------------------------

class GameSession:
    """Holds the current game and caches the legal moves for its position.

    The cache matters: legal_moves() is the most expensive call in the
    engine, and a single page render asks for it several times (once for
    the selectable squares, once for the status line, once per click).
    Computing it once per position keeps clicks feeling instant.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = initial_state()
        self._cache_key = None
        self._cache_moves: list[Move] = []
        self._undo = []
        self.revision = 0
        self.scenario = 'opening'

    def _snapshot(self):
        return {**serialize_state(self._state, self._moves(self._state)), 'revision': self.revision, 'scenario': self.scenario}

    def _check_revision(self, revision):
        if revision is not None and revision != self.revision:
            raise ApiError('The position changed. Reload the board and try again.', 409)

    def _moves(self, state: GameState) -> list[Move]:
        key = state.position_key()
        if key != self._cache_key:
            self._cache_moves = legal_moves(state)
            self._cache_key = key
        return self._cache_moves

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot()

    def legal_moves_from(self, square_id: str) -> list[dict]:
        with self._lock:
            try:
                square = square_id_to_square(square_id)
            except (ValueError, IndexError):
                raise ApiError(f"'{square_id}' is not a square on the board", 400)
            if result(self._state) is not GameOutcome.ONGOING:
                return []
            return [
                serialize_legal_move(move)
                for move in self._moves(self._state)
                if move.from_sq == square
            ]

    def play(self, from_id: str, to_id: str, revision=None) -> dict:
        with self._lock:
            self._check_revision(revision)
            if result(self._state) is not GameOutcome.ONGOING:
                raise ApiError('This game has ended. Start a new game.', 409)
            try:
                from_sq = square_id_to_square(from_id)
                to_sq = square_id_to_square(to_id)
            except (ValueError, IndexError):
                raise ApiError(f"'{from_id}' or '{to_id}' is not a square on the board", 400)

            match = next(
                (m for m in self._moves(self._state)
                 if m.from_sq == from_sq and m.to_sq == to_sq),
                None,
            )
            if match is None:
                # The engine is the authority: an illegal request is
                # rejected rather than quietly applied.
                raise ApiError(f"{from_id}-{to_id} is not a legal move in this position", 409)

            self._undo.append(self._state)
            self._state = apply_move(self._state, match)
            self.revision += 1
            return self._snapshot()

    def reset(self, scenario='opening') -> dict:
        with self._lock:
            from integration.scenarios import make_position
            try:
                self._state = make_position(scenario)
            except ValueError as error:
                raise ApiError(str(error), 400)
            self.scenario = scenario
            self._undo = []
            self.revision += 1
            return self._snapshot()

    def undo(self, plies=1, revision=None):
        with self._lock:
            self._check_revision(revision)
            for _ in range(min(plies, len(self._undo))):
                self._state = self._undo.pop()
            self.revision += 1
            return self._snapshot()

    def analyze(self, engine, simulations=64, play=False, revision=None):
        with self._lock:
            self._check_revision(revision)
            state, version = self._state, self.revision
        move, analysis = engine.analyze(state, simulations)
        with self._lock:
            self._check_revision(version)
            if play:
                if move is None:
                    raise ApiError('This game has ended. Start a new game.', 409)
                self._undo.append(self._state)
                self._state = apply_move(self._state, move)
                self.revision += 1
            return {'state': self._snapshot(), 'analysis': analysis}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


# --------------------------------------------------------------------------
# Serialization: GameState -> the BoardState shape the React app expects.
# --------------------------------------------------------------------------

_SIDE_LABEL = {Side.WHITE: "Ivory", Side.BLACK: "Jade"}

_RESULT_TEXT = {
    GameOutcome.WHITE_WINS: "Ivory wins by checkmate",
    GameOutcome.BLACK_WINS: "Jade wins by checkmate",
    GameOutcome.DRAW: "Draw",
}


def serialize_legal_move(move: Move) -> dict:
    return {
        "from": square_to_id(move.from_sq),
        "to": square_to_id(move.to_sq),
        "isCapture": move.is_capture,
    }


def serialize_move(move: Move, ply: int) -> dict:
    payload = {
        "id": f"move-{ply}",
        "ply": ply,
        "side": move.side.value,
        "piece": move.piece_kind.value,
        "from": square_to_id(move.from_sq),
        "to": square_to_id(move.to_sq),
        "notation": move.to_notation(),
    }
    if move.captured_kind is not None:
        payload["captured"] = move.captured_kind.value
    return payload


def _status(state: GameState, moves: list[Move]) -> dict:
    turn_label = _SIDE_LABEL[state.side_to_move]
    in_check = is_in_check(state, state.side_to_move)

    if not moves:
        outcome = result(state)
        detail = (
            f"{_SIDE_LABEL[state.side_to_move.opponent]} delivered checkmate."
            if in_check
            else f"{turn_label} has no legal moves. Stalemate."
        )
        return {
            "code": "terminal",
            "label": "Game over",
            "detail": detail,
            "result": _RESULT_TEXT[outcome],
        }

    outcome = result(state)
    if outcome is GameOutcome.DRAW:
        # Reached the no-progress limit while moves are still available.
        return {
            "code": "terminal",
            "label": "Game over",
            "detail": "The same position occurred three times." if state.repetition_count >= 3 else "No capture or padati move for 50 moves.",
            "result": "Draw by repetition" if state.repetition_count >= 3 else "Draw by no progress",
        }

    if in_check:
        return {
            "code": "check",
            "label": f"{turn_label} is in check",
            "detail": "You must respond to the check.",
        }

    return {
        "code": "playing",
        "label": "Game in progress",
        "detail": f"{turn_label} to move.",
    }


def serialize_state(state: GameState, moves: list[Move]) -> dict:
    history = [serialize_move(move, index + 1) for index, move in enumerate(state.move_history)]

    return {
        "pieces": [
            {
                "id": piece.id,
                "kind": piece.kind.value,
                "side": piece.side.value,
                "square": square_to_id(piece.square),
            }
            for piece in state.pieces
        ],
        "currentTurn": state.side_to_move.value,
        # Only squares that actually have a legal move are clickable, so
        # the UI can never offer a selection the engine would reject.
        "selectableSquares": sorted({square_to_id(move.from_sq) for move in moves}) if result(state) is GameOutcome.ONGOING else [],
        "lastMove": history[-1] if history else None,
        "moveHistory": history,
        "status": _status(state, moves),
        "isAiThinking": False,
    }


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------

SESSION = GameSession()
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()
ENGINE = None
ENGINE_LOCK = threading.Lock()
TRAINING = None


def get_engine():
    global ENGINE, TRAINING
    with ENGINE_LOCK:
        if ENGINE is None:
            from integration.engine import Engine, TrainingManager
            ENGINE = Engine()
            TRAINING = TrainingManager()
    return ENGINE


def get_session(token):
    if not token:
        return SESSION
    if not re.fullmatch(r'[a-zA-Z0-9-]{8,80}', token):
        raise ApiError('Invalid session identifier', 400)
    with SESSIONS_LOCK:
        now = time.monotonic()
        expired = [key for key, (_, seen) in SESSIONS.items() if now - seen > 86400]
        for key in expired:
            del SESSIONS[key]
        if token not in SESSIONS:
            if len(SESSIONS) >= 128:
                raise ApiError('Session capacity reached. Restart the local server.', 503)
            SESSIONS[token] = (GameSession(), now)
        session = SESSIONS[token][0]
        SESSIONS[token] = (session, now)
        return session


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        sys.stderr.write("  api  %s\n" % (fmt % args))

    # -- helpers --

    def _send(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise ApiError('Invalid content length', 400)
        if not 0 <= length <= 65536:
            self.close_connection = True
            raise ApiError('Request too large', 413)
        if length == 0:
            return {}
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ApiError('Request body must be a JSON object', 400)
            return body
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiError("Request body was not valid JSON", 400)

    def _session(self):
        return get_session(self.headers.get('X-Session-ID', ''))

    def _static(self, path):
        root = PROJECT_ROOT / 'shared' / 'ui' / 'dist'
        if not root.exists():
            raise ApiError('Build the UI with npm run build in shared/ui, or open the development server on port 5173.', 404)
        file = (root / path.lstrip('/')).resolve()
        if not file.is_relative_to(root.resolve()):
            raise ApiError('Invalid path', 400)
        if not file.is_file():
            if Path(path).suffix:
                raise ApiError('Asset not found', 404)
            file = root / 'index.html'
        body = file.read_bytes()
        self.send_response(200)
        content_type = 'application/javascript' if file.suffix == '.js' else mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    # -- routes --

    def do_OPTIONS(self) -> None:
        self._send({}, 204)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        try:
            if route.path == "/api/health":
                self._send({"ok": True, 'app': 'ChaturangZero', 'version': '1.0.0', 'model': get_engine().info()})
            elif route.path == "/api/state":
                self._send(self._session().snapshot())
            elif route.path == "/api/legal-moves":
                square = parse_qs(route.query).get("square", [""])[0]
                if not square:
                    raise ApiError("Missing required 'square' query parameter", 400)
                self._send(self._session().legal_moves_from(square))
            elif route.path == '/api/model':
                self._send(get_engine().info())
            elif route.path == '/api/training':
                get_engine()
                self._send(TRAINING.snapshot())
            elif not route.path.startswith('/api/'):
                self._static(route.path)
            else:
                raise ApiError(f"Unknown endpoint: {route.path}", 404)
        except ApiError as error:
            self._send({"error": error.message}, error.status)
        except Exception as error:  # pragma: no cover - last-resort guard
            self._send({"error": f"Internal engine error: {error}"}, 500)

    def do_POST(self) -> None:
        route = urlparse(self.path)
        try:
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                self.close_connection = True
                raise ApiError('Use the app from the same origin as its API.', 403)
            body = self._read_json()
            if route.path == "/api/move":
                origin, destination = body.get("from"), body.get("to")
                if not origin or not destination:
                    raise ApiError("Body must include 'from' and 'to' squares", 400)
                self._send(self._session().play(origin, destination, body.get('revision')))
            elif route.path == "/api/reset":
                self._send(self._session().reset(body.get('scenario', 'opening')))
            elif route.path == '/api/undo':
                plies = body.get('plies', 1)
                if type(plies) is not int or not 1 <= plies <= 2:
                    raise ApiError('plies must be 1 or 2', 400)
                self._send(self._session().undo(plies, body.get('revision')))
            elif route.path in ('/api/analyze', '/api/ai-move'):
                simulations = body.get('simulations', 64)
                if type(simulations) is not int or not 1 <= simulations <= 256:
                    raise ApiError('simulations must be an integer from 1 to 256', 400)
                self._send(self._session().analyze(get_engine(), simulations, route.path == '/api/ai-move', body.get('revision')))
            elif route.path == '/api/training/start':
                get_engine()
                if not TRAINING.start():
                    raise ApiError('A training run is already active', 409)
                self._send(TRAINING.snapshot(), 202)
            else:
                raise ApiError(f"Unknown endpoint: {route.path}", 404)
        except ApiError as error:
            self._send({"error": error.message}, error.status)
        except Exception as error:  # pragma: no cover - last-resort guard
            self._send({"error": f"Internal engine error: {error}"}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChaturangZero game core API")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ChaturangZero engine API listening on http://{args.host}:{args.port}")
    print("Endpoints: /api/health  /api/state  /api/legal-moves?square=e2  /api/move  /api/reset")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
