"""The browser's machine mode uses a real P2/P3 search behind the core API."""

import json
import threading
from types import SimpleNamespace
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from integration.ai_player import MachinePlayer
from p1_representation import legal_action_mask, move_to_action
from p2_neural_network import ModelConfig, PolicyValueNetwork, save_checkpoint
from shared.game_core import IllegalMoveError, legal_moves
from shared.game_core.server import GameApiHandler, GameSession


@pytest.fixture
def machine(tmp_path):
    path = tmp_path / "latest_model.pt"
    model = PolicyValueNetwork(ModelConfig(
        channels=16, residual_blocks=1, policy_channels=2, value_hidden=32,
    ))
    save_checkpoint(model, path)
    return MachinePlayer(path, simulations=3)


def test_machine_search_uses_legal_core_moves_and_real_root_statistics(machine):
    session = GameSession(machine=machine)
    assert session.reset("machine")["machineReady"] is True
    after_human = session.apply("e2", "e3")
    assert after_human["currentTurn"] == "black"
    before_search = session.game.state
    legal = set(legal_moves(before_search))

    with pytest.raises(IllegalMoveError, match="machine's turn"):
        session.apply("e7", "e6")

    result = session.apply_machine()
    progress = session.search_status()
    assert progress["done"] is True
    assert progress["simulationsCompleted"] == 3
    analysis = result["analysis"]
    assert progress["legalActionCount"] == analysis["legalActionCount"]
    assert any("MCTS simulations completed" in event for event in progress["events"])
    selected = analysis["selectedAction"]
    assert result["currentTurn"] == "white"
    assert len(result["moveHistory"]) == 2
    assert result["lastMove"]["side"] == "black"
    assert (selected["from"], selected["to"]) == (
        result["lastMove"]["from"], result["lastMove"]["to"]
    )
    assert analysis["legalActionCount"] == int(legal_action_mask(before_search).sum())
    assert analysis["rootVisits"] == analysis["simulations"] == 3
    assert analysis['searchElapsedSeconds'] > 0
    assert analysis['simulationsPerSecond'] == pytest.approx(3 / analysis['searchElapsedSeconds'])
    assert analysis['pipelineElapsedSeconds'] >= analysis['searchElapsedSeconds']
    assert progress['searchElapsedSeconds'] >= analysis['searchElapsedSeconds']
    assert result['engine']['simulations'] == 3
    assert len(result['engine']['sha256']) == 64
    assert result['engine']['training'] is None
    assert result['repetition'] == {'occurrences': 1, 'drawAt': 3}
    assert -1 <= analysis["value"] <= 1
    assert all(0 <= item["prior"] <= 1 for item in analysis["policyPriors"])
    assert all(item["visits"] >= 0 for item in analysis["candidates"])
    assert selected["actionId"] in {move_to_action(move) for move in legal}


def test_http_machine_move_returns_analysis_and_rejects_out_of_turn_human_move(machine):
    class Handler(GameApiHandler):
        session = GameSession(machine=machine)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(route, payload):
        request = Request(
            base + route, json.dumps(payload).encode(),
            {"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    try:
        initial = post("/api/game/reset", {"mode": "machine"})
        assert initial["mode"] == "machine"
        post("/api/game/move", {"from": "e2", "to": "e3"})
        with pytest.raises(HTTPError) as rejected:
            post("/api/game/move", {"from": "e7", "to": "e6"})
        assert rejected.value.code == 400
        result = post("/api/game/ai/move", {})
        assert result["analysis"]["selectedAction"]["actionId"] >= 0
        assert result["moveHistory"][-1]["side"] == "black"
        with urlopen(base + "/api/game/ai/status", timeout=5) as response:
            progress = json.load(response)
        assert progress["done"] is True
        assert progress["simulationsCompleted"] == result["analysis"]["simulations"]
        assert post("/api/game/reset", {"mode": "human"})["analysis"] is None
        assert Handler.session.state_dict()['engine'] is None
        assert Handler.session.state_dict()['repetition'] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_missing_selected_checkpoint_is_reported_without_resetting_the_game(tmp_path, monkeypatch):
    session = GameSession()
    session.apply("e2", "e3")
    monkeypatch.setenv("CHATURANGZERO_CHECKPOINT", str(tmp_path / "missing.pt"))
    with pytest.raises(RuntimeError, match="Machine checkpoint not found"):
        session.reset("machine")
    assert session.mode == "human"
    assert len(session.history) == 1


def test_search_status_remains_readable_and_reset_discards_stale_search():
    searching = threading.Event()
    release = threading.Event()

    class WaitingMachine:
        config = SimpleNamespace(num_simulations=4)
        label = "waiting test model"
        is_smoke = False

        def choose(self, state, on_progress=None):
            on_progress("state", {})
            on_progress("legal", {"legalActionCount": len(legal_moves(state))})
            on_progress("mcts", {"totalSimulations": 4})
            searching.set()
            assert release.wait(timeout=5)
            return legal_moves(state)[0], {}

    session = GameSession(machine=WaitingMachine())
    session.reset("machine")
    session.apply("e2", "e3")
    errors = []

    def search():
        try:
            session.apply_machine()
        except RuntimeError as error:
            errors.append(str(error))

    worker = threading.Thread(target=search)
    worker.start()
    try:
        assert searching.wait(timeout=5)
        progress = session.search_status()
        assert progress["done"] is False
        assert progress["legalActionCount"] > 0
        assert progress["events"][-1] == "Running 4 MCTS simulations with PUCT..."
        assert len(session.state_dict()["moveHistory"]) == 1
        session.reset("machine")
    finally:
        release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert errors == ["Machine search was superseded by a new game"]
    assert session.state_dict()["moveHistory"] == []
    assert session.search_status()["phase"] == "idle"
