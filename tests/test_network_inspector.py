"""Real inspector tensors must agree with P1/P2 and leave gameplay untouched."""
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
import torch

from integration.ai_player import MachinePlayer
from integration.network_inspector import inspect_position
from p1_representation import PLANE_NAMES, encode_state, legal_action_mask
from p2_neural_network import ModelConfig, NetworkEvaluator, PolicyValueNetwork, save_checkpoint
from shared.game_core import GameState, Piece, PieceType, Side, initial_state
from shared.game_core.server import GameApiHandler, GameSession


@pytest.mark.parametrize("blocks", [1, 3])
def test_inspector_matches_real_forward_and_runtime_architecture(blocks):
    model = PolicyValueNetwork(ModelConfig(channels=8, residual_blocks=blocks, value_hidden=16))
    evaluator = NetworkEvaluator(model)
    state = initial_state()
    expected = evaluator.evaluate(state)
    model.train()
    norm_before = model.stem[1].running_mean.clone()
    snapshot = inspect_position(state, evaluator)
    assert model.training
    assert torch.equal(norm_before, model.stem[1].running_mean)
    layers = {layer["id"]: layer for layer in snapshot["layers"]}
    assert layers["input"]["channelNames"] == list(PLANE_NAMES)
    np.testing.assert_array_equal(np.array(layers["input"]["channels"]).reshape(13, 8, 8), encode_state(state))
    assert len([key for key in layers if key.startswith("residual-")]) == blocks
    assert layers["stem"]["shape"] == [8, 8, 8]
    assert layers["value"]["shape"] == [1, 1]
    assert layers["value"]["channels"][0][0] == pytest.approx(expected.value)
    assert snapshot["value"] == pytest.approx(expected.value)
    reconstructed = np.zeros(4096)
    for action in snapshot["policy"]:
        reconstructed[action["actionId"]] = action["prior"]
    np.testing.assert_allclose(reconstructed, expected.policy, atol=1e-7)
    destinations = np.array(layers["policy"]["channels"])[0]
    np.testing.assert_allclose(destinations, expected.policy.reshape(64, 64).sum(axis=0), atol=1e-7)
    assert destinations.sum() == pytest.approx(1)
    assert snapshot["legalActionCount"] == int(legal_action_mask(state).sum())
    # Endpoint content must serialize as strict JSON, including every tensor.
    json.dumps(snapshot, allow_nan=False)


def test_missing_checkpoint_returns_only_exact_encoding_and_does_not_mutate_game(monkeypatch):
    monkeypatch.setattr("integration.ai_player.configured_checkpoint", lambda: None)
    session = GameSession()
    session.apply("e2", "e3")
    before = session.state_dict()
    snapshot = session.network_snapshot()
    assert session.state_dict() == before
    assert snapshot["positionKey"] == before["positionKey"]
    assert snapshot["checkpoint"] is None
    assert snapshot["value"] is None
    assert [layer["id"] for layer in snapshot["layers"]] == ["input"]
    assert snapshot["layers"][0]["channels"][-1] == [0.0] * 64


def test_terminal_policy_is_zero_and_value_keeps_its_negative_sign():
    model = PolicyValueNetwork(ModelConfig(channels=8, residual_blocks=1))
    with torch.no_grad():
        model.value_head[2].weight.zero_()
        model.value_head[2].bias.fill_(-1)
    state = GameState.from_mapping({"a1": Piece(Side.WHITE, PieceType.RAJA),
                                    "h8": Piece(Side.BLACK, PieceType.RAJA),
                                    "h7": Piece(Side.BLACK, PieceType.MANTRI)})
    snapshot = inspect_position(state, NetworkEvaluator(model))
    assert snapshot["legalActionCount"] == 0
    assert snapshot["policy"] == []
    layers = {layer["id"]: layer for layer in snapshot["layers"]}
    assert layers["policy"]["channels"] == [[0.0] * 64]
    assert snapshot["value"] < 0
    assert layers["value"]["channels"][0][0] == snapshot["value"]


def test_inspector_rejects_snapshot_if_a_move_arrives_during_inference(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    save_checkpoint(PolicyValueNetwork(ModelConfig(channels=8, residual_blocks=1)), path)
    machine = MachinePlayer(path, simulations=2)
    session = GameSession(machine=machine)
    original = MachinePlayer.inspect

    def concurrent_move(self, state):
        session.apply("e2", "e3")
        return original(self, state)

    monkeypatch.setattr(MachinePlayer, "inspect", concurrent_move)
    with pytest.raises(RuntimeError, match="board changed"):
        session.network_snapshot()
    assert len(session.history) == 1


def test_inspector_http_endpoint_and_search_busy_response(monkeypatch):
    monkeypatch.setattr("integration.ai_player.configured_checkpoint", lambda: None)

    class Handler(GameApiHandler):
        session = GameSession()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/game/network"
        with urlopen(url, timeout=5) as response:
            assert json.load(response)["layers"][0]["shape"] == [13, 8, 8]
        Handler.session.search_status_data = {"done": False}
        with pytest.raises(HTTPError) as error:
            urlopen(url, timeout=5)
        assert error.value.code == 503
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=2)
