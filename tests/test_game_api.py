from shared.game_core import IllegalMoveError
from shared.game_core.server import GameSession


def test_api_state_exposes_every_movable_initial_piece():
    session = GameSession()
    state = session.state_dict()
    assert len(state["pieces"]) == 32
    assert set(state["selectableSquares"]) == {
        "a2", "b1", "b2", "c1", "c2", "d2",
        "e2", "f1", "f2", "g1", "g2", "h2",
    }


def test_api_legal_moves_and_state_are_generated_by_core():
    session = GameSession()
    assert session.legal_moves_from("c1") == [
        {"from": "c1", "to": "a3", "isCapture": False},
        {"from": "c1", "to": "e3", "isCapture": False},
    ]
    state = session.apply("c1", "a3")
    assert state["currentTurn"] == "black"
    assert state["lastMove"]["piece"] == "gaja"
    assert any(piece["square"] == "a3" for piece in state["pieces"])


def test_api_cannot_bypass_core_validation():
    session = GameSession()
    try:
        session.apply("a2", "a4")
    except IllegalMoveError:
        pass
    else:
        raise AssertionError("API accepted a forbidden Padati double-step")


def test_api_reset_restores_authoritative_initial_state():
    session = GameSession()
    session.apply("b1", "a3")
    state = session.reset()
    assert state["currentTurn"] == "white"
    assert state["moveHistory"] == []
    assert len(state["pieces"]) == 32
