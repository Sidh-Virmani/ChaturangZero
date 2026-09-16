from dataclasses import replace

import numpy as np
import pytest

from p1_representation import (
    ACTION_SPACE_SIZE,
    action_to_move,
    legal_action_mask,
    move_to_action,
)
from shared.game_core import (
    GameState,
    Move,
    Piece,
    PieceType,
    Side,
    apply_move,
    initial_state,
    legal_moves,
)


def position(extra: dict[str, Piece], side: Side = Side.WHITE) -> GameState:
    pieces = {
        "a1": Piece(Side.WHITE, PieceType.RAJA),
        "h2": Piece(Side.WHITE, PieceType.PADATI),
        "h7": Piece(Side.BLACK, PieceType.PADATI),
        "h8": Piece(Side.BLACK, PieceType.RAJA),
    }
    pieces.update(extra)
    return GameState.from_mapping(pieces, side)


def test_action_ids_have_a_fixed_size_and_worked_square_examples():
    assert ACTION_SPACE_SIZE == 4096
    assert move_to_action(Move.from_names("a1", "b1")) == 1
    assert move_to_action(Move.from_names("b1", "c3")) == 82
    assert move_to_action(Move.from_names("a2", "a3")) == 528
    assert move_to_action(Move.from_names("h8", "a1")) == 4032
    assert move_to_action(Move.from_names("h8", "h8")) == 4095
    assert action_to_move(528) == Move.from_names("a2", "a3")
    assert action_to_move(np.int64(82)) == Move.from_names("b1", "c3")


def test_every_square_pair_has_a_unique_id_and_round_trips():
    ids = set()
    for source in range(64):
        for destination in range(64):
            move = Move(source, destination)
            action = move_to_action(move)
            assert action_to_move(action) == move
            ids.add(action)
    assert ids == set(range(ACTION_SPACE_SIZE))


@pytest.mark.parametrize("bad_move", [Move(-1, 0), Move(0, 64), Move(True, 2)])
def test_invalid_move_squares_are_rejected(bad_move):
    with pytest.raises(ValueError):
        move_to_action(bad_move)


@pytest.mark.parametrize("bad_action", [-1, 4096, True, 1.5, "82"])
def test_invalid_action_ids_are_rejected(bad_action):
    with pytest.raises(ValueError):
        action_to_move(bad_action)


@pytest.mark.parametrize(
    ("state", "move"),
    [
        (position({}), ("a1", "b1")),  # Raja
        (position({"d4": Piece(Side.WHITE, PieceType.MANTRI)}), ("d4", "e5")),
        (position({"d4": Piece(Side.WHITE, PieceType.GAJA)}), ("d4", "f6")),
        (position({"d4": Piece(Side.WHITE, PieceType.ASHVA)}), ("d4", "f5")),
        (position({"d4": Piece(Side.WHITE, PieceType.RATHA)}), ("d4", "d5")),
        (position({"d4": Piece(Side.WHITE, PieceType.PADATI)}), ("d4", "d5")),
        (position({"d4": Piece(Side.WHITE, PieceType.PADATI),
                   "e5": Piece(Side.BLACK, PieceType.ASHVA)}), ("d4", "e5")),
        (position({"b7": Piece(Side.WHITE, PieceType.PADATI),
                   "c8": Piece(Side.BLACK, PieceType.ASHVA)}), ("b7", "c8")),
        (position({"d7": Piece(Side.BLACK, PieceType.PADATI)}, Side.BLACK), ("d7", "d6")),
    ],
)
def test_masks_include_core_legal_moves_for_each_piece_capture_and_promotion(state, move):
    core_move = Move.from_names(*move)
    assert core_move in legal_moves(state)
    assert legal_action_mask(state)[move_to_action(core_move)] == 1
    assert action_to_move(move_to_action(core_move)) == core_move


def test_black_promotion_is_a_single_from_to_action():
    state = GameState.from_mapping({
        "h1": Piece(Side.WHITE, PieceType.RAJA),
        "h2": Piece(Side.WHITE, PieceType.PADATI),
        "h8": Piece(Side.BLACK, PieceType.RAJA),
        "b2": Piece(Side.BLACK, PieceType.PADATI),
        "c1": Piece(Side.WHITE, PieceType.ASHVA),
    }, Side.BLACK)
    move = Move.from_names("b2", "c1")
    assert move in legal_moves(state)
    assert legal_action_mask(state)[move_to_action(move)] == 1
    assert apply_move(state, move).piece_at("c1") == Piece(Side.BLACK, PieceType.MANTRI)


def test_mask_matches_core_exactly_for_initial_and_midgame_positions():
    states = [initial_state()]
    for move in (("a2", "a3"), ("a7", "a6"), ("b1", "c3"), ("b8", "c6")):
        states.append(apply_move(states[-1], move))

    for state in states:
        before = replace(state)
        mask = legal_action_mask(state)
        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)) <= {0, 1}
        assert set(np.flatnonzero(mask)) == {move_to_action(move) for move in legal_moves(state)}
        assert state == before
    assert np.count_nonzero(legal_action_mask(states[0])) == 16
    assert legal_action_mask(states[0])[move_to_action(Move.from_names("a2", "a4"))] == 0
    assert legal_action_mask(states[0])[move_to_action(Move.from_names("a1", "a1"))] == 0


def test_terminal_position_has_no_legal_actions_and_masks_are_independent():
    state = GameState.from_mapping({
        "a1": Piece(Side.WHITE, PieceType.RAJA),
        "h8": Piece(Side.BLACK, PieceType.RAJA),
        "h7": Piece(Side.BLACK, PieceType.MANTRI),
    })
    mask = legal_action_mask(state)
    assert np.count_nonzero(mask) == 0
    mask[:] = 1
    assert np.count_nonzero(legal_action_mask(state)) == 0
