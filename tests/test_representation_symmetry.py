from dataclasses import replace

import numpy as np
import pytest

from p1_representation import (
    ACTION_SPACE_SIZE,
    encode_state,
    legal_action_mask,
    move_to_action,
    rotate_action,
    rotate_state,
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
    terminal_status,
)


def test_rotation_preserves_initial_setup_and_round_trips_every_action():
    start = initial_state()
    assert rotate_state(start).board == start.board
    assert rotate_state(start).side_to_move is Side.BLACK
    assert rotate_state(rotate_state(start)) == start
    assert rotate_action(move_to_action(Move.from_names("a1", "b1"))) == move_to_action(
        Move.from_names("h8", "g8")
    )
    assert {rotate_action(action) for action in range(ACTION_SPACE_SIZE)} == set(
        range(ACTION_SPACE_SIZE)
    )
    assert all(rotate_action(rotate_action(action)) == action for action in range(ACTION_SPACE_SIZE))


def test_rotated_state_and_encoded_planes_agree_without_mutation():
    state = apply_move(initial_state(), ("a2", "a3"))
    before = replace(state)
    rotated = rotate_state(state)

    assert state == before
    assert rotate_state(rotated) == state
    assert rotated.side_to_move is Side.WHITE
    assert rotated.last_move == Move.from_names("h7", "h6")
    assert rotated.piece_at("h6") == Piece(Side.BLACK, PieceType.PADATI)

    original_planes = encode_state(state)
    rotated_planes = encode_state(rotated)
    np.testing.assert_array_equal(rotated_planes[:6], original_planes[6:12, ::-1, ::-1])
    np.testing.assert_array_equal(rotated_planes[6:12], original_planes[:6, ::-1, ::-1])
    np.testing.assert_array_equal(rotated_planes[12], 1 - original_planes[12, ::-1, ::-1])


@pytest.mark.parametrize("moves", [
    (),
    (("a2", "a3"),),
    (("a2", "a3"), ("a7", "a6"), ("b1", "c3"), ("b8", "c6")),
])
def test_rotation_preserves_core_legality_and_move_application(moves):
    state = initial_state()
    for move in moves:
        state = apply_move(state, move)
    rotated = rotate_state(state)

    rotated_ids = {rotate_action(move_to_action(move)) for move in legal_moves(state)}
    assert rotated_ids == set(np.flatnonzero(legal_action_mask(rotated)))
    for move in legal_moves(state):
        transformed_move = Move(63 - move.from_square, 63 - move.to_square)
        assert transformed_move in legal_moves(rotated)
        assert rotate_state(apply_move(state, move)) == apply_move(rotated, transformed_move)


def test_rotation_preserves_promotion_and_terminal_winner():
    promotion = GameState.from_mapping({
        "a1": Piece(Side.WHITE, PieceType.RAJA),
        "b7": Piece(Side.WHITE, PieceType.PADATI),
        "h8": Piece(Side.BLACK, PieceType.RAJA),
        "h7": Piece(Side.BLACK, PieceType.PADATI),
    })
    move = Move.from_names("b7", "b8")
    rotated = rotate_state(promotion)
    rotated_move = Move.from_names("g2", "g1")
    assert move in legal_moves(promotion)
    assert rotated_move in legal_moves(rotated)
    assert rotate_action(move_to_action(move)) == move_to_action(rotated_move)
    assert rotate_state(apply_move(promotion, move)) == apply_move(rotated, rotated_move)

    terminal = GameState.from_mapping({
        "a1": Piece(Side.WHITE, PieceType.RAJA),
        "h8": Piece(Side.BLACK, PieceType.RAJA),
        "h7": Piece(Side.BLACK, PieceType.MANTRI),
    })
    assert terminal_status(terminal).winner is Side.BLACK
    assert terminal_status(rotate_state(terminal)).winner is Side.WHITE
    assert np.count_nonzero(legal_action_mask(rotate_state(terminal))) == 0


@pytest.mark.parametrize("kind", list(PieceType))
def test_rotation_preserves_each_piece_kinds_legal_moves(kind):
    pieces = {
        "h2": Piece(Side.WHITE, PieceType.PADATI),
        "d4": Piece(Side.WHITE, kind),
        "h7": Piece(Side.BLACK, PieceType.PADATI),
        "h8": Piece(Side.BLACK, PieceType.RAJA),
    }
    if kind is not PieceType.RAJA:
        pieces["a1"] = Piece(Side.WHITE, PieceType.RAJA)
    state = GameState.from_mapping(pieces)
    rotated = rotate_state(state)
    assert {rotate_action(move_to_action(move)) for move in legal_moves(state)} == {
        move_to_action(move) for move in legal_moves(rotated)
    }
    for move in legal_moves(state):
        rotated_move = Move(63 - move.from_square, 63 - move.to_square)
        assert rotate_state(apply_move(state, move)) == apply_move(rotated, rotated_move)
