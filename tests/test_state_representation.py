from dataclasses import replace

import numpy as np

from p1_representation import PLANE_NAMES, STATE_SHAPE, encode_state
from shared.game_core import Side, apply_move, initial_state, parse_square


def cell(square: str) -> tuple[int, int]:
    index = parse_square(square)
    return divmod(index, 8)


def test_initial_position_has_expected_shape_type_and_piece_planes():
    encoded = encode_state(initial_state())

    assert encoded.shape == STATE_SHAPE == (13, 8, 8)
    assert encoded.dtype == np.float32
    assert encoded.flags.c_contiguous
    assert PLANE_NAMES == (
        "white_raja", "white_mantri", "white_gaja", "white_ashva",
        "white_ratha", "white_padati", "black_raja", "black_mantri",
        "black_gaja", "black_ashva", "black_ratha", "black_padati",
        "white_to_move",
    )

    expected_squares = (
        ("e1",), ("d1",), ("c1", "f1"), ("b1", "g1"),
        ("a1", "h1"), tuple(f"{file}2" for file in "abcdefgh"),
        ("d8",), ("e8",), ("c8", "f8"), ("b8", "g8"),
        ("a8", "h8"), tuple(f"{file}7" for file in "abcdefgh"),
    )
    for plane, squares in enumerate(expected_squares):
        expected = np.zeros((8, 8), dtype=np.float32)
        for square in squares:
            expected[cell(square)] = 1.0
        np.testing.assert_array_equal(encoded[plane], expected)
    np.testing.assert_array_equal(encoded[-1], np.ones((8, 8), dtype=np.float32))


def test_position_after_several_legal_moves_tracks_squares_and_turn():
    state = initial_state()
    for move in (("a2", "a3"), ("a7", "a6"), ("b1", "c3")):
        state = apply_move(state, move)

    encoded = encode_state(state)
    assert state.side_to_move is Side.BLACK
    assert encoded[PLANE_NAMES.index("white_padati")][cell("a3")] == 1.0
    assert encoded[PLANE_NAMES.index("black_padati")][cell("a6")] == 1.0
    assert encoded[PLANE_NAMES.index("white_ashva")][cell("c3")] == 1.0
    assert encoded[:, *cell("a2")].sum() == 0.0
    assert encoded[:, *cell("a7")].sum() == 0.0
    assert encoded[:, *cell("b1")].sum() == 0.0
    np.testing.assert_array_equal(encoded[-1], np.zeros((8, 8), dtype=np.float32))


def test_turn_plane_distinguishes_same_board_with_different_side_to_move():
    white = initial_state()
    black = replace(white, side_to_move=Side.BLACK)

    white_planes = encode_state(white)
    black_planes = encode_state(black)
    np.testing.assert_array_equal(white_planes[:-1], black_planes[:-1])
    assert np.all(white_planes[-1] == 1.0)
    assert np.all(black_planes[-1] == 0.0)


def test_encoding_is_deterministic_and_does_not_mutate_state():
    state = apply_move(initial_state(), ("a2", "a3"))
    equivalent = replace(state, ply_count=99, last_move=None)
    before = replace(state)

    first = encode_state(state)
    np.testing.assert_array_equal(first, encode_state(state))
    np.testing.assert_array_equal(first, encode_state(equivalent))
    assert state == before
    first[:] = 0.0
    assert state == before
    assert np.count_nonzero(encode_state(state)) > 0
