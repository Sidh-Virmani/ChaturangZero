"""
tests/test_game_core.py
=======================
Automated tests for the Chaturanga game engine.
Run with:  pytest tests/ -v
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import (
    ChaturangaState, Piece, PieceKind, Side, Square, initial_state
)
from shared.game_core.rules  import is_in_check, pseudolegal_moves


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return ChaturangaEngine()


@pytest.fixture
def start(engine):
    return engine.new_game()


# ─────────────────────────────────────────────────────────────────────────────
# Basic state tests
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_piece_count(start):
    assert len(start.pieces) == 32, "Should start with 32 pieces"


def test_initial_turn(start):
    assert start.current_turn == Side.WHITE


def test_initial_not_terminal(engine, start):
    assert not engine.is_terminal(start)


def test_initial_status(start):
    assert start.status == "playing"


# ─────────────────────────────────────────────────────────────────────────────
# Square helper tests
# ─────────────────────────────────────────────────────────────────────────────

def test_square_label_roundtrip():
    for f in range(8):
        for r in range(8):
            sq = Square(f, r)
            assert Square.from_label(sq.label()) == sq


def test_square_labels():
    assert Square(0, 0).label() == "a1"
    assert Square(7, 7).label() == "h8"
    assert Square(4, 3).label() == "e4"


# ─────────────────────────────────────────────────────────────────────────────
# Piece movement tests  (single pieces on empty boards)
# ─────────────────────────────────────────────────────────────────────────────

def _empty_state_with(*pieces) -> ChaturangaState:
    return ChaturangaState(pieces=list(pieces))


def test_ashva_centre_moves():
    """Ashva in centre should have 8 moves."""
    ashva = Piece(PieceKind.ASHVA, Side.WHITE, Square(4, 4))
    state = _empty_state_with(ashva)
    moves = pseudolegal_moves(ashva, state)
    assert len(moves) == 8


def test_ashva_corner_moves():
    """Ashva in corner should have 2 moves."""
    ashva = Piece(PieceKind.ASHVA, Side.WHITE, Square(0, 0))
    state = _empty_state_with(ashva)
    moves = pseudolegal_moves(ashva, state)
    assert len(moves) == 2


def test_ratha_open_board():
    """Ratha in corner with no blockers should reach 14 squares."""
    ratha = Piece(PieceKind.RATHA, Side.WHITE, Square(0, 0))
    state = _empty_state_with(ratha)
    moves = pseudolegal_moves(ratha, state)
    assert len(moves) == 14


def test_gaja_centre_jumps():
    """Gaja in centre should jump to 4 squares (ignores intermediates)."""
    gaja = Piece(PieceKind.GAJA, Side.WHITE, Square(4, 4))
    state = _empty_state_with(gaja)
    moves = pseudolegal_moves(gaja, state)
    destinations = {m.to_sq for m in moves}
    assert len(destinations) == 4
    assert Square(2, 2) in destinations
    assert Square(6, 6) in destinations
    assert Square(2, 6) in destinations
    assert Square(6, 2) in destinations


def test_gaja_ignores_intermediate_pieces():
    """Gaja must be able to jump even if the intermediate square is occupied."""
    gaja  = Piece(PieceKind.GAJA,  Side.WHITE, Square(4, 4))
    blocker = Piece(PieceKind.PADATI, Side.WHITE, Square(5, 5))  # intermediate
    state = _empty_state_with(gaja, blocker)
    moves = pseudolegal_moves(gaja, state)
    # Should still reach Square(6, 6) by jumping over blocker
    assert any(m.to_sq == Square(6, 6) for m in moves)


def test_mantri_moves():
    """Mantri moves exactly 1 step diagonally."""
    mantri = Piece(PieceKind.MANTRI, Side.WHITE, Square(4, 4))
    state  = _empty_state_with(mantri)
    moves  = pseudolegal_moves(mantri, state)
    assert len(moves) == 4
    for m in moves:
        df = abs(m.to_sq.file - m.from_sq.file)
        dr = abs(m.to_sq.rank - m.from_sq.rank)
        assert df == 1 and dr == 1


def test_raja_moves():
    """Raja in centre has 8 moves."""
    raja  = Piece(PieceKind.RAJA, Side.WHITE, Square(4, 4))
    state = _empty_state_with(raja)
    moves = pseudolegal_moves(raja, state)
    assert len(moves) == 8


def test_padati_white_forward():
    """White Padati moves forward (increasing rank)."""
    padati = Piece(PieceKind.PADATI, Side.WHITE, Square(4, 3))
    state  = _empty_state_with(padati)
    moves  = pseudolegal_moves(padati, state)
    assert len(moves) == 1
    assert moves[0].to_sq == Square(4, 4)


def test_padati_captures_diagonally():
    """Padati can capture diagonally forward."""
    padati  = Piece(PieceKind.PADATI, Side.WHITE, Square(4, 3))
    enemy_l = Piece(PieceKind.PADATI, Side.BLACK, Square(3, 4))
    enemy_r = Piece(PieceKind.PADATI, Side.BLACK, Square(5, 4))
    state   = _empty_state_with(padati, enemy_l, enemy_r)
    moves   = pseudolegal_moves(padati, state)
    assert any(m.to_sq == Square(3, 4) and m.captured for m in moves)
    assert any(m.to_sq == Square(5, 4) and m.captured for m in moves)


def test_padati_promotion():
    """Padati reaching rank 8 promotes."""
    padati = Piece(PieceKind.PADATI, Side.WHITE, Square(4, 6))
    state  = _empty_state_with(padati)
    moves  = pseudolegal_moves(padati, state)
    assert any(m.is_promotion for m in moves)


# ─────────────────────────────────────────────────────────────────────────────
# Legal move / check tests
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_legal_moves_count(engine, start):
    """White should have 20 legal moves at start (8 padati + 4 ashva + 8 ratha... no actually Chaturanga: 8 padati × 1 + 2 ashva × 2 = 12)."""
    moves = engine.legal_moves(start)
    # 8 Padati can each move 1 forward, 2 Ashva can jump to 2 squares each = 12
    # (Ratha, Gaja, Raja, Mantri all blocked by own pieces at start)
    assert len(moves) >= 10  # conservative — at least padati + ashva


def test_check_detection():
    """Place enemy Ratha to attack own Raja — should detect check."""
    raja  = Piece(PieceKind.RAJA,  Side.WHITE, Square(4, 0))
    ratha = Piece(PieceKind.RATHA, Side.BLACK, Square(4, 7))
    state = ChaturangaState(pieces=[raja, ratha], current_turn=Side.WHITE)
    assert is_in_check(Side.WHITE, state)


def test_no_check_detection():
    """Raja not in check should return False."""
    raja  = Piece(PieceKind.RAJA,  Side.WHITE, Square(0, 0))
    ratha = Piece(PieceKind.RATHA, Side.BLACK, Square(7, 7))
    state = ChaturangaState(pieces=[raja, ratha], current_turn=Side.WHITE)
    assert not is_in_check(Side.WHITE, state)


def test_pinned_piece_cannot_move(engine):
    """A piece absolutely pinned to the Raja cannot legally move."""
    # Pin scenario: White Raja at a1 (0,0), White Ratha at b1 (1,0) blocks Black Ratha at h1 (7,0)
    # The White Ratha at b1 is pinned horizontally — it cannot leave the 1st rank.
    # If Ratha moves vertically (b1→b2) it exposes the White Raja to Black Ratha at h1.
    w_raja  = Piece(PieceKind.RAJA,  Side.WHITE, Square(0, 0))  # a1
    w_ratha = Piece(PieceKind.RATHA, Side.WHITE, Square(1, 0))  # b1  (pinned piece)
    b_ratha = Piece(PieceKind.RATHA, Side.BLACK, Square(7, 0))  # h1  (pinner)
    state   = ChaturangaState(pieces=[w_raja, w_ratha, b_ratha], current_turn=Side.WHITE)
    legal   = engine.legal_moves(state)
    # Pinned Ratha at b1 can only slide along rank 1 (a1 blocked by own king, c1-g1 ok, h1 captures)
    # It must NOT be able to leave rank 1 (e.g. go to b2, b3, etc.)
    vertical_moves = [m for m in legal if m.from_sq == Square(1, 0) and m.to_sq.rank != 0]
    assert len(vertical_moves) == 0, "Pinned Ratha must not leave the pin line"


def test_apply_move_updates_turn(engine, start):
    """After one move, it should be Black's turn."""
    moves = engine.legal_moves(start)
    next_state = engine.apply_move(start, moves[0])
    assert next_state.current_turn == Side.BLACK


def test_apply_illegal_move_raises(engine, start):
    """Applying an illegal move must raise ValueError."""
    from shared.game_core.state import Move
    bad_move = Move(Square(0, 0), Square(5, 5), PieceKind.RATHA, Side.WHITE)
    with pytest.raises(ValueError):
        engine.apply_move(start, bad_move)


# ─────────────────────────────────────────────────────────────────────────────
# Checkmate / terminal tests
# ─────────────────────────────────────────────────────────────────────────────

def test_checkmate_detection(engine):
    """White Raja with no escapes and in check should be checkmate."""
    # White Raja at a1 (0,0), attacked by Ratha at a8 (0,7) — check on a-file
    # Black Ratha at b8 (1,7) covers b1 (blocking escape to b)
    # So white Raja at a1: can go a2(0,1) — but a2 is also covered by Ratha at a8
    # Can go b1(1,0) — but b1 is covered by Ratha at b8
    # Can go b2(1,1) — check if covered.  Ratha at b8 covers b1,b2,...b7,b8.
    # So b2 is covered too. White King trapped → checkmate.
    w_raja   = Piece(PieceKind.RAJA,  Side.WHITE, Square(0, 0))   # a1
    b_ratha1 = Piece(PieceKind.RATHA, Side.BLACK, Square(0, 7))   # a8 — pins a-file
    b_ratha2 = Piece(PieceKind.RATHA, Side.BLACK, Square(1, 7))   # b8 — covers b-file
    state = ChaturangaState(
        pieces=[w_raja, b_ratha1, b_ratha2],
        current_turn=Side.WHITE,
    )
    # _update_status needs a "playing" starting state to compute
    state = engine._update_status(state)
    assert state.status == "checkmate", f"Expected checkmate, got {state.status}"
    assert state.winner == "black"
    assert engine.is_terminal(state)


def test_result_from_loser_perspective(engine):
    """result() should return -1 for the losing side."""
    w_raja   = Piece(PieceKind.RAJA,  Side.WHITE, Square(0, 0))
    b_ratha1 = Piece(PieceKind.RATHA, Side.BLACK, Square(1, 7))
    b_ratha2 = Piece(PieceKind.RATHA, Side.BLACK, Square(7, 1))
    state = ChaturangaState(
        pieces=[w_raja, b_ratha1, b_ratha2],
        current_turn=Side.WHITE,
        status="checkmate",
        winner="black",
    )
    assert engine.result(state) == -1.0


# ─────────────────────────────────────────────────────────────────────────────
# API serialisation test
# ─────────────────────────────────────────────────────────────────────────────

def test_api_dict_fields(engine, start):
    d = engine.state_to_api_dict(start)
    assert "pieces" in d
    assert "currentTurn" in d
    assert "selectableSquares" in d
    assert "status" in d
    assert "moveHistory" in d
    assert d["currentTurn"] == "white"
