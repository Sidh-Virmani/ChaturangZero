import pytest

from shared.game_core import (
    GameOverError,
    GameState,
    IllegalMoveError,
    Move,
    Piece,
    PieceType,
    Side,
    TerminalReason,
    apply_move,
    initial_state,
    is_in_check,
    is_valid_gaja_jump,
    is_valid_padati_move,
    legal_moves,
    parse_square,
    result,
    terminal_status,
)


def piece(side: Side, kind: PieceType) -> Piece:
    return Piece(side, kind)


def state(pieces, turn=Side.WHITE):
    return GameState.from_mapping(pieces, turn)


def uci_moves(position):
    return {move.uci for move in legal_moves(position)}


def kings(**extra):
    return {
        "a1": piece(Side.WHITE, PieceType.RAJA),
        "h2": piece(Side.WHITE, PieceType.PADATI),
        "h7": piece(Side.BLACK, PieceType.PADATI),
        "h8": piece(Side.BLACK, PieceType.RAJA),
        **extra,
    }


def test_initial_setup_is_asymmetric_and_has_sixteen_legal_moves():
    position = initial_state()
    assert position.piece_at("d1") == piece(Side.WHITE, PieceType.MANTRI)
    assert position.piece_at("e1") == piece(Side.WHITE, PieceType.RAJA)
    assert position.piece_at("d8") == piece(Side.BLACK, PieceType.RAJA)
    assert position.piece_at("e8") == piece(Side.BLACK, PieceType.MANTRI)
    assert len([p for p in position.board if p and p.side is Side.WHITE]) == 16
    assert len([p for p in position.board if p and p.side is Side.BLACK]) == 16
    assert len(legal_moves(position)) == 16


def test_padati_has_no_double_step_and_captures_only_forward_diagonally():
    position = state(kings(d2=piece(Side.WHITE, PieceType.PADATI), c3=piece(Side.BLACK, PieceType.ASHVA)))
    moves = uci_moves(position)
    assert "d2d3" in moves
    assert "d2c3" in moves
    assert "d2d4" not in moves
    assert "d2e3" not in moves


def test_black_padati_moves_toward_rank_one():
    position = state(kings(d7=piece(Side.BLACK, PieceType.PADATI)), Side.BLACK)
    moves = uci_moves(position)
    assert "d7d6" in moves
    assert "d7d5" not in moves


def test_gaja_jumps_over_occupied_intervening_square_and_captures_destination():
    position = state(kings(c1=piece(Side.WHITE, PieceType.GAJA), b2=piece(Side.WHITE, PieceType.PADATI), a3=piece(Side.BLACK, PieceType.ASHVA)))
    assert "c1a3" in uci_moves(position)
    moved = apply_move(position, ("c1", "a3"))
    assert moved.piece_at("a3") == piece(Side.WHITE, PieceType.GAJA)
    assert moved.piece_at("b2") == piece(Side.WHITE, PieceType.PADATI)
    assert is_valid_gaja_jump("c1", "a3")
    assert not is_valid_gaja_jump("c1", "b2")


def test_modular_padati_validator_explicitly_rejects_double_step():
    position = initial_state()
    assert is_valid_padati_move(position, ("a2", "a3"), Side.WHITE)
    assert not is_valid_padati_move(position, ("a2", "a4"), Side.WHITE)


def test_all_non_sliding_piece_movements_are_exact():
    mantri = state(kings(d4=piece(Side.WHITE, PieceType.MANTRI)))
    assert {m for m in uci_moves(mantri) if m.startswith("d4")} == {"d4c3", "d4c5", "d4e3", "d4e5"}
    ashva = state(kings(d4=piece(Side.WHITE, PieceType.ASHVA)))
    assert len({m for m in uci_moves(ashva) if m.startswith("d4")}) == 8
    gaja = state(kings(d4=piece(Side.WHITE, PieceType.GAJA)))
    assert {m for m in uci_moves(gaja) if m.startswith("d4")} == {"d4b2", "d4b6", "d4f2", "d4f6"}


def test_ratha_slides_but_cannot_cross_a_blocker():
    position = state(kings(d4=piece(Side.WHITE, PieceType.RATHA), d6=piece(Side.WHITE, PieceType.PADATI)))
    moves = uci_moves(position)
    assert "d4d5" in moves
    assert "d4d6" not in moves
    assert "d4d7" not in moves


def test_move_exposing_own_raja_is_filtered_out():
    position = state({
        "e1": piece(Side.WHITE, PieceType.RAJA),
        "e2": piece(Side.WHITE, PieceType.RATHA),
        "e8": piece(Side.BLACK, PieceType.RATHA),
        "a8": piece(Side.BLACK, PieceType.RAJA),
    })
    assert "e2d2" not in uci_moves(position)
    assert "e2e3" in uci_moves(position)
    with pytest.raises(IllegalMoveError):
        apply_move(position, ("e2", "d2"))


def test_rajas_may_not_move_adjacent_and_raja_is_never_captured():
    position = state({
        "e1": piece(Side.WHITE, PieceType.RAJA),
        "e3": piece(Side.BLACK, PieceType.RAJA),
        "a1": piece(Side.WHITE, PieceType.RATHA),
        "h8": piece(Side.BLACK, PieceType.RATHA),
    })
    assert "e1e2" not in uci_moves(position)


def test_promotion_is_immediate_and_only_to_mantri():
    position = state(kings(b7=piece(Side.WHITE, PieceType.PADATI)))
    promoted = apply_move(position, Move.from_names("b7", "b8"))
    assert promoted.piece_at("b8") == piece(Side.WHITE, PieceType.MANTRI)


def test_checkmate_awards_opponent_plus_one_and_loser_minus_one():
    position = state({
        "a1": piece(Side.WHITE, PieceType.RAJA),
        "h2": piece(Side.WHITE, PieceType.PADATI),
        "a2": piece(Side.BLACK, PieceType.RATHA),
        "b2": piece(Side.BLACK, PieceType.RATHA),
        "h3": piece(Side.BLACK, PieceType.MANTRI),
        "h8": piece(Side.BLACK, PieceType.RAJA),
    })
    outcome = terminal_status(position)
    assert is_in_check(position)
    assert outcome.reason is TerminalReason.CHECKMATE
    assert outcome.winner is Side.BLACK
    assert result(position, Side.BLACK) == 1
    assert result(position, Side.WHITE) == -1


def test_bare_king_is_an_immediate_win_even_if_bare_raja_has_moves():
    position = state({
        "a1": piece(Side.WHITE, PieceType.RAJA),
        "h8": piece(Side.BLACK, PieceType.RAJA),
        "h7": piece(Side.BLACK, PieceType.MANTRI),
    })
    outcome = terminal_status(position)
    assert outcome == terminal_status(position)  # deterministic evaluation
    assert outcome.reason is TerminalReason.BARE_KING
    assert outcome.winner is Side.BLACK
    assert legal_moves(position) == ()
    with pytest.raises(GameOverError):
        apply_move(position, ("a1", "a2"))


def test_capturing_last_non_raja_piece_triggers_bare_king():
    position = state({
        "a1": piece(Side.WHITE, PieceType.RAJA),
        "b2": piece(Side.WHITE, PieceType.RATHA),
        "h8": piece(Side.BLACK, PieceType.RAJA),
        "b7": piece(Side.BLACK, PieceType.MANTRI),
    })
    ended = apply_move(position, ("b2", "b7"))
    outcome = terminal_status(ended)
    assert outcome.reason is TerminalReason.BARE_KING
    assert outcome.winner is Side.WHITE


def test_stalemate_is_a_win_for_the_player_who_delivered_it():
    position = state({
        "a1": piece(Side.WHITE, PieceType.RAJA),
        "h2": piece(Side.WHITE, PieceType.PADATI),
        "b3": piece(Side.BLACK, PieceType.RATHA),
        "c2": piece(Side.BLACK, PieceType.RATHA),
        "h3": piece(Side.BLACK, PieceType.MANTRI),
        "h8": piece(Side.BLACK, PieceType.RAJA),
    })
    outcome = terminal_status(position)
    assert not is_in_check(position)
    assert legal_moves(position) == ()
    assert outcome.reason is TerminalReason.STALEMATE
    assert outcome.winner is Side.BLACK


def test_no_castling_move_is_generated():
    position = state({
        "e1": piece(Side.WHITE, PieceType.RAJA),
        "h1": piece(Side.WHITE, PieceType.RATHA),
        "a8": piece(Side.BLACK, PieceType.RAJA),
        "h8": piece(Side.BLACK, PieceType.RATHA),
    })
    assert "e1g1" not in uci_moves(position)
