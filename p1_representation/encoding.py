from __future__ import annotations

import numpy as np
from shared.game_core import GameState, Move, PieceKind, Side, legal_moves

KINDS = (PieceKind.RAJA, PieceKind.MANTRI, PieceKind.RATHA, PieceKind.ASHVA, PieceKind.GAJA, PieceKind.PADATI)
CHANNELS = 14
ACTION_SIZE = 4096


def square_index(square: tuple[int, int], side: Side) -> int:
    x, y = square
    return (y if side is Side.WHITE else 7 - y) * 8 + x


def encode_state(state: GameState) -> np.ndarray:
    """12 piece planes + normalized no-progress clock + repetition count.

    Own army is always in channels 0..5, advancing toward rank index 7.
    Black positions flip ranks, not files. No future information is encoded.
    """
    planes = np.zeros((CHANNELS, 8, 8), dtype=np.float32)
    for piece in state.pieces:
        channel = KINDS.index(piece.kind) + (0 if piece.side is state.side_to_move else 6)
        index = square_index(piece.square, state.side_to_move)
        planes[channel, index // 8, index % 8] = 1
    planes[12].fill(min(state.halfmove_clock / 100, 1))
    planes[13].fill(min((state.repetition_count - 1) / 2, 1))
    return planes


def encode_action(move: Move, perspective: Side | None = None) -> int:
    side = perspective or move.side
    return square_index(move.from_sq, side) * 64 + square_index(move.to_sq, side)


def decode_action(index: int, state: GameState) -> Move:
    if not isinstance(index, (int, np.integer)) or not 0 <= index < ACTION_SIZE:
        raise ValueError('Action must be an integer in [0, 4096)')
    for move in legal_moves(state):
        if encode_action(move, state.side_to_move) == index:
            return move
    raise ValueError('Action is illegal in this position')


def legal_action_mask(state: GameState, moves: list[Move] | None = None) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
    for move in legal_moves(state) if moves is None else moves:
        mask[encode_action(move, state.side_to_move)] = True
    return mask


def reflect_example(planes: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """File reflection is a rules-preserving augmentation; reflect both endpoints."""
    return planes[:, :, ::-1].copy(), policy.reshape(8, 8, 8, 8)[:, ::-1, :, ::-1].reshape(-1).copy()
