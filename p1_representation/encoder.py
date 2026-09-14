"""
p1_representation/encoder.py
============================
Converts Chaturanga board positions and moves into tensors/integers
that the neural network can consume.

Encoding scheme
---------------
State → 13-channel 8×8 float32 tensor:
  Channels 0-5  : White pieces (Raja, Mantri, Ratha, Gaja, Ashva, Padati)
  Channels 6-11 : Black pieces (same order)
  Channel 12    : Whose turn (all-1 = white to move, all-0 = black to move)

  Each channel is an 8×8 binary plane where a 1 indicates a piece of that
  type on that square.

Action → integer in [0, 4095]:
  action_index = from_square_index * 64 + to_square_index
  square_index = file * 8 + rank   (file: 0=a…7=h, rank: 0=rank1…7=rank8)
"""

from __future__ import annotations

import numpy as np

# ── Try importing torch; fall back to numpy-only mode if not installed ──────
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

import sys, pathlib
# Make shared importable when running from repo root
_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.state import ChaturangaState, Move, PieceKind, Side, Square


# ── Constants ────────────────────────────────────────────────────────────────
NUM_CHANNELS  = 13
BOARD_SIZE    = 8
ACTION_SIZE   = 64 * 64   # 4096

_PIECE_ORDER = [
    PieceKind.RAJA,
    PieceKind.MANTRI,
    PieceKind.RATHA,
    PieceKind.GAJA,
    PieceKind.ASHVA,
    PieceKind.PADATI,
]
_WHITE_OFFSET = 0
_BLACK_OFFSET = 6


def _sq_idx(sq: Square) -> int:
    return sq.file * 8 + sq.rank


# ─────────────────────────────────────────────────────────────────────────────


class StateEncoder:
    """Encodes Chaturanga states and actions for the neural network."""

    # ── State encoding ────────────────────────────────────────────────────

    @staticmethod
    def encode_state(state: ChaturangaState) -> np.ndarray:
        """
        Returns an (13, 8, 8) float32 numpy array representing `state`.

        Axis ordering: (channel, file, rank)
          - file 0 = 'a', file 7 = 'h'
          - rank 0 = rank 1, rank 7 = rank 8
        """
        tensor = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        for piece in state.pieces:
            offset = _WHITE_OFFSET if piece.side == Side.WHITE else _BLACK_OFFSET
            ch = offset + _PIECE_ORDER.index(piece.kind)
            tensor[ch, piece.square.file, piece.square.rank] = 1.0

        # Turn channel
        if state.current_turn == Side.WHITE:
            tensor[12, :, :] = 1.0

        return tensor

    @staticmethod
    def encode_state_torch(state: ChaturangaState):
        """Returns a (1, 13, 8, 8) float tensor ready for the model."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Run: pip install torch")
        arr = StateEncoder.encode_state(state)
        return torch.from_numpy(arr).unsqueeze(0)   # add batch dim

    # ── Action encoding ───────────────────────────────────────────────────

    @staticmethod
    def encode_action(move: Move) -> int:
        """Move → integer in [0, 4095]."""
        return _sq_idx(move.from_sq) * 64 + _sq_idx(move.to_sq)

    @staticmethod
    def decode_action(index: int) -> tuple[Square, Square]:
        """Integer → (from_square, to_square)."""
        from_idx = index // 64
        to_idx   = index % 64
        from_sq  = Square(from_idx // 8, from_idx % 8)
        to_sq    = Square(to_idx // 8,   to_idx % 8)
        return from_sq, to_sq

    # ── Legal action mask ─────────────────────────────────────────────────

    @staticmethod
    def legal_action_mask(state: ChaturangaState, legal_moves: list[Move]) -> np.ndarray:
        """
        Returns a boolean numpy array of shape (4096,).
        True at index i means action i is legal in this state.
        """
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        for move in legal_moves:
            mask[StateEncoder.encode_action(move)] = True
        return mask

    @staticmethod
    def legal_action_mask_torch(state: ChaturangaState, legal_moves: list[Move]):
        """Returns a (4096,) bool tensor."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed.")
        mask = StateEncoder.legal_action_mask(state, legal_moves)
        return torch.from_numpy(mask)

    # ── Canonical encoding (current-player perspective) ───────────────────

    @staticmethod
    def encode_canonical(state: ChaturangaState) -> np.ndarray:
        """
        Encode from the *current player's* perspective:
        - White's pieces are always channels 0-5 (own)
        - Black's pieces are always channels 6-11 (opponent)
        If it is Black's turn, we flip the board and swap sides so the
        network always sees 'its own pieces' in channels 0-5.
        This improves learning symmetry.
        """
        if state.current_turn == Side.WHITE:
            return StateEncoder.encode_state(state)

        # Black's turn: flip ranks and swap side assignments
        tensor = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for piece in state.pieces:
            # swap sides: black → offset 0, white → offset 6
            offset = _WHITE_OFFSET if piece.side == Side.BLACK else _BLACK_OFFSET
            ch = offset + _PIECE_ORDER.index(piece.kind)
            # flip rank axis (rank 7 becomes rank 0 etc.)
            flipped_rank = 7 - piece.square.rank
            tensor[ch, piece.square.file, flipped_rank] = 1.0
        # Turn channel = all-1 (own pieces to move)
        tensor[12, :, :] = 1.0
        return tensor

    # ── Data augmentation ─────────────────────────────────────────────────

    @staticmethod
    def mirror_horizontal(encoded: np.ndarray) -> np.ndarray:
        """
        Reflect the board left-right (file axis).
        Chaturanga has no castling so left-right symmetry is valid.
        """
        return np.flip(encoded, axis=1).copy()

    @staticmethod
    def augment_example(
        encoded_state: np.ndarray,
        action_index: int,
        value: float,
    ) -> list[tuple[np.ndarray, int, float]]:
        """
        Return original + horizontally mirrored training example.
        The mirrored action index is computed by swapping the file
        coordinate of both from and to squares.
        """
        examples = [(encoded_state, action_index, value)]

        from_sq, to_sq = StateEncoder.decode_action(action_index)
        mirrored_from = Square(7 - from_sq.file, from_sq.rank)
        mirrored_to   = Square(7 - to_sq.file,   to_sq.rank)
        mirrored_idx  = _sq_idx(mirrored_from) * 64 + _sq_idx(mirrored_to)

        mirrored_enc = StateEncoder.mirror_horizontal(encoded_state)
        examples.append((mirrored_enc, mirrored_idx, value))
        return examples
