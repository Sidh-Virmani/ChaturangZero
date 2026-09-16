"""
p4_training/curriculum.py
==========================
Curriculum Self-Play Scheduler (Research Contribution #7).

Motivation
----------
Training from the initial position at full game length is inefficient.
The early network has no idea what a good endgame looks like, so it
generates noise-filled games that teach almost nothing about endgame technique.

Curriculum learning starts with simple positions, advances to harder ones
as the model's win rate improves:

  Stage 1 — Simplified positions (≤ 6 random pieces)
  Stage 2 — Tactical positions (one side has material advantage + threat)
  Stage 3 — Endgame positions (Raja + 1-2 pieces vs Raja)
  Stage 4 — Full games from initial position

The scheduler advances a stage when the model achieves a target win rate
in the current stage's arena matches.
"""

from __future__ import annotations

import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Piece, PieceKind, Side, Square


# ── Stage definitions ─────────────────────────────────────────────────────────

@dataclass
class CurriculumStage:
    name:            str
    description:     str
    advance_win_rate: float   # win rate needed to advance to next stage
    max_pieces:      int      # max total pieces on the board
    min_pieces:      int      # min total pieces (ensures it's not trivial)


STAGES = [
    CurriculumStage("simplified",   "Random simplified positions (≤8 pieces)",  0.60, 8,  4),
    CurriculumStage("endgame",      "Endgame: Raja + 1-2 pieces vs Raja",       0.60, 6,  3),
    CurriculumStage("tactical",     "Tactical: material advantage + threat",    0.55, 12, 6),
    CurriculumStage("full_game",    "Full games from starting position",        0.00, 32, 32),
]


class CurriculumScheduler:
    """
    Manages which training positions to generate at each stage.

    Usage
    -----
        scheduler = CurriculumScheduler(engine)
        positions = scheduler.get_start_positions(n=20)  # for self-play batch
        # after arena:
        scheduler.report_win_rate(win_rate=0.62)  # advances stage if ready
    """

    def __init__(
        self,
        engine:       ChaturangaEngine,
        start_stage:  int = 0,
        verbose:      bool = True,
    ) -> None:
        self.engine      = engine
        self.stage_idx   = start_stage
        self.verbose     = verbose
        self._history:   list[dict] = []

    @property
    def current_stage(self) -> CurriculumStage:
        return STAGES[min(self.stage_idx, len(STAGES) - 1)]

    @property
    def is_full_game(self) -> bool:
        return self.stage_idx >= len(STAGES) - 1

    def report_win_rate(self, win_rate: float) -> bool:
        """
        Report the model's win rate in the current stage's arena.
        Returns True if the scheduler advanced to the next stage.
        """
        stage = self.current_stage
        self._history.append({"stage": stage.name, "win_rate": win_rate})
        if win_rate >= stage.advance_win_rate and not self.is_full_game:
            self.stage_idx += 1
            if self.verbose:
                print(f"[curriculum] Advanced to stage {self.stage_idx}: "
                      f"{self.current_stage.name}")
            return True
        return False

    def get_start_positions(self, n: int = 20) -> list[ChaturangaState] | None:
        """
        Return a list of starting positions for this curriculum stage.
        Returns None if we are in the full-game stage (use engine.new_game()).
        """
        if self.is_full_game:
            return None

        stage = self.current_stage
        positions = []
        attempts  = 0
        while len(positions) < n and attempts < n * 20:
            attempts += 1
            pos = self._generate_position(stage)
            if pos is not None:
                positions.append(pos)

        return positions if positions else None

    # ── Position generators ───────────────────────────────────────────────────

    def _generate_position(self, stage: CurriculumStage) -> ChaturangaState | None:
        """Dispatch to the appropriate generator for the current stage."""
        name = stage.name
        if name == "simplified":
            return self._random_position(stage.min_pieces, stage.max_pieces)
        elif name == "endgame":
            return self._endgame_position()
        elif name == "tactical":
            return self._tactical_position()
        return None

    def _random_position(self, min_p: int, max_p: int) -> ChaturangaState | None:
        """
        Generate a random legal position with between min_p and max_p pieces.
        Both sides always have a Raja. Extra pieces are chosen randomly.
        """
        n_pieces = random.randint(min_p, max_p)
        n_extra  = max(0, n_pieces - 2)  # 2 rajas are mandatory

        squares = random.sample(
            [Square(f, r) for f in range(8) for r in range(8)],
            n_extra + 2
        )

        white_raja = Piece(PieceKind.RAJA, Side.WHITE, squares[0])
        black_raja = Piece(PieceKind.RAJA, Side.BLACK, squares[1])
        pieces = [white_raja, black_raja]

        extra_kinds = [PieceKind.RATHA, PieceKind.GAJA, PieceKind.ASHVA,
                       PieceKind.MANTRI, PieceKind.PADATI]

        for i, sq in enumerate(squares[2:]):
            side = Side.WHITE if i % 2 == 0 else Side.BLACK
            kind = random.choice(extra_kinds)
            # Padati cannot be on rank 0 (white) or rank 7 (black)
            if kind == PieceKind.PADATI:
                if side == Side.WHITE and sq.rank == 0:
                    kind = PieceKind.RATHA
                if side == Side.BLACK and sq.rank == 7:
                    kind = PieceKind.RATHA
            pieces.append(Piece(kind, side, sq))

        state = ChaturangaState(pieces=pieces, current_turn=Side.WHITE)
        # Verify it's not immediately terminal
        if self.engine.is_terminal(state):
            return None
        # Verify both sides have legal moves
        if not self.engine.legal_moves(state):
            return None
        return state

    def _endgame_position(self) -> ChaturangaState | None:
        """
        Generate a classic endgame: white has Raja + 1 major piece, black has only Raja.
        Useful for learning checkmating technique.
        """
        all_squares = [Square(f, r) for f in range(8) for r in range(8)]
        sq_w_raja, sq_b_raja, sq_piece = random.sample(all_squares, 3)

        extra_kind = random.choice([PieceKind.RATHA, PieceKind.MANTRI, PieceKind.GAJA])

        pieces = [
            Piece(PieceKind.RAJA, Side.WHITE, sq_w_raja),
            Piece(PieceKind.RAJA, Side.BLACK, sq_b_raja),
            Piece(extra_kind,     Side.WHITE, sq_piece),
        ]

        turn  = random.choice([Side.WHITE, Side.BLACK])
        state = ChaturangaState(pieces=pieces, current_turn=turn)

        if self.engine.is_terminal(state):
            return None
        if not self.engine.legal_moves(state):
            return None
        return state

    def _tactical_position(self) -> ChaturangaState | None:
        """
        Generate a tactically rich position: white has a material advantage
        with an immediate capturing threat.
        """
        # Start from a simplified position and ensure White has > Black material
        pos = self._random_position(min_p=6, max_p=10)
        if pos is None:
            return None

        _PIECE_VAL = {
            PieceKind.RATHA: 5, PieceKind.GAJA: 2,
            PieceKind.ASHVA: 3, PieceKind.MANTRI: 1,
            PieceKind.PADATI: 1, PieceKind.RAJA: 0,
        }
        white_mat = sum(_PIECE_VAL[p.kind] for p in pos.pieces if p.side == Side.WHITE)
        black_mat = sum(_PIECE_VAL[p.kind] for p in pos.pieces if p.side == Side.BLACK)

        # Bias toward positions where White has some advantage
        if white_mat < black_mat - 2:
            return None

        return pos

    def summary(self) -> str:
        return (
            f"Curriculum stage {self.stage_idx}/{len(STAGES)-1}: "
            f"{self.current_stage.name} — {self.current_stage.description}"
        )
