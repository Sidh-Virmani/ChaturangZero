"""
p5_endgame/retrograde.py
=========================
Chaturanga Endgame Tablebase via Retrograde Analysis (Research Contribution #10).

What is retrograde analysis?
-----------------------------
Start from all known terminal positions (checkmate, stalemate, bare-Raja)
and work BACKWARDS:

  1. All checkmate positions: DTM (Distance To Mate) = 0, result = LOSS for side to move
  2. A position is WIN-in-N if the side to move can reach a LOSS-in-(N-1) for opponent
  3. A position is LOSS-in-N if ALL moves lead to WIN-in-(N-1) for opponent
  4. Remaining unclassified positions after exhaustion = DRAW

This produces an exact game-theoretic solution for all positions with the
given piece configuration. No neural network — just pure combinatorial search.

Why Chaturanga?
---------------
No published Chaturanga endgame tablebases exist (as of 2026). This is
genuinely new territory. Even simple configurations like Raja+Ratha vs Raja
are unverified for Chaturanga rules (vs standard chess where it's a proven win).

Supported configurations
------------------------
  "KR-K"   : White Raja + Ratha vs Black Raja
  "KM-K"   : White Raja + Mantri vs Black Raja
  "KG-K"   : White Raja + Gaja vs Black Raja
  "KA-K"   : White Raja + Ashva vs Black Raja
  "KRR-K"  : White Raja + 2 Rathas vs Black Raja
  Custom   : any list of (PieceKind, Side) tuples

Usage
-----
    solver = RetrogradeAnalyzer(engine)
    results = solver.solve("KR-K", verbose=True)
    # results: {state_hash: {"dtm": int, "result": "win"/"loss"/"draw"}}
    solver.save("checkpoints/KR-K.egtb")
"""

from __future__ import annotations

import sys
import pathlib
import json
import hashlib
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Piece, PieceKind, Side, Square
from shared.game_core.rules  import is_in_check


# ── Piece configuration shortcuts ────────────────────────────────────────────

_CONFIG_MAP: dict[str, list[tuple[PieceKind, Side]]] = {
    "KR-K":  [(PieceKind.RAJA, Side.WHITE), (PieceKind.RATHA, Side.WHITE), (PieceKind.RAJA, Side.BLACK)],
    "KM-K":  [(PieceKind.RAJA, Side.WHITE), (PieceKind.MANTRI, Side.WHITE), (PieceKind.RAJA, Side.BLACK)],
    "KG-K":  [(PieceKind.RAJA, Side.WHITE), (PieceKind.GAJA, Side.WHITE), (PieceKind.RAJA, Side.BLACK)],
    "KA-K":  [(PieceKind.RAJA, Side.WHITE), (PieceKind.ASHVA, Side.WHITE), (PieceKind.RAJA, Side.BLACK)],
    "KRR-K": [(PieceKind.RAJA, Side.WHITE), (PieceKind.RATHA, Side.WHITE),
              (PieceKind.RATHA, Side.WHITE), (PieceKind.RAJA, Side.BLACK)],
}

# ── State hashing ─────────────────────────────────────────────────────────────

def _hash_state(state: ChaturangaState) -> str:
    """Compact hash of a board position: piece positions + side to move."""
    parts = sorted(
        f"{p.kind.value}{p.side.value}{p.square.file}{p.square.rank}"
        for p in state.pieces
    )
    parts.append(state.current_turn.value)
    return "|".join(parts)


# ── Main analyzer ─────────────────────────────────────────────────────────────

@dataclass
class TablebaseEntry:
    result: str   # "win", "loss", "draw"
    dtm:    int   # distance to mate in half-moves (0 = already mated)


class RetrogradeAnalyzer:
    """
    Builds an exact endgame tablebase for a given piece configuration.

    The tablebase maps state_hash → TablebaseEntry.
    All positions are evaluated from White's perspective:
      - "win"  means the side to move WINS with best play
      - "loss" means the side to move LOSES with best play
      - "draw" means best play leads to a draw
    """

    def __init__(self, engine: ChaturangaEngine | None = None) -> None:
        self.engine = engine or ChaturangaEngine()
        self._table: dict[str, TablebaseEntry] = {}
        self._total_positions = 0

    def solve(
        self,
        config: str | list[tuple[PieceKind, Side]],
        verbose: bool = True,
    ) -> dict[str, TablebaseEntry]:
        """
        Run retrograde analysis for the given piece configuration.

        Parameters
        ----------
        config : either a shorthand string like "KR-K" or a list of
                 (PieceKind, Side) tuples
        verbose: print progress

        Returns
        -------
        dict mapping state_hash → TablebaseEntry
        """
        if isinstance(config, str):
            pieces_spec = _CONFIG_MAP.get(config)
            if pieces_spec is None:
                raise ValueError(f"Unknown config '{config}'. Use: {list(_CONFIG_MAP)}")
        else:
            pieces_spec = config

        if verbose:
            print(f"[retrograde] Solving {config if isinstance(config, str) else 'custom'}...")

        # Phase 1: enumerate all legal positions
        all_states = self._enumerate_positions(pieces_spec, verbose)
        self._total_positions = len(all_states)
        if verbose:
            print(f"[retrograde] Enumerated {self._total_positions:,} positions")

        # Phase 2: retrograde BFS from terminal states
        self._table = self._backward_induction(all_states, verbose)

        # Phase 3: mark remaining as draw
        for h, state in all_states.items():
            if h not in self._table:
                self._table[h] = TablebaseEntry(result="draw", dtm=9999)

        if verbose:
            wins   = sum(1 for e in self._table.values() if e.result == "win")
            losses = sum(1 for e in self._table.values() if e.result == "loss")
            draws  = sum(1 for e in self._table.values() if e.result == "draw")
            print(f"[retrograde] Done. W={wins:,} L={losses:,} D={draws:,}")

        return self._table

    def lookup(self, state: ChaturangaState) -> Optional[TablebaseEntry]:
        """Look up a position in the solved tablebase."""
        return self._table.get(_hash_state(state))

    def save(self, path: str) -> None:
        """Save the tablebase to a JSON file."""
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            h: {"result": e.result, "dtm": e.dtm}
            for h, e in self._table.items()
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[retrograde] Saved {len(data):,} entries → {path}")

    def load(self, path: str) -> None:
        """Load a previously solved tablebase."""
        with open(path) as f:
            data = json.load(f)
        self._table = {
            h: TablebaseEntry(result=v["result"], dtm=v["dtm"])
            for h, v in data.items()
        }
        print(f"[retrograde] Loaded {len(self._table):,} entries ← {path}")

    # ── Private: enumeration ──────────────────────────────────────────────────

    def _enumerate_positions(
        self,
        pieces_spec: list[tuple[PieceKind, Side]],
        verbose: bool,
    ) -> dict[str, ChaturangaState]:
        """
        Enumerate all board positions for the given piece configuration.
        Filters out illegal positions (both Rajas adjacent, etc.).
        """
        n_pieces = len(pieces_spec)
        all_squares = [(f, r) for f in range(8) for r in range(8)]

        # Enumerate all ways to place n_pieces on the board (no two on same square)
        positions: dict[str, ChaturangaState] = {}

        for square_combo in itertools.permutations(all_squares, n_pieces):
            for turn in [Side.WHITE, Side.BLACK]:
                pieces = [
                    Piece(kind, side, Square(f, r))
                    for (kind, side), (f, r) in zip(pieces_spec, square_combo)
                ]
                state = ChaturangaState(pieces=pieces, current_turn=turn)

                # Filter: must not already be in check on the wrong side
                # (the side that JUST moved must not be in check)
                other = turn.opponent()
                if is_in_check(other, state):
                    continue  # Opponent in check = last player left them in check → illegal

                h = _hash_state(state)
                if h not in positions:
                    positions[h] = state

        return positions

    # ── Private: backward induction ───────────────────────────────────────────

    def _backward_induction(
        self,
        all_states: dict[str, ChaturangaState],
        verbose: bool,
    ) -> dict[str, TablebaseEntry]:
        """
        BFS from terminal states (DTM=0) backwards to classify all reachable positions.
        """
        table: dict[str, TablebaseEntry]  = {}
        # remaining_moves[h] = number of unclassified successors
        remaining: dict[str, int]         = {}
        # predecessors[h] = list of (pred_hash, pred_state)
        predecessors: dict[str, list[str]] = {}

        # Build predecessor graph + identify terminals
        queue: deque[str] = deque()

        for h, state in all_states.items():
            if self.engine.is_terminal(state):
                r = state.result()
                if r is None:
                    entry = TablebaseEntry("draw", 0)
                elif r > 0:
                    entry = TablebaseEntry("win", 0)
                else:
                    entry = TablebaseEntry("loss", 0)
                table[h] = entry
                queue.append(h)
                continue

            legal = self.engine.legal_moves(state)
            remaining[h] = len(legal)

            for move in legal:
                try:
                    next_state = self.engine.apply_move(state, move)
                    nh = _hash_state(next_state)
                    if nh not in predecessors:
                        predecessors[nh] = []
                    predecessors[nh].append(h)
                except Exception:
                    remaining[h] -= 1

        # BFS: propagate results backwards
        iteration = 0
        while queue:
            h = queue.popleft()
            entry = table[h]
            iteration += 1

            # Notify predecessors
            for pred_h in predecessors.get(h, []):
                if pred_h in table:
                    continue  # already resolved

                pred_state = all_states.get(pred_h)
                if pred_state is None:
                    continue

                # If this successor is a LOSS for the side to move there,
                # then the predecessor (opposite side to move) has a WIN
                if entry.result == "loss":
                    # Predecessor can move to this loss → predecessor is a WIN
                    table[pred_h] = TablebaseEntry("win", entry.dtm + 1)
                    queue.append(pred_h)
                else:
                    # Successor is win or draw — decrement remaining moves counter
                    remaining[pred_h] = remaining.get(pred_h, 1) - 1
                    if remaining[pred_h] <= 0:
                        # All successors are wins for opponent → predecessor is LOSS
                        table[pred_h] = TablebaseEntry("loss", entry.dtm + 1)
                        queue.append(pred_h)

        return table

    @property
    def stats(self) -> dict:
        if not self._table:
            return {"solved": False}
        wins   = sum(1 for e in self._table.values() if e.result == "win")
        losses = sum(1 for e in self._table.values() if e.result == "loss")
        draws  = sum(1 for e in self._table.values() if e.result == "draw")
        return {
            "solved": True,
            "total":  len(self._table),
            "wins":   wins,
            "losses": losses,
            "draws":  draws,
            "max_dtm": max((e.dtm for e in self._table.values() if e.dtm < 9999), default=0),
        }
