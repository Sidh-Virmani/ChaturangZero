"""
p4_training/opening_book.py
===========================
Records which opening sequences the AI learns to prefer during training.

After self-play games, the first N moves of every game are extracted and
tallied.  The most-played openings emerge naturally from the learned policy.

The book is saved as JSON and can be viewed in the UI to show "what the
AI has learned to open with" — a unique educational feature.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict


class OpeningBook:
    """
    Collects and ranks opening move sequences from self-play games.

    Parameters
    ----------
    depth     : number of plies to record (default 6 = 3 moves each side)
    save_path : where to save the book
    """

    def __init__(
        self,
        depth:     int = 6,
        save_path: str = "checkpoints/opening_book.json",
    ) -> None:
        self.depth     = depth
        self.save_path = pathlib.Path(save_path)
        self._counts:  dict[str, int] = defaultdict(int)

    def record_game(self, move_history: list) -> None:
        """
        Record the opening sequence from a completed game's move history.
        `move_history` is a list of Move objects.
        """
        opening = tuple(m.label() for m in move_history[: self.depth])
        if len(opening) == self.depth:
            self._counts[" ".join(opening)] += 1

    def top_openings(self, n: int = 10) -> list[tuple[str, int]]:
        """Return the top-n most frequent opening sequences."""
        sorted_openings = sorted(
            self._counts.items(), key=lambda kv: kv[1], reverse=True
        )
        return sorted_openings[:n]

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "depth":    self.depth,
            "openings": [
                {"sequence": seq, "count": cnt}
                for seq, cnt in self.top_openings(50)
            ],
        }
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[opening-book] Saved {len(self._counts)} openings → {self.save_path}")

    def load(self) -> None:
        if self.save_path.exists():
            with open(self.save_path) as f:
                data = json.load(f)
            self.depth = data.get("depth", self.depth)
            for item in data.get("openings", []):
                self._counts[item["sequence"]] = item["count"]
            print(f"[opening-book] Loaded {len(self._counts)} openings")

    def print_summary(self) -> None:
        print("\n=== Opening Book (top 10) ===")
        for i, (seq, cnt) in enumerate(self.top_openings(10), 1):
            print(f"  {i:2d}. [{cnt:4d}x]  {seq}")
        print()
