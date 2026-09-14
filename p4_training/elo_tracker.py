"""
p4_training/elo_tracker.py
==========================
Tracks and visualises the AI's Elo rating progression across training
iterations.  After each candidate-vs-best evaluation, the rating is
updated using the standard Elo formula.

The progression is saved to a JSON file and can be plotted with
matplotlib (optional dependency).
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EloRecord:
    iteration: int
    elo:       float
    games:     int
    wins:      int
    losses:    int
    draws:     int
    win_rate:  float


class EloTracker:
    """
    Simple Elo rating tracker for the ChaturangZero training loop.

    Parameters
    ----------
    initial_elo : starting Elo for the very first model (default 1000)
    k_factor    : Elo K-factor (default 32 — standard for games)
    save_path   : where to persist the rating history as JSON
    """

    def __init__(
        self,
        initial_elo: float = 1000.0,
        k_factor:    float = 32.0,
        save_path:   str   = "checkpoints/elo_history.json",
    ) -> None:
        self.current_elo = initial_elo
        self.k_factor    = k_factor
        self.save_path   = pathlib.Path(save_path)
        self.history:    list[EloRecord] = []

    # ── Elo update ────────────────────────────────────────────────────────

    def update(
        self,
        iteration: int,
        wins:      int,
        losses:    int,
        draws:     int,
        opponent_elo: float = 1000.0,
    ) -> float:
        """
        Update the current model's Elo based on match results.

        Parameters
        ----------
        iteration    : training iteration number (for logging)
        wins/losses/draws : arena match results (candidate perspective)
        opponent_elo : Elo of the opponent model (usually the best model's last rating)

        Returns
        -------
        new Elo rating
        """
        games = wins + losses + draws
        if games == 0:
            return self.current_elo

        actual_score = (wins + 0.5 * draws) / games
        expected     = self._expected_score(self.current_elo, opponent_elo)
        new_elo      = self.current_elo + self.k_factor * (actual_score - expected)
        win_rate     = actual_score

        record = EloRecord(
            iteration=iteration,
            elo=round(new_elo, 1),
            games=games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=round(win_rate, 3),
        )
        self.history.append(record)
        self.current_elo = new_elo
        self._save()

        print(
            f"  [elo] iter={iteration}  Elo: {self.current_elo:.0f}  "
            f"W/L/D={wins}/{losses}/{draws}  win_rate={win_rate:.1%}"
        )
        return new_elo

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(
                [vars(r) for r in self.history], f, indent=2
            )

    def load(self) -> None:
        if self.save_path.exists():
            with open(self.save_path) as f:
                data = json.load(f)
            self.history = [EloRecord(**d) for d in data]
            if self.history:
                self.current_elo = self.history[-1].elo

    # ── Plotting (optional) ───────────────────────────────────────────────

    def plot(self, save_fig: Optional[str] = None) -> None:
        """Plot Elo progression using matplotlib (if installed)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[elo] matplotlib not installed — skipping plot")
            return

        iters = [r.iteration for r in self.history]
        elos  = [r.elo       for r in self.history]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iters, elos, marker="o", color="#4C72B0", linewidth=2)
        ax.fill_between(iters, elos, min(elos) - 10, alpha=0.15, color="#4C72B0")
        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Elo Rating")
        ax.set_title("ChaturangZero — AI Strength Progression")
        ax.grid(True, linestyle="--", alpha=0.5)

        if save_fig:
            fig.savefig(save_fig, dpi=150, bbox_inches="tight")
            print(f"[elo] Plot saved → {save_fig}")
        else:
            plt.show()

    # ── Helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
