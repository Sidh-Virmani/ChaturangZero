"""
p3_mcts/node.py
===============
MCTS tree node.  Each node represents one game state.

Stored per node:
  N  — visit count
  W  — total value (sum of backed-up values)
  Q  — mean value  (W / N)
  P  — prior probability from the neural network policy
  children — dict mapping action index → child MCTSNode
"""

from __future__ import annotations
from typing import Optional


class MCTSNode:
    """One node in the Monte Carlo search tree."""

    __slots__ = ("state", "parent", "action", "P", "N", "W", "children", "_expanded")

    def __init__(
        self,
        state,
        parent:  Optional["MCTSNode"] = None,
        action:  Optional[int]        = None,  # action index that led here
        prior:   float                = 0.0,
    ) -> None:
        self.state    = state
        self.parent   = parent
        self.action   = action   # action index that was taken from parent
        self.P        = prior    # prior probability (from neural network)
        self.N        = 0        # visit count
        self.W        = 0.0      # total backed-up value
        self.children: dict[int, "MCTSNode"] = {}
        self._expanded = False

    @property
    def Q(self) -> float:
        """Mean action value."""
        return self.W / self.N if self.N > 0 else 0.0

    def is_leaf(self) -> bool:
        return not self._expanded

    def is_root(self) -> bool:
        return self.parent is None

    def expand(self, action_priors: list[tuple[int, float]], states: list) -> None:
        """
        Create child nodes for each (action, prior) pair.

        Parameters
        ----------
        action_priors : list of (action_index, prior_probability)
        states        : list of corresponding next game states
        """
        for (action, prior), child_state in zip(action_priors, states):
            self.children[action] = MCTSNode(
                state=child_state,
                parent=self,
                action=action,
                prior=prior,
            )
        self._expanded = True

    def update(self, value: float) -> None:
        """Backpropagate a value up the tree."""
        self.N += 1
        self.W += value
        if self.parent:
            # Negate because parent is the opponent
            self.parent.update(-value)

    def puct_score(self, c_puct: float, total_visits: int) -> float:
        """
        PUCT score used by AlphaZero:
            PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N)
        """
        exploration = c_puct * self.P * (total_visits ** 0.5) / (1 + self.N)
        return self.Q + exploration

    def __repr__(self) -> str:
        return f"MCTSNode(N={self.N}, Q={self.Q:.3f}, P={self.P:.3f}, children={len(self.children)})"
