"""
p3_mcts/node.py
===============
MCTS tree node — now with uncertainty-aware PUCT (Research Contribution #4).

ChaturangZero-X additions
--------------------------
  - `uncertainty` slot: stores the value-std returned by the network for this
    node's state. Populated during _expand_node() when the model has an
    uncertainty head.
  - `puct_score()` gains a `c_uncertainty` parameter. When > 0, the PUCT
    formula adds an uncertainty bonus:

      PUCT_U(s,a) = Q(s,a)
                  + c₁ · P(s,a) · √N(s) / (1 + N(s,a))   ← original
                  + c₂ · U(s,a) · √N(s) / (1 + N(s,a))   ← uncertainty bonus

    Effect: positions the network is UNCERTAIN about receive MORE exploration.
    This is especially useful for a historically under-studied game like
    Chaturanga where the network may systematically misestimate rare positions.
"""

from __future__ import annotations
from typing import Optional


class MCTSNode:
    """One node in the Monte Carlo search tree."""

    __slots__ = ("state", "parent", "action", "P", "N", "W", "uncertainty", "children", "_expanded")

    def __init__(
        self,
        state,
        parent:      Optional["MCTSNode"] = None,
        action:      Optional[int]        = None,
        prior:       float                = 0.0,
        uncertainty: float                = 0.0,
    ) -> None:
        self.state       = state
        self.parent      = parent
        self.action      = action
        self.P           = prior
        self.N           = 0
        self.W           = 0.0
        self.uncertainty = uncertainty    # value_std from the network
        self.children:   dict[int, "MCTSNode"] = {}
        self._expanded   = False

    @property
    def Q(self) -> float:
        """Mean action value."""
        return self.W / self.N if self.N > 0 else 0.0

    def is_leaf(self) -> bool:
        return not self._expanded

    def is_root(self) -> bool:
        return self.parent is None

    def expand(
        self,
        action_priors:  list[tuple[int, float]],
        states:         list,
        uncertainties:  list[float] | None = None,
    ) -> None:
        """
        Create child nodes.

        Parameters
        ----------
        action_priors  : list of (action_index, prior_prob)
        states         : corresponding next game states
        uncertainties  : per-child value_std estimates (same length as states)
        """
        if uncertainties is None:
            uncertainties = [0.0] * len(states)

        for (action, prior), child_state, unc in zip(action_priors, states, uncertainties):
            self.children[action] = MCTSNode(
                state=child_state,
                parent=self,
                action=action,
                prior=prior,
                uncertainty=unc,
            )
        self._expanded = True

    def update(self, value: float) -> None:
        """Backpropagate a value up the tree (negated for parent = opponent)."""
        self.N += 1
        self.W += value
        if self.parent:
            self.parent.update(-value)

    def puct_score(
        self,
        c_puct:       float,
        total_visits:  int,
        c_uncertainty: float = 0.0,
    ) -> float:
        """
        Uncertainty-aware PUCT score (Research Contribution #4).

          Standard:     PUCT = Q + c₁ · P · √N_parent / (1 + N)
          With uncertainty:
                        PUCT_U = Q + c₁ · P · √N_parent / (1 + N)
                                   + c₂ · U · √N_parent / (1 + N)

        where U = self.uncertainty (value_std from the neural network).
        """
        explore_base = (total_visits ** 0.5) / (1 + self.N)
        policy_bonus  = c_puct       * self.P           * explore_base
        uncert_bonus  = c_uncertainty * self.uncertainty * explore_base
        return self.Q + policy_bonus + uncert_bonus

    def __repr__(self) -> str:
        return (
            f"MCTSNode(N={self.N}, Q={self.Q:.3f}, P={self.P:.3f}, "
            f"U={self.uncertainty:.3f}, children={len(self.children)})"
        )
