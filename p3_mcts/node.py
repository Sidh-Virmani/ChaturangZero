"""Tree node used by Monte Carlo Tree Search."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Sequence

import numpy as np

from p1_representation import ACTION_SPACE_SIZE, move_to_action
from shared.game_core import GameState, Move
from shared.game_core.core import _analyze_for_search, _apply_known_legal_move


@dataclass(slots=True)
class PendingChild:
    """An unvisited legal edge; no child board or terminal analysis yet."""

    move: Move
    prior: float

    @property
    def visit_count(self) -> int:
        return 0

    @property
    def value_sum(self) -> float:
        return 0.0

    @property
    def q_value(self) -> float:
        return 0.0

    def puct_score(self, c_puct: float, total_parent_visits: int) -> float:
        return float(c_puct) * self.prior * math.sqrt(max(0, total_parent_visits))


@dataclass(slots=True)
class MCTSNode:
    """A position and its incoming edge statistics in the search tree.

    ``value_sum`` and ``q_value`` use this node's side-to-move perspective.
    Consequently, selection negates a child's Q value before comparing moves
    from the parent position.

    children retains all legal edges in insertion order. Unvisited edges are
    PendingChild records; selection/step replaces only the chosen record with
    a full node. Both expose prior, visits and Q for analysis and policy code.
    legal_moves belongs only to this immutable state, never a global cache.
    """

    state: GameState
    parent: Optional["MCTSNode"] = field(default=None, repr=False)
    action_taken: Optional[int] = None
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode | PendingChild"] = field(default_factory=dict)
    legal_moves: tuple[Move, ...] = field(init=False, repr=False)
    is_terminal: bool = field(init=False)
    terminal_value: Optional[float] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, GameState):
            raise TypeError("state must be a shared.game_core.GameState")
        if not math.isfinite(float(self.prior)) or self.prior < 0:
            raise ValueError("prior must be a finite non-negative number")

        self.legal_moves, outcome = _analyze_for_search(self.state)
        self.is_terminal = outcome.terminal
        self.terminal_value = (
            float(outcome.reward(self.state.side_to_move)) if outcome.terminal else None
        )

    @property
    def q_value(self) -> float:
        """Mean backed-up value from this node's side-to-move perspective."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def is_expanded(self) -> bool:
        """Whether selection should stop at this node."""
        return bool(self.children) or self.is_terminal

    def puct_score(self, c_puct: float, total_parent_visits: int) -> float:
        """Return this edge's PUCT score from its parent's perspective."""
        exploration = (
            float(c_puct)
            * self.prior
            * math.sqrt(max(0, total_parent_visits))
            / (1 + self.visit_count)
        )
        return -self.q_value + exploration

    def select_best_child(self, c_puct: float) -> tuple[int, "MCTSNode"]:
        """Select the child with maximum PUCT, breaking ties by action ID."""
        if not self.children:
            raise ValueError("Cannot select a child from an unexpanded node")
        action, _ = max(
            self.children.items(),
            key=lambda item: (
                item[1].puct_score(c_puct, self.visit_count),
                -item[0],
            ),
        )
        return action, self.materialize_child(action)

    def materialize_child(self, action: int) -> "MCTSNode":
        """Create an already-known legal child once, preserving its edge prior."""
        child = self.children[action]
        if isinstance(child, PendingChild):
            child = MCTSNode(
                state=_apply_known_legal_move(self.state, child.move),
                parent=self,
                action_taken=action,
                prior=child.prior,
            )
            self.children[action] = child
        return child

    def expand(
        self,
        legal_actions: Sequence[tuple[int, Move]],
        priors: np.ndarray,
    ) -> None:
        """Store one lightweight edge for each supplied legal action.

        Priors are sanitized and normalized over the legal actions. This keeps
        search well-defined for simple test evaluators as well as real models.
        """
        if self.is_terminal:
            return
        if self.children:
            raise RuntimeError("A node cannot be expanded more than once")

        policy = np.asarray(priors)
        if policy.shape != (ACTION_SPACE_SIZE,):
            raise ValueError(f"priors must have shape ({ACTION_SPACE_SIZE},)")
        if not legal_actions:
            raise ValueError("A non-terminal node must have at least one legal action")

        action_ids = [action for action, _ in legal_actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("legal_actions contains duplicate action IDs")
        if any(
            not isinstance(action, (int, np.integer))
            or isinstance(action, (bool, np.bool_))
            or not 0 <= int(action) < ACTION_SPACE_SIZE
            for action in action_ids
        ):
            raise ValueError("legal action IDs must be integers in the action space")

        known_moves = set(self.legal_moves)
        if any(move not in known_moves or action != move_to_action(move)
               for action, move in legal_actions):
            raise ValueError("Expansion requires matching core-legal moves and action IDs")

        legal_priors = np.asarray(policy[action_ids], dtype=np.float64)
        legal_priors = np.where(
            np.isfinite(legal_priors) & (legal_priors > 0), legal_priors, 0.0
        )
        total = float(legal_priors.sum())
        if total <= 0:
            legal_priors.fill(1.0 / len(legal_actions))
        else:
            legal_priors /= total

        for (action, move), prior in zip(legal_actions, legal_priors, strict=True):
            action_id = int(action)
            self.children[action_id] = PendingChild(move=move, prior=float(prior))

    def update(self, value: float) -> None:
        """Accumulate one simulation result for this node."""
        if not math.isfinite(float(value)):
            raise ValueError("value must be finite")
        self.visit_count += 1
        self.value_sum += float(value)
