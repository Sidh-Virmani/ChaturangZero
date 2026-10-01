"""AlphaZero-style Monte Carlo Tree Search for Classical Chaturanga."""

from __future__ import annotations

from numbers import Integral
from typing import Callable, Optional

import numpy as np

from p1_representation import ACTION_SPACE_SIZE, action_to_move, move_to_action
from p2_neural_network import NetworkEvaluator
from shared.game_core import GameState, Move, apply_move, legal_moves

from .config import MCTSConfig
from .node import MCTSNode


class MCTS:
    """Run policy/value-guided search and retain the resulting tree."""

    def __init__(
        self,
        evaluator: NetworkEvaluator,
        config: Optional[MCTSConfig] = None,
    ) -> None:
        if not hasattr(evaluator, "evaluate") or not callable(evaluator.evaluate):
            raise TypeError("evaluator must provide an evaluate(state) method")
        self.evaluator = evaluator
        self.config = MCTSConfig() if config is None else config
        if not isinstance(self.config, MCTSConfig):
            raise TypeError("config must be an MCTSConfig")
        self.root: Optional[MCTSNode] = None

    def search(
        self,
        state: GameState,
        config: Optional[MCTSConfig] = None,
        on_simulation: Optional[Callable[[int], None]] = None,
    ) -> tuple[np.ndarray, Move, int]:
        """Search ``state`` and return policy, selected move, and action ID.

        A terminal position has no legal move and therefore cannot satisfy the
        return contract; it raises before the neural evaluator is queried.
        """
        if not isinstance(state, GameState):
            raise TypeError("state must be a shared.game_core.GameState")
        search_config = self.config if config is None else config
        if not isinstance(search_config, MCTSConfig):
            raise TypeError("config must be an MCTSConfig")

        # ``step`` can place an already-searched child at the root. Preserve
        # that subtree when the caller searches the resulting position.
        if self.root is not None and self.root.state == state:
            root = self.root
        else:
            root = MCTSNode(state=state)
            self.root = root
        if root.is_terminal:
            raise ValueError("Cannot search a terminal position")

        if not root.is_expanded:
            self._evaluate_and_expand(root)
        if search_config.add_noise:
            self._add_dirichlet_noise(root, search_config)

        for simulation in range(1, search_config.num_simulations + 1):
            leaf = root
            while leaf.is_expanded and not leaf.is_terminal:
                _, leaf = leaf.select_best_child(search_config.c_puct)

            if leaf.is_terminal:
                if leaf.terminal_value is None:  # pragma: no cover - invariant guard
                    raise RuntimeError("Terminal node is missing its exact value")
                value = leaf.terminal_value
            else:
                value = self._evaluate_and_expand(leaf)
            self._backpropagate(leaf, value)
            if on_simulation is not None:
                on_simulation(simulation)

        policy = self._policy_from_visits(root, search_config.temperature)
        chosen_action = self._select_action(policy, search_config.temperature)
        return policy, action_to_move(chosen_action), chosen_action

    def step(self, action: int | Move) -> None:
        """Advance the retained tree after a played move, reusing its subtree."""
        if self.root is None:
            raise RuntimeError("Cannot step before a search has established a root")

        if isinstance(action, Move):
            action_id = move_to_action(action)
            move = action
        elif isinstance(action, Integral) and not isinstance(action, bool):
            action_id = int(action)
            move = action_to_move(action_id)
        else:
            raise TypeError("action must be an action ID or Move")

        child = self.root.children.get(action_id)
        if child is None:
            child = MCTSNode(
                state=apply_move(self.root.state, move),
                action_taken=action_id,
            )
        child.parent = None
        self.root = child

    def _evaluate_and_expand(self, node: MCTSNode) -> float:
        evaluation = self.evaluator.evaluate(node.state)
        policy = np.asarray(evaluation.policy)
        if policy.shape != (ACTION_SPACE_SIZE,):
            raise ValueError(
                f"Evaluator policy must have shape ({ACTION_SPACE_SIZE},)"
            )
        value = float(evaluation.value)
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("Evaluator value must be finite and in [-1, 1]")
        moves = legal_moves(node.state)
        actions = tuple((move_to_action(move), move) for move in moves)
        node.expand(actions, policy)
        return value

    @staticmethod
    def _backpropagate(leaf: MCTSNode, value: float) -> None:
        node: Optional[MCTSNode] = leaf
        backed_up_value = float(value)
        while node is not None:
            node.update(backed_up_value)
            backed_up_value = -backed_up_value
            node = node.parent

    @staticmethod
    def _add_dirichlet_noise(root: MCTSNode, config: MCTSConfig) -> None:
        children = tuple(root.children.values())
        if not children:
            return
        noise = np.random.dirichlet(
            np.full(len(children), config.dirichlet_alpha, dtype=np.float64)
        )
        keep = 1.0 - config.dirichlet_epsilon
        for child, sample in zip(children, noise, strict=True):
            child.prior = keep * child.prior + config.dirichlet_epsilon * float(sample)

    @staticmethod
    def _policy_from_visits(root: MCTSNode, temperature: float) -> np.ndarray:
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        actions = np.fromiter(root.children.keys(), dtype=np.int64)
        counts = np.fromiter(
            (child.visit_count for child in root.children.values()), dtype=np.float64
        )
        if actions.size == 0:
            raise RuntimeError("Search root has no legal children")

        if temperature < 1e-4:
            policy[int(actions[int(np.argmax(counts))])] = 1.0
            return policy

        exponent = 1.0 / float(temperature)
        positive = counts > 0
        if not np.any(positive):  # Defensive fallback for externally-built trees.
            weights = np.ones_like(counts)
        else:
            # Scaling before exponentiation avoids overflow for low temperatures.
            scaled = counts / counts.max()
            weights = np.power(scaled, exponent, where=positive, out=np.zeros_like(counts))
        weight_sum = float(weights.sum())
        policy[actions] = (weights / weight_sum).astype(np.float32)
        # Correct the tiny float32 normalization residue on the largest entry.
        largest = int(actions[int(np.argmax(weights))])
        policy[largest] += np.float32(1.0 - float(policy.sum(dtype=np.float64)))
        return policy

    @staticmethod
    def _select_action(policy: np.ndarray, temperature: float) -> int:
        if temperature < 1e-4:
            return int(np.argmax(policy))
        return int(np.random.choice(ACTION_SPACE_SIZE, p=policy.astype(np.float64)))
