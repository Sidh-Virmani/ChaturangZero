"""
p3_mcts/search.py
=================
Monte Carlo Tree Search using the PUCT algorithm (AlphaZero style).

Algorithm per move
------------------
1. Start at the root (current game state).
2. Selection: follow the highest PUCT child until a leaf is reached.
3. Expansion + Evaluation: call the neural network for the leaf state.
   - Policy head gives prior probabilities for each action.
   - Value head gives an estimated value.
4. Backpropagation: propagate the value back up the visited path.
5. Repeat for `num_simulations` iterations.
6. Return an improved policy: proportional to visit counts N^(1/temperature).

Exploration noise
-----------------
At the root node, Dirichlet noise is added to the priors to encourage
exploration, exactly as in AlphaZero.
"""

from __future__ import annotations

import sys
import pathlib
import math
import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Move
from p1_representation.encoder import StateEncoder, ACTION_SIZE
from p3_mcts.node import MCTSNode

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class MCTSSearch:
    """
    Neural-guided Monte Carlo Tree Search.

    Parameters
    ----------
    engine          : ChaturangaEngine instance
    model           : ChaturangaNet (None = random rollouts for testing)
    num_simulations : how many simulations per move (default 50)
    c_puct          : exploration constant (default 1.5)
    temperature     : move selection temperature:
                        > 1  = more random  (exploration phase)
                        → 0  = deterministic (evaluation phase)
    dirichlet_alpha : Dirichlet noise alpha for root exploration (0.3)
    dirichlet_eps   : weight of noise vs prior at root (0.25)
    device          : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        engine,
        model                       = None,
        num_simulations: int        = 50,
        c_puct:          float      = 1.5,
        temperature:     float      = 1.0,
        dirichlet_alpha: float      = 0.3,
        dirichlet_eps:   float      = 0.25,
        device:          str        = "cpu",
    ) -> None:
        self.engine          = engine
        self.model           = model
        self.num_simulations = num_simulations
        self.c_puct          = c_puct
        self.temperature     = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps   = dirichlet_eps
        self.device          = device
        self.encoder         = StateEncoder()

    # ── Public API ───────────────────────────────────────────────────────────

    def search(
        self, state: ChaturangaState
    ) -> tuple[np.ndarray, list[tuple[int, float, float]]]:
        """
        Run MCTS from `state` and return:

        Returns
        -------
        improved_policy : (4096,) float array — visit-count distribution over actions
        top_moves       : list of (action_idx, visit_fraction, Q_value) for top 5 moves
                          (used by the AI Insights panel in the UI)
        """
        root = MCTSNode(state)
        self._expand_node(root, add_noise=True)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not self.engine.is_terminal(node.state):
                self._expand_node(node, add_noise=False)
            value = self._evaluate(node)
            node.update(value)

        return self._build_policy(root), self._top_moves(root)

    def best_move(self, state: ChaturangaState) -> Move | None:
        """Run MCTS and return the single best Move object."""
        if self.engine.is_terminal(state):
            return None
        policy, _ = self.search(state)

        # Pick the action with the highest visit count
        legal = self.engine.legal_moves(state)
        if not legal:
            return None

        legal_indices = {StateEncoder.encode_action(m): m for m in legal}
        best_idx = max(legal_indices.keys(), key=lambda i: policy[i])
        return legal_indices[best_idx]

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk down the tree following max PUCT until a leaf."""
        while not node.is_leaf() and not self.engine.is_terminal(node.state):
            total_visits = node.N
            best_score   = -math.inf
            best_child   = None
            for child in node.children.values():
                score = child.puct_score(self.c_puct, total_visits)
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            node = best_child
        return node

    def _expand_node(self, node: MCTSNode, add_noise: bool) -> None:
        """Evaluate with the network and create child nodes."""
        state  = node.state
        legal  = self.engine.legal_moves(state)
        if not legal:
            return

        priors = self._get_priors(state, legal)

        if add_noise and self.dirichlet_eps > 0:
            noise  = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            priors = [
                (1 - self.dirichlet_eps) * p + self.dirichlet_eps * n
                for (_, p), n in zip(priors, noise)
            ]
            priors = list(zip([a for a, _ in self._get_priors(state, legal)], priors))

        next_states = [self.engine.apply_move(state, m) for m in legal]
        node.expand(priors, next_states)
        node.N = 1  # mark as visited

    def _get_priors(
        self, state: ChaturangaState, legal: list[Move]
    ) -> list[tuple[int, float]]:
        """
        Query the neural network (or use uniform priors if no model).
        Returns [(action_index, prior_prob), ...] for each legal move.
        """
        if self.model is None or not _TORCH_AVAILABLE:
            # Uniform priors (for testing without a trained model)
            p = 1.0 / len(legal)
            return [(StateEncoder.encode_action(m), p) for m in legal]

        enc   = self.encoder.encode_state_torch(state).to(self.device)
        mask  = self.encoder.legal_action_mask_torch(state, legal).to(self.device)
        probs, _ = self.model.predict(enc, mask)
        probs_np = probs.cpu().numpy()

        return [
            (StateEncoder.encode_action(m), float(probs_np[StateEncoder.encode_action(m)]))
            for m in legal
        ]

    def _evaluate(self, node: MCTSNode) -> float:
        """Return value estimate for `node.state` from network or terminal result."""
        state = node.state
        if self.engine.is_terminal(state):
            r = self.engine.result(state)
            return r if r is not None else 0.0

        if self.model is None or not _TORCH_AVAILABLE:
            return 0.0  # uniform evaluation without model

        enc = self.encoder.encode_state_torch(state).to(self.device)
        _, value = self.model.predict(enc)
        return float(value)

    def _build_policy(self, root: MCTSNode) -> np.ndarray:
        """Build the improved policy from visit counts."""
        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action_idx, child in root.children.items():
            counts[action_idx] = child.N

        if self.temperature == 0:
            # Deterministic — all mass on most visited
            best = np.argmax(counts)
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[best] = 1.0
            return policy

        # Softmax with temperature
        counts = counts ** (1.0 / self.temperature)
        total  = counts.sum()
        return counts / total if total > 0 else counts

    def _top_moves(
        self, root: MCTSNode, k: int = 5
    ) -> list[tuple[int, float, float]]:
        """Return top-k (action_idx, visit_fraction, Q_value) for AI Insights."""
        if not root.children:
            return []
        total = sum(c.N for c in root.children.values())
        ranked = sorted(
            root.children.items(), key=lambda kv: kv[1].N, reverse=True
        )
        return [
            (action, child.N / max(total, 1), child.Q)
            for action, child in ranked[:k]
        ]
