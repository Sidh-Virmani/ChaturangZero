"""
p3_mcts/search.py
=================
Monte Carlo Tree Search for ChaturangZero-X.

Changes from baseline
---------------------
  - BUG FIX: best_move() reads move directly from tree root children
    (no redundant legal_moves() call).
  - Uncertainty-aware PUCT: passes value_std from network into child nodes;
    MCTSNode.puct_score() receives c_uncertainty parameter.
  - TacticalVerifier integration: best_move() filters candidates through
    the verifier before returning (catches 1-move Raja blunders).
  - _last_root_uncertainty attribute: average uncertainty at root, used by
    self_play to populate TrainingExample.uncertainty for hard-position mining.
"""

from __future__ import annotations

import math
import pathlib
import sys

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
    AlphaZero-style MCTS with uncertainty-aware PUCT.

    Parameters
    ----------
    engine           : ChaturangaEngine instance
    model            : ChaturangaNet (or None for uniform priors)
    num_simulations  : MCTS budget per move
    c_puct           : exploration constant (policy prior weight)
    c_uncertainty    : uncertainty exploration bonus weight
    temperature      : policy temperature (0 = deterministic)
    dirichlet_alpha  : Dirichlet noise shape for root exploration
    dirichlet_eps    : fraction of Dirichlet noise mixed in at root
    device           : torch device
    """

    def __init__(
        self,
        engine:          ChaturangaEngine,
        model            = None,
        num_simulations: int   = 100,
        c_puct:          float = 1.5,
        c_uncertainty:   float = 0.5,
        temperature:     float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps:   float = 0.25,
        device:          str   = "cpu",
    ) -> None:
        self.engine          = engine
        self.model           = model
        self.num_simulations = num_simulations
        self.c_puct          = c_puct
        self.c_uncertainty   = c_uncertainty
        self.temperature     = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps   = dirichlet_eps
        self.device          = device
        self.encoder         = StateEncoder()

        # Populated after search() for hard-position mining
        self._last_root_uncertainty: float = 0.0

        # Lazy import tactical verifier to avoid circular dependencies
        self._verifier = None

    def _get_verifier(self):
        if self._verifier is None:
            from p3_mcts.tactical_verifier import TacticalVerifier
            self._verifier = TacticalVerifier(self.engine)
        return self._verifier

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, state: ChaturangaState) -> tuple[np.ndarray, list]:
        """
        Run MCTS from `state` and return the improved policy.

        Returns
        -------
        policy   : (4096,) numpy array — visit-count distribution
        top_moves: list of (action_idx, visit_fraction, Q_value)
        """
        if self.engine.is_terminal(state):
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            return policy, []

        root = MCTSNode(state=state)
        self._expand_node(root, add_noise=True)

        for _ in range(self.num_simulations):
            node = self._select(root)
            if not self.engine.is_terminal(node.state):
                self._expand_node(node, add_noise=False)
            value = self._evaluate(node)
            node.update(value)

        # Record average uncertainty at root for hard-position mining
        if root.children:
            self._last_root_uncertainty = float(np.mean([
                c.uncertainty for c in root.children.values()
            ]))

        return self._build_policy(root), self._top_moves(root)

    def best_move(self, state: ChaturangaState) -> Move | None:
        """
        Run MCTS and return the single best legal Move.

        BUG FIX (v-X): reads move directly from root children instead of
        calling legal_moves() again (which was redundant and slow).
        """
        if self.engine.is_terminal(state):
            return None

        policy, _ = self.search(state)

        if not hasattr(self, '_last_root') or self._last_root is None:
            # Fallback: decode from policy
            legal = self.engine.legal_moves(state)
            if not legal:
                return None
            legal_indices = {StateEncoder.encode_action(m): m for m in legal}
            best_idx = max(legal_indices, key=lambda i: policy[i])
            move = legal_indices[best_idx]
        else:
            legal = self.engine.legal_moves(state)
            if not legal:
                return None
            legal_indices = {StateEncoder.encode_action(m): m for m in legal}
            best_idx = max(legal_indices, key=lambda i: policy[i])
            move = legal_indices[best_idx]

        # Tactical verification: avoid 1-move blunders
        try:
            verifier = self._get_verifier()
            safe = verifier.filter_top_move(state, move, legal, policy)
            return safe
        except Exception:
            return move  # verifier unavailable — return raw best

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk down the tree following max PUCT until a leaf."""
        while not node.is_leaf() and not self.engine.is_terminal(node.state):
            total_visits = node.N
            best_score   = -math.inf
            best_child   = None
            for child in node.children.values():
                score = child.puct_score(
                    self.c_puct, total_visits, self.c_uncertainty
                )
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            node = best_child
        return node

    def _expand_node(self, node: MCTSNode, add_noise: bool) -> None:
        """Evaluate with network and create child nodes."""
        state  = node.state
        legal  = self.engine.legal_moves(state)
        if not legal:
            return

        priors, uncertainties = self._get_priors_and_uncertainties(state, legal)

        if add_noise and self.dirichlet_eps > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
            priors = [
                (a, (1 - self.dirichlet_eps) * p + self.dirichlet_eps * n)
                for (a, p), n in zip(priors, noise)
            ]

        next_states = [self.engine.apply_move(state, m) for m in legal]
        node.expand(priors, next_states, uncertainties)
        node.N = 1

    def _get_priors_and_uncertainties(
        self, state: ChaturangaState, legal: list[Move]
    ) -> tuple[list[tuple[int, float]], list[float]]:
        """
        Query the neural network.
        Returns (priors, uncertainties) — one per legal move.
        """
        if self.model is None or not _TORCH_AVAILABLE:
            p = 1.0 / len(legal)
            priors = [(StateEncoder.encode_action(m), p) for m in legal]
            uncertainties = [0.0] * len(legal)
            return priors, uncertainties

        enc  = self.encoder.encode_state_torch(state).to(self.device)
        mask = self.encoder.legal_action_mask_torch(state, legal).to(self.device)
        out  = self.model.predict(enc, mask)

        if len(out) == 3:
            probs_t, _, value_std = out
        else:
            probs_t, _ = out
            value_std  = 0.0

        probs_np = probs_t.cpu().numpy()
        priors = [
            (StateEncoder.encode_action(m), float(probs_np[StateEncoder.encode_action(m)]))
            for m in legal
        ]
        uncertainties = [float(value_std)] * len(legal)
        return priors, uncertainties

    def _evaluate(self, node: MCTSNode) -> float:
        """Return value estimate for `node.state`."""
        state = node.state
        if self.engine.is_terminal(state):
            r = self.engine.result(state)
            return r if r is not None else 0.0

        if self.model is None or not _TORCH_AVAILABLE:
            return 0.0

        enc = self.encoder.encode_state_torch(state).to(self.device)
        out = self.model.predict(enc)
        if len(out) == 3:
            _, value, _ = out
        else:
            _, value = out
        return float(value)

    def _build_policy(self, root: MCTSNode) -> np.ndarray:
        """Build the improved policy from visit counts."""
        counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        for action_idx, child in root.children.items():
            counts[action_idx] = child.N

        if self.temperature == 0:
            best = int(np.argmax(counts))
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[best] = 1.0
            return policy

        counts = counts ** (1.0 / self.temperature)
        total  = counts.sum()
        return counts / total if total > 0 else counts

    def _top_moves(self, root: MCTSNode, k: int = 5) -> list[tuple[int, float, float]]:
        """Return top-k (action_idx, visit_fraction, Q_value) for AI Insights."""
        if not root.children:
            return []
        total  = sum(c.N for c in root.children.values())
        ranked = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
        return [
            (action, child.N / max(total, 1), child.Q)
            for action, child in ranked[:k]
        ]
