"""PUCT search with side-to-move values and alternating-sign backup."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
import numpy as np
from shared.game_core import GameState, Move, apply_move, legal_moves, is_in_check
from p1_representation import ACTION_SIZE, encode_action


def terminal_value(state, moves):
    if not moves:
        return -1.0 if is_in_check(state, state.side_to_move) else 0.0
    if state.halfmove_clock >= 100 or state.repetition_count >= 3:
        return 0.0
    return None


@dataclass
class Node:
    state: GameState
    prior: float = 1.0
    move: Move | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: list['Node'] = field(default_factory=list)
    expanded: bool = False
    terminal: float | None = None

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    root: Node
    simulations: int
    elapsed_ms: float
    depth: int

    def policy(self, temperature=1.0):
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        if not self.root.children:
            return policy
        counts = np.array([n.visits for n in self.root.children], dtype=np.float64)
        if temperature <= 0:
            counts = np.eye(len(counts))[int(np.argmax(counts))]
        else:
            counts = (counts / max(float(counts.max()), 1)) ** (1 / temperature)
        if counts.sum() == 0:
            counts = np.array([n.prior for n in self.root.children])
        counts /= counts.sum()
        for child, probability in zip(self.root.children, counts):
            policy[encode_action(child.move)] = probability
        return policy

    def choose(self, temperature=0.0, rng=None):
        if not self.root.children:
            raise ValueError('Cannot choose a move from a terminal position')
        probabilities = self.policy(temperature)
        weights = [probabilities[encode_action(n.move)] for n in self.root.children]
        index = (rng or np.random.default_rng()).choice(len(weights), p=np.array(weights) / sum(weights))
        return self.root.children[index].move


class MCTS:
    def __init__(self, evaluator, simulations=64, c_puct=1.6, seed=0):
        if simulations < 1:
            raise ValueError('Search requires at least one simulation')
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.rng = np.random.default_rng(seed)

    def _expand(self, node):
        moves = legal_moves(node.state)
        node.expanded = True
        node.terminal = terminal_value(node.state, moves)
        if node.terminal is not None:
            return node.terminal
        priors, value = self.evaluator.predict(node.state, moves)
        node.children = [Node(apply_move(node.state, m), float(p), m) for m, p in zip(moves, priors)]
        return value

    def search(self, state, add_noise=False, time_limit=None):
        start = time.perf_counter()
        root = Node(state)
        self._expand(root)
        if root.terminal is not None:
            return SearchResult(root, 0, (time.perf_counter() - start) * 1000, 0)
        # Exact one-ply terminal checks keep even a small search from overlooking
        # immediate mate because an early network assigned it a tiny prior.
        # This uses the rules, not a material or positional evaluation function.
        mating = []
        for child in root.children:
            if is_in_check(child.state, child.state.side_to_move) and not legal_moves(child.state):
                child.expanded, child.terminal = True, -1.0
                mating.append(child)
        if add_noise:
            noise = self.rng.dirichlet(np.full(len(root.children), 0.3))
            for child, sample in zip(root.children, noise):
                child.prior = 0.75 * child.prior + 0.25 * float(sample)
        depth, completed = 0, 0
        for _ in range(self.simulations):
            node, path = root, [root]
            if mating:
                node = mating[completed % len(mating)]
                path.append(node)
            while node.expanded and node.children:
                factor = self.c_puct * math.sqrt(max(node.visits, 1))
                node = max(node.children, key=lambda child: -child.value + factor * child.prior / (1 + child.visits))
                path.append(node)
            value = self._expand(node) if not node.expanded else node.terminal
            for entry in reversed(path):
                entry.visits += 1
                entry.value_sum += value
                value = -value
            depth = max(depth, len(path) - 1)
            completed += 1
            if time_limit and time.perf_counter() - start >= time_limit:
                break
        return SearchResult(root, completed, (time.perf_counter() - start) * 1000, depth)
