"""Rule-safe inference boundary used by MCTS and other game clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from p1_representation import ACTION_SPACE_SIZE, encode_state, legal_action_mask, move_to_action
from shared.game_core import GameState, Move

from .model import PolicyValueNetwork


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Legal policy probabilities plus a side-to-move value estimate."""

    policy: np.ndarray
    value: float


class NetworkEvaluator:
    """Evaluate core states without exposing model layers to search code."""

    def __init__(self, model: PolicyValueNetwork, device: torch.device | str = "cpu") -> None:
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def evaluate(self, state: GameState) -> Evaluation:
        """Evaluate one state and return a legal normalized policy and value."""
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        """Evaluate states together; results remain in the supplied order."""
        return self._evaluate_batch(states)

    def evaluate_with_legal_moves(
        self, state: GameState, moves: Sequence[Move]
    ) -> Evaluation:
        """Reuse core-generated legal moves for this exact state.

        Trusted optimization: the caller must supply the complete legal move
        list from the same immutable state. This method translates moves with
        P1; it deliberately does not recheck legality. Ordinary callers should
        use evaluate(state). An empty list produces an all-zero policy.
        """
        mask = np.zeros((1, ACTION_SPACE_SIZE), dtype=bool)
        for move in moves:
            mask[0, move_to_action(move)] = True
        return self._evaluate_batch((state,), mask)[0]

    def _evaluate_batch(
        self, states: Sequence[GameState], masks_np: np.ndarray | None = None
    ) -> tuple[Evaluation, ...]:
        if not states:
            return ()
        if any(not isinstance(state, GameState) for state in states):
            raise TypeError("states must contain only shared.game_core.GameState values")

        encoded = np.stack([encode_state(state) for state in states])
        inputs = torch.from_numpy(encoded).to(self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                logits, values = self.model(inputs)
                policies = self._legal_probabilities(logits, states, masks_np)
        finally:
            self.model.train(was_training)

        policies_np = policies.cpu().numpy()
        values_np = values.cpu().numpy()
        return tuple(
            Evaluation(policy.copy(), float(value))
            for policy, value in zip(policies_np, values_np, strict=True)
        )

    def _legal_probabilities(
        self, logits: torch.Tensor, states: Sequence[GameState],
        masks_np: np.ndarray | None = None,
    ) -> torch.Tensor:
        if masks_np is None:
            masks_np = np.stack([legal_action_mask(state) for state in states]).astype(bool)
        masks = torch.from_numpy(masks_np).to(self.device)
        policies = torch.zeros_like(logits)
        nonterminal = masks.any(dim=1)
        if nonterminal.any():
            masked_logits = logits[nonterminal].masked_fill(~masks[nonterminal], -torch.inf)
            policies[nonterminal] = torch.softmax(masked_logits, dim=1)
        return policies
