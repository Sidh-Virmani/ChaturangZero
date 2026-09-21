"""Loss calculation owned by the policy-value network layer."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F

from p1_representation import ACTION_SPACE_SIZE


class PolicyValueLoss(NamedTuple):
    total: Tensor
    policy: Tensor
    value: Tensor


def policy_value_loss(
    policy_logits: Tensor,
    values: Tensor,
    target_policy: Tensor,
    target_values: Tensor,
) -> PolicyValueLoss:
    """Return mean soft-target policy loss plus mean squared value loss."""
    batch_size = policy_logits.shape[0] if policy_logits.ndim == 2 else -1
    if policy_logits.shape != (batch_size, ACTION_SPACE_SIZE):
        raise ValueError("policy_logits must have shape (batch, ACTION_SPACE_SIZE)")
    if target_policy.shape != policy_logits.shape:
        raise ValueError("target_policy must have the same shape as policy_logits")
    if values.shape != (batch_size,) or target_values.shape != (batch_size,):
        raise ValueError("values and target_values must each have shape (batch,)")
    if batch_size == 0:
        raise ValueError("loss requires at least one training example")

    policy_loss = -(target_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, target_values)
    return PolicyValueLoss(policy_loss + value_loss, policy_loss, value_loss)
