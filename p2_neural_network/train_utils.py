"""
p2_neural_network/train_utils.py
=================================
Loss functions, optimizer setup, and training step helper.

AlphaZero loss:
    L = MSE(z, v) + cross_entropy(π, p) + λ·‖θ‖²
where:
    z = game result from self-play (target value)
    v = network value estimate
    π = MCTS improved policy (target)
    p = network policy (log-softmax)
    λ = L2 regularisation weight (default 1e-4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def compute_loss(
    policy_logits: torch.Tensor,   # (batch, 4096)
    value:         torch.Tensor,   # (batch, 1)
    target_policy: torch.Tensor,   # (batch, 4096)  — MCTS visit fractions
    target_value:  torch.Tensor,   # (batch, 1)     — game result ±1 / 0
) -> tuple[torch.Tensor, float, float]:
    """
    Compute combined AlphaZero loss.

    Returns
    -------
    total_loss   : differentiable tensor
    policy_loss  : float (for logging)
    value_loss   : float (for logging)
    """
    # Value loss: mean squared error
    value_loss = nn.functional.mse_loss(value, target_value)

    # Policy loss: cross-entropy over legal move distribution
    # target_policy may have zeros for illegal moves (masked externally)
    log_probs   = nn.functional.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()

    total_loss = value_loss + policy_loss
    return total_loss, policy_loss.item(), value_loss.item()


def build_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4):
    """Return an Adam optimizer with L2 regularisation."""
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, step_size: int = 50, gamma: float = 0.5):
    """Halve the LR every `step_size` training iterations."""
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def train_epoch(
    model:          nn.Module,
    optimizer:      optim.Optimizer,
    states:         torch.Tensor,   # (N, 13, 8, 8)
    policies:       torch.Tensor,   # (N, 4096)
    values:         torch.Tensor,   # (N, 1)
    batch_size:     int = 256,
    device:         str = "cpu",
) -> dict[str, float]:
    """
    Train for one epoch over the provided dataset.

    Returns a dict with average losses for logging.
    """
    model.train()
    dataset = TensorDataset(states, policies, values)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss_sum   = 0.0
    policy_loss_sum  = 0.0
    value_loss_sum   = 0.0
    n_batches        = 0

    for s_batch, p_batch, v_batch in loader:
        s_batch = s_batch.to(device)
        p_batch = p_batch.to(device)
        v_batch = v_batch.to(device)

        optimizer.zero_grad()
        policy_logits, value = model(s_batch)
        loss, ploss, vloss = compute_loss(policy_logits, value, p_batch, v_batch)
        loss.backward()
        optimizer.step()

        total_loss_sum  += loss.item()
        policy_loss_sum += ploss
        value_loss_sum  += vloss
        n_batches       += 1

    n = max(n_batches, 1)
    return {
        "total_loss":  total_loss_sum  / n,
        "policy_loss": policy_loss_sum / n,
        "value_loss":  value_loss_sum  / n,
    }
