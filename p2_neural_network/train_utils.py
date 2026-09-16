"""
p2_neural_network/train_utils.py
=================================
Training utilities for ChaturangaNet.

ChaturangZero-X additions
--------------------------
  - gaussian_nll_loss(): replaces MSE for uncertainty-head models.
    Loss = (v_mean - target)² / (2σ²) + log(σ) + constant.
    The network learns both the value estimate AND its own uncertainty.
  - compute_loss() now accepts an optional `value_logvar` argument.
  - uncertainty_weight: scales the NLL term relative to the policy loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def gaussian_nll_loss(
    value_mean:   torch.Tensor,
    value_logvar: torch.Tensor,
    target:       torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood of a Gaussian distribution.

    L = (mean - target)² / (2 * σ²) + 0.5 * log(σ²)

    Parameters
    ----------
    value_mean   : (batch, 1) predicted mean
    value_logvar : (batch, 1) predicted log-variance (unconstrained)
    target       : (batch, 1) ground-truth value ∈ {-1, 0, +1}

    Returns
    -------
    Scalar loss (mean over batch).
    """
    # Clamp log_var for numerical stability
    value_logvar = value_logvar.clamp(-10, 5)
    inv_var  = torch.exp(-value_logvar)           # 1 / σ²
    mse_term = 0.5 * inv_var * (value_mean - target) ** 2
    reg_term = 0.5 * value_logvar                 # penalise overconfidence
    return (mse_term + reg_term).mean()


def compute_loss(
    model,
    states:       torch.Tensor,          # (B, 13, 8, 8)
    policies:     torch.Tensor,          # (B, 4096)
    values:       torch.Tensor,          # (B,)
    uncertainty_weight: float = 1.0,     # scaling for NLL term vs policy loss
) -> dict[str, torch.Tensor]:
    """
    Compute the combined AlphaZero training loss.

    Loss = policy_cross_entropy + uncertainty_weight * value_loss

    Where:
      - value_loss = MSE              (if uncertainty_head=False)
      - value_loss = Gaussian NLL     (if uncertainty_head=True)

    Returns dict with keys: total_loss, policy_loss, value_loss
    """
    model.train()
    out = model(states)

    has_uncertainty = model.uncertainty_head and len(out) == 3

    if has_uncertainty:
        policy_logits, value_mean, value_logvar = out
    else:
        policy_logits, value_mean = out
        value_logvar = None

    # Policy loss — cross-entropy between MCTS distribution and network output
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policies * log_probs).sum(dim=1).mean()

    # Value loss
    targets = values.unsqueeze(1).float()
    if has_uncertainty and value_logvar is not None:
        value_loss = gaussian_nll_loss(value_mean, value_logvar, targets)
    else:
        value_loss = F.mse_loss(value_mean, targets)

    total_loss = policy_loss + uncertainty_weight * value_loss

    return {
        "total_loss":  total_loss,
        "policy_loss": policy_loss.detach(),
        "value_loss":  value_loss.detach(),
    }


def build_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> optim.Optimizer:
    """Adam optimiser with weight decay."""
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: optim.Optimizer, T_max: int = 50) -> CosineAnnealingLR:
    """Cosine annealing LR schedule."""
    return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-5)


def train_epoch(
    model:              nn.Module,
    optimizer:          optim.Optimizer,
    states:             torch.Tensor,
    policies:           torch.Tensor,
    values:             torch.Tensor,
    batch_size:         int   = 256,
    device:             str   = "cpu",
    uncertainty_weight: float = 1.0,
) -> dict[str, float]:
    """
    Run one training epoch over the provided data.

    Data is shuffled and processed in mini-batches.

    Returns dict of average losses over the epoch.
    """
    model.train()
    model.to(device)

    n       = len(states)
    perm    = torch.randperm(n)
    states  = states[perm]
    policies= policies[perm]
    values  = values[perm]

    total_loss_acc  = 0.0
    policy_loss_acc = 0.0
    value_loss_acc  = 0.0
    num_batches     = 0

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        s_b  = states[start:end].to(device)
        p_b  = policies[start:end].to(device)
        v_b  = values[start:end].to(device)

        optimizer.zero_grad()
        losses = compute_loss(model, s_b, p_b, v_b, uncertainty_weight)
        losses["total_loss"].backward()
        # Gradient clipping — important for stability with NLL loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_acc  += losses["total_loss"].item()
        policy_loss_acc += losses["policy_loss"].item()
        value_loss_acc  += losses["value_loss"].item()
        num_batches += 1

    nb = max(num_batches, 1)
    return {
        "total_loss":  total_loss_acc  / nb,
        "policy_loss": policy_loss_acc / nb,
        "value_loss":  value_loss_acc  / nb,
    }
