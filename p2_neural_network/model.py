"""
p2_neural_network/model.py
==========================
ChaturangaNet — residual policy-value network.

ChaturangZero-X additions
--------------------------
  - uncertainty_head: when True, the value head outputs both a mean and
    a log-variance (Gaussian NLL). The network learns to say "I'm not sure."
  - predict() returns (policy_probs, value_mean, value_std) when uncertainty
    is enabled; otherwise (policy_probs, value_mean) as before.
  - Checkpoint format extended to store uncertainty_head flag.
  - All existing code using the 2-tuple return works unchanged (uncertainty=False).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Two conv layers with a skip connection (He 2016)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChaturangaNet(nn.Module):
    """
    Policy + value (+ optional uncertainty) network for ChaturangZero-X.

    Parameters
    ----------
    in_channels      : number of input planes (13 by default)
    num_filters      : number of convolutional filters per layer
    num_res_blocks   : number of residual blocks in the trunk
    action_size      : total number of possible actions (4096)
    uncertainty_head : if True, value head also outputs log-variance
                       enabling uncertainty-aware PUCT (Research Contribution #3)
    """

    def __init__(
        self,
        in_channels:      int  = 13,
        num_filters:      int  = 64,
        num_res_blocks:   int  = 4,
        action_size:      int  = 4096,
        uncertainty_head: bool = True,
    ) -> None:
        super().__init__()
        self.action_size      = action_size
        self.uncertainty_head = uncertainty_head

        # ── Input convolution ─────────────────────────────────────────────────
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # ── Residual trunk ────────────────────────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # ── Policy head ───────────────────────────────────────────────────────
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * 8 * 8, action_size)

        # ── Value head (shared feature extractor) ─────────────────────────────
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)          # mean

        # ── Uncertainty head (log-variance) — Research Contribution #3 ───────
        if uncertainty_head:
            self.value_fc2_logvar = nn.Linear(64, 1)   # log σ²

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        x : (batch, 13, 8, 8) float tensor

        Returns (without uncertainty_head)
        -------
        policy_logits : (batch, 4096)
        value         : (batch, 1)    tanh-squashed ∈ (−1, +1)

        Returns (with uncertainty_head)
        -------
        policy_logits : (batch, 4096)
        value_mean    : (batch, 1)    tanh-squashed ∈ (−1, +1)
        value_logvar  : (batch, 1)    unconstrained log-variance
        """
        x = self.input_conv(x)
        x = self.res_blocks(x)

        # Policy head
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value_mean = torch.tanh(self.value_fc2(v))

        if self.uncertainty_head:
            value_logvar = self.value_fc2_logvar(v)   # unconstrained log σ²
            return policy_logits, value_mean, value_logvar

        return policy_logits, value_mean

    @torch.no_grad()
    def predict(
        self,
        x:    torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple:
        """
        Run a forward pass in eval mode for a single state.

        Parameters
        ----------
        x    : (1, 13, 8, 8) float tensor
        mask : (4096,) bool tensor of legal actions (optional)

        Returns (uncertainty_head=False)
        -------
        policy_probs : (4096,) masked + softmax
        value        : scalar float

        Returns (uncertainty_head=True)
        -------
        policy_probs : (4096,) masked + softmax
        value_mean   : scalar float   ∈ (−1, +1)
        value_std    : scalar float   ∈ (0, +∞)  — epistemic uncertainty
        """
        self.eval()
        out = self(x)

        if self.uncertainty_head:
            logits, value_mean, value_logvar = out
        else:
            logits, value_mean = out

        logits = logits.squeeze(0)   # (4096,)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        policy_probs = torch.softmax(logits, dim=0)

        if self.uncertainty_head:
            # std = exp(0.5 * log_var), clipped for numerical stability
            value_std = torch.exp(0.5 * value_logvar.clamp(-10, 5)).item()
            return policy_probs, value_mean.item(), value_std
        else:
            return policy_probs, value_mean.item()


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(model: ChaturangaNet, path: str) -> None:
    """Save model weights to `path`."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":       model.state_dict(),
        "in_channels":      model.input_conv[0].in_channels,
        "num_filters":      model.input_conv[0].out_channels,
        "num_res_blocks":   len(model.res_blocks),
        "action_size":      model.action_size,
        "uncertainty_head": model.uncertainty_head,
    }, path)
    print(f"[checkpoint] Saved → {path}")


def load_checkpoint(path: str, device: str = "cpu") -> ChaturangaNet:
    """Load a model from a checkpoint file."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = ChaturangaNet(
        in_channels=ckpt.get("in_channels", 13),
        num_filters=ckpt.get("num_filters", 64),
        num_res_blocks=ckpt.get("num_res_blocks", 4),
        action_size=ckpt.get("action_size", 4096),
        uncertainty_head=ckpt.get("uncertainty_head", True),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    print(f"[checkpoint] Loaded ← {path}")
    return model
