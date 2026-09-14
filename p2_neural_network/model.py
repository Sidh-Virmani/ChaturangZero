"""
p2_neural_network/model.py
==========================
ChaturangaNet — a lightweight residual policy-value network.

Architecture
------------
  Input:   (batch, 13, 8, 8) float32 tensor
  Body:    Conv 3×3 → N residual blocks (each block: Conv+BN+ReLU × 2)
  Policy head: flatten → FC → 4096 logits (one per possible from/to pair)
  Value head:  flatten → FC → 1 scalar in (−1, +1)

  Default: 4 residual blocks, 64 channels.
  This is intentionally smaller than the full AlphaZero network so it
  trains on CPU in a reasonable time.
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
    Policy-value network for ChaturangZero.

    Parameters
    ----------
    in_channels    : number of input planes (13 by default)
    num_filters    : number of convolutional filters per layer
    num_res_blocks : number of residual blocks in the trunk
    action_size    : total number of possible actions (4096)
    """

    def __init__(
        self,
        in_channels:    int = 13,
        num_filters:    int = 64,
        num_res_blocks: int = 4,
        action_size:    int = 4096,
    ) -> None:
        super().__init__()
        self.action_size = action_size

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # Residual trunk
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * 8 * 8, action_size)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, 13, 8, 8) float tensor

        Returns
        -------
        policy_logits : (batch, 4096) — raw logits; apply softmax + mask externally
        value         : (batch, 1)   — tanh-squashed value in (−1, +1)
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
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, float]:
        """
        Run a forward pass in eval mode.

        Parameters
        ----------
        x    : (1, 13, 8, 8) float tensor (single state)
        mask : (4096,) bool tensor of legal actions (optional)

        Returns
        -------
        policy_probs : (4096,) float tensor — masked, softmax-normalised
        value        : scalar float
        """
        self.eval()
        logits, value = self(x)
        logits = logits.squeeze(0)  # (4096,)

        if mask is not None:
            # Set illegal actions to a large negative number before softmax
            logits = logits.masked_fill(~mask, -1e9)

        policy_probs = torch.softmax(logits, dim=0)
        return policy_probs, value.item()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model: ChaturangaNet, path: str) -> None:
    """Save model weights to `path`."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict":   model.state_dict(),
        "in_channels":  model.input_conv[0].in_channels,
        "num_filters":  model.input_conv[0].out_channels,
        "num_res_blocks": len(model.res_blocks),
        "action_size":  model.action_size,
    }, path)
    print(f"[checkpoint] Saved → {path}")


def load_checkpoint(path: str, device: str = "cpu") -> ChaturangaNet:
    """Load a model from a checkpoint file."""
    ckpt = torch.load(path, map_location=device)
    model = ChaturangaNet(
        in_channels=ckpt.get("in_channels", 13),
        num_filters=ckpt.get("num_filters", 64),
        num_res_blocks=ckpt.get("num_res_blocks", 4),
        action_size=ckpt.get("action_size", 4096),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    print(f"[checkpoint] Loaded ← {path}")
    return model
