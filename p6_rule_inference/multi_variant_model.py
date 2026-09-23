"""
p6_rule_inference/multi_variant_model.py
========================================
Rule-Conditioned Network (Contribution B).

Instead of training 4 separate networks (one per rule variant), we train
ONE network that plays all rule variants.

The network receives the board state PLUS a Rule Embedding vector (r ∈ R^8).
- The shared ResNet trunk learns universal Chaturanga strategy.
- The rule embedding is concatenated into the value head (and optionally policy).
- The network learns what is rule-invariant and what is rule-specific.
"""

from __future__ import annotations

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass  # Allow import without torch for testing environments


class MultiVariantChaturangaNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 13,
        num_filters: int = 64,
        num_res_blocks: int = 4,
        action_size: int = 4096,
        num_variants: int = 4,
        rule_embed_dim: int = 8,
        uncertainty_head: bool = True,
    ):
        super().__init__()
        self.action_size = action_size
        self.uncertainty_head = uncertainty_head

        # Rule Embedding Layer (maps variant index 0..3 to an 8-d vector)
        self.rule_embedding = nn.Embedding(num_variants, rule_embed_dim)

        # 1. Shared Convolutional Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        # 2. Shared Residual Trunk
        self.res_blocks = nn.ModuleList([
            ResBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # 3. Policy Head (Shared)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        # Policy FC takes flattened board + rule embedding
        self.policy_fc = nn.Linear(2 * 8 * 8 + rule_embed_dim, action_size)

        # 4. Rule-Conditioned Value Head
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        # Value FC takes flattened board + rule embedding
        self.value_fc1 = nn.Sequential(
            nn.Linear(1 * 8 * 8 + rule_embed_dim, 256),
            nn.ReLU()
        )
        self.value_fc2_mean = nn.Linear(256, 1)

        if self.uncertainty_head:
            self.value_fc2_logvar = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, variant_idx: torch.Tensor) -> tuple:
        """
        x: (batch_size, 13, 8, 8)
        variant_idx: (batch_size,) integer indices [0..num_variants-1]
        """
        batch_size = x.size(0)

        # Trunk
        out = self.conv_block(x)
        for block in self.res_blocks:
            out = block(out)

        # Get rule embedding
        r_embed = self.rule_embedding(variant_idx)  # (batch_size, rule_embed_dim)

        # Policy Head
        pol = self.policy_conv(out).view(batch_size, -1)
        # Concatenate rule embedding
        pol = torch.cat([pol, r_embed], dim=1)
        policy_logits = self.policy_fc(pol)

        # Value Head
        val = self.value_conv(out).view(batch_size, -1)
        # Concatenate rule embedding
        val = torch.cat([val, r_embed], dim=1)
        val = self.value_fc1(val)
        
        value_mean = torch.tanh(self.value_fc2_mean(val))

        if self.uncertainty_head:
            value_logvar = self.value_fc2_logvar(val)
            return policy_logits, value_mean, value_logvar
        
        return policy_logits, value_mean


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return F.relu(out)
