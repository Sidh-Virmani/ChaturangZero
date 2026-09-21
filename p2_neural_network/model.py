"""PyTorch policy-value network for P1 encoded Chaturanga states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from p1_representation import ACTION_SPACE_SIZE, STATE_SHAPE


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Sizes that define a PolicyValueNetwork checkpoint."""

    channels: int = 64
    residual_blocks: int = 4
    policy_channels: int = 2
    value_hidden: int = 128

    def __post_init__(self) -> None:
        for name, value in (
            ("channels", self.channels),
            ("residual_blocks", self.residual_blocks),
            ("policy_channels", self.policy_channels),
            ("value_hidden", self.value_hidden),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class ResidualBlock(nn.Module):
    """Two spatial convolutions with the input added back before activation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        features = self.activation(self.norm1(self.conv1(inputs)))
        features = self.norm2(self.conv2(features))
        return self.activation(features + residual)


class PolicyValueNetwork(nn.Module):
    """Shared board feature extractor with policy-logit and value outputs.

    Input is a batch shaped ``(batch, *STATE_SHAPE)``. The returned policy
    logits have shape ``(batch, ACTION_SPACE_SIZE)``. Values have shape
    ``(batch,)`` and use the side-to-move perspective in the range [-1, 1].
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else config
        channels = self.config.channels
        input_channels, rows, columns = STATE_SHAPE

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(self.config.residual_blocks))
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, self.config.policy_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.config.policy_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.config.policy_channels * rows * columns, ACTION_SPACE_SIZE),
        )
        self.value_features = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(rows * columns, self.config.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.value_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, encoded_states: Tensor) -> tuple[Tensor, Tensor]:
        if encoded_states.ndim != 4 or tuple(encoded_states.shape[1:]) != STATE_SHAPE:
            raise ValueError(
                f"Expected encoded states shaped (batch, {STATE_SHAPE[0]}, "
                f"{STATE_SHAPE[1]}, {STATE_SHAPE[2]}), got {tuple(encoded_states.shape)}"
            )
        if not torch.is_floating_point(encoded_states):
            raise TypeError("encoded_states must use a floating-point dtype")

        features = self.backbone(self.stem(encoded_states))
        policy_logits = self.policy_head(features)
        values = self.value_head(self.value_features(features)).squeeze(1)
        return policy_logits, values

    @property
    def parameter_count(self) -> int:
        """Number of trainable scalar parameters in this model."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
