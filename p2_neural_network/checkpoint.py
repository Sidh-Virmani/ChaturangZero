"""Portable model-only checkpoint save and load helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from .model import ModelConfig, PolicyValueNetwork


def save_checkpoint(model: PolicyValueNetwork, path: str | Path) -> None:
    """Save model configuration and learned parameters."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(model.config), "model_state": model.state_dict()},
        destination,
    )


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu"
) -> PolicyValueNetwork:
    """Load a saved model onto the requested device in evaluation mode."""
    resolved_device = torch.device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
    model = PolicyValueNetwork(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved_device)
    model.eval()
    return model
