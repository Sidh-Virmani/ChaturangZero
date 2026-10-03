"""Read-only, checkpoint-backed tensors for the educational 3D inspector."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from p1_representation import PLANE_NAMES, encode_state, legal_action_mask
from p2_neural_network import NetworkEvaluator
from shared.game_core import GameState, square_name


def _layer(key: str, label: str, operation: str, explanation: str,
           tensor: torch.Tensor | np.ndarray, names: list[str] | None = None) -> dict[str, Any]:
    array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
    if array.ndim == 4:
        array = array[0]
    return {
        "id": key, "label": label, "operation": operation, "explanation": explanation,
        "shape": list(array.shape), "channels": array.reshape(array.shape[0], -1).tolist(),
        "channelNames": names, "minimum": float(array.min()), "maximum": float(array.max()),
    }


def inspect_position(state: GameState, evaluator: NetworkEvaluator | None) -> dict[str, Any]:
    """Capture actual intermediate outputs, preserving model mode and BN statistics.

    Channel maps retain P1's physical orientation: row zero is rank 1. Policy
    cells aggregate legal probabilities by destination; they are not logits or
    MCTS visit counts. Terminal positions have an all-zero legal policy.
    The caller serializes this operation with other uses of the evaluator.
    """
    encoded = encode_state(state)
    layers = [_layer("input", "Board encoding", "13 × 8 × 8", "Six piece planes for each side, plus the side-to-move plane. Values are occupancy, not learned features.", encoded, list(PLANE_NAMES))]
    result: dict[str, Any] = {
        "sideToMove": state.side_to_move.value, "layers": layers,
        "value": None, "legalActionCount": int(legal_action_mask(state).sum()),
        "policy": [], "parameterCount": None,
    }
    if evaluator is None:
        return result
    model = evaluator.model
    inputs = torch.from_numpy(encoded).unsqueeze(0).to(evaluator.device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            features = model.stem(inputs)
            layers.append(_layer("stem", "Feature stem", "3×3 convolution · batch norm · ReLU", "Learned local patterns. A bright cell means strong activation in this channel; it does not establish why a move was chosen.", features))
            for index, block in enumerate(model.backbone):
                features = block(features)
                layers.append(_layer(f"residual-{index + 1}", f"Residual block {index + 1}", "Conv · norm · ReLU · conv · norm + skip · ReLU", "The skip connection adds the incoming features to the learned transformation. These are the actual outputs after the addition and ReLU.", features))
            policy_features = model.policy_head[:3](features)
            logits = model.policy_head[3:](policy_features)
            probabilities = evaluator._legal_probabilities(logits, (state,))[0]
            value_features = model.value_features[:3](features)
            value = model.value_head(model.value_features[3:](value_features))[0, 0]
            layers.append(_layer("policy-features", "Policy features", "1×1 convolution · batch norm · ReLU", "Compact feature maps feed the dense layer that produces 4,096 from–to action logits. Illegal actions are masked before softmax.", policy_features))
            destinations = probabilities.reshape(64, 64).sum(dim=0).reshape(1, 8, 8)
            layers.append(_layer("policy", "Legal policy", "4,096 logits → legal mask → softmax", "Each cell sums the normalized priors of all legal moves ending on that square. Priors guide MCTS; search visit counts are shown separately in the game sidebar.", destinations, ["Legal probability by destination"]))
            layers.append(_layer("value-features", "Value features", "1×1 convolution · batch norm · ReLU", "This spatial map feeds a dense hidden layer, then a tanh value output. It is an activation map, not a square-by-square win estimate.", value_features))
            layers.append(_layer("value", "Outcome value", f"64 → {model.config.value_hidden} → 1 · tanh", "A single learned outcome estimate from the side to move, bounded by −1 and +1. This is not a win probability or a per-square score.", value.reshape(1, 1), ["Side-to-move outcome value"]))
            result["value"] = float(value)
            policy_np = probabilities.cpu().numpy()
            result["policy"] = [{"from": square_name(int(action) // 64), "to": square_name(int(action) % 64), "actionId": int(action), "prior": float(policy_np[action])} for action in np.flatnonzero(legal_action_mask(state))]
    finally:
        model.train(was_training)
    result["parameterCount"] = model.parameter_count
    return result
