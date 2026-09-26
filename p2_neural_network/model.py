"""A compact residual policy/value network; all displayed activations are real."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from p1_representation import CHANNELS, encode_state, encode_action

# Small 8x8 convolutions are faster without a large thread pool.
torch.set_num_threads(int(os.environ.get('CHATURANGZERO_TORCH_THREADS', '2')))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int = 32, blocks: int = 2):
        super().__init__()
        self.config = {'channels': channels, 'blocks': blocks}
        self.stem = nn.Sequential(nn.Conv2d(CHANNELS, channels, 3, padding=1), nn.ReLU())
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])
        # Each source square predicts 64 destination logits.
        self.policy_head = nn.Conv2d(channels, 64, 1)
        self.value_head = nn.Sequential(nn.Conv2d(channels, 4, 1), nn.ReLU(), nn.Flatten(),
                                        nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x, return_activations: bool = False):
        x = self.stem(x)
        activations = [x]
        for block in self.blocks:
            x = block(x)
            activations.append(x)
        logits = self.policy_head(x).permute(0, 2, 3, 1).reshape(-1, 4096)
        value = self.value_head(x).squeeze(-1)
        if return_activations:
            return logits, value, activations
        return logits, value


class NeuralEvaluator:
    def __init__(self, model: PolicyValueNet, device: str = 'cpu'):
        self.model = model.to(device).eval()
        self.device = device

    def predict(self, state, moves):
        with torch.inference_mode():
            logits, value = self.model(torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device))
            indices = [encode_action(m, state.side_to_move) for m in moves]
            probs = torch.softmax(logits[0, indices], dim=0).cpu().numpy()
        return probs, float(value.item())

    def inspect(self, state):
        from shared.game_core import legal_moves, result, GameOutcome
        encoded = encode_state(state)
        with torch.inference_mode():
            logits, value, activations = self.model(torch.from_numpy(encoded).unsqueeze(0).to(self.device), True)
        layers = []
        for i, activation in enumerate(activations):
            heat = activation[0].abs().mean(0).cpu().numpy()
            scale = float(heat.max()) or 1
            heat = heat / scale
            # Return physical board orientation for frontend overlays.
            if state.side_to_move.value == 'black':
                heat = heat[::-1]
            channels = activation[0].cpu().numpy()
            if state.side_to_move.value == 'black':
                channels = channels[:, ::-1, :]
            layers.append({'name': 'Input convolution' if i == 0 else f'Residual block {i}',
                           'shape': [self.model.config['channels'], 8, 8],
                           'heatmap': heat.reshape(-1).tolist(), 'maxActivation': scale,
                           'channels': channels.reshape(self.model.config['channels'], 64).tolist()})
        prior_heatmap = np.zeros(64, dtype=np.float32)
        moves = legal_moves(state) if result(state) is GameOutcome.ONGOING else []
        if moves:
            indices = [encode_action(move, state.side_to_move) for move in moves]
            probabilities = torch.softmax(logits[0, indices], dim=0).cpu().numpy()
            for move, probability in zip(moves, probabilities):
                file, rank = move.to_sq
                prior_heatmap[rank * 8 + file] += probability
        return {'value': float(value.item()), 'layers': layers,
                'input': encoded.tolist(), 'inputShape': [CHANNELS, 8, 8],
                'policyPriorHeatmap': prior_heatmap.tolist()}


def save_checkpoint(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'format': 1, 'config': model.config, 'state_dict': model.state_dict(),
                'metadata': metadata or {}}, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('format') != 1:
        raise ValueError('Unsupported checkpoint format')
    model = PolicyValueNet(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint.get('metadata', {})
