"""Iterative self-play, replay sampling, network updates, and resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
import math

import numpy as np
import torch

from p1_representation import ACTION_SPACE_SIZE, STATE_SHAPE
from p2_neural_network import (
    ModelConfig, NetworkEvaluator, PolicyValueNetwork, policy_value_loss,
    save_checkpoint,
)
from shared.game_core import GameState, Side

from .config import TrainingConfig
from .augmentation import augment_example
from .data import ReplayBuffer
from .self_play import play_game


@dataclass(frozen=True, slots=True)
class IterationMetrics:
    iteration: int
    games_completed: int
    games_truncated: int
    examples_added: int
    replay_size: int
    average_plies: float
    white_wins: int
    black_wins: int
    updates: int
    policy_loss: float | None
    value_loss: float | None
    total_loss: float | None
    draws: int = 0
    training_passes: float | None = None
    training_steps: int | None = None
    batch_size: int | None = None


class Trainer:
    """Run additional training iterations using the current P1/P2/P3 APIs."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        np.random.seed(config.seed)  # P3 uses NumPy's module-level RNG.
        torch.manual_seed(config.seed)
        self.sample_rng = Random(config.seed)

        self.model = PolicyValueNetwork(config.model).to(self.device)
        self.evaluator = NetworkEvaluator(self.model, self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.replay = ReplayBuffer(config.replay_capacity)
        self.completed_iterations = 0
        self.history: list[IterationMetrics] = []

    def run(self, start_state: GameState | None = None) -> tuple[IterationMetrics, ...]:
        """Run ``config.iterations`` more cycles and checkpoint after each one."""
        new_metrics: list[IterationMetrics] = []
        for _ in range(self.config.iterations):
            game_results = [
                play_game(self.evaluator, self.config, start_state)
                for _ in range(self.config.games_per_iteration)
            ]
            for game in game_results:
                self.replay.add_game(game.examples)

            examples_added = sum(len(game.examples) for game in game_results)
            updates = self.update_count(examples_added)
            losses = self._train_steps(updates) if updates else []
            completed = [game for game in game_results if not game.truncated]
            self.completed_iterations += 1
            metrics = IterationMetrics(
                iteration=self.completed_iterations,
                games_completed=len(completed),
                games_truncated=len(game_results) - len(completed),
                examples_added=examples_added,
                replay_size=len(self.replay),
                average_plies=sum(game.plies for game in game_results) / len(game_results),
                white_wins=sum(game.outcome.winner is Side.WHITE for game in completed),
                black_wins=sum(game.outcome.winner is Side.BLACK for game in completed),
                updates=len(losses),
                policy_loss=_mean_or_none([item[0] for item in losses]),
                value_loss=_mean_or_none([item[1] for item in losses]),
                total_loss=_mean_or_none([item[2] for item in losses]),
                draws=sum(game.outcome.draw for game in completed),
                training_passes=self.config.training_passes,
                training_steps=self.config.training_steps,
                batch_size=self.config.batch_size,
            )
            self.history.append(metrics)
            new_metrics.append(metrics)
            self.save()
        return tuple(new_metrics)

    def update_count(self, new_examples: int) -> int:
        """Budget updates from new data, or an explicitly selected fixed schedule."""
        if not isinstance(new_examples, int) or isinstance(new_examples, bool) or new_examples < 0:
            raise ValueError("new_examples must be a non-negative integer")
        if not len(self.replay):
            return 0
        if self.config.training_steps is not None:
            return self.config.training_steps
        return math.ceil(new_examples * self.config.training_passes / self.config.batch_size)

    def _train_steps(self, updates: int) -> list[tuple[float, float, float]]:
        self.model.train()
        recorded: list[tuple[float, float, float]] = []
        for _ in range(updates):
            examples = self.replay.sample(self.config.batch_size, self.sample_rng)
            examples = tuple(
                augment_example(example, self.sample_rng, self.config.symmetry_probability)
                for example in examples
            )
            states = torch.from_numpy(
                np.stack([example.state for example in examples]).astype(np.float32)
            ).to(self.device)
            policies = torch.from_numpy(
                np.stack([example.policy() for example in examples])
            ).to(self.device)
            values = torch.tensor(
                [example.value for example in examples], dtype=torch.float32, device=self.device
            )

            self.optimizer.zero_grad(set_to_none=True)
            logits, predictions = self.model(states)
            loss = policy_value_loss(logits, predictions, policies, values)
            if not torch.isfinite(loss.total):
                raise FloatingPointError("Training produced a non-finite loss")
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            recorded.append((
                float(loss.policy.detach()),
                float(loss.value.detach()),
                float(loss.total.detach()),
            ))
        return recorded

    def save(self) -> None:
        """Save a P2-loadable model and full restart state in checkpoint_dir."""
        directory = self.config.checkpoint_dir
        directory.mkdir(parents=True, exist_ok=True)

        model_temp = directory / "latest_model.tmp"
        save_checkpoint(self.model, model_temp)
        model_temp.replace(directory / "latest_model.pt")

        numpy_state = np.random.get_state()
        config_data = asdict(self.config)
        config_data["checkpoint_dir"] = str(self.config.checkpoint_dir)
        payload = {
            "version": 1,
            "state_shape": STATE_SHAPE,
            "action_space_size": ACTION_SPACE_SIZE,
            "config": config_data,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "replay": self.replay.to_checkpoint(),
            "completed_iterations": self.completed_iterations,
            "history": [asdict(item) for item in self.history],
            "python_rng": self.sample_rng.getstate(),
            "numpy_rng": (
                numpy_state[0], torch.from_numpy(numpy_state[1].copy()),
                numpy_state[2], numpy_state[3], numpy_state[4],
            ),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        }
        state_temp = directory / "training_state.tmp"
        torch.save(payload, state_temp)
        state_temp.replace(directory / "training_state.pt")

    @classmethod
    def resume(
        cls, checkpoint_dir: Path, iterations: int, device: str = "cpu"
    ) -> "Trainer":
        """Restore model, optimizer, replay, and RNG; run more iterations later."""
        path = checkpoint_dir / "training_state.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (
            saved["version"] != 1
            or tuple(saved["state_shape"]) != STATE_SHAPE
            or saved["action_space_size"] != ACTION_SPACE_SIZE
        ):
            raise ValueError("Checkpoint is incompatible with the current P1/P4 format")

        config_data = dict(saved["config"])
        if "training_passes" not in config_data:
            # Old configs always included a default step count. Migrate to
            # data-scaled work so zero-data iterations do not fit stale replay.
            config_data["training_steps"] = None
            config_data["training_passes"] = 1.0
        # Older runs did not augment; retain their original random stream and
        # training behavior unless a new run is explicitly configured.
        config_data.setdefault("symmetry_probability", 0.0)
        config_data["model"] = ModelConfig(**config_data["model"])
        config_data["checkpoint_dir"] = checkpoint_dir
        config_data["device"] = device
        config_data["iterations"] = iterations
        trainer = cls(TrainingConfig(**config_data))
        trainer.model.load_state_dict(saved["model_state"])
        trainer.optimizer.load_state_dict(saved["optimizer_state"])
        trainer.replay = ReplayBuffer.from_checkpoint(
            trainer.config.replay_capacity, saved["replay"]
        )
        trainer.completed_iterations = int(saved["completed_iterations"])
        trainer.history = [IterationMetrics(**item) for item in saved["history"]]
        trainer.sample_rng.setstate(saved["python_rng"])
        numpy_state = saved["numpy_rng"]
        np.random.set_state((
            numpy_state[0], numpy_state[1].numpy(),
            numpy_state[2], numpy_state[3], numpy_state[4],
        ))
        torch.set_rng_state(saved["torch_rng"])
        if device.startswith("cuda") and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        return trainer


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
