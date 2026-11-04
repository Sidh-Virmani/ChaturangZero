"""Fixed Phase 0 benchmark for the unoptimized production pipeline.

The workload deliberately calls the public P1/P2/P3/game-core interfaces. It
does not contain alternative fast paths, save checkpoints, or invoke P4
self-play/training. The training-step measurement updates only an in-memory
copy of the loaded model and discards it immediately.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
import torch

from p1_representation import ACTION_SPACE_SIZE, encode_state, legal_action_mask
from p2_neural_network import (
    NetworkEvaluator,
    PolicyValueNetwork,
    load_checkpoint,
    policy_value_loss,
)
from p3_mcts import MCTS, MCTSConfig
from shared.game_core import (
    GameState,
    Move,
    Piece,
    PieceType,
    Side,
    apply_move,
    initial_state,
    legal_moves,
    terminal_status,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class NamedPosition:
    """A stable position included in every full baseline run."""

    name: str
    state: GameState


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Repeat counts and fixed workload sizes for one benchmark run."""

    checkpoint: Path = Path("checkpoints/benchmark/hard.pt")
    device: str = "cpu"
    seed: int = 20261105
    batch_sizes: tuple[int, ...] = (1, 16, 64)
    core_warmups: int = 5
    core_rounds: int = 5
    core_iterations: int = 100
    encoding_iterations: int = 1_000
    inference_warmups: int = 5
    inference_rounds: int = 5
    inference_iterations: int = 30
    evaluator_warmups: int = 2
    evaluator_rounds: int = 3
    evaluator_iterations: int = 5
    mcts_simulations: tuple[int, ...] = (5, 10, 25)
    mcts_warmups: int = 1
    mcts_repeats: int = 3
    self_play_simulations: int = 5
    self_play_plies: int = 4
    self_play_repeats: int = 2
    training_batch_size: int = 64
    training_warmups: int = 2
    training_rounds: int = 3
    training_steps_per_round: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, Path):
            raise TypeError("checkpoint must be a pathlib.Path")
        integer_fields = (
            "core_warmups", "core_rounds", "core_iterations", "encoding_iterations",
            "inference_warmups", "inference_rounds", "inference_iterations",
            "evaluator_warmups", "evaluator_rounds", "evaluator_iterations",
            "mcts_warmups", "mcts_repeats", "self_play_simulations",
            "self_play_plies", "self_play_repeats", "training_batch_size",
            "training_warmups", "training_rounds", "training_steps_per_round",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if any(value <= 0 for value in (
            self.core_rounds, self.core_iterations, self.encoding_iterations,
            self.inference_rounds, self.inference_iterations, self.evaluator_rounds,
            self.evaluator_iterations, self.mcts_repeats, self.self_play_simulations,
            self.self_play_plies, self.self_play_repeats, self.training_batch_size,
            self.training_rounds, self.training_steps_per_round,
        )):
            raise ValueError("measured workload sizes and repeat counts must be positive")
        if not self.batch_sizes or any(size <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must contain positive integers")
        if not self.mcts_simulations or any(size <= 0 for size in self.mcts_simulations):
            raise ValueError("mcts_simulations must contain positive integers")

    @classmethod
    def quick(cls, checkpoint: Path, device: str = "cpu") -> "BenchmarkConfig":
        """Return a tiny command-validation workload, not a performance baseline."""
        return cls(
            checkpoint=checkpoint,
            device=device,
            batch_sizes=(1,),
            core_warmups=0,
            core_rounds=1,
            core_iterations=1,
            encoding_iterations=1,
            inference_warmups=0,
            inference_rounds=1,
            inference_iterations=1,
            evaluator_warmups=0,
            evaluator_rounds=1,
            evaluator_iterations=1,
            mcts_simulations=(1,),
            mcts_warmups=0,
            mcts_repeats=1,
            self_play_simulations=1,
            self_play_plies=1,
            self_play_repeats=1,
            training_batch_size=2,
            training_warmups=0,
            training_rounds=1,
            training_steps_per_round=1,
        )


def fixed_positions() -> tuple[NamedPosition, ...]:
    """Build the exact valid positions shared by future benchmark runs."""
    initial = initial_state()
    middle = initial
    for uci in (
        "a2a3", "a7a6", "b1c3", "b8c6", "d2d3",
        "d7d6", "c1e3", "c8e6", "g1f3", "g8f6",
    ):
        middle = apply_move(middle, Move.from_names(uci[:2], uci[2:]))

    endgame = GameState.from_mapping(
        {
            "a1": Piece(Side.WHITE, PieceType.RAJA),
            "b2": Piece(Side.WHITE, PieceType.RATHA),
            "c4": Piece(Side.WHITE, PieceType.PADATI),
            "h8": Piece(Side.BLACK, PieceType.RAJA),
            "g7": Piece(Side.BLACK, PieceType.RATHA),
            "f5": Piece(Side.BLACK, PieceType.PADATI),
        },
        side_to_move=Side.WHITE,
    )
    positions = (
        NamedPosition("initial", initial),
        NamedPosition("midgame_10_ply", middle),
        NamedPosition("simple_endgame", endgame),
    )
    for position in positions:
        if terminal_status(position.state).terminal or not legal_moves(position.state):
            raise RuntimeError(f"Benchmark position {position.name} must be non-terminal")
    return positions


def run_baseline(config: BenchmarkConfig) -> dict[str, object]:
    """Run the fixed workload and return a JSON-serializable report."""
    checkpoint = config.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Benchmark checkpoint does not exist: {checkpoint}")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but torch.cuda.is_available() is false")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    positions = fixed_positions()
    states = tuple(position.state for position in positions)
    state_snapshots = tuple(states)
    checkpoint_hash_before = _sha256(checkpoint)

    model = load_checkpoint(checkpoint, device=device)
    model_snapshot = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    evaluator = NetworkEvaluator(model, device=device)
    metrics: list[dict[str, object]] = []

    metrics.extend(_core_metrics(config, states))
    metrics.extend(_raw_inference_metrics(config, model, device, states))
    metrics.extend(_evaluator_metrics(config, evaluator, states))
    metrics.extend(_mcts_metrics(config, evaluator, states[0]))
    metrics.append(_self_play_metric(config, evaluator, states[0]))
    metrics.append(_training_metric(config, model, states))

    if tuple(states) != state_snapshots:
        raise RuntimeError("Benchmark mutated a fixed GameState")
    for name, expected in model_snapshot.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), expected):
            raise RuntimeError("Benchmark mutated the loaded evaluation model")
    checkpoint_hash_after = _sha256(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("Benchmark modified its source checkpoint")

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "phase_0_baseline",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": _system_information(device),
        "git": _git_information(),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash_before,
            "size_bytes": checkpoint.stat().st_size,
            "model_config": asdict(model.config),
            "trainable_parameters": model.parameter_count,
        },
        "configuration": _config_dict(config),
        "workload": {
            "positions": [
                {
                    "name": item.name,
                    "ply_count": item.state.ply_count,
                    "side_to_move": item.state.side_to_move.value,
                    "piece_count": sum(piece is not None for piece in item.state.board),
                    "legal_moves": len(legal_moves(item.state)),
                }
                for item in positions
            ],
            "mcts_position": "initial",
            "self_play_start_position": "initial",
        },
        "metrics": metrics,
    }


def _core_metrics(
    config: BenchmarkConfig, states: Sequence[GameState]
) -> list[dict[str, object]]:
    operations: tuple[tuple[str, Callable[[GameState], object]], ...] = (
        ("legal_moves", legal_moves),
        ("terminal_status", terminal_status),
        ("legal_action_mask", legal_action_mask),
    )
    metrics = []
    for name, operation in operations:
        samples = _time_state_operation(
            operation,
            states,
            warmups=config.core_warmups,
            rounds=config.core_rounds,
            iterations=config.core_iterations,
        )
        metrics.append(_timing_metric(name, "game_and_representation", samples, "ms/state"))

    encoding_samples = _time_state_operation(
        encode_state,
        states,
        warmups=config.core_warmups,
        rounds=config.core_rounds,
        iterations=config.encoding_iterations,
    )
    metrics.append(
        _timing_metric("encode_state", "game_and_representation", encoding_samples, "ms/state")
    )
    return metrics


def _raw_inference_metrics(
    config: BenchmarkConfig,
    model: PolicyValueNetwork,
    device: torch.device,
    states: Sequence[GameState],
) -> list[dict[str, object]]:
    model.eval()
    metrics = []
    for batch_size in config.batch_sizes:
        batch_states = _cycled_states(states, batch_size)
        encoded = np.stack([encode_state(state) for state in batch_states])
        inputs = torch.from_numpy(encoded).to(device)
        with torch.inference_mode():
            for _ in range(config.inference_warmups):
                model(inputs)
            _synchronize(device)
            samples = []
            for _ in range(config.inference_rounds):
                _synchronize(device)
                start = perf_counter()
                for _ in range(config.inference_iterations):
                    model(inputs)
                _synchronize(device)
                elapsed = perf_counter() - start
                samples.append(elapsed / (config.inference_iterations * batch_size))
        metric = _timing_metric(
            f"raw_network_batch_{batch_size}", "network", samples, "ms/state"
        )
        median_seconds = statistics.median(samples)
        metric["details"] = {
            "batch_size": batch_size,
            "median_ms_per_batch": median_seconds * batch_size * 1_000.0,
            "states_per_second": 1.0 / median_seconds,
        }
        metrics.append(metric)
    return metrics


def _evaluator_metrics(
    config: BenchmarkConfig,
    evaluator: NetworkEvaluator,
    states: Sequence[GameState],
) -> list[dict[str, object]]:
    metrics = []
    for batch_size in config.batch_sizes:
        batch_states = _cycled_states(states, batch_size)
        for _ in range(config.evaluator_warmups):
            evaluator.evaluate_batch(batch_states)
        samples = []
        for _ in range(config.evaluator_rounds):
            start = perf_counter()
            for _ in range(config.evaluator_iterations):
                evaluator.evaluate_batch(batch_states)
            elapsed = perf_counter() - start
            samples.append(elapsed / (config.evaluator_iterations * batch_size))
        metric = _timing_metric(
            f"complete_evaluator_batch_{batch_size}",
            "evaluator",
            samples,
            "ms/state",
        )
        median_seconds = statistics.median(samples)
        metric["details"] = {
            "batch_size": batch_size,
            "median_ms_per_batch": median_seconds * batch_size * 1_000.0,
            "states_per_second": 1.0 / median_seconds,
        }
        metrics.append(metric)
    return metrics


def _mcts_metrics(
    config: BenchmarkConfig,
    evaluator: NetworkEvaluator,
    state: GameState,
) -> list[dict[str, object]]:
    for _ in range(config.mcts_warmups):
        MCTS(
            evaluator,
            MCTSConfig(num_simulations=1, temperature=0.0, add_noise=False),
        ).search(state)

    metrics = []
    for simulations in config.mcts_simulations:
        samples = []
        for repeat in range(config.mcts_repeats):
            np.random.seed(config.seed + simulations * 100 + repeat)
            search = MCTS(
                evaluator,
                MCTSConfig(
                    num_simulations=simulations,
                    temperature=0.0,
                    add_noise=False,
                ),
            )
            start = perf_counter()
            search.search(state)
            samples.append(perf_counter() - start)
        median_seconds = statistics.median(samples)
        metrics.append({
            "name": f"mcts_{simulations}_simulations",
            "category": "mcts",
            "value": median_seconds,
            "unit": "seconds/search",
            "samples": len(samples),
            "minimum": min(samples),
            "maximum": max(samples),
            "details": {
                "simulations": simulations,
                "simulations_per_second": simulations / median_seconds,
                "temperature": 0.0,
                "dirichlet_noise": False,
                "fresh_tree_each_sample": True,
            },
        })
    return metrics


def _self_play_metric(
    config: BenchmarkConfig,
    evaluator: NetworkEvaluator,
    start_state: GameState,
) -> dict[str, object]:
    samples = []
    completed_plies = []
    for repeat in range(config.self_play_repeats):
        np.random.seed(config.seed + 50_000 + repeat)
        state = start_state
        search = MCTS(
            evaluator,
            MCTSConfig(
                num_simulations=config.self_play_simulations,
                temperature=0.0,
                add_noise=False,
            ),
        )
        plies = 0
        start = perf_counter()
        while plies < config.self_play_plies and not terminal_status(state).terminal:
            _, move, _ = search.search(state)
            state = apply_move(state, move)
            search.step(move)
            plies += 1
        elapsed = perf_counter() - start
        samples.append(plies / elapsed)
        completed_plies.append(plies)
    return {
        "name": "short_self_play",
        "category": "self_play",
        "value": statistics.median(samples),
        "unit": "plies/second",
        "samples": len(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "details": {
            "simulations_per_move": config.self_play_simulations,
            "requested_plies_per_sample": config.self_play_plies,
            "completed_plies_per_sample": completed_plies,
            "temperature": 0.0,
            "dirichlet_noise": False,
            "subtree_reuse": True,
        },
    }


def _training_metric(
    config: BenchmarkConfig,
    source_model: PolicyValueNetwork,
    states: Sequence[GameState],
) -> dict[str, object]:
    model = deepcopy(source_model).cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    batch_states = _cycled_states(states, config.training_batch_size)
    inputs = torch.from_numpy(np.stack([encode_state(state) for state in batch_states]))
    target_policy = torch.from_numpy(np.stack([
        _uniform_legal_policy(state) for state in batch_states
    ]))
    target_values = torch.tensor(
        [(-1.0, 0.0, 1.0)[index % 3] for index in range(config.training_batch_size)],
        dtype=torch.float32,
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(inputs)
        loss = policy_value_loss(logits, values, target_policy, target_values)
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    for _ in range(config.training_warmups):
        step()
    samples = []
    for _ in range(config.training_rounds):
        start = perf_counter()
        for _ in range(config.training_steps_per_round):
            step()
        elapsed = perf_counter() - start
        samples.append(config.training_steps_per_round / elapsed)
    median_steps_per_second = statistics.median(samples)
    return {
        "name": "cpu_training_step",
        "category": "training",
        "value": median_steps_per_second,
        "unit": "steps/second",
        "samples": len(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "details": {
            "batch_size": config.training_batch_size,
            "median_ms_per_step": 1_000.0 / median_steps_per_second,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "model_copy_discarded": True,
        },
    }


def _uniform_legal_policy(state: GameState) -> np.ndarray:
    policy = legal_action_mask(state).astype(np.float32)
    total = float(policy.sum())
    if total <= 0:
        raise ValueError("Training benchmark requires non-terminal states")
    policy /= total
    return policy


def _cycled_states(states: Sequence[GameState], count: int) -> tuple[GameState, ...]:
    return tuple(states[index % len(states)] for index in range(count))


def _time_state_operation(
    operation: Callable[[GameState], object],
    states: Sequence[GameState],
    *,
    warmups: int,
    rounds: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmups):
        for state in states:
            operation(state)
    samples = []
    calls = iterations * len(states)
    for _ in range(rounds):
        start = perf_counter()
        for _ in range(iterations):
            for state in states:
                operation(state)
        samples.append((perf_counter() - start) / calls)
    return samples


def _timing_metric(
    name: str,
    category: str,
    samples_seconds: Sequence[float],
    unit: str,
) -> dict[str, object]:
    scale = 1_000.0 if unit.startswith("ms/") else 1.0
    values = [sample * scale for sample in samples_seconds]
    return {
        "name": name,
        "category": category,
        "value": statistics.median(values),
        "unit": unit,
        "samples": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "details": {},
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_information() -> dict[str, object]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _system_information(device: torch.device) -> dict[str, object]:
    device_name = "CPU"
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "device_name": device_name,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _config_dict(config: BenchmarkConfig) -> dict[str, object]:
    data = asdict(config)
    data["checkpoint"] = str(config.checkpoint)
    return data


def write_json(report: dict[str, object], path: Path) -> None:
    """Write a benchmark report without altering checkpoints or model files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def print_report(report: dict[str, object]) -> None:
    """Print the concise comparison table used during optimization work."""
    git = report["git"]
    system = report["system"]
    checkpoint = report["checkpoint"]
    print("ChaturangZero Phase 0 performance baseline")
    print(f"Commit:     {git['commit']} ({git['branch']}, dirty={git['dirty']})")
    print(f"Python:     {system['python']} | PyTorch {system['torch']}")
    print(f"Device:     {system['device']} ({system['device_name']})")
    print(f"Checkpoint: {checkpoint['path']}")
    print(f"Seed:       {report['configuration']['seed']}")
    print()
    print(f"{'Benchmark':<34} {'Median':>12}  {'Unit':<18} Details")
    print("-" * 92)
    for metric in report["metrics"]:
        details = metric["details"]
        detail_text = ""
        if "simulations_per_second" in details:
            detail_text = f"{details['simulations_per_second']:.2f} simulations/s"
        elif "states_per_second" in details:
            detail_text = f"{details['states_per_second']:.1f} states/s"
        elif "median_ms_per_step" in details:
            detail_text = f"{details['median_ms_per_step']:.2f} ms/step"
        elif metric["name"] == "short_self_play":
            detail_text = f"{details['simulations_per_move']} simulations/move"
        print(
            f"{metric['name']:<34} {metric['value']:>12.4f}  "
            f"{metric['unit']:<18} {detail_text}"
        )
