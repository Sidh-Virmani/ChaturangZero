"""Milestone-oriented orchestration for a continuous P4 training run."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Callable

from shared.game_core import GameState

from .config import TrainingConfig
from .trainer import IterationMetrics, Trainer


@dataclass(frozen=True, slots=True)
class LevelProgress:
    """Timing and checkpoint information for one completed iteration."""

    metrics: IterationMetrics
    elapsed_seconds: float
    estimated_remaining_seconds: float
    milestone: str | None


def train_levels(
    config: TrainingConfig,
    *,
    easy: int,
    medium: int,
    hard: int,
    start_state: GameState | None = None,
    on_iteration: Callable[[LevelProgress], None] | None = None,
) -> Trainer:
    """Start or resume one run, save fixed level snapshots, and stop at hard."""
    _validate_milestones(easy, medium, hard)
    output = config.checkpoint_dir
    state_path = output / "training_state.pt"
    milestones = {easy: "easy", medium: "medium", hard: "hard"}

    if state_path.exists():
        trainer = Trainer.resume(output, iterations=1, device=config.device)
    else:
        existing = list(output.glob('*.pt'))
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"Found {names} without training_state.pt; use a clean output directory"
            )
        trainer = Trainer(replace(config, iterations=1))

    if trainer.completed_iterations > hard:
        raise ValueError(
            f"Run already completed iteration {trainer.completed_iterations}, beyond hard={hard}"
        )

    for iteration, name in milestones.items():
        path = output / f"{name}.pt"
        if iteration < trainer.completed_iterations and not path.exists():
            raise FileNotFoundError(
                f"Cannot recreate missing {path.name}: iteration {iteration} has already passed"
            )
        if iteration == trainer.completed_iterations and not path.exists():
            _copy_without_overwrite(output / "latest_model.pt", path)

    durations: list[float] = []
    while trainer.completed_iterations < hard:
        started = time.perf_counter()
        metrics = trainer.run(start_state)[0]
        milestone = milestones.get(metrics.iteration)
        if milestone is not None:
            _copy_without_overwrite(
                output / "latest_model.pt", output / f"{milestone}.pt"
            )
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        remaining = hard - trainer.completed_iterations
        estimate = (sum(durations) / len(durations)) * remaining
        if on_iteration is not None:
            on_iteration(LevelProgress(metrics, elapsed, estimate, milestone))

    return trainer


def _validate_milestones(easy: int, medium: int, hard: int) -> None:
    values = (easy, medium, hard)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise TypeError("easy, medium, and hard must be integers")
    if not 0 < easy < medium < hard:
        raise ValueError("milestones must satisfy 0 < easy < medium < hard")


def _copy_without_overwrite(source: Path, destination: Path) -> bool:
    """Copy a loadable P2 checkpoint once; an existing milestone always wins."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return False
    temporary: Path | None = None
    try:
        with source.open("rb") as input_file, tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.stem}-", suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary = Path(output_file.name)
            shutil.copyfileobj(input_file, output_file)
        os.link(temporary, destination)
    except FileExistsError:
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True
