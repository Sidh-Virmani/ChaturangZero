"""Command line entry point for self-play training and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from p2_neural_network import ModelConfig, NetworkEvaluator, load_checkpoint

from .arena import compare_models
from .config import TrainingConfig
from .levels import LevelProgress, train_levels
from .self_play import smoke_position
from .trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="ChaturangZero self-play training")
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke", help="One short complete game and update")
    smoke.add_argument("--output", type=Path, default=Path("checkpoints/p4_smoke"))

    train = commands.add_parser("train", help="Start a new run from a random model")
    _add_training_arguments(train, default_output=Path("checkpoints/p4_training"))
    train.add_argument("--iterations", type=int, default=1)

    levels = commands.add_parser(
        "train-levels", help="Start/resume one run and save Easy, Medium, and Hard models"
    )
    _add_training_arguments(levels, default_output=Path("checkpoints/final_models"))
    levels.add_argument("--easy", type=int, required=True, help="Easy milestone iteration")
    levels.add_argument("--medium", type=int, required=True, help="Medium milestone iteration")
    levels.add_argument("--hard", type=int, required=True, help="Final iteration and stopping point")

    resume = commands.add_parser("resume", help="Continue an existing run")
    resume.add_argument("--output", type=Path, default=Path("checkpoints/p4_training"))
    resume.add_argument("--iterations", type=int, default=1, help="Additional iterations")
    resume.add_argument("--device", default="cpu")

    arena = commands.add_parser("arena", help="Compare two model checkpoints")
    arena.add_argument("--candidate", type=Path, required=True)
    arena.add_argument("--reference", type=Path, required=True)
    arena.add_argument("--games", type=int, default=2)
    arena.add_argument("--simulations", type=int, default=25)
    arena.add_argument("--max-plies", type=int, default=512)
    arena.add_argument("--device", default="cpu")

    args = parser.parse_args()
    if args.command == "arena":
        result = compare_models(
            load_checkpoint(args.candidate, args.device),
            load_checkpoint(args.reference, args.device),
            games=args.games,
            simulations_per_move=args.simulations,
            max_game_plies=args.max_plies,
            device=args.device,
        )
        print(json.dumps(asdict(result)))
        return

    if args.command == "train-levels":
        config = _training_config(args, iterations=1)
        if (args.output / "training_state.pt").exists():
            print("mode=resume (restoring saved model, optimizer, replay, and training settings)")
        else:
            print("mode=new")
        trainer = train_levels(
            config,
            easy=args.easy,
            medium=args.medium,
            hard=args.hard,
            on_iteration=lambda progress: _print_level_progress(progress, args.hard),
        )
        metrics = ()
    elif args.command == "resume":
        trainer = Trainer.resume(args.output, args.iterations, args.device)
        metrics = trainer.run()
    else:
        if (args.output / "training_state.pt").exists():
            parser.error("training_state.pt already exists; use resume or a new --output")
        if args.command == "smoke":
            config = TrainingConfig.smoke(args.output)
            trainer = Trainer(config)
            metrics = trainer.run(smoke_position())
            if metrics[0].games_completed != 1 or metrics[0].updates != 1:
                raise RuntimeError("Smoke run did not finish a game and update the model")
            loaded = load_checkpoint(args.output / "latest_model.pt")
            NetworkEvaluator(loaded).evaluate(smoke_position())
        else:
            config = _training_config(args, iterations=args.iterations)
            trainer = Trainer(config)
            metrics = trainer.run()

    for item in metrics:
        print(json.dumps(asdict(item)))
    print(f"model={trainer.config.checkpoint_dir / 'latest_model.pt'}")
    print(f"resume={trainer.config.checkpoint_dir / 'training_state.pt'}")
    if args.command == "train-levels":
        for name in ("easy", "medium", "hard"):
            print(f"{name}={trainer.config.checkpoint_dir / f'{name}.pt'}")


def _add_training_arguments(parser: argparse.ArgumentParser, *, default_output: Path) -> None:
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--games-per-iteration", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--training-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--opening-plies", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)


def _training_config(args: argparse.Namespace, *, iterations: int) -> TrainingConfig:
    return TrainingConfig(
        iterations=iterations,
        games_per_iteration=args.games_per_iteration,
        simulations_per_move=args.simulations,
        max_game_plies=args.max_plies,
        opening_temperature_plies=args.opening_plies,
        training_steps=args.training_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        checkpoint_dir=args.output,
        model=ModelConfig(channels=args.channels, residual_blocks=args.blocks),
    )


def _print_level_progress(progress: LevelProgress, hard: int) -> None:
    iteration = progress.metrics.iteration
    milestone = f" milestone={progress.milestone}.pt" if progress.milestone else ""
    print(
        f"iteration={iteration}/{hard} "
        f"elapsed={_format_duration(progress.elapsed_seconds)} "
        f"eta={_format_duration(progress.estimated_remaining_seconds)}{milestone}",
        flush=True,
    )
    print(json.dumps(asdict(progress.metrics)), flush=True)


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == "__main__":
    main()
