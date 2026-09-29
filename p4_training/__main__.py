"""Command line entry point: smoke, train, resume, and optional arena."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from p2_neural_network import ModelConfig, NetworkEvaluator, load_checkpoint

from .arena import compare_models
from .config import TrainingConfig
from .self_play import smoke_position
from .trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="ChaturangZero self-play training")
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke", help="One short complete game and update")
    smoke.add_argument("--output", type=Path, default=Path("checkpoints/p4_smoke"))

    train = commands.add_parser("train", help="Start a new run from a random model")
    train.add_argument("--output", type=Path, default=Path("checkpoints/p4_training"))
    train.add_argument("--iterations", type=int, default=1)
    train.add_argument("--games-per-iteration", type=int, default=2)
    train.add_argument("--simulations", type=int, default=25)
    train.add_argument("--max-plies", type=int, default=512)
    train.add_argument("--training-steps", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--replay-capacity", type=int, default=20_000)
    train.add_argument("--opening-plies", type=int, default=12)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--channels", type=int, default=64)
    train.add_argument("--blocks", type=int, default=4)

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

    if args.command == "resume":
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
            config = TrainingConfig(
                iterations=args.iterations,
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
            trainer = Trainer(config)
            metrics = trainer.run()

    for item in metrics:
        print(json.dumps(asdict(item)))
    print(f"model={trainer.config.checkpoint_dir / 'latest_model.pt'}")
    print(f"resume={trainer.config.checkpoint_dir / 'training_state.pt'}")


if __name__ == "__main__":
    main()
