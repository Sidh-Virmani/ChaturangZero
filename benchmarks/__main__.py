"""Command-line entry point for repository benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .baseline import BenchmarkConfig, print_report, run_baseline, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChaturangZero performance benchmarks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser(
        "baseline", help="measure the fixed Phase 0 production workload"
    )
    baseline.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/benchmark/hard.pt"),
        help="P2 model checkpoint to evaluate (default: checkpoints/benchmark/hard.pt)",
    )
    baseline.add_argument("--device", default="cpu", help="inference device")
    baseline.add_argument("--json", type=Path, help="optional JSON result path")
    baseline.add_argument(
        "--quick",
        action="store_true",
        help="minimal smoke workload for validating the command, not for comparisons",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "baseline":  # pragma: no cover - argparse owns this invariant
        raise AssertionError(f"Unhandled benchmark command: {args.command}")

    config = BenchmarkConfig.quick(args.checkpoint, args.device) if args.quick else BenchmarkConfig(
        checkpoint=args.checkpoint,
        device=args.device,
    )
    report = run_baseline(config)
    print_report(report)
    if args.json is not None:
        write_json(report, args.json)
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a real command
    raise SystemExit(main())
