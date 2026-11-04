import json

import torch

from benchmarks.baseline import (
    BenchmarkConfig,
    fixed_positions,
    run_baseline,
    write_json,
)
from p2_neural_network import ModelConfig, PolicyValueNetwork, save_checkpoint
from shared.game_core import legal_moves, terminal_status


def _tiny_checkpoint(path):
    torch.manual_seed(41)
    model = PolicyValueNetwork(
        ModelConfig(channels=8, residual_blocks=1, policy_channels=2, value_hidden=16)
    )
    save_checkpoint(model, path)


def test_fixed_benchmark_positions_are_valid_and_reproducible():
    first = fixed_positions()
    second = fixed_positions()

    assert [item.name for item in first] == [
        "initial", "midgame_10_ply", "simple_endgame"
    ]
    assert first == second
    assert [item.state.ply_count for item in first] == [0, 10, 0]
    assert all(legal_moves(item.state) for item in first)
    assert all(not terminal_status(item.state).terminal for item in first)


def test_quick_benchmark_is_structured_and_preserves_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    _tiny_checkpoint(checkpoint)
    before = checkpoint.read_bytes()

    report = run_baseline(BenchmarkConfig.quick(checkpoint))

    assert report["schema_version"] == 1
    assert report["benchmark"] == "phase_0_baseline"
    assert report["checkpoint"]["trainable_parameters"] > 0
    assert len(report["workload"]["positions"]) == 3
    assert report["metrics"]
    assert all(metric["value"] >= 0 for metric in report["metrics"])
    assert checkpoint.read_bytes() == before
    assert list(tmp_path.glob("*.pt")) == [checkpoint]

    output = tmp_path / "result.json"
    write_json(report, output)
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["checkpoint"]["sha256"] == report["checkpoint"]["sha256"]
