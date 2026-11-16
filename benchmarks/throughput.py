"""Bounded hardware and self-play experiments; never writes input checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter
import subprocess
from threading import Event, Thread

import numpy as np
import torch

from p1_representation import encode_state, legal_action_mask
from p2_neural_network import load_checkpoint, policy_value_loss, NetworkEvaluator
from p3_mcts import MCTS, MCTSConfig
from shared.game_core import initial_state
from p4_training.recording import file_sha256, provenance


def network_measurements(checkpoint, device, threads, repeats=3):
    torch.set_num_threads(threads)
    model = load_checkpoint(checkpoint, device)
    state = initial_state()
    encoded = encode_state(state)
    mask = legal_action_mask(state).astype(np.float32)
    mask /= mask.sum()
    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize()
    def measure(fn, count):
        for _ in range(5):
            fn()
        sync()
        samples = []
        for _ in range(repeats):
            start = perf_counter()
            for _ in range(count):
                fn()
            sync()
            samples.append((perf_counter() - start) / count)
        return statistics.median(samples)
    rows = []
    for batch in (1, 16, 64, 128, 256):
        x = torch.from_numpy(np.repeat(encoded[None], batch, axis=0)).to(device)
        p = torch.from_numpy(np.repeat(mask[None], batch, axis=0)).to(device)
        v = torch.zeros(batch, device=device)
        model.eval()
        with torch.inference_mode():
            seconds = measure(lambda: model(x), 20)
        row = {'batch': batch, 'inference_ms': seconds * 1000,
               'inference_states_per_second': batch / seconds}
        if batch >= 16:
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            model.train()
            def step():
                optimizer.zero_grad(set_to_none=True)
                logits, values = model(x)
                loss = policy_value_loss(logits, values, p, v)
                loss.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if not torch.isfinite(loss.total):
                    raise FloatingPointError('Nonfinite benchmark loss')
            seconds = measure(step, 10)
            row.update(training_ms=seconds * 1000, training_examples_per_second=batch / seconds)
        rows.append(row)
    # Reload original weights before search (the training benchmark modified its copy).
    evaluator = NetworkEvaluator(load_checkpoint(checkpoint, device), device)
    searches = []
    for sims in (25, 50, 100):
        config = MCTSConfig(num_simulations=sims, temperature=0)
        seconds = measure(lambda: MCTS(evaluator, config).search(state), 2)
        searches.append({'simulations': sims, 'seconds': seconds, 'simulations_per_second': sims / seconds})
    return {'network': rows, 'search': searches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['network', 'self-play'])
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/benchmark/hard.pt'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--simulations', type=int, default=25)
    parser.add_argument('--max-plies', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20261115)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file')
    torch.set_num_threads(args.threads)
    report = {'config': {**vars(args), 'checkpoint': str(args.checkpoint), 'output': str(args.output)},
              'environment': provenance(args.device), 'checkpoint_sha256': file_sha256(args.checkpoint)}
    monitor_stop, gpu_samples = Event(), []
    def monitor_gpu():
        while not monitor_stop.is_set():
            try:
                values = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'], timeout=5).decode().strip().split(',')
                gpu_samples.append([float(value) for value in values])
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            monitor_stop.wait(1)
    monitor = Thread(target=monitor_gpu, daemon=True)
    if args.device.startswith('cuda'):
        monitor.start()
    if args.mode == 'network':
        report.update(network_measurements(args.checkpoint, args.device, args.threads))
    else:
        from p4_training.config import TrainingConfig
        from p4_training.workers import generate_games, game_record
        model = load_checkpoint(args.checkpoint)
        config = TrainingConfig(model=model.config, games_per_iteration=args.games,
            simulations_per_move=args.simulations, max_game_plies=args.max_plies, seed=args.seed,
            self_play_workers=args.workers, worker_threads=args.threads, self_play_device=args.device)
        start = perf_counter()
        results = generate_games(model, config, 1, on_game=lambda i, result: print(
            f'game={i} plies={result.plies} examples={len(result.examples)}', flush=True))
        seconds = perf_counter() - start
        completed = sum(not game.truncated for game in results)
        examples = sum(len(game.examples) for game in results)
        plies = sum(game.plies for game in results)
        report.update(seconds=seconds, completed_games=completed, examples=examples,
            completed_games_per_hour=completed * 3600 / seconds,
            examples_per_hour=examples * 3600 / seconds,
            plies_per_second=plies / seconds, simulations_per_second=plies * args.simulations / seconds,
            average_busy_cpu_cores=sum(game.cpu_seconds for game in results) / seconds,
            games=[game_record(game, config, 1, i) for i, game in enumerate(results)])
    monitor_stop.set()
    if monitor.is_alive():
        monitor.join(timeout=6)
    if gpu_samples:
        report['gpu_samples'] = {'columns': ['utilization_percent', 'memory_mib', 'temperature_c', 'power_w'],
                                 'samples': gpu_samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('environment', 'games')}, indent=2))


if __name__ == '__main__':
    main()
