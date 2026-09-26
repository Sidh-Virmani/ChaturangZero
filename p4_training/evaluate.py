"""Held-out, reproducible comparison against random and seeded initial weights.

    python -m p4_training.evaluate --games 24 --simulations 64 --max-plies 200
"""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from p2_neural_network import PolicyValueNet, load_checkpoint
from .pipeline import ROOT, evaluate, write_json


def paired_interval(game_results, seed=817):
    """Bootstrap complete color-swapped pairs, not correlated individual games."""
    pairs = np.array([(game_results[i]['score'] + game_results[i + 1]['score']) / 2
                      for i in range(0, len(game_results) - 1, 2)])
    if not len(pairs):
        return [0.0, 1.0]
    rng = np.random.default_rng(seed)
    means = pairs[rng.integers(0, len(pairs), (5000, len(pairs)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def main():
    parser = argparse.ArgumentParser(description='Evaluate the learned ChaturangZero checkpoint on held-out seeds')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints' / 'best.pt')
    parser.add_argument('--games', type=int, default=24, help='Even number, for color-swapped pairs')
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--max-plies', type=int, default=200)
    parser.add_argument('--seed', type=int, default=713731)
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'evaluation_report.json')
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.simulations < 1 or args.max_plies < 1:
        parser.error('Use an even positive --games value and positive search/game budgets.')
    candidate, metadata = load_checkpoint(args.checkpoint)
    torch.manual_seed(42)
    initial = PolicyValueNet(**candidate.config).eval()
    report = {'createdAt': datetime.now(timezone.utc).isoformat(), 'checkpoint': args.checkpoint.name,
              'sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), 'model': metadata,
              'seed': args.seed, 'initialWeightsSeed': 42, 'searchVersion': 2,
              'gamesPerComparison': args.games, 'simulations': args.simulations, 'maxPlies': args.max_plies,
              'comparisons': [],
              'interpretation': 'Scores count draws as 0.5. Ply-cap draws are reported separately. Confidence intervals resample color-swapped pairs. Endgame results do not establish full-board strength.'}
    for curriculum in [True, False]:
        scope = 'curriculum' if curriculum else 'full-board'
        seed = args.seed if curriculum else args.seed + 10000
        for name, model, opponent in [('trained vs random', candidate, None),
                                       ('initial vs random', initial, None),
                                       ('trained vs initial', candidate, initial)]:
            print(f'[{scope}] {name}: {args.games} games', flush=True)
            measured = evaluate(model, opponent, args.games, args.simulations, args.max_plies, seed, curriculum)
            measured['name'] = name
            measured['paired95Interval'] = paired_interval(measured['gameResults'])
            report['comparisons'].append(measured)
            write_json(args.output, report)
            print(f"  {measured['wins']}W {measured['draws']}D {measured['losses']}L; score {measured['score']:.3f}; capped {measured['truncated']}", flush=True)
    report['complete'] = True
    write_json(args.output, report)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
