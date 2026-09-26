"""Run with: python -m p4_training.train --games 12 --epochs 6."""
import argparse
from .pipeline import run_training

def main():
    parser = argparse.ArgumentParser(description='ChaturangZero self-play training')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--games', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--simulations', type=int, default=16)
    parser.add_argument('--max-plies', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arena-games', type=int, default=4)
    parser.add_argument('--full-start', action='store_true', help='Use complete initial boards instead of endgame curriculum')
    args = parser.parse_args()
    for name in ('rounds', 'games', 'epochs', 'simulations', 'max_plies', 'arena_games'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    run_training(rounds=args.rounds, games=args.games, epochs=args.epochs, simulations=args.simulations,
                 max_plies=args.max_plies, seed=args.seed, curriculum=not args.full_start, arena_games=args.arena_games)

if __name__ == '__main__':
    main()
