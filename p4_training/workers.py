"""Independent CPU self-play processes; parent alone owns replay and checkpoints."""
from __future__ import annotations

import multiprocessing as mp
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from p2_neural_network import NetworkEvaluator, PolicyValueNetwork
from shared.game_core import GameState
from .config import TrainingConfig
from .self_play import SelfPlayResult, play_game

_evaluator = None
_config = None


def game_seed(seed: int, iteration: int, index: int) -> int:
    """Version 1 seed assignment is independent of worker count/completion order."""
    return int(np.random.SeedSequence([seed, iteration, index, 1]).generate_state(1)[0])


def _initialize(model_config, weights, config):
    global _evaluator, _config
    torch.set_num_threads(config.worker_threads)
    torch.set_num_interop_threads(1)
    model = PolicyValueNetwork(model_config)
    model.load_state_dict(weights)
    model.eval()
    _evaluator, _config = NetworkEvaluator(model, 'cpu'), config


def _task(job):
    index, seed, state = job
    np.random.seed(seed)
    return index, play_game(_evaluator, _config, state)


def generate_games(model: PolicyValueNetwork, config: TrainingConfig, iteration: int,
                   start_state: GameState | None = None,
                   on_game: Callable[[int, SelfPlayResult], None] | None = None,
                   on_wait: Callable[[], None] | None = None) -> tuple[SelfPlayResult, ...]:
    """All-or-nothing collection, in game-index order. Exceptions abort the pool."""
    if not config.seeded_games:
        raise ValueError('generate_games requires the seeded game protocol')
    jobs = [(i, game_seed(config.seed, iteration, i), start_state)
            for i in range(config.games_per_iteration)]
    # An immutable CPU weight snapshot; workers never access the live optimizer/model.
    weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if config.self_play_workers == 1:
        numpy_state, threads = np.random.get_state(), torch.get_num_threads()
        with torch.random.fork_rng(devices=[]):
            copy = PolicyValueNetwork(model.config)
        copy.load_state_dict(weights)
        evaluator = NetworkEvaluator(copy, config.self_play_device)
        results = []
        try:
            torch.set_num_threads(config.worker_threads)
            for index, seed, state in jobs:
                np.random.seed(seed)
                result = play_game(evaluator, config, state)
                results.append(result)
                if on_game:
                    on_game(index, result)
        finally:
            np.random.set_state(numpy_state)
            torch.set_num_threads(threads)
        return tuple(results)
    results = {}
    # Pool's context terminates and joins children on interruption, worker error,
    # or a stalled/dead worker. No partial iteration reaches training/save.
    with mp.get_context('spawn').Pool(
        min(config.self_play_workers, len(jobs)), initializer=_initialize,
        initargs=(model.config, weights, config),
    ) as pool:
        iterator = pool.imap_unordered(_task, jobs, chunksize=1)
        last_result = perf_counter()
        while len(results) < len(jobs):
            try:
                index, result = iterator.next(timeout=min(15, config.worker_timeout_seconds))
            except mp.TimeoutError:
                if perf_counter() - last_result >= config.worker_timeout_seconds:
                    raise TimeoutError('Self-play workers stalled; iteration not committed. Resume the last checkpoint.')
                if on_wait:
                    on_wait()
                continue
            results[index] = result
            last_result = perf_counter()
            if on_game:
                on_game(index, result)
    return tuple(results[i] for i in range(len(jobs)))


def game_record(game: SelfPlayResult, config: TrainingConfig, iteration: int, index: int) -> dict:
    return {'game': index, 'seed': game_seed(config.seed, iteration, index) if config.seeded_games else None,
            'plies': game.plies, 'examples': len(game.examples),
            'winner': game.outcome.winner.value if game.outcome and game.outcome.winner else None,
            'reason': game.outcome.reason.value if game.outcome else 'truncation',
            'moves': list(game.moves), 'mean_policy_entropy': game.mean_policy_entropy,
            'mean_visited_actions': game.mean_visited_actions,
            'elapsed_seconds': game.elapsed_seconds, 'cpu_seconds': game.cpu_seconds}
