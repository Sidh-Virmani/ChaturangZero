from __future__ import annotations

import copy
import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from shared.game_core import GameState, Piece, PieceKind, Side, initial_state, legal_moves, apply_move, is_in_check
from p1_representation import encode_state, legal_action_mask, reflect_example
from p2_neural_network import PolicyValueNet, NeuralEvaluator, save_checkpoint, load_checkpoint
from p3_mcts import MCTS, terminal_value

ROOT = Path(__file__).resolve().parents[1]


class ReplayBuffer:
    def __init__(self, capacity=12000):
        self.capacity = capacity
        self.examples = []

    def extend(self, examples):
        self.examples.extend(examples)
        self.examples = self.examples[-self.capacity:]

    def save(self, path):
        if not self.examples:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        states, policies, values, masks = map(np.array, zip(*self.examples))
        np.savez_compressed(path, states=states, policies=policies, values=values, masks=masks)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.extend(list(zip(data['states'], data['policies'], data['values'], data['masks'])))


def curriculum_position(seed: int):
    """Random legal two-chariot endgames, generated without expert move targets.

    Curriculum games teach terminal rewards faster than long opening draws.
    This is explicitly separate from full-start evaluation.
    """
    rng = np.random.default_rng(seed)
    strong = Side.WHITE if seed % 2 == 0 else Side.BLACK
    for _ in range(1000):
        squares = [(int(i % 8), int(i // 8)) for i in rng.choice(64, 4, replace=False)]
        pieces = (Piece('strong-raja', PieceKind.RAJA, strong, squares[0]),
                  Piece('weak-raja', PieceKind.RAJA, strong.opponent, squares[1]),
                  Piece('ratha-1', PieceKind.RATHA, strong, squares[2]),
                  Piece('ratha-2', PieceKind.RATHA, strong, squares[3]))
        state = GameState(pieces, strong)
        if not is_in_check(state, strong) and not is_in_check(state, strong.opponent):
            return state
    raise RuntimeError('Could not sample a legal curriculum position')


def self_play(model, simulations=16, max_plies=100, seed=0, curriculum=False):
    rng = np.random.default_rng(seed)
    state = curriculum_position(seed) if curriculum else initial_state()
    searcher = MCTS(NeuralEvaluator(model), simulations, seed=seed)
    records, moves_text = [], []
    outcome, truncated = None, False
    for ply in range(max_plies):
        moves = legal_moves(state)
        outcome = terminal_value(state, moves)
        if outcome is not None:
            break
        search = searcher.search(state, add_noise=True)
        policy = search.policy(1.0)
        records.append((encode_state(state), policy, state.side_to_move, legal_action_mask(state, moves)))
        move = search.choose(1.0 if ply < 12 else 0.25, rng)
        moves_text.append(move.to_notation())
        state = apply_move(state, move)
    if outcome is None:
        outcome = terminal_value(state, legal_moves(state))
    if outcome is None:
        outcome, truncated = 0.0, True
    examples = []
    for encoded, policy, side, mask in records:
        value = outcome if side is state.side_to_move else -outcome
        examples.append((encoded, policy, value, mask))
        reflected, reflected_policy = reflect_example(encoded, policy)
        _, reflected_mask = reflect_example(encoded, mask)
        examples.append((reflected, reflected_policy, value, reflected_mask))
    winner = None if outcome == 0 else (state.side_to_move if outcome > 0 else state.side_to_move.opponent).value
    return examples, {'plies': len(records), 'winner': winner, 'truncated': truncated,
                      'source': 'curriculum' if curriculum else 'full-start', 'moves': moves_text}


def train_model(model, replay, epochs=4, batch_size=64, seed=0, device='cpu'):
    if not replay.examples:
        raise ValueError('Replay buffer is empty')
    rng = np.random.default_rng(seed)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(replay.examples))
        totals = np.zeros(3)
        count = 0
        for start in range(0, len(order), batch_size):
            batch = [replay.examples[i] for i in order[start:start + batch_size]]
            states, targets, values, masks = map(np.array, zip(*batch))
            states = torch.tensor(states, dtype=torch.float32, device=device)
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            values = torch.tensor(values, dtype=torch.float32, device=device)
            masks = torch.tensor(masks, dtype=torch.bool, device=device)
            logits, predictions = model(states)
            logits = logits.masked_fill(~masks, -1e9)
            policy_loss = -(targets * F.log_softmax(logits, dim=1)).sum(1).mean()
            value_loss = F.mse_loss(predictions, values)
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals += np.array([loss.item(), policy_loss.item(), value_loss.item()]) * len(batch)
            count += len(batch)
        totals /= count
        history.append({'epoch': epoch + 1, 'loss': float(totals[0]),
                        'policyLoss': float(totals[1]), 'valueLoss': float(totals[2])})
    return model.cpu().eval(), history


def evaluate(candidate, opponent=None, games=8, simulations=16, max_plies=100, seed=10000, curriculum=False):
    """Paired seeds/colors. Draw score is 0.5; capped games are counted explicitly."""
    result = {'wins': 0, 'draws': 0, 'losses': 0, 'truncated': 0, 'games': games,
              'gameResults': [],
              'opponent': 'random' if opponent is None else 'previous checkpoint',
              'scope': 'two-chariot endgames' if curriculum else 'full board after 8 seeded opening plies',
              'simulations': simulations, 'maxPlies': max_plies, 'seed': seed}
    for game in range(games):
        candidate_side = Side.WHITE if game % 2 == 0 else Side.BLACK
        state = curriculum_position(seed + game // 2) if curriculum else initial_state()
        rng = np.random.default_rng(seed + game // 2)
        if not curriculum:
            # Each pair shares an opening and swaps model colors. Without this,
            # deterministic temperature-zero play repeats the same two games.
            for _ in range(8):
                opening_moves = legal_moves(state)
                if terminal_value(state, opening_moves) is not None:
                    break
                state = apply_move(state, opening_moves[int(rng.integers(len(opening_moves)))])
        candidate_search = MCTS(NeuralEvaluator(candidate), simulations, seed=seed + game)
        opponent_search = MCTS(NeuralEvaluator(opponent), simulations, seed=seed + game) if opponent is not None else None
        terminal = None
        played = 0
        for _ in range(max_plies):
            moves = legal_moves(state)
            terminal = terminal_value(state, moves)
            if terminal is not None:
                break
            if state.side_to_move is candidate_side:
                move = candidate_search.search(state).choose(0, rng)
            elif opponent_search:
                move = opponent_search.search(state).choose(0, rng)
            else:
                move = moves[int(rng.integers(len(moves)))]
            state = apply_move(state, move)
            played += 1
        if terminal is None:
            terminal = terminal_value(state, legal_moves(state))
        capped = terminal is None
        if terminal is None:
            terminal = 0.0
            result['truncated'] += 1
        score = terminal if state.side_to_move is candidate_side else -terminal
        result['wins' if score > 0 else 'losses' if score < 0 else 'draws'] += 1
        result['gameResults'].append({'pair': game // 2, 'candidateSide': candidate_side.value,
                                      'score': (score + 1) / 2, 'plies': played, 'truncated': capped})
    result['score'] = (result['wins'] + result['draws'] * 0.5) / max(games, 1)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def _run_training(rounds=1, games=6, epochs=4, simulations=16, max_plies=80,
                  seed=42, curriculum=True, output_dir=ROOT, progress=None, arena_games=4):
    output_dir = Path(output_dir)
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint = output_dir / 'checkpoints' / 'best.pt'
    if checkpoint.exists():
        best, previous_meta = load_checkpoint(checkpoint)
    else:
        best, previous_meta = PolicyValueNet(), {}
    candidate = copy.deepcopy(best)
    replay = ReplayBuffer()
    replay_path = output_dir / 'data' / 'replay.npz'
    if replay_path.exists():
        replay.load(replay_path)
    metrics_path = output_dir / 'data' / 'training_metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {'runs': [], 'epochs': [], 'games': []}
    run_id = max((r['id'] for r in metrics['runs']), default=0) + 1
    started = time.perf_counter()
    def emit(phase, detail, fraction):
        if progress:
            progress({'phase': phase, 'detail': detail, 'progress': fraction})
        print(f'[{phase}] {detail}', flush=True)
    for round_index in range(rounds):
        new_games = []
        for game in range(games):
            emit('self-play', f'Round {round_index + 1}/{rounds}, game {game + 1}/{games}', game / games * 0.65)
            examples, summary = self_play(best, simulations, max_plies, seed + run_id * 100 + round_index * games + game, curriculum)
            replay.extend(examples)
            new_games.append(summary)
        emit('training', f'Optimizing {len(replay.examples)} replay examples', 0.7)
        candidate, losses = train_model(candidate, replay, epochs, seed=seed + round_index)
        for entry in losses:
            entry['step'] = len(metrics['epochs']) + 1
            metrics['epochs'].append(entry)
        emit('evaluation', 'Paired evaluation against the previous network', 0.85)
        evaluation_seed = seed + run_id * 1000 + round_index * 100
        arena = evaluate(candidate, best, arena_games, simulations, max_plies, evaluation_seed + 50000, curriculum)
        baseline = evaluate(candidate, None, arena_games, simulations, max_plies, evaluation_seed + 90000, curriculum)
        first_checkpoint = not checkpoint.exists()
        promoted = first_checkpoint or arena['score'] > 0.55
        metadata = {'name': f'CZ-{run_id:03d}.{round_index + 1}', 'trained': True,
                    'createdAt': datetime.now(timezone.utc).isoformat(),
                    'examples': len(replay.examples), 'trainingGames': len(metrics['games']) + len(new_games),
                    'epochs': len(metrics['epochs']), 'curriculum': curriculum,
                    'seed': seed, 'arena': arena, 'baseline': baseline,
                    'searchVersion': 2,
                    'limitations': 'Small CPU pilot. Strength and improvement are not established by this sample.'}
        save_checkpoint(output_dir / 'checkpoints' / 'candidate.pt', candidate, metadata)
        if promoted:
            best = copy.deepcopy(candidate)
            save_checkpoint(checkpoint, best, metadata)
            previous_meta = metadata
        metrics['games'].extend(new_games)
        metrics['runs'].append({'id': run_id, 'round': round_index + 1, 'promoted': promoted,
                                'bootstrap': first_checkpoint, 'arena': arena, 'baseline': baseline,
                                'seconds': round(time.perf_counter() - started, 2), 'metadata': metadata})
        metrics['model'] = previous_meta
        metrics['replayExamples'] = len(replay.examples)
        metrics['latestCandidate'] = metadata
        metrics['updatedAt'] = datetime.now(timezone.utc).isoformat()
        write_json(metrics_path, metrics)
        replay.save(replay_path)
    emit('complete', 'Checkpoint, replay data, and measured metrics saved', 1)
    return metrics


def run_training(*args, output_dir=ROOT, **kwargs):
    """One writer per project, including separate CLI and web processes."""
    lock = Path(output_dir) / 'data' / 'training.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Training is already locked at {lock}. If an earlier process crashed, verify it has stopped before removing this lock.')
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(json.dumps({'pid': os.getpid(), 'startedAt': datetime.now(timezone.utc).isoformat()}))
        return _run_training(*args, output_dir=output_dir, **kwargs)
    finally:
        lock.unlink(missing_ok=True)
