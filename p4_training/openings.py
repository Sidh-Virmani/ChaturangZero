"""Versioned opening suites made only by playing validated core moves."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from random import Random

from shared.game_core import GameState, Move, apply_move, initial_state, legal_moves, terminal_status

DEFAULT_SUITE = Path(__file__).with_name('evaluation_openings.json')


@dataclass(frozen=True)
class Opening:
    id: str
    moves: tuple[str, ...]

    def state(self) -> GameState:
        state = initial_state()
        for uci in self.moves:
            if not isinstance(uci, str) or len(uci) != 4:
                raise ValueError('Opening moves must use four-character square notation')
            state = apply_move(state, Move.from_names(uci[:2], uci[2:]))
        return state


def load_openings(path: Path = DEFAULT_SUITE) -> tuple[Opening, ...]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload['version'] != 1:
        raise ValueError('Unsupported opening suite version')
    openings = tuple(Opening(item['id'], tuple(item['moves'])) for item in payload['openings'])
    validate_openings(openings)
    return openings


def validate_openings(openings: tuple[Opening, ...]) -> None:
    if not openings:
        raise ValueError('Opening suite must not be empty')
    ids, positions = set(), set()
    for opening in openings:
        if not isinstance(opening.id, str) or not opening.id or opening.id in ids:
            raise ValueError('Opening IDs must be unique nonempty strings')
        state = opening.state()
        key = (state.board, state.side_to_move)
        if key in positions or terminal_status(state).terminal:
            raise ValueError('Openings must be distinct nonterminal positions')
        ids.add(opening.id)
        positions.add(key)


def generate_openings(count: int, seed: int) -> dict:
    """Create a model-independent suite; freeze the exported moves before evaluating."""
    if count <= 0 or seed < 0:
        raise ValueError('count must be positive and seed non-negative')
    rng = Random(seed)
    openings = [Opening('initial', ())]
    seen = {(initial_state().board, initial_state().side_to_move)}
    for attempt in range(count * 100):
        if len(openings) == count:
            break
        state, moves = initial_state(), []
        for _ in range(4 + attempt % 9):
            if terminal_status(state).terminal:
                break
            move = rng.choice(sorted(legal_moves(state), key=lambda move: move.uci))
            state = apply_move(state, move)
            moves.append(move.uci)
        key = (state.board, state.side_to_move)
        if key not in seen and not terminal_status(state).terminal:
            openings.append(Opening(f'opening-{len(openings):03d}', tuple(moves)))
            seen.add(key)
    if len(openings) != count:
        raise ValueError('Could not generate enough unique nonterminal openings')
    return {'version': 1, 'generator': 'sorted-legal-uniform-v1', 'seed': seed,
            'description': 'Initial position plus fixed 4–12-ply legal random openings; not expert openings.',
            'openings': [{'id': item.id, 'moves': list(item.moves)} for item in openings]}
