"""Checkpoint-backed machine player and truthful search summary for the UI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable
from dataclasses import asdict
from time import perf_counter
from .model_info import checkpoint_identity

from p1_representation import ACTION_SPACE_SIZE, action_to_move, legal_action_mask, move_to_action
from p2_neural_network import NetworkEvaluator, load_checkpoint
from p3_mcts import MCTS, MCTSConfig
from shared.game_core import GameState, Move, legal_moves, square_name


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "p4_smoke" / "latest_model.pt"


def configured_checkpoint() -> Path | None:
    """Use the selected model, or the included smoke model if present."""
    specified = os.environ.get("CHATURANGZERO_CHECKPOINT")
    if specified:
        path = Path(specified).expanduser()
        return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    return DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT.is_file() else None


def _move_dict(move: Move) -> dict[str, str | int]:
    return {
        "from": square_name(move.from_square),
        "to": square_name(move.to_square),
        "actionId": move_to_action(move),
    }


class MachinePlayer:
    """Load a P4 checkpoint through P2 and choose core moves through P3."""

    def __init__(self, checkpoint: Path, simulations: int = 32) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Machine checkpoint not found: {checkpoint}")
        self.checkpoint = checkpoint.resolve()
        self.evaluator = NetworkEvaluator(load_checkpoint(self.checkpoint, device="cpu"))
        self.config = MCTSConfig(
            num_simulations=simulations, temperature=0.0, add_noise=False
        )
        self.label = str(self.checkpoint.relative_to(PROJECT_ROOT)) if self.checkpoint.is_relative_to(PROJECT_ROOT) else self.checkpoint.name
        self.is_smoke = self.checkpoint.parent.name == "p4_smoke"
        digest, training = checkpoint_identity(self.checkpoint)
        self.engine_info = {
            'checkpoint': self.label, 'sha256': digest, 'inferenceDevice': 'cpu',
            'simulations': self.config.num_simulations, 'cPuct': self.config.c_puct,
            'model': asdict(self.evaluator.model.config), 'training': training,
        }

    def choose(
        self,
        state: GameState,
        on_progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> tuple[Move, dict[str, Any]]:
        """Return a legal selected move and measurements from this exact search."""
        started = perf_counter()
        search_started = None
        def report(stage: str, **data: Any) -> None:
            if search_started is not None:
                data['searchElapsedSeconds'] = perf_counter() - search_started
            if on_progress is not None:
                on_progress(stage, data)

        report("state")
        moves = legal_moves(state)
        if not moves:
            raise ValueError("Cannot search a position without legal moves")
        mask = legal_action_mask(state)
        report("legal", legalActionCount=int(mask.sum()))
        report("network")
        evaluation = self.evaluator.evaluate(state)
        report("network_ready", value=evaluation.value)
        search = MCTS(self.evaluator, self.config)
        report("mcts", totalSimulations=self.config.num_simulations)
        search_started = perf_counter()
        _, selected_move, selected_id = search.search(
            state,
            on_simulation=lambda count: report("mcts", simulationsCompleted=count),
        )
        search_seconds = perf_counter() - search_started
        report("search_complete", simulationsCompleted=self.config.num_simulations)
        root = search.root
        if root is None:  # pragma: no cover - MCTS establishes a root
            raise RuntimeError("MCTS did not produce a search root")

        priors = sorted(moves, key=lambda move: (-float(evaluation.policy[move_to_action(move)]), move_to_action(move)))[:5]
        candidates = sorted(root.children.items(), key=lambda item: (-item[1].visit_count, item[0]))[:5]
        selected_node = root.children[selected_id]
        analysis: dict[str, Any] = {
            "checkpoint": self.label,
            "smokeCheckpoint": self.is_smoke,
            "sideToMove": state.side_to_move.value,
            "actionSpaceSize": ACTION_SPACE_SIZE,
            "legalActionCount": int(mask.sum()),
            "value": evaluation.value,
            "simulations": self.config.num_simulations,
            "cPuct": self.config.c_puct,
            "rootVisits": root.visit_count,
            "searchElapsedSeconds": search_seconds,
            "simulationsPerSecond": self.config.num_simulations / search_seconds if search_seconds else 0.,
            "pipelineElapsedSeconds": perf_counter() - started,
            "policyPriors": [
                {**_move_dict(move), "prior": float(evaluation.policy[move_to_action(move)])}
                for move in priors
            ],
            "candidates": [
                {**_move_dict(action_to_move(action_id)), "visits": child.visit_count, "prior": child.prior}
                for action_id, child in candidates
            ],
            "selectedAction": {
                **_move_dict(selected_move),
                "visits": selected_node.visit_count,
                "prior": selected_node.prior,
            },
        }
        report("selected", selectedAction=analysis["selectedAction"])
        return selected_move, analysis
