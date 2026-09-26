from __future__ import annotations

import json
import threading
from pathlib import Path
import torch
from p2_neural_network import PolicyValueNet, NeuralEvaluator, load_checkpoint
from p3_mcts import MCTS
from shared.game_core.types import square_to_id

ROOT = Path(__file__).resolve().parents[1]


def node_payload(node, depth=0):
    move = node.move
    payload = {'move': move.to_notation() if move else 'Position', 'visits': node.visits,
               'value': round(-node.value if move else node.value, 4), 'prior': round(node.prior, 5)}
    if depth < 3:
        payload['children'] = [node_payload(child, depth + 1) for child in
                               sorted(node.children, key=lambda n: n.visits, reverse=True)[:5 if depth == 0 else 2]
                               if child.visits > 0]
    return payload


class Engine:
    def __init__(self):
        self.lock = threading.RLock()
        self.stamp = None
        self.model = None
        self.metadata = {}
        self.reload()

    def reload(self):
        with self.lock:
            path = ROOT / 'checkpoints' / 'best.pt'
            stamp = path.stat().st_mtime_ns if path.exists() else 0
            if self.model is not None and stamp == self.stamp:
                return
            if stamp:
                self.model, self.metadata = load_checkpoint(path)
            else:
                torch.manual_seed(42)
                self.model = PolicyValueNet().eval()
                self.metadata = {'name': 'CZ-untrained', 'trained': False,
                                 'limitations': 'Random initial weights. Run training to produce a learned checkpoint.'}
            self.evaluator = NeuralEvaluator(self.model)
            self.stamp = stamp

    def info(self):
        with self.lock:
            self.reload()
            return {**self.metadata, 'parameters': sum(p.numel() for p in self.model.parameters()),
                    'channels': self.model.config['channels'], 'blocks': self.model.config['blocks'],
                    'inputShape': [14, 8, 8], 'actions': 4096, 'device': 'CPU',
                    'architecture': 'Residual policy–value network'}

    def analyze(self, state, simulations=64):
        with self.lock:
            self.reload()
            model_info = self.info()
            search = MCTS(self.evaluator, simulations, seed=42).search(state, time_limit=12)
            inspected = self.evaluator.inspect(state)
            children = sorted(search.root.children, key=lambda n: (n.visits, -n.value), reverse=True)
            candidates = [{'from': square_to_id(n.move.from_sq), 'to': square_to_id(n.move.to_sq),
                           'notation': n.move.to_notation(), 'visits': n.visits, 'prior': n.prior,
                           'probability': n.visits / max(search.simulations, 1),
                           'value': -n.value} for n in children]
            pv, node = [], search.root
            while node.children and len(pv) < 8:
                node = max(node.children, key=lambda n: n.visits)
                if not node.visits:
                    break
                pv.append(node.move.to_notation())
            heatmap = [0.0] * 64
            for child in candidates:
                square = child['to']
                index = (int(square[1]) - 1) * 8 + 'abcdefgh'.index(square[0])
                heatmap[index] += child['probability']
            payload = {**inspected, 'searchValue': search.root.terminal if search.root.terminal is not None else search.root.value,
                       'simulations': search.simulations, 'elapsedMs': round(search.elapsed_ms, 1),
                       'depth': search.depth, 'candidates': candidates, 'tree': node_payload(search.root),
                       'principalVariation': pv, 'policyHeatmap': heatmap,
                       'model': model_info, 'positionPly': len(state.move_history),
                       'positionTurn': state.side_to_move.value,
                       'positionPieces': [{'square': square_to_id(p.square), 'side': p.side.value,
                                           'kind': p.kind.value} for p in state.pieces]}
            return (children[0].move if children else None), payload


class TrainingManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.job = {'running': False, 'phase': 'idle', 'detail': 'Ready for self-play', 'progress': 0}

    def snapshot(self):
        with self.lock:
            job = dict(self.job)
        path = ROOT / 'data' / 'training_metrics.json'
        report = ROOT / 'data' / 'evaluation_report.json'
        return {'job': job, 'metrics': json.loads(path.read_text(encoding='utf-8')) if path.exists() else None,
                'evaluation': json.loads(report.read_text(encoding='utf-8')) if report.exists() else None}

    def start(self):
        with self.lock:
            if self.job['running'] or (ROOT / 'data' / 'training.lock').exists():
                return False
            self.job = {'running': True, 'phase': 'starting', 'detail': 'Preparing training', 'progress': 0}
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self):
        from p4_training.pipeline import run_training
        def progress(update):
            with self.lock:
                self.job.update(update)
        try:
            run_training(games=4, epochs=4, simulations=12, max_plies=60, arena_games=4, progress=progress)
        except Exception as error:
            progress({'phase': 'failed', 'detail': str(error)})
        finally:
            with self.lock:
                self.job['running'] = False
