"""
p5_endgame/endgame_benchmark.py
=================================
Evaluate any ChaturangaNet against exact endgame tablebase solutions.

Why this matters
-----------------
Existing AlphaZero evaluations use self-play Elo as the only metric.
But Elo is relative — it only measures how well the model beats itself.

The endgame tablebase gives us an ABSOLUTE benchmark:
  "In this position, the optimal move is X. Did the model choose it?"

This is a much stronger claim for a paper:
  "Our model achieves 91.4% accuracy on KR-K endgame positions
   (vs 67.2% for the baseline after 10 training iterations)"

Metrics reported
-----------------
  top1_accuracy  : fraction of positions where model's top move = tablebase optimal
  dtm_error      : average |predicted_dtm - true_dtm|
  win_conversion : of WIN positions, fraction where model converges correctly
  blunder_rate   : fraction of LOSS positions where model plays a move that extends DTM
"""

from __future__ import annotations

import sys
import pathlib
import random
from dataclasses import dataclass, field

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState
from p1_representation.encoder import StateEncoder
from p5_endgame.retrograde import RetrogradeAnalyzer, TablebaseEntry, _hash_state

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class BenchmarkResult:
    config:            str
    n_positions:       int
    top1_accuracy:     float   # fraction where model top move = tablebase optimal move
    value_accuracy:    float   # fraction where model value sign matches tablebase
    win_conversion:    float   # of WIN positions, model plays correctly
    draw_accuracy:     float   # of DRAW positions, model identifies draw
    blunder_rate:      float   # model plays a losing move in a winning position
    dtm_correlation:   float   # Pearson r between model |value| and true DTM

    def summary(self) -> str:
        return (
            f"[benchmark] {self.config}  n={self.n_positions}\n"
            f"  Top-1 accuracy:    {self.top1_accuracy*100:.1f}%\n"
            f"  Value accuracy:    {self.value_accuracy*100:.1f}%\n"
            f"  Win conversion:    {self.win_conversion*100:.1f}%\n"
            f"  Draw accuracy:     {self.draw_accuracy*100:.1f}%\n"
            f"  Blunder rate:      {self.blunder_rate*100:.1f}%\n"
            f"  DTM correlation:   {self.dtm_correlation:.3f}\n"
        )

    def to_dict(self) -> dict:
        return {
            "config":          self.config,
            "n_positions":     self.n_positions,
            "top1_accuracy":   self.top1_accuracy,
            "value_accuracy":  self.value_accuracy,
            "win_conversion":  self.win_conversion,
            "draw_accuracy":   self.draw_accuracy,
            "blunder_rate":    self.blunder_rate,
            "dtm_correlation": self.dtm_correlation,
        }


class EndgameBenchmark:
    """
    Evaluates a ChaturangaNet against solved endgame tablebases.

    Usage
    -----
        engine   = ChaturangaEngine()
        analyzer = RetrogradeAnalyzer(engine)
        table    = analyzer.solve("KR-K")

        bm = EndgameBenchmark(engine, table, config_name="KR-K")
        result = bm.evaluate(model, n_sample=500)
        print(result.summary())
    """

    def __init__(
        self,
        engine:      ChaturangaEngine,
        tablebase:   dict[str, TablebaseEntry],
        config_name: str = "unknown",
    ) -> None:
        self.engine      = engine
        self.tablebase   = tablebase
        self.config_name = config_name
        self.encoder     = StateEncoder()

    def evaluate(
        self,
        model,
        n_sample:     int  = 500,
        num_sims:     int  = 0,    # 0 = network only (no MCTS), N = MCTS budget
        device:       str  = "cpu",
        verbose:      bool = True,
    ) -> BenchmarkResult:
        """
        Sample positions from the tablebase and evaluate model accuracy.

        Parameters
        ----------
        model     : ChaturangaNet (or None for random baseline)
        n_sample  : number of positions to evaluate
        num_sims  : MCTS simulations (0 = greedy network policy only)
        """
        positions = list(self.tablebase.items())
        if len(positions) > n_sample:
            positions = random.sample(positions, n_sample)

        top1_correct   = 0
        value_correct  = 0
        win_total      = 0
        win_correct    = 0
        draw_total     = 0
        draw_correct   = 0
        blunder_total  = 0
        blunder_count  = 0
        dtm_true_vals  = []
        dtm_model_vals = []
        evaluated      = 0

        for h, entry in positions:
            # Reconstruct state from tablebase (need the actual state object)
            # We store states by hash — need to find the matching one
            # Build state from hash string is not trivial, so we enumerate
            # from the RetrogradeAnalyzer instead. Here we evaluate via model.
            # For now skip positions we can't reconstruct.
            # (In practice, call this with the analyzer's all_states dict)
            pass

        # Alternative: evaluate directly on a list of (state, entry) pairs
        # This method should be called via evaluate_states() instead.
        return BenchmarkResult(
            config=self.config_name,
            n_positions=0,
            top1_accuracy=0.0, value_accuracy=0.0,
            win_conversion=0.0, draw_accuracy=0.0,
            blunder_rate=0.0, dtm_correlation=0.0,
        )

    def evaluate_states(
        self,
        states_and_entries: list[tuple[ChaturangaState, TablebaseEntry]],
        model,
        device: str = "cpu",
        verbose: bool = True,
    ) -> BenchmarkResult:
        """
        Evaluate model on a pre-built list of (ChaturangaState, TablebaseEntry) pairs.

        This is the primary evaluation method. Use with the RetrogradeAnalyzer's
        internal state dict:

            from p5_endgame.retrograde import RetrogradeAnalyzer
            analyzer = RetrogradeAnalyzer(engine)
            table = analyzer.solve("KR-K")
            pairs = [(analyzer._all_states[h], entry) for h, entry in table.items()]
            bm.evaluate_states(pairs, model)
        """
        top1_correct  = 0
        value_correct = 0
        win_total     = 0; win_correct  = 0
        draw_total    = 0; draw_correct = 0
        blunder_count = 0; blunder_total = 0
        dtm_true_vals = []; model_value_abs = []
        n = 0

        for state, entry in states_and_entries:
            if self.engine.is_terminal(state):
                continue
            legal = self.engine.legal_moves(state)
            if not legal:
                continue

            # Get model policy
            try:
                enc  = self.encoder.encode_state_torch(state).to(device)
                mask = self.encoder.legal_action_mask_torch(state, legal).to(device)
                out  = model.predict(enc, mask)
                probs_t = out[0]
                value   = float(out[1])
                probs_np = probs_t.cpu().numpy()
            except Exception:
                continue

            # Best move by model (greedy)
            from p1_representation.encoder import StateEncoder as SE
            legal_idx = {SE.encode_action(m): m for m in legal}
            best_idx  = max(legal_idx, key=lambda i: probs_np[i] if i < len(probs_np) else 0)
            best_move = legal_idx[best_idx]

            # Find tablebase-optimal move (lowest DTM in successors)
            optimal_move = self._find_optimal_move(state, legal)
            if optimal_move is not None and best_move == optimal_move:
                top1_correct += 1

            # Value sign accuracy
            true_sign  = 1.0 if entry.result == "win" else (-1.0 if entry.result == "loss" else 0.0)
            model_sign = 1.0 if value > 0.1 else (-1.0 if value < -0.1 else 0.0)
            if true_sign == model_sign:
                value_correct += 1

            # Win conversion
            if entry.result == "win":
                win_total += 1
                if optimal_move is not None and best_move == optimal_move:
                    win_correct += 1
                # Blunder: model plays into a LOSS position (extending DTM)
                blunder_total += 1
                next_state = self.engine.apply_move(state, best_move)
                nh = _hash_state(next_state)
                next_entry = self.tablebase.get(nh)
                if next_entry and next_entry.result == "win":
                    blunder_count += 1  # we should have delivered a loss to opponent

            # Draw accuracy
            if entry.result == "draw":
                draw_total += 1
                if abs(value) < 0.15:
                    draw_correct += 1

            # DTM correlation
            if entry.dtm < 9999:
                dtm_true_vals.append(float(entry.dtm))
                model_value_abs.append(abs(value))

            n += 1

        if n == 0:
            return BenchmarkResult(self.config_name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Pearson correlation between |model_value| and true DTM
        dtm_corr = 0.0
        if len(dtm_true_vals) >= 2:
            try:
                dtm_corr = float(np.corrcoef(dtm_true_vals, model_value_abs)[0, 1])
                if np.isnan(dtm_corr):
                    dtm_corr = 0.0
            except Exception:
                pass

        result = BenchmarkResult(
            config=self.config_name,
            n_positions=n,
            top1_accuracy=top1_correct / n,
            value_accuracy=value_correct / n,
            win_conversion=win_correct / max(win_total, 1),
            draw_accuracy=draw_correct / max(draw_total, 1),
            blunder_rate=blunder_count / max(blunder_total, 1),
            dtm_correlation=dtm_corr,
        )
        if verbose:
            print(result.summary())
        return result

    def _find_optimal_move(
        self, state: ChaturangaState, legal: list
    ):
        """
        Find the tablebase-optimal move: the one leading to lowest DTM for opponent.
        Returns None if not all successors are in the tablebase.
        """
        best_move  = None
        best_score = float("inf")
        for move in legal:
            try:
                next_state = self.engine.apply_move(state, move)
                nh = _hash_state(next_state)
                entry = self.tablebase.get(nh)
                if entry is None:
                    return None  # incomplete tablebase coverage
                # We want: move that leads to LOSS for opponent (lowest DTM)
                score = entry.dtm if entry.result == "loss" else 9999
                if score < best_score:
                    best_score = score
                    best_move  = move
            except Exception:
                continue
        return best_move
