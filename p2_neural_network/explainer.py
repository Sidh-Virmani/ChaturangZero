"""
p2_neural_network/explainer.py
================================
Explainable AI — MoveExplainer (Research Contribution #9).

What it does
------------
Given a position and the move the AI chose, explain in plain English WHY
it chose that move. The explanation uses strategic feature deltas:

  feature_delta = features(state_after) - features(state_before)

Positive delta on a feature = the move improved that strategic concept.

Example output:
  Recommended move: Ratha e1 → e5
  Confidence: 87%   Uncertainty: low
  ✓ improves piece activity (+0.18)
  ✓ improves opponent Raja exposure (+0.14)
  ✓ improves Ratha on open files (+0.12)
  Alternative: Gaja c1 → e3  (visit: 23%,  Q: +0.31)

This output is also returned as structured data via the FastAPI /ai_move
endpoint so the React UI can render it.
"""

from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass, field

import numpy as np

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Move, PieceKind, Side
from p1_representation.features import StrategicFeatureExtractor, FEATURE_NAMES, _FEATURE_LABELS


@dataclass
class AlternativeMove:
    """One alternative move shown in the explanation."""
    notation:       str
    visit_fraction: float    # MCTS visit %
    q_value:        float    # MCTS Q-value


@dataclass
class MoveExplanation:
    """
    Structured explanation of a single AI move.
    Returned by MoveExplainer and serialised to JSON by the API.
    """
    move_notation:   str
    confidence:      float         # visit fraction of chosen move (0–1)
    uncertainty:     float         # value_std from network (0 = certain)
    uncertainty_label: str         # "low" / "medium" / "high"
    reasons:         list[str]     # human-readable reasons (up to 5)
    alternatives:    list[AlternativeMove] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "move":             self.move_notation,
            "confidence":       round(self.confidence * 100, 1),
            "uncertainty":      round(self.uncertainty, 3),
            "uncertaintyLabel": self.uncertainty_label,
            "reasons":          self.reasons,
            "alternatives": [
                {
                    "move":          a.notation,
                    "visitPct":      round(a.visit_fraction * 100, 1),
                    "qValue":        round(a.q_value, 3),
                }
                for a in self.alternatives
            ],
        }

    def __str__(self) -> str:
        lines = [
            f"  Recommended: {self.move_notation}",
            f"  Confidence:  {self.confidence*100:.0f}%   "
            f"Uncertainty: {self.uncertainty_label} ({self.uncertainty:.3f})",
        ]
        for r in self.reasons:
            lines.append(f"  ✓ {r}")
        if self.alternatives:
            lines.append("  Alternatives:")
            for alt in self.alternatives:
                lines.append(
                    f"    {alt.notation}  "
                    f"(visit {alt.visit_fraction*100:.0f}%,  Q={alt.q_value:+.3f})"
                )
        return "\n".join(lines)


class MoveExplainer:
    """
    Generates human-readable explanations for AI moves.

    Parameters
    ----------
    engine       : ChaturangaEngine (needed to compute next states)
    threshold    : minimum |delta| to report a feature change
    max_reasons  : maximum number of reasons to include
    """

    def __init__(
        self,
        engine:       ChaturangaEngine,
        threshold:    float = 0.04,
        max_reasons:  int   = 5,
    ) -> None:
        self.engine      = engine
        self.extractor   = StrategicFeatureExtractor()
        self.threshold   = threshold
        self.max_reasons = max_reasons

    def explain(
        self,
        state:          ChaturangaState,
        chosen_move:    Move,
        top_moves:      list[tuple[int, float, float]],   # (action_idx, visit_frac, Q)
        value_std:      float = 0.0,
    ) -> MoveExplanation:
        """
        Explain `chosen_move` in the context of `state`.

        Parameters
        ----------
        state        : current game state
        chosen_move  : move the AI chose (highest visit count)
        top_moves    : list from MCTSSearch._top_moves()
        value_std    : network's value uncertainty estimate

        Returns
        -------
        MoveExplanation with reasons and alternatives
        """
        from p1_representation.encoder import StateEncoder
        encoder = StateEncoder()

        # Compute feature delta
        try:
            next_state = self.engine.apply_move(state, chosen_move)
            delta = self.extractor.extract_delta(state, next_state, state.current_turn)
            reasons = self.extractor.explain_delta(delta, threshold=self.threshold)
            reasons = reasons[:self.max_reasons]
        except Exception:
            reasons = []

        # Tactical bonuses as extra reasons
        try:
            from p3_mcts.tactical_verifier import TacticalVerifier
            verifier = TacticalVerifier(self.engine)
            tscore = verifier.tactical_score(state, chosen_move)
            if tscore >= 3:
                reasons.insert(0, "delivers checkmate ✓✓✓")
            elif tscore >= 2:
                reasons.insert(0, "promotes Padati to Mantri ✓✓")
            elif tscore >= 1:
                reasons.insert(0, "gives check ✓")
        except Exception:
            pass

        # Confidence from visit fraction of chosen move
        chosen_idx = StateEncoder.encode_action(chosen_move)
        confidence = 0.0
        for idx, visit_frac, _ in top_moves:
            if idx == chosen_idx:
                confidence = visit_frac
                break

        # Uncertainty label
        if value_std < 0.1:
            unc_label = "low"
        elif value_std < 0.3:
            unc_label = "medium"
        else:
            unc_label = "high"

        # Alternatives: top moves OTHER than chosen
        legal = self.engine.legal_moves(state)
        legal_map = {StateEncoder.encode_action(m): m for m in legal}
        alternatives = []
        for idx, visit_frac, q in top_moves:
            if idx == chosen_idx:
                continue
            move_obj = legal_map.get(idx)
            if move_obj is None:
                continue
            alternatives.append(AlternativeMove(
                notation=move_obj.label(),
                visit_fraction=visit_frac,
                q_value=q,
            ))
            if len(alternatives) >= 3:
                break

        return MoveExplanation(
            move_notation=chosen_move.label(),
            confidence=confidence,
            uncertainty=value_std,
            uncertainty_label=unc_label,
            reasons=reasons,
            alternatives=alternatives,
        )
