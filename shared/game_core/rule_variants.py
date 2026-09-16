"""
shared/game_core/rule_variants.py
==================================
Rule Hypothesis Engine (Research Contribution #6).

Historical context
------------------
Different scholarly reconstructions of Chaturanga disagree on several rules.
Rather than picking one and assuming it is correct, ChaturangZero-X models
multiple rule hypotheses explicitly. This lets us study:

  1. How sensitive is learned strategy to historical rule uncertainty?
  2. Do the same openings / tactical patterns emerge across variants?
  3. Which variant produces the most decisive play?

Rule variants implemented
--------------------------
  STANDARD      R1: Checkmate wins. Stalemate = draw.   (our default)
  CAPTURE_RAJA  R2: Capturing the Raja wins immediately,
                    even if the king wasn't technically in check.
  BARE_RAJA     R3: Reducing opponent to Raja only = win
                    (in addition to checkmate).
  NO_STALEMATE  R4: Stalemate = loss for the stalemated player (not draw).

Usage
-----
    from shared.game_core.rule_variants import RuleVariant, RuleConfig
    config = RuleConfig.from_variant(RuleVariant.BARE_RAJA)
    engine = ChaturangaEngine(rule_config=config)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuleVariant(str, Enum):
    STANDARD      = "standard"       # R1 — checkmate wins, stalemate draws
    CAPTURE_RAJA  = "capture_raja"   # R2 — any Raja capture wins
    BARE_RAJA     = "bare_raja"      # R3 — bare Raja = loss
    NO_STALEMATE  = "no_stalemate"   # R4 — stalemate = loss


@dataclass
class RuleConfig:
    """
    Collection of boolean flags that parameterise one rule variant.
    All combinations are supported; the engine reads these flags in
    _update_status() to determine terminal conditions.
    """
    variant:              RuleVariant

    # Terminal conditions
    checkmate_wins:       bool = True   # Standard: raja has no legal moves in check
    capture_raja_wins:    bool = False  # R2: opponent captures raja directly
    bare_raja_loses:      bool = False  # R3: only raja left = loss
    stalemate_is_loss:    bool = False  # R4: stalemate = loss (not draw)

    # Draw conditions
    fifty_move_draw:      bool = True   # Standard 50-move (100 half-move) rule
    halfmove_limit:       int  = 100    # Number of half-moves before draw

    # Win conditions
    promotion_piece:      str  = "mantri"  # Padati promotes to Mantri (historical)

    @classmethod
    def standard(cls) -> "RuleConfig":
        """R1 — standard checkmate / stalemate draw rules."""
        return cls(variant=RuleVariant.STANDARD)

    @classmethod
    def capture_raja(cls) -> "RuleConfig":
        """R2 — Raja capture wins immediately."""
        return cls(
            variant=RuleVariant.CAPTURE_RAJA,
            capture_raja_wins=True,
        )

    @classmethod
    def bare_raja(cls) -> "RuleConfig":
        """R3 — bare Raja condition (in addition to checkmate)."""
        return cls(
            variant=RuleVariant.BARE_RAJA,
            bare_raja_loses=True,
        )

    @classmethod
    def no_stalemate(cls) -> "RuleConfig":
        """R4 — stalemate is a loss, not a draw."""
        return cls(
            variant=RuleVariant.NO_STALEMATE,
            stalemate_is_loss=True,
        )

    @classmethod
    def from_variant(cls, variant: RuleVariant) -> "RuleConfig":
        """Factory: create RuleConfig from a RuleVariant enum value."""
        _map = {
            RuleVariant.STANDARD:     cls.standard,
            RuleVariant.CAPTURE_RAJA: cls.capture_raja,
            RuleVariant.BARE_RAJA:    cls.bare_raja,
            RuleVariant.NO_STALEMATE: cls.no_stalemate,
        }
        return _map[variant]()

    def describe(self) -> str:
        """Short human-readable description for logging / reports."""
        lines = [f"Rule variant: {self.variant.value}"]
        if self.capture_raja_wins:
            lines.append("  • Raja capture wins immediately")
        if self.bare_raja_loses:
            lines.append("  • Bare Raja = loss")
        if self.stalemate_is_loss:
            lines.append("  • Stalemate = loss (not draw)")
        if self.fifty_move_draw:
            lines.append(f"  • {self.halfmove_limit}-halfmove draw rule")
        return "\n".join(lines)
