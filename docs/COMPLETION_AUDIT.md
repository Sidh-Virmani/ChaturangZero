# Completion audit

The local Chaturanga application, learning pipeline, and visual study tools
are implemented. The delivered checkpoint is a small trained pilot; the
separate question of playing strength is documented in
[MODEL_CARD.md](MODEL_CARD.md).

| Requirement | Evidence | Status |
|---|---|---|
| Legal two-player games and fixed CZ v1 rules | Immutable rules, perft, fuzz, check/draw tests | Complete |
| Canonical state/action representation | 14 planes, 4,096 actions, legal mask, reflection tests | Complete |
| Learned network and neural-guided search | 59,813-parameter checkpoint, MCTS and mate tests | Complete as a small pilot |
| Self-play, replay, optimization, and promotion | 88 self-play games, 40 optimization epochs in all runs; live UI cycle completed | Complete and extensible |
| Independent strength measurement | Six 16-game held-out comparisons, color-swapped pairs, checksums and capped draws | Measured; broad improvement remains unproven |
| Human/AI, local play, and AI self-play | Live browser play, 3D move/undo, checkmate demo, session tests | Complete |
| Reference-inspired luxury visual design | Sculpted pieces, blush/navy board, gold inlay, cyan edge light, responsive layout | Complete |
| Inspectable network and search | Six stage views, 14 input planes, 32 feature channels per layer, priors, value, visit tree, numeric hover/tap | Complete for major stages |
| Accessible alternatives | Labeled 2D squares, keyboard controls, reduced-motion handling, WebGL fallback | Complete for local use |
| Reproducible presentation | One-process launcher, bundled UI/model, README, demo guide, model card, release script | Complete |

Verification: **80 unit/integration tests passed**, the production TypeScript
build passed, `run.py --check` reported ready, and the browser loaded the
trained checkpoint through the local API. Desktop and 390-pixel layouts were
inspected; the mobile document width stayed within its viewport. The direct 3D
move `g2-g3`, undo, neural layer focus, channel selection, and raw-cell value
readout were exercised. The browser also completed a real training cycle and
a one-move AI checkmate. See `output/verification` in the release archive for
test, build, and environment logs. This audit does not substitute for the raw
evidence or claim exhaustive device coverage.

The engine's held-out results are draw-heavy. The delivered checkpoint scored
62.5% against random on sampled two-chariot endgames and 43.8% directly
against initial weights on that same set. Full-board tests had many ply-cap
draws. Stronger play requires further research, larger diverse training runs,
and new evaluation. The six 3D stages show major network outputs and
interpretations; they do not claim a causal explanation of every neuron.
