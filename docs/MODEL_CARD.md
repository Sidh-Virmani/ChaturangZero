# ChaturangZero model card

## Delivered checkpoint

- Name: **CZ-001.1**, a learned CPU pilot.
- Architecture: 14 × 8 × 8 input, 32-channel stem, two residual blocks,
  a 4,096-action policy head and a scalar value head; **59,813 parameters**.
- Checkpoint: `checkpoints/best.pt`.
- SHA-256: `5ec8c69da8277af8966d5478d5dc8c0dfb4324fb12956a551e44c14c78f7d126`.
- That checkpoint was trained on **12 self-play games, 1,456 reflected replay
  examples, and 8 epochs**. It bootstrapped the first playable learned model.
- The project has subsequently executed **88 self-play games and
  40 optimization epochs in total**. The replay currently holds
  **12,000 examples**. Those totals include rejected candidates;
  they must not be attributed to the delivered checkpoint.
- Latest candidate: **CZ-004.1**. The arena retained
  the existing best checkpoint. The final four-game cycle was started and
  completed through the app's Training button.

## Independent evaluation

The checkpoint was fixed before evaluation. Each comparison used **16 games**
(8 color-swapped position pairs), **64 MCTS simulations per move**, and a
**200-ply cap**. Seeds were 713731 for two-chariot endgames and 723731 for full
boards. Full-board tests use eight seeded legal opening plies so pairs cover
different positions. Initial weights use seed 42 and the same architecture,
search implementation, and budget as the trained model. Random play samples
legal moves uniformly.

| Position set | Comparison | W / D / L | Score | Paired 95% bootstrap interval | Ply-cap draws |
|---|---|---|---|---|---|
| Two-chariot endgames | trained vs random | 5 / 10 / 1 | 62.5% | 53.1%–71.9% | 1 / 16 |
| Two-chariot endgames | initial vs random | 2 / 14 / 0 | 56.2% | 50.0%–65.6% | 0 / 16 |
| Two-chariot endgames | trained vs initial | 2 / 10 / 4 | 43.8% | 34.4%–50.0% | 0 / 16 |
| Full board, 8 opening plies | trained vs random | 1 / 15 / 0 | 53.1% | 50.0%–59.4% | 14 / 16 |
| Full board, 8 opening plies | initial vs random | 2 / 13 / 1 | 53.1% | 43.8%–62.5% | 13 / 16 |
| Full board, 8 opening plies | trained vs initial | 1 / 15 / 0 | 53.1% | 50.0%–59.4% | 15 / 16 |

Score = (wins + 0.5 × draws) / games. The intervals resample complete color
pairs, using 5,000 bootstrap samples; eight pairs are a small sample. These
intervals describe the tested conditions and do not establish general skill.
All individual pair results, budgets, and seeds are preserved in
`data/evaluation_report.json`. The Training page also displays this report.

**Interpretation:** the trained model scored 62.5% against random on the
held-out endgames, but scored 43.8% directly against initial weights on that
same position set. Full-board results were mostly ply-capped draws. Broad
improvement over the untrained network and strong full-game play are **not
established**. This is a working learning/search system with an honest small
pilot, not a completed strength breakthrough.

## Training method

Training examples contain the canonical position tensor, legal-action mask,
MCTS visit distribution, and final outcome from the current player's
perspective. Self-play is generated from the rules; external expert games and
handcrafted material scores are not used. File reflection augments the data.
A bounded replay buffer supplies masked policy cross-entropy and value MSE;
AdamW updates the weights with gradient clipping.

A practical promotion gate requires a score above 55% against the current
checkpoint; the first trained model is an explicit bootstrap exception.
Small arenas can accept or reject candidates by chance. Historical full-start
run 002 used repeated deterministic openings and short caps, so it is not
credible evidence of general improvement. The independent benchmark addresses
opening diversity and reports each game and cap.

The search uses exact legal moves and detects an immediate root checkmate
before allocating simulations. This tactical check is shared by trained and
initial-weight comparisons. Deeper search uses neural-guided PUCT and
alternating-perspective backup.

## What the visualizations mean

- Input planes: exact encoded values. The 3D view maps them back to physical
  board coordinates; the 2D tensor inspector explicitly labels canonical coordinates.
- Convolution and residual views: actual post-ReLU feature channels. Mean mode
  averages absolute channel values; individual mode shows the selected channel.
  Bar height and color are normalized within the displayed map. Hover numbers
  are the raw values, not invented importance scores.
- Policy head: legal neural priors aggregated by destination, before search.
- Arena policy: visit shares after MCTS, aggregated by destination.
- Value head: the raw tanh prediction for the player to move. The Arena search
  value is a backed-up estimate; neither is a calibrated win probability.
- Search tree: actual visited nodes, with values from the parent player's perspective.
- Replay animation: an illustrative replay of the last computed result. Its
  timing, light pulses, layer spacing, and zoom are presentation choices, not
  a measurement of execution time or a causal account of the network's intent.
  The policy and value heads are fed in parallel from the shared residual features.

## Limitations and next research steps

The small dataset, limited search, draw-heavy outcomes, and ply caps leave
substantial room for stronger play. Many early examples have zero-valued draw
targets. Improvement needs larger diverse self-play runs, more terminal games,
separate validation seeds, stronger baselines, and sufficiently large paired
arenas. Increasing the network alone is not evidence of progress.

The UI exposes six major stages and the outputs of each residual block, not
every individual multiply-add or an attention/causal explanation. The project
has no calibrated Elo rating. It targets local play and teaching, not public
production hosting. See README.md for the prioritized engineering roadmap.
