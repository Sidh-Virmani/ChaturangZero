# ChaturangZero: five-minute live demonstration

## Before presenting

1. Run `python run.py --check`, then `python run.py --open`.
2. Use a browser with WebGL enabled; keep **2D** available as the accessible
   fallback. The UI, fonts, pieces, and trained model are local.
3. Open the Training view once to confirm saved metrics are present. Training
   takes real CPU time; present existing results before starting another cycle.

## 0:00 — The idea

“ChaturangZero learns a policy and a value from self-play, then uses Monte Carlo
tree search to explore possible moves. We made those computations visible.”

Show the ivory and midnight 3D armies. Drag to orbit, scroll to zoom, and use Reset
camera to return to the presentation view. The six original sculptures include
a crowned raja, counselor, wheeled chariot, horse, elephant with trunk/tusks,
and helmeted foot soldier.

## 0:45 — A real legal game

Start a traditional opening with Human vs. AI, Ivory, Balanced search.
Switch to 2D briefly and select **c1**: the gaja can leap to **a3** or **e3**,
even over its own pawn. Play **c1-e3** and let the AI answer. The journal records
both moves. Undo returns to the previous human decision point.

Explain the differences from modern chess: elephants leap two diagonally;
mantri move one diagonal; pawns advance once and promote only to mantri.

## 1:30 — Make thinking visible

Click **Analyze position**. The panel displays actual simulation counts,
search depth, runtime, candidate visit shares, and a signed value for the
named player. That value is not a calibrated win probability.

Turn on **Policy heatmap**, **Search paths**, and **Reveal neural layers**.
The top board remains playable in the overview. Six lower displays expose
input, convolution, both residual blocks, policy, and value. Choose a stage to
zoom in; use All layers to return. Hover or tap cells for measured values.
Move arcs represent the three most visited candidates.

## 2:20 — Open the neural lab

Open **Neural lab**. Select **Own padati** in the input-plane selector.
Show the 14-channel input, convolution, two residual blocks, and policy/value
heads. In a focused feature stage, change the channel selector. Replay inference
to illustrate the signal flow; explain that its timing is a presentation aid
and that both heads receive the shared residual features in parallel. The feature maps average absolute channel activations and normalize
within each layer; they are observations, not a causal explanation of intent.

Scroll beside the board to inspect the real search tree and principal variation.
Explain the four search operations: select with PUCT, expand, evaluate, and
back up values with alternating player perspective.

## 3:15 — Demonstrate a decisive result

New game -> **Watch AI self-play** -> **Checkmate challenge** -> Start.
Start self-play. The engine finds **d1-d8**, delivers checkmate, and the game
stops with a result banner. This uses the same rules and search as normal play;
the demonstration position simply makes the result quick to inspect.

Return to New game -> Traditional opening -> Human vs. AI for the next player.

## 4:00 — Show real learning evidence

Open **Training**. Explain that each example contains a position, the MCTS
visit distribution, and the eventual outcome. File reflection augments the
data. Policy cross-entropy and value MSE train the residual network.

Show the measured loss curves and candidate decisions. A newer model is not
automatically better: a candidate must clear the arena threshold. Point out
the number of games, the scope (endgame/full board), and capped draws.
The separate `data/evaluation_report.json` records held-out color-swapped
comparisons and a checksum for the evaluated model.

If time permits, start a training cycle. The UI reports self-play, optimization,
and evaluation as they actually run, then reloads the best checkpoint.

## Claims to make carefully

- It is a complete, functioning AlphaZero-inspired pipeline.
- Visualizations are derived from real model/search outputs.
- It is a compact CPU research project, not an unbeatable engine.
- A decreasing training loss is not proof of better play.
- Endgame gains cannot be generalized to full games without a full-board test.
- Draw-heavy or ply-capped results must be shown, not omitted.

Use the model card and evaluation report for exact observed results, rather
than memorizing a number that may change after another training run.
