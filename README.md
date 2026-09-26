# ChaturangZero

**Ancient game. Evolving mind.** A complete, local AlphaZero-inspired learning
system and interactive 3D Chaturanga lab, built with Python, PyTorch, React,
TypeScript, and Three.js.

Play against a real neural-guided search engine, watch self-play, or explore
the tensors, move probabilities, and search branches behind a decision.
The original procedural pieces represent the raja, mantri, ratha, ashva,
gaja, and padati in warm ivory and midnight lacquer with champagne-gold inlays.

## Run

Recommended: **Python 3.12** and **Node.js 22.12+ or 24+**. First-time setup
downloads CPU PyTorch and frontend packages into this project. A GPU, API key,
external game service, or cloud account is not required.

On Windows, double-click **`start-chaturangzero.bat`**. It uses `.venv` when
available, installs missing dependencies, and opens the arena.

Or, from the project root:

```sh
python run.py --setup --open
```

Subsequent runs:

```sh
python run.py --open
```

Open **http://127.0.0.1:8000**. One Python process serves both the production
UI and the API. Keep its terminal open; Ctrl+C stops it. If that port is busy,
use `python run.py --port 8001 --open`.

The launcher automatically uses the project's virtual environment, so it does
not install packages globally. To use it directly on Windows:

```powershell
.\.venv\Scripts\python.exe run.py --check
.\.venv\Scripts\python.exe run.py --open
```

On macOS/Linux the environment executable is `.venv/bin/python`.
The release archive includes a built UI and trained checkpoint; rebuilding the
UI requires Node.js. All fonts and piece geometry are bundled locally.

## What to explore

- **Arena:** human vs. AI, local two-player, or AI self-play; either army;
  32/96/192 search simulations; animated moves; undo; flip; orbit/zoom;
  sound; fullscreen; move journal and JSON export.
- **Three.js board:** six redesigned sculptures, rounded plinths, porcelain/lacquer
  finishes, gold inlays, cyan underlighting, shadows, and eased move animations.
  Use **2D** for labeled, keyboard-accessible squares or when WebGL is unavailable.
- **Policy heatmap:** destination-square share of actual MCTS visits.
- **Search paths:** the three most visited root moves, drawn as curved paths.
- **Neural lab:** smoothly expand into six major stages: input, convolution,
  residual 1, residual 2, policy, and value. Choose a stage to glide into a
  focused view; inspect all 14 input planes or any of 32 feature channels.
  Hover or tap cells for real numbers and plain-language interpretations.
- **Inference replay:** follow an explicitly illustrative signal animation;
  both heads receive the shared residual features in parallel. Zoom, orbit,
  return to all layers, or reset the camera at any time.
- **Study aids:** piece-hover movement rules, legal destinations, a “Read this
  decision” guide, actual search branches, and the principal variation.
- **Training:** measured loss curves, evaluation results, replay counts,
  checkpoint promotion decisions, the held-out benchmark, and a button that
  runs a real training cycle.
- **The game:** the fixed rules and an illustrated guide to all six pieces.
- **Presentation positions:** choose the full opening, a mate-in-one challenge,
  or an eight-piece endgame in New game.

See **[the presentation walkthrough](docs/PRESENTATION.md)** for a concise live demo.


## Current delivery and what changed

The original workspace had a rules engine and a basic board. The representation,
network, search, training, and integration folders are now implemented. The
project runs end to end with a saved learned checkpoint and a production UI.

The latest visual pass follows the supplied luxury-board reference: a quieter
midnight/plum interface, restrained blush tiles, warm ivory and dark lacquer
pieces, fine gold lines, and soft cyan light under the board. Each of the six
piece types has an original three-dimensional silhouette. These are procedural
models, not a flattened image or a dependency on remote 3D assets.

The network explorer uses an eased 1.1-second camera transition, smooth layer
separation and fading, focused views, cell outlines, and both mouse-hover and
tap inspection. Instanced activation bars keep the grids efficient. Camera
motion respects the system's reduced-motion preference, and labeled 2D play
remains available. The interface preserves human/AI, local play, self-play,
undo, settings, sound, fullscreen, export, policy overlays, and search paths.

The project has completed **88 self-play games and 40 training epochs** across
all candidates. The delivered best checkpoint remains **CZ-001.1** (12 games,
8 epochs); later training candidates did not clear the promotion gate. An
independent **96-game benchmark** evaluates it against random and initial
weights under matched budgets. General strength improvement is not established;
read the [model card](docs/MODEL_CARD.md) before presenting performance claims.

## How to study a decision

1. Play a move or choose a presentation position, then **Analyze position**.
2. Read visit shares and the signed search value in **Inside the mind**. Expand
   **Read this decision** for their meaning and limitations.
3. Enable **Policy heatmap** or **Search paths** for the current board.
4. Enable **Reveal neural layers**, or open **Neural lab**. Use the layer
   selector or numbered stage buttons to focus a layer.
5. In Input, choose an encoded plane. In a convolution/residual view, compare
   the channel mean with individual channels. Hover or tap a cell for its raw
   activation, physical square, and interpretation.
6. Compare **Policy head** (neural priors before search) with the Arena heatmap
   (MCTS visit shares after search). Inspect **Value head** separately.
7. Use **Replay inference** to follow the explanatory animation. Data stays tied
   to the stated analyzed ply; refresh after moving to study a new position.

Height and color are normalized within the selected feature map. They indicate
activation magnitude, not strategic importance. Layer spacing and animation
speed are display choices. Policy and value are parallel heads, even though
their displays are arranged vertically for inspection.

## Learning system

```text
Immutable Chaturanga rules
    -> 14 x 8 x 8 canonical tensor + legal-action mask
    -> 32-channel convolution + 2 residual blocks
    -> 4096 policy logits + side-to-move value
    -> PUCT Monte Carlo tree search
    -> self-play (state, visit policy, outcome) examples
    -> bounded replay + file-reflection augmentation
    -> policy cross-entropy + value MSE + AdamW
    -> paired candidate-vs-best evaluation -> checkpoint promotion
```

The default network has **59,813 trainable parameters**. Black's ranks are
flipped in the canonical representation. The input has six own-piece planes,
six opponent-piece planes, a normalized no-progress clock, and repetition count.
The policy encodes `from_square * 64 + to_square`; illegal actions are masked.
Search backs up alternating player perspectives and checks immediate root mates
using exact game rules. No external expert games or handcrafted material
evaluation are used.

The UI never implements a second rules engine. It requests legal moves and
validated state transitions from Python. Sessions are isolated by tab token;
revision checks reject stale search results and stale move requests.

## Train and evaluate

Use the virtual-environment Python for these commands (shown below as `python`).

```sh
# Fast reward-rich endgames; resumes the existing best checkpoint and replay
python -m p4_training.train --rounds 2 --games 24 --epochs 8 --simulations 64 --max-plies 160 --arena-games 12 --seed 271

# Full starting boards
python -m p4_training.train --full-start --games 12 --epochs 6 --simulations 32 --max-plies 200

# Independent color-swapped held-out evaluation against random and initial weights
python -m p4_training.evaluate --games 24 --simulations 64 --max-plies 200
```

`checkpoints/best.pt` is served by the app. `candidate.pt` holds the last trained
candidate. The first trained checkpoint bootstraps the demo; later candidates
must score above 55% against the current best to replace it. Checkpoints are
written atomically and reloaded automatically. Only one trainer can write the
project at a time. If a process crashes, verify it has stopped before removing
its `data/training.lock`.

`data/replay.npz` contains numeric arrays and is loaded without pickle.
`data/training_metrics.json` contains measured losses, game summaries, and arena
decisions. `data/evaluation_report.json` records held-out results, seeds,
budgets, color pairs, confidence intervals, and the checkpoint hash.

**Interpret results honestly:** this is a small CPU research model, not a
grandmaster-strength engine. Endgame curriculum results do not establish
full-board strength. Ply-capped games are explicitly counted as draws. Training
loss is not playing strength, and a small promotion arena is a practical gate,
not a statistical proof. See [the model card](docs/MODEL_CARD.md) for measured
results and limitations.

## Ruleset CZ v1

| Piece | Movement |
|---|---|
| Raja | One square in any direction; cannot enter or remain in check |
| Mantri | One square diagonally |
| Ratha | Any distance orthogonally, blocked by pieces |
| Ashva | A standard knight's L-shaped leap |
| Gaja | Exactly two squares diagonally, leaping over the intervening square |
| Padati | One square forward; captures diagonally; promotes to mantri |

Checkmate wins. Stalemate, threefold repetition, or 100 plies without a capture
or padati move draw. Checkmate takes precedence over a draw threshold. There is
no castling, opening pawn double-step, en passant, or automatic bare-king win.
This is the project's frozen two-player variant, not a claim that every
historical Chaturanga tradition used the same rules.

## Verify and develop

```sh
python -m unittest discover -s tests -v
python run.py --check
cd shared/ui
npm ci
npm run build
```

For live frontend development, run `python output/server/api_server.py` from
the root and `npm run dev` in `shared/ui`. Vite proxies `/api` from port 5173 to
8000. After source changes, rebuild the served production UI with
`python run.py --build --check`.

Tests cover piece movement, check evasions, promotion, perft, random-game
invariants, draw rules, action/representation roundtrips, MCTS value perspective,
mate selection, checkpoint fidelity, actual learning, replay persistence,
session isolation, stale-result rejection, and live HTTP integration.

The server binds to loopback by default. It is a local presentation/development
server; public multi-user hosting would require authentication, resource quotas,
and a production server. Browser sessions are in memory and end when the server
restarts; export a game to keep its record.

## Source map

| Directory | Responsibility |
|---|---|
| `shared/game_core` | Immutable state, movement, checks, terminal rules |
| `p1_representation` | Canonical tensors, action encoding, masks, symmetry |
| `p2_neural_network` | Residual policy/value model, inference, inspection, checkpoints |
| `p3_mcts` | PUCT selection, expansion, backup, visit policies |
| `p4_training` | Self-play, replay, training, paired evaluation, independent benchmark |
| `integration` | Application engine, training jobs, presentation scenarios |
| `output/server` | Session API and built-UI serving |
| `shared/ui/src` | Arena, 3D sculptures, tensor/search inspection, training UI |
| `tests` | Rules, learning, search, sessions, and HTTP tests |
| `docs` | Presentation, model card, completion audit, third-party notices |
| `scripts` | Reproducible release packaging with file checksums |

The original [architecture proposal](Architecture_plan.md) is retained for
context. This workspace is linked to the supplied ChaturangZero Git repository.

## What remains and suggested next steps

The local application and complete learning pipeline are implemented. The
remaining work is chiefly model research and optional product expansion:

1. **Establish playing strength.** Generate substantially more diverse self-play
   with longer games and enough terminal outcomes. Evaluate on seeds kept out
   of training; compare against initial weights, random play, a stronger search
   baseline, and human opponents. Keep caps and draws visible. Do not infer
   stronger play from training loss or a tiny promotion arena.
2. **Improve training throughput.** Batch neural evaluations across search
   workers, profile the rules engine, and add optional accelerator support.
   Larger training budgets should be justified by held-out performance.
3. **Expand study tools.** Add curated tactical exercises, a board editor,
   saved variations, and game import/review with Chaturanga-aware notation.
4. **Broaden device validation.** Test physical phones, tablets, and low-end
   GPUs; add an explicit quality preset and more automated browser coverage.
   The current browser checks cover desktop and narrow layouts; they do not
   establish universal device compatibility.
5. **Prepare public hosting only if needed.** Add durable sessions, authentication,
   job/resource limits, monitoring, and a production deployment. The shipped
   server is intentionally a local presentation app.
6. **Publish and review the feature branch.** The completed work is committed
   against the supplied repository's `main` history. Push the branch and review
   the changes before merging them into `main`.

There is no claim of unbeatable play, measured Elo, causal neural explanations,
or production-grade public hosting. Those are separate milestones.

## Package the project

After building, testing, and completing an evaluation of the current best model:

```sh
python scripts/package_release.py
```

This creates `output/releases/ChaturangZero-v1.0.0.zip`, including source, built
UI, model weights, replay data, measured reports, documentation, and verification
logs. It excludes environments, node_modules, process logs, and private config.
The archive contains `SHA256SUMS.json`; a checksum for the archive is saved
beside it. The packager refuses to bundle during training or with an incomplete
or mismatched benchmark. Bundled third-party licenses are in
[docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md).
