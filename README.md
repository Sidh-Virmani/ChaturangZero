# ChaturangZero

An AlphaZero-style Chaturanga project with an authoritative Python rules engine,
a PyTorch policy/value network, Monte Carlo tree search, self-play training, and
a React/Three.js interface for playing and studying the model.

## Current project status

| Part | Implemented | Purpose |
| --- | --- | --- |
| `shared/game_core` | Legal moves, check, promotion, terminal outcomes, HTTP game session | One source of truth for the game |
| `p1_representation` | 13 input planes, 4,096 action IDs, legal masks, symmetry transforms | Translate the game into numbers |
| `p2_neural_network` | Configurable residual network, policy/value heads, inference, loss, checkpoint loading | Evaluate a position and propose moves |
| `p3_mcts` | PUCT search, legal children, visit distributions, tree reuse | Search ahead using network predictions |
| `p4_training` | Self-play, compact replay, AdamW updates, full resume, arena comparisons, milestone checkpoints | Learn from completed games |
| `integration` | Checkpoint-backed machine opponent and model inspection | Connect the existing modules to the UI |
| `shared/ui` | Human/human and human/machine play, 2D and 3D views, search progress, move history, reset, flip, results | Play and explain the project |

This integration branch starts from `main` at `468ce5d`. The independent visual
prototype is preserved at
[`codex/complete-chaturangzero`](https://github.com/Sidh-Virmani/ChaturangZero/tree/codex/complete-chaturangzero).
The prototype uses a different backend and checkpoint format. Merge the
main-based `codex/threejs-main-integration` branch to adopt these visuals; do not
replace the team's modules with the prototype's backend.

## What this branch adds

- Beveled board tiles, blush and midnight colours, champagne inlays, porcelain
  and onyx materials, restrained cyan illumination, environment lighting, shadows.
- Original procedural Raja, Mantri, Ratha, Gaja, Ashva, and Padati sculptures:
  lotus crowns, turban ribs, chariot spokes, horse bridles, elephant ornaments,
  and infantry shields and spears. Geometry is cached and merged by material.
- Six distinct capture sequences: Ashva crescent leap, Gaja lotus shockwave,
  Ratha chariot charge, Mantri astral sigil, Raja royal eclipse, and Padati spear
  strike. Approach, impact, victim retirement and landing share an elapsed-time
  clock; particles are deterministic, decorative and pooled.
- A replayable piece gallery with movement rules and clearly labelled illustrative
  positions. It never changes the current match or calls the game API.
- Staggered piece arrivals, selection lift, check rings, mineral veining, gold
  frame rails, damped orbiting/zoom, camera flips and animated network focus.
- A real model inspector: input planes, stem, every configured residual block,
  policy features, legally masked policy, value features, and the scalar value.
- Individual channel selection, numeric canvas hover/tap readings, a keyboard
  activation grid, original tensor values, and explanations of every operation.
- A clearly labelled illustrative signal animation. Both heads branch from the
  same backbone. Animation speed is not a measurement of inference time.
- Local fonts, responsive controls, reduced-motion support, GPU resource cleanup,
  hidden-tab rendering suspension, lazy 3D loading, and a 2D WebGL fallback.
- Read-only `/api/game/network`, position revision checks, and serialized model
  inference to avoid stale snapshots and concurrent model-mode changes.

Moves still come from the existing game API. The scene reconciles pieces by
their squares and the last move because `main` assigns position-based piece IDs.
No alternative rules engine, action numbering, trainer, or checkpoint format
is introduced.

Fast AI replies queue behind the preceding visual move. Input unlocks after
the sequence settles, and terminal result dialogs wait for the final capture.
Reset, inspection, 2D switching and reduced-motion mode settle safely. Motion
does not affect engine legality, rewards, model predictions or search results.

## Run locally

Use Python 3.11+ and a current Node.js release compatible with the locked Vite
version (Node 24 was used for verification).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd shared/ui
npm ci
cd ../..
.\.venv\Scripts\python.exe run_chaturangzero.py
```

The launcher starts the game API on `127.0.0.1:8000` and the interface on
`http://127.0.0.1:5173/`. The Windows batch launcher is also available after
installing the dependencies. Only one server/launcher should use these ports.
To stop it, press Ctrl+C in its terminal.

For a quick, real training/inference demonstration:

```powershell
.\.venv\Scripts\python.exe -m p4_training smoke --output checkpoints/p4_smoke
```

Run smoke training in a new output directory if it already contains a training
state. This completes one short forced-terminal game and one optimizer update;
it demonstrates the pipeline, not strong playing ability. The default smoke
model is small. The inspector discovers its actual channels and blocks rather
than drawing a fictional fixed architecture. Local demo weights are ignored by Git.

To use a trained P2/P4 checkpoint, set the environment before starting the API:

```powershell
$env:CHATURANGZERO_CHECKPOINT = 'checkpoints/final_models/hard.pt'
$env:CHATURANGZERO_SIMULATIONS = '64'
.\.venv\Scripts\python.exe run_chaturangzero.py
```

Without a checkpoint, local human play works and the inspector displays the
exact 13 input planes. Learned activations and machine play require a checkpoint.
Selected checkpoints must come from this project's P2/P4 format.

## How to study the model

1. Start a match. Click a piece, then a legal destination. Drag to orbit and
   scroll to zoom; use **2D board** for keyboard play.
2. Choose **Explore the network**. The current position is inspected on the CPU.
3. Choose a stage using the canvas selector or the buttons below the scene.
   The camera eases towards that layer. Select a channel and hover/tap a cell.
4. Read the original value and explanation in the canvas card or activation grid.
5. Use **All layers** and **Animate signal** to explain the forward-pass order,
   then **Return to game** to continue playing.

The input has six piece planes per side plus `white_to_move`. Tensor coordinates
always follow P1: rank 1 is row zero, independent of the camera orientation.
The HTML grid displays rank 1 at the top and labels every square explicitly.
Channel colours and bar heights normalize by the displayed map's maximum
magnitude; the numeric readings retain their original values. The scalar value
keeps its sign. Activations are descriptive measurements, not causal proof of
why a move was selected.

The policy map sums **legal network priors by destination square**. Individual
from–to priors appear below it. These are different from the MCTS visit counts
in the machine sidebar. The value is a learned side-to-move outcome estimate in
`[-1, +1]`, not a calibrated win probability.

## Rules and limitations

This is Chaturanga, not modern chess. The Mantri moves one diagonal square;
the Gaja jumps two squares diagonally; the Ashva moves like a knight; the Ratha
moves along files and ranks; the Padati advances one square and captures
diagonally, promoting to Mantri. There is no castling, en passant, or initial
double pawn move. The core treats stalemate and a bare opposing Raja as wins.
It currently has no repetition or move-count draw rule. Self-play games that
reach the configured ply cap without a core outcome are discarded as truncated.

The API is a single shared local game session. This is a presentation/study
application, not an authenticated multiplayer service. The scene displays one
channel or channel aggregate per stage; it does not render every neuron at once.
The 3D chunk is lazy loaded and is approximately 160 kB gzipped. Rendering
performance depends on the browser and GPU; the 2D fallback remains available.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd shared/ui
npm run build
npm test
```

`pytest.ini` limits discovery to the actual `tests/` directory, excluding local
comparison snapshots. Inspector tests compare its policy and value against
ordinary P2 inference, check dynamic block counts and exact P1 planes, preserve
batch-normalization state, reject stale/busy inspection, and verify terminal
masking and the HTTP endpoint. Browser verification covers the interactive views,
camera controls, channel selection, and responsive layouts.
The frontend tests check every character's timing/endpoints, history continuity,
bounded effect pools, finite transforms, reduced motion and resource disposal.
Use Node 24 for the native TypeScript test runner. While Vite runs, open
`http://127.0.0.1:5173/tests/scene-harness.html` and run the scene lifecycle
checks; repeat with the browser's reduced-motion emulation enabled.

## Presentation and next steps

Present this as an integrated AI pipeline with a visual explanation interface.
Credit the team's existing core and P1–P4 modules; your contribution in this
branch is the main-compatible 3D experience and educational inspection boundary.
Show a legal move, a machine search, the 13 input planes, a residual channel,
the two heads, and the difference between a network prior and a search visit.
Use the piece gallery to demonstrate the six capture styles without needing
to reach every capture in a live match.
Call the demonstration checkpoint a smoke model, without claiming measured strength.

The main remaining work is model quality: run longer self-play training,
evaluate checkpoints across alternating colours and enough games, report
truncation rates and uncertainty, and choose demonstration weights based on
evidence. Then add position setup/replay for lessons, controlled performance
benchmarks on lower-end GPUs, and optional captured-piece displays or audio.
Changing draw rules requires a team decision and matching core/training changes.
See [the integration guide](docs/THREEJS_INTEGRATION.md) for the merge boundary.

Further technical reports: [P1](docs/dev_reports/p1_representation.md),
[P2](docs/dev_reports/p2_neural_network.md), [P3](p3_mcts/GUIDE.md),
[P4](docs/dev_reports/p4_training.md), and
[machine integration](docs/dev_reports/ui_machine_integration.md).
