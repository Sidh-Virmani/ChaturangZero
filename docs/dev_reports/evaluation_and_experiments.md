# Measuring whether ChaturangZero learned

## The question this layer answers

A lower training loss means the network fits its examples more closely. It does
not prove that it plays better. A later checkpoint must play actual games against
a frozen opponent under recorded, equal conditions.

Previously, deterministic arena games repeatedly started from the initial board.
Fifty games could repeat just two trajectories. The new `evaluate` command plays
from a fixed collection of different openings, with **two games per opening**:

```text
Opening A, including its move history
    -> candidate White / reference Black
    -> reference White / candidate Black
Opening B, including its move history
    -> candidate White / reference Black
    -> reference White / candidate Black
```

The *candidate* is the model being tested. The *reference* is its frozen opponent.
Reversing their colours does not rotate or otherwise change the opening board.
Both models experience the same advantage or disadvantage of that starting board.
An opening pair is the useful comparison unit; its two games are related.

## The protocol we actually implement

`p4_training/evaluation_openings.json` contains 12 distinct nonterminal positions:
the initial position and eleven openings reached after 4–12 legal moves. These
were generated independently of any network, using a fixed seed and uniformly
choosing among sorted legal moves. They are **not expert or balanced openings**.
Some deliberately include retreats or unusual development. Both sides to move
occur in the suite. No opening is repeated to inflate the game count.

The JSON stores the actual square-to-square moves, not just a seed. Loading it
replays every move through public `game_core.apply_move()`. This validates the
position and preserves the complete repetition history. Terminal positions,
duplicate visible board/turn positions, duplicate IDs and illegal moves are
rejected. Saved move sequences are authoritative if a generator changes later.

Each move uses the existing P2 evaluator and optimized P3 MCTS. Both players get
the same simulation count and PUCT setting. Temperature is zero (choose the most
visited action), exploration noise is off, and ties follow existing P3 ordering.
We start a fresh search each move, as the old arena did. In particular, we never
carry a tree evaluated by one model into the other model's search. There is no
new game logic, representation, or search algorithm here.

The ply cap counts moves **after the opening**. One ply means one player's move.
If the last allowed move ends the game, the core result still counts normally.
Otherwise it is a truncation: we stopped observing before knowing the outcome.

## Commands

Use the repository's Python environment (`.\.venv\Scripts\python.exe` on Windows,
or `python` with that environment activated). All example output directories must
be new. The commands refuse to overwrite an existing experiment or baseline.

Freeze a standard-size random, untrained network once:

```powershell
python -m p4_training freeze-baseline --output checkpoints/baselines/untrained_seed0 --seed 0
```

This creates `model.pt` plus `baseline.json`, recording its seed, architecture,
file hash and environment. A file hash is a fingerprint of the exact bytes.
Future evaluations should reuse this file, rather than regenerate random weights
under possibly different PyTorch versions. Baseline creation preserves the
caller's PyTorch random stream. An instance already exists locally from validation.

Compare saved checkpoints on all 12 default openings (24 games):

```powershell
python -m p4_training evaluate --candidate checkpoints/benchmark/hard.pt --reference checkpoints/benchmark/medium.pt --output evaluation_results/hard_medium_full --simulations 25 --max-plies 512
```

Against the untrained baseline, substitute
`--reference checkpoints/baselines/untrained_seed0/model.pt`.
Use `--pairs 3 --simulations 5 --max-plies 128` for a short infrastructure check.
Small budgets and short caps are **not** adequate final strength evidence.

Freeze a larger suite before comparing serious models:

```powershell
python -m p4_training make-openings --output evaluation_results/openings24.json --count 24 --seed 20261114
python -m p4_training evaluate --candidate checkpoints/run1/latest_model.pt --reference checkpoints/benchmark/easy.pt --openings evaluation_results/openings24.json --output evaluation_results/run1_easy --simulations 25 --max-plies 512
```

The parent directory of `make-openings --output` must exist. The same opening
file and settings should be used across comparisons. For final reporting, also
use a separately frozen held-out suite that was not used to choose checkpoints.
Do not repeatedly choose favourable opening seeds after seeing results.

Verify an experiment without running either neural network again:

```powershell
python -m p4_training verify-evaluation --output evaluation_results/run1_easy
```

This replays opening and game moves through the core and checks game/pair counts,
colours, results, reasons, caps, checkpoint IDs, search settings and totals. It
also checks that CSV and JSONL agree. It detects inconsistent records, not
deliberate fabrication: it cannot prove a recorded move came from a neural search
without rerunning that search. The tests additionally rerun real searches and
compare trajectories.

To rerun exactly the saved workload, use the exported `openings.json`, the model
files whose hashes match the manifest, and the manifest's simulation/PUCT/cap
settings. Keep the same code, dependency versions, device and thread settings.
Deterministic CPU tests reproduce move sequences; timings naturally differ.
Bit-for-bit agreement across hardware/PyTorch versions is not promised.

The older `arena` command and `compare_models()` API remain available for small
compatibility checks and explicitly warn about repeated starts. Use `evaluate`
for strength evidence. No automatic best-model promotion is performed.

## What gets saved

Each new evaluation directory contains:

- `manifest.json`: protocol version, start/end time, completion status, both
  checkpoint paths/SHA-256 fingerprints/architectures, all MCTS settings, cap,
  device, Python/PyTorch/NumPy versions, threads, Git commit/status and an AI source
  fingerprint. It also fingerprints the exact exported opening suite.
- `openings.json`: the actual openings used, including any `--pairs` subset.
- `games.jsonl`: one complete record per finished game, flushed to disk after
  every game. JSONL means one readable JSON object per line. Each record includes
  opening, colour, checkpoint fingerprints, search settings, full move sequence,
  result, winner, terminal reason, moves after opening, total final ply count and
  elapsed seconds.
- `games.csv`: the same records in a spreadsheet-friendly format. Lists and search
  settings are JSON text inside cells.
- `summary.json`: totals, lengths, terminal-reason counts and results per pair.

Interrupted runs keep their completed JSONL rows and a `running` manifest; they
are not silently called complete. Automated mid-experiment resume is not yet
implemented. Start a fresh output directory for a rerun. Output directories are
never reused. Checkpoints are read only; evaluation rejects a checkpoint that
changes during loading. Prefer immutable milestones or frozen copies, rather
than a `latest_model.pt` being replaced by a simultaneous training process.

A dirty source fingerprint identifies the code but does not archive its contents.
Commit/archive the code before a serious run; do not rely on a hash to reconstruct
uncommitted source later. Evaluation artifacts from development explicitly record
that the working tree was dirty.

## Reading the score honestly

A candidate win earns 1 point, a genuine core draw earns 0.5, and a loss earns 0.
The completed-game score is these points divided by completed games. It is `null`
when no game finishes. Truncations are shown separately, never as draws or losses.

We also give a range for the score across **all** scheduled games. Its lower end
assumes every unresolved game would become a loss; its upper end assumes they
would all become wins. Example: one win, one loss, one draw and one truncation
give 1.5 known points out of four games. The range is 1.5/4 to 2.5/4, or 37.5–62.5%.
The completed-game score alone would be 50% and hide the missing information.

These are truncation bounds, **not a statistical confidence interval**. The suite
is fixed, pairs are related, and different openings can lead to similar games.
We do not pretend that 24 games are 24 independent random observations. There is
no Elo or IID win-rate interval. Raw results and every pair remain visible.
Repeating a deterministic experiment checks reproducibility; it does not increase
the sample size. A credible strength claim needs enough varied, resolved pairs,
consistent results against frozen opponents, and confirmation on held-out starts.

## Training records outside the resume checkpoint

Normal training commands are unchanged. After every committed iteration:

```text
self-play -> replay -> data-scaled updates
    -> latest_model.pt + training_state.pt
    -> metrics.jsonl + run.json
```

`metrics.jsonl` contains one row per completed iteration: completed/truncated games,
White/Black wins, draws, examples, replay size, mean plies, actual updates, losses,
schedule/batch settings, self-play seconds, optimizer seconds and total compute
seconds. Timing excludes checkpoint/log I/O. Each new row also contains its actual
config, learning rates, starting-state specification (normally `initial`), UTC
timestamp, source/environment identity and resulting model-file fingerprint.
Positions/hour or plies/second can be derived from these counts and times without
inventing a second set of counters.

`run.json` records the latest completed iteration, active config/environment and
hashes of both model and resume checkpoint. The resume checkpoint still contains
optimizer, replay, RNG state and full history. Logging adds no randomness and does
not change sampling, augmentation, losses or update scheduling.

The checkpoint is authoritative: after it is saved, the small text history is
atomically rebuilt from that committed history. This avoids duplicated rows on
resume and repairs stale/partial text logs on the next successful save. A crash
between checkpoint and text exports may temporarily leave logs one iteration
behind. Old checkpoint rows load normally and export unknown audit/timing fields
as `null`; we do not invent old provenance. The existing checkpoint pair is not
a single cross-file transaction: after interruption, resume from
`training_state.pt`, not by guessing from the latest-model file.

Historical model hashes identify snapshots; they do not retain old weights.
Keep milestone/checkpoint copies you want to compare. Evaluation output records
their fingerprints separately; no evaluation is automatically launched during
training, and losses do not trigger model promotion.

## Code map and tests

- `openings.py`: generates, loads and validates fixed opening sequences.
- `evaluation_openings.json`: the committed 12-position development suite.
- `arena.py`: one shared game runner, plus the preserved small comparison API.
- `evaluation.py`: freezes baselines, runs pairs, exports/summarizes records and
  replays records for verification.
- `recording.py`: SHA-256 fingerprints, environment/source metadata and atomic
  text writes; no framework or external service.
- `trainer.py`: adds timing/provenance to existing history and exports it after
  saving. `__main__.py` exposes the commands and short progress summaries.
- `test_evaluation_experiments.py`: distinct legal openings/history, deterministic
  generation, fixed baseline/RNG preservation, no overwrites, actual paired
  searches, replayable exports, draws versus caps, conservative scoring,
  interruption, tamper detection and CLI.
- `test_training_recording.py`: checkpoint identities, config/timing exports,
  repair on resumed save and honest migration of old history. Existing training
  tests still check exact optimizer/RNG continuation and augmentation.

## Validation on 2026-11-14

Full regression command: `python -m pytest -q` — **165 passed in 55.93s**.
The pre-change baseline was **150 passed in 40.79s**; these timings include different
test workloads and are not a performance comparison.

Real comparisons used the first three saved openings in both colours, five MCTS
simulations per move and a cap of 128 moves after each opening:

| Candidate | Reference | Wins | Losses | Draws | Truncations | Mean plies | Search/game time |
|---|---|---:|---:|---:|---:|---:|---:|
| hard.pt | easy.pt | 0 | 0 | 5 | 1 | 83.3 | 29.1 s |
| hard.pt | medium.pt | 0 | 0 | 2 | 4 | 108.2 | 35.1 s |
| hard.pt | untrained seed 0 | 0 | 0 | 4 | 2 | 70.2 | 23.2 s |

All draws were core repetition draws. The corresponding overall score bounds were
41.7–58.3%, 16.7–83.3% and 33.3–66.7%. **None establishes a strength advantage.**
All 18 exported games passed `verify-evaluation`. Different openings produced
different trajectories and completion lengths. The complete files are under
`evaluation_results/20261114_hard_vs_easy`, `20261114_hard_vs_medium` and
`20261114_hard_vs_untrained`. Existing benchmark SHA-256 hashes were unchanged.

The new `checkpoints/baselines/untrained_seed0/model.pt` is the standard 64-channel,
four-block network before any training. Its adjacent metadata and each comparison
manifest identify its exact bytes; no benchmark file was replaced.

One real smoke run in `checkpoints/measurement_smoke_20261114` completed a one-ply
game and one finite update. A separate CLI resume completed iteration 2 with one
four-ply truncation, zero examples and zero updates. Both rows were preserved in
the exported history and hashes matched their checkpoint files. No long training
was run. Automated tests separately verify exact resumed weights and RNG streams.

## Before the serious run

Use the fixed untrained model and `benchmark/easy.pt` as frozen references; compare
against the previous selected checkpoint too. `medium.pt` and `hard.pt` remain
milestone names, not proven difficulty levels. Choose search budget, suite and cap
before evaluation. Investigate high truncation rates rather than hiding them.

Next measure conservative self-play parallelism and CPU thread use, then run a
small instrumented pilot. Compare useful completed examples/hour and training
time, not just simulated moves. Keep the current model size and value convention.
Only scale to long training after throughput, completion rates and saved records
are verified. Parallelism, CUDA, gating and long training are outside this change.
