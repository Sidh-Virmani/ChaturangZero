# Choosing a practical training run

For the current start/resume/monitor/evaluate commands and interruption behavior,
use [the serious-training runbook](../SERIOUS_TRAINING.md). The historical commands
below describe the measured pilot; `resume --until 200` now supports an exact total.

## What changed

P4 can play independent games in CPU worker processes. Each worker receives a
frozen CPU copy of the current model, calls the existing P3/P2/P1/core pipeline,
and returns a complete `SelfPlayResult`. Only the parent owns the optimizer,
replay, training history and checkpoint files. There is no alternative search or
rule implementation.

New runs use game seeds derived from `(run seed, iteration, game index, protocol 1)`.
Changing worker count or completion order therefore does not deliberately change
the workload. Completed results enter replay in game-index order. Serial seeded
self-play uses the same seeds; tests compare actual moves and policy targets.
Existing checkpoints without `seeded_games` retain the old serial NumPy stream.
The model still uses 64 channels/four blocks and side-to-move values.

New configuration/CLI controls:

- `--self-play-workers`: independent games in flight; default 1.
- `--worker-threads`: Torch CPU threads per self-play process; default 1.
- `--cpu-threads`: parent Torch threads for training; default 8. The parent does
  not train while workers are playing, so these do not add to worker contention.
- `--device cpu|cuda`: training device. CPU remains fully supported.
- `--self-play-device cpu`: independent setting; parallel workers require CPU.
  Single-worker self-play can use CUDA, but the measured search favoured CPU.
- `--worker-timeout-seconds`: abort if no game result arrives for this long,
  including startup; default 3600. CLI prints waiting messages every 15 seconds.
- `--snapshot-interval N`: retain a separate, immutable `iteration_000010.pt`,
  etc., every N iterations. Default 0 disables additional snapshots. Latest
  model and full resume state still save every iteration.

Workers are spawned with one inter-op thread. Pools are bounded and recreated
per iteration, keeping weight synchronization and failure recovery simple.
The measurements include model transfer, process startup, result transfer and
pool shutdown. No persistent distributed inference service was introduced.

## Reliability and measurements

Worker exceptions, interruption and timeout terminate/join the pool. An incomplete
collection is not added to replay or committed as a completed iteration. Resume
from the preceding `training_state.pt`; in-flight games are recomputed using the
same seeds. A hard OS/power failure is recoverable at iteration boundaries, not
at each ply. Do not run two trainers against the same output directory.

New `train`/`smoke` commands reject nonempty output directories. Periodic snapshots
reuse the existing no-overwrite milestone-copy helper. Latest/resume writes retain
the existing temporary-file-and-replace mechanism. They are not a cross-file
transaction; `training_state.pt` remains authoritative after interruption.

Training records now include per-game seeds, moves, outcomes, plies, policy
entropy, visited-action counts, wall/CPU seconds, and the number of distinct
encoded positions retained in the iteration. Entropy measures how spread out the
visit target is; it does **not** measure whether those moves are good. Simulations
per second count the configured P3 simulations on every played ply. Retained
examples exclude capped unfinished games, and draws still receive value zero.

## Hardware and isolated CUDA setup

Measured laptop: Ryzen 7 7840HS, 8 cores/16 logical processors, about 15.2 GiB
usable RAM; RTX 4060 Laptop GPU with 8 GiB VRAM, driver 581.86. Power status was
AC/100% battery. GPU power limit reported 80 W. The working `.venv` remains
Python 3.13.1, PyTorch 2.14.0+cpu and NumPy 2.5.3.

CUDA was installed in **`.venv-cuda`**, leaving `.venv` unchanged. The official
CUDA 12.8 wheel index supplied PyTorch 2.11.0+cu128 for this interpreter. We did
not claim that switching from CPU 2.14 to CUDA 2.11 isolates GPU speed; the direct
CPU/GPU comparison runs both devices in the same 2.11 environment.

Recreate the GPU environment from the repository root:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

`requirements-cuda.txt` pins Torch, NumPy and pytest. The wheel includes the needed
CUDA runtime; no system CUDA toolkit or driver was replaced. This follows the
[official PyTorch installation workflow](https://pytorch.org/get-started/locally/).
Use `.venv\Scripts\python.exe` and `--device cpu` to retain the original CPU setup.
Cross-device or cross-version bitwise training reproducibility is not promised.

## Rerunning the bounded experiments

Use fresh JSON paths. The benchmark never changes its input checkpoint. Network
training measurements update only a temporary in-memory model, then reload the
checkpoint before measuring search. Inference/training rows use warmups and the
median of three rounds; CUDA timings synchronize before/after measured work.
Training microbenchmarks use prepared tensors, so the real pilot is needed to
include replay sampling, augmentation, transfers, startup and checkpoint overhead.

```powershell
python -m benchmarks baseline --checkpoint checkpoints/benchmark/hard.pt --json benchmark_results/new_baseline.json
python -m benchmarks.throughput self-play --workers 4 --threads 1 --games 4 --simulations 25 --max-plies 512 --output benchmark_results/new_selfplay.json
.\.venv-cuda\Scripts\python.exe -m benchmarks.throughput network --device cuda --threads 8 --output benchmark_results/new_cuda.json
.\.venv-cuda\Scripts\python.exe -m benchmarks.throughput network --device cpu --threads 8 --output benchmark_results/new_cpu.json
```

The self-play command uses frozen `benchmark/hard.pt`, seed 20261115, the standard
initial position and a 512-ply cap by default. JSON records versions, config,
checkpoint hash, code identity and every game. CPU-seconds/wall-seconds estimates
busy CPU cores inside games; it excludes process-import overhead. GPU samples use
`nvidia-smi` once per second and can miss brief utilization peaks.

## Batching decision

The network has large measured gains for batches. That does not mean six
independent searches can immediately supply a batch of 64 leaves: each search
waits for its current leaf value before selecting its next leaf. A shared batch
server would require queues, waiting/deadline policy, error propagation and new
tests. Searching several leaves within one tree would also require decisions
about temporary visits and search semantics.

For this iteration we retain independent CPU search and use normal batching for
training. We do not claim a measured speedup from an unimplemented inference
batcher. The practical next test, if self-play remains limiting, is batching at
most the active games' leaf requests; large-batch raw inference rates are an
upper bound, not expected whole-search throughput.

## Measurements on 2026-11-15

The unchanged standard benchmark (original CPU environment) measured legal moves
at 2.87 ms/state, raw batch-1 inference at 1.61 ms/state, and 25-simulation MCTS at
0.154 s/search (162 simulations/s). Results are in `throughput_before.json`.

Worker scaling below uses the identical frozen Hard checkpoint, seed, move cap
and per-game seed assignment. Rates extrapolate these short workloads; they are
not promises of sustained hourly production. All 25-simulation games here drew
by core repetition, so their value targets are zero.

| Games | Workers | Torch threads/worker | Seconds | Completed games/hour | Retained examples/hour |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 1 | 71.11 | 202 | 37,057 |
| 4 | 2 | 1 | 57.77 | 249 | 45,618 |
| 4 | 4 | 1 | 26.45 | 544 | 99,639 |
| 6 | 4 | 1 | 31.77 | 680 | 103,330 |
| 6 | 6 | 1 | 26.17 | 826 | 125,476 |
| 6 | 6 | 2 | 28.58 | 756 | 114,873 |

Four-worker throughput improved 2.69× over the clean serial repeat. Six workers
improved a matched six-game workload by 21%; two threads per worker were slower
and used about eight busy CPU cores instead of 3.82. Choose one thread/worker.
The first serial measurement took 152.63 s during environment installation; it
is retained as `selfplay_w1_25.json` but excluded from the speedup claim. Laptop
power/thermal/background effects remain a source of noise, and these process
scaling rows have only one measured workload each.

Search-budget comparison used four workers/one thread and identical four game
seeds, with a 512-ply cap. Different budgets naturally play different moves.

| Simulations | Seconds | Decisive | Draws | Truncated | Retained examples | Examples/hour |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 26.45 | 0 | 4 | 0 | 732 | 99,639 |
| 50 | 94.34 | 2 | 1 | 1 | 776 | 29,613 |
| 100 | 178.38 | 2 | 0 | 2 | 676 | 13,643 |

Fifty simulations supplied nonzero win/loss targets in this sample; 100 doubled
the truncations without increasing decisive games. Twenty-five is the throughput
winner, but counting its draw positions alone as superior data quality would be
misleading. Fifty is a provisional quality/throughput compromise, not a proven
strength winner. Confirmation requires a larger controlled training comparison.
Policy entropy and visited-action counts are recorded, but different trajectories
prevent interpreting their raw averages as causal evidence of better search.

Same-build CPU/GPU measurements use PyTorch 2.11.0+cu128, eight parent threads,
full precision, AdamW and the existing clipped policy/value loss:

| Batch | CPU training ms | CUDA training ms | Speedup | CUDA training positions/s |
|---:|---:|---:|---:|---:|
| 16 | 12.79 | 6.88 | 1.86× | 2,325 |
| 64 | 24.84 | 6.45 | 3.85× | 9,915 |
| 128 | 42.00 | 5.99 | 7.01× | 21,375 |
| 256 | 83.22 | 8.26 | 10.08× | 31,004 |

Batch 128 gives a short GPU update and a modest number of optimizer steps for our
pilot-sized datasets. Batch 256 maximizes raw sample throughput but halves the
number of updates at the same training-pass budget; no learning comparison proves
that better. Keep one training pass and the existing optimizer/LR/weight decay
until a controlled experiment justifies changing them.

CPU versus CUDA raw inference at batch 1 was 0.756 versus 1.257 ms; at batch 64,
5.666 versus 1.101 ms for the entire batch. Initial-state 25-simulation MCTS was
74.6 versus 94.1 ms. Therefore CPU workers are the measured choice for individual
searches; GPU is the choice for training. GPU samples peaked at 63% utilization,
299 MiB reported memory and 51°C in this microbenchmark. These are sampled values,
not peak-memory profiling. Training is a small part of total runtime, so a 7×
optimizer speedup does not imply a 7× end-to-end improvement.

## Representative pilot and proposed run

The fresh standard-size model pilot is stored separately at
`checkpoints/throughput_pilot_20261115`. It used six one-thread CPU workers,
50 simulations, six games/iteration, cap 512, CUDA training, batch 128, one pass,
20,000 replay capacity, seed 20261115 and symmetry probability 0.5. Iteration 1
finished six decisive games (two White wins, four Black wins), no draws or
truncations. It retained 1,204 examples / 1,193 distinct encoded positions and
made ten updates. Self-play took 63.97 s and training 1.34 s, including the first
CUDA update's startup cost; policy/value/total losses were 8.071/0.698/8.769.
The compute-time rate was 331 completed games/hour and 66,361 examples/hour.
Aggregate search throughput was approximately 941 simulations/s, with about
4.18 busy CPU cores inside games. Startup, task imbalance and unoccupied tail
workers explain why this does not mean six fully busy cores throughout.

A separate CLI resume completed iteration 2: six decisive games (three White,
three Black wins), no draws/truncations, 1,419 new examples / 1,406 distinct
encoded positions, replay 2,623 and twelve updates. Self-play took 79.93 s and
training 1.22 s; policy/value/total losses were 7.477/0.495/7.972. These decreasing
losses do not establish playing strength. Across both iterations the compute-time
rates were **295 games/hour and 64,469 retained examples/hour**, excluding
checkpoint I/O. Both immutable snapshots and latest/resume files were saved.
The full game records remain in that run's `metrics.jsonl`; they can be inspected
without loading the replay checkpoint.

Four paired evaluation games (two openings, 10 simulations, cap 128) against the
frozen untrained seed-0 model took 11.18 s and produced zero wins/losses, three
repetition draws and one truncation. Overall score bounds are 37.5–62.5%, so this
shows **no proven strength improvement**. Files are in
`evaluation_results/throughput_pilot_20261115`. It is a runtime/integration pilot,
not the final 24-game protocol or a controlled comparison of trained search
budgets. No strong claim about the optimal simulation budget follows from it.

Choose the following configuration for the next serious run, retaining the
existing network, AdamW LR 0.001, weight decay 0.0001 and symmetry probability 0.5:

```powershell
.\.venv-cuda\Scripts\python.exe -m p4_training train --output checkpoints/serious_20261115 --iterations 200 --games-per-iteration 6 --self-play-workers 6 --worker-threads 1 --cpu-threads 8 --self-play-device cpu --device cuda --simulations 50 --max-plies 512 --batch-size 128 --training-passes 1 --replay-capacity 20000 --opening-plies 12 --symmetry-probability 0.5 --learning-rate 0.001 --weight-decay 0.0001 --seed 20261115 --snapshot-interval 10
```

This starts a **new** model, not an old benchmark or the pilot. It has not been
launched. Use a new dated directory if that name already exists. Six games were
directly measured; twelve might improve tail balancing but is not yet tested as
a complete training iteration. Keep replay at 20,000 for now rather than tune
an unmeasured capacity: it holds roughly 16–17 pilot-sized iterations, and metrics
will expose how this changes. Keep the cap at 512 and report discarded games;
we have not measured whether extending it provides enough extra finished data.

Latest model and resume state save every iteration; permanent snapshots save
every ten. Inspect metrics at least every ten iterations, particularly examples,
decisive/draw/truncation mix, updates and elapsed time. Evaluate saved snapshots
against the frozen untrained model and Easy/previous selected model using the
fixed paired suite at 25 simulations and cap 512. Evaluation is a separate command,
not an automatic gate; run it between training segments or account for CPU
contention. Confirm gains on held-out openings before claiming strength.

Resume for additional iterations with the saved worker/configuration settings:

```powershell
.\.venv-cuda\Scripts\python.exe -m p4_training resume --output checkpoints/serious_20261115 --iterations 200 --device cuda
```

At the combined pilot rate 200 iterations would take about 4.1 compute hours;
that is an extrapolation, not a duration promise. Longer/capped games can increase
it substantially. Review the first 10–20 iterations before leaving it unattended
for days. A growing loss or a high draw rate alone is not a strength verdict;
use the recorded games and paired evaluation. Pause and investigate persistently
high truncation/zero-example rates before spending days generating unusable data.

## Validation and file map

Full original CPU environment: `python -m pytest -q` — **178 passed, 1 skipped in
54.36s**. The skip is the CUDA hardware test. Full isolated CUDA environment:
the same command — **179 passed in 61.58s**. Tests cover actual spawned processes,
game-index seeds/order and equal targets, parameter/RNG isolation, exact CPU
continuous-versus-resumed updates, failures/timeouts with child cleanup, legacy
resume settings, immutable snapshots, CPU/GPU model portability and updates,
benchmark JSON validity, and existing core/P1/P2/P3/training/integration behavior.
The pilot's four evaluation records also passed core replay verification.

Main new code: `p4_training/workers.py` owns bounded process collection and seed
assignment; `benchmarks/throughput.py` runs reproducible network and self-play
measurements. `config.py`, `trainer.py` and `__main__.py` expose the device/thread/
worker settings, durable per-game audit and progress. `self_play.py` adds diagnostic
counts and timing to the existing game loop; `levels.py` reuses checkpoint copy
safety and rejects orphan model files before new level runs. `recording.py`
adds actual CUDA runtime/device metadata. `requirements-cuda.txt` and `.gitignore`
keep the isolated setup reproducible and outside version control.

Measurements are saved under `benchmark_results/`; pilot model/replay/history
are under `checkpoints/throughput_pilot_20261115`; paired pilot evaluation is under
`evaluation_results/throughput_pilot_20261115`. Existing benchmark model and resume
SHA-256 hashes were checked unchanged. No UI, model architecture or MCTS semantics
were altered, and no serious training run was started.

The exact standard benchmark was repeated afterward in the original CPU
environment, saved as `throughput_after.json`:

| Workload | Before | After |
|---|---:|---:|
| Legal moves, ms/state | 2.8665 | 2.7842 |
| Complete evaluator batch 1, ms/state | 5.3031 | 5.1623 |
| MCTS 5 simulations, seconds | 0.0361 | 0.0335 |
| MCTS 10 simulations, seconds | 0.0618 | 0.0702 |
| MCTS 25 simulations, seconds | 0.1540 | 0.1463 |
| 25-simulation throughput, simulations/s | 162.36 | 170.83 |
| Fixed short self-play, plies/s | 22.74 | 23.07 |
| CPU batch-64 updates/s | 22.96 | 23.21 |

Search/core code was unchanged. The 10-simulation timing regressed by about 14%
while other timings improved; these short-run fluctuations are reported rather
than attributed to worker orchestration. The measured gain claimed here is
concurrent game generation, not an additional single-search optimization.
