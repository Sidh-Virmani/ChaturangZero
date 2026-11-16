# Serious training from a normal PowerShell terminal

Run everything below from the repository root. Codex is not needed after launch.
The measured configuration is unchanged: six CPU workers with one Torch thread
each, CPU self-play, CUDA training, 50 simulations, six games/iteration, batch 128,
one training pass, replay 20,000, and snapshots every ten iterations. The two-
iteration pilot projected roughly 295 games/hour and 64,500 examples/hour, but
these are short-run estimates. Game length, truncation and thermals affect them.

## Readiness and start

The original CPU `.venv` remains usable. The installed `.venv-cuda` uses the pinned
`requirements-cuda.txt` setup from `dev_reports/throughput_and_pilot.md`.

```powershell
.\.venv-cuda\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
git status --short
git rev-parse HEAD
Test-Path checkpoints/serious_20261116
```

The last check should print False. The command below refuses a nonempty existing
run directory. Keep the laptop plugged in, ventilated, and set Windows sleep on
AC power to Never for the run. Keep this terminal open. The GPU can be mostly idle
while CPU self-play runs; CUDA updates happen afterward, not concurrently with
self-play. This is intentional and was measured in the pilot.

**Start once:**

```powershell
.\.venv-cuda\Scripts\python.exe -u -m p4_training train --output checkpoints/serious_20261116 --iterations 200 --games-per-iteration 6 --self-play-workers 6 --worker-threads 1 --cpu-threads 8 --self-play-device cpu --device cuda --simulations 50 --max-plies 512 --batch-size 128 --training-passes 1 --replay-capacity 20000 --snapshot-interval 10 --seed 20261115
```

Other settings remain the measured defaults: opening exploration 12 plies,
symmetry probability 0.5, AdamW LR 0.001, weight decay 0.0001, standard 64×4 model.
The seed matches the pilot; the output directory is new. No serious job was run
while preparing this workflow. Do not use `train` again to continue this run.
The rough 200-iteration estimate is 4.1 compute hours; it is not a guarantee.

The terminal prints the absolute target, completed games, a waiting heartbeat
while parallel games run, wall seconds for each committed iteration, elapsed time,
ETA, and concise JSON metrics. ETA averages completed iterations in the current
invocation, including checkpoint writes; before the first completion it is unknown.
The initial checkpoint is saved before the first game, once initialization succeeds.

Readiness validation on 2026-11-16: full CPU suite **184 passed, 1 skipped in
80.26s**; full CUDA-environment suite **185 passed in 96.48s**. Tests include
first-game interruption recovery, duplicate-writer rejection/release, exact-target
resume/no-op, real worker cleanup and GPU update/checkpoint portability. Existing
benchmark checkpoint hashes are unchanged. The serious output directory was
confirmed absent after validation; no 200-iteration job was started.

## Stop and resume

Use Ctrl+C once, then wait for the terminal prompt and worker cleanup. The CLI
exits with code 130 and prints the recovery path. It does not save partial optimizer
work or half-finished games. The preceding committed iteration is authoritative;
resume repeats uncommitted work with the saved game seeds and RNG streams.

**Resume to the original total of 200:**

```powershell
.\.venv-cuda\Scripts\python.exe -u -m p4_training resume --output checkpoints/serious_20261116 --until 200 --device cuda
```

This loads the existing model, optimizer, replay, history and RNG state. Missing,
corrupt or incompatible resume state raises an error; there is no fresh-model
fallback. Already at/past 200 means no updates. Existing `--iterations N` still
means N **additional** iterations; use `--until` to avoid extending the run by
accident. Saved architecture/worker/thread/schedule settings are restored.

The same resume command applies after terminal closure, Windows restart or an
intentional pause. A forced termination/power failure skips Python cleanup and
may leave temporary files; resume uses `training_state.pt`, not temporary files
or a newer-looking `latest_model.pt`. If there is no committed checkpoint because
termination happened during initial initialization/save, stop and inspect the
directory rather than overwrite it. Keep completed checkpoints backed up.

One trainer at a time may write to a directory. CLI training commands hold an OS
writer lock in `.training.lock`. The OS releases the lock on process exit, including
crash/restart. The file intentionally remains; its existence does not mean a
trainer is alive. Do not delete it to bypass an active lock. This protects CLI
commands; custom scripts that call `Trainer` directly must coordinate themselves.

Windows normally sends terminal-close events to attached child processes, but
abrupt OS termination is not a graceful-pause guarantee. Prefer Ctrl+C. If CPU
work persists after closing the terminal, inspect the command lines before
restarting (do not kill unrelated Python programs):

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Select-Object ProcessId,ParentProcessId,CommandLine
```

## Files and recovery boundary

- `latest_model.pt`: model-only P2 checkpoint refreshed each iteration.
- `training_state.pt`: complete committed restart state, also initially iteration 0.
- `iteration_000010.pt`, `iteration_000020.pt`, etc.: no-overwrite model snapshots.
- `metrics.jsonl`: one JSON object per committed iteration, including game counts,
  examples, losses, replay size, updates, compute timings, per-game seeds/moves/
  outcomes, config and code/model fingerprints. Empty before iteration 1.
- `run.json`: readable latest committed iteration/config/environment and file hashes.
- `.training.lock`: persistent file backing the automatically released writer lock.

Checkpoints and logs use temporary files followed by replacement. They are separate
files, not a single transaction. A crash after the resume checkpoint commits but
before text logs are written may leave text records behind; the next successful
save rebuilds them from checkpoint history. If interrupted between checkpoint and
snapshot copy, that interval's snapshot may be missing; inspect snapshots after
resume rather than assuming every name exists. Resume state still retains the
committed weights. A corrupted committed resume checkpoint requires a backup;
model-only snapshots cannot restore optimizer/replay/RNG. No automatic recovery
from disk corruption is claimed.

## Monitor from another PowerShell terminal

```powershell
# Last recorded committed iteration and config (lightweight; no model load)
Get-Content checkpoints/serious_20261116/run.json -Raw | ConvertFrom-Json | Select-Object completed_iterations,config,latest_model

# Recent health metrics
Get-Content checkpoints/serious_20261116/metrics.jsonl -Tail 10 | ForEach-Object { $_ | ConvertFrom-Json } | Select-Object iteration,games_completed,games_truncated,white_wins,black_wins,draws,examples_added,replay_size,updates,policy_loss,value_loss,total_loss,self_play_seconds,training_seconds,elapsed_seconds | Format-Table -AutoSize

# Available permanent snapshots
Get-ChildItem checkpoints/serious_20261116/iteration_*.pt | Select-Object Name,Length,LastWriteTime

# GPU samples (Ctrl+C stops only this monitor)
nvidia-smi -l 5
```

Rerun the first two commands to refresh. Logs are atomically replaced rather than
appended in place; `Get-Content -Wait` can follow an old file handle, so use repeated
reads instead. `elapsed_seconds` in JSON excludes save/log I/O; terminal wall time
includes it. Neither `run.json` nor metrics claims that a process is currently
alive. The running terminal provides the live heartbeat.

At iterations 10–20, check that counts advance; new examples and replay grow
(replay eventually caps at 20,000); losses are finite; updates agree with
`ceil(examples_added / 128)`; and snapshots at 10/20 exist and load. Zero examples
correctly mean zero normal updates, but repeated zero-example iterations or many
512-ply truncations warrant pausing. Inspect decisive/draw mix and time per game.
Do not expect monotonically falling losses or interpret falling loss as strength.
Watch for sustained thermal throttling, disk pressure and worker errors.

## Evaluate and choose models

Use immutable snapshots, not a changing latest file. Pause training to evaluate
without CPU contention. The default frozen development suite supplies 12 openings
in both colours: 24 games, same search budget, no noise, deterministic selection.
Each output directory must be new. The following are instructions for later;
the serious snapshots do not yet exist.

```powershell
# First health comparison: iteration 20 against frozen untrained seed 0
.\.venv-cuda\Scripts\python.exe -m p4_training evaluate --candidate checkpoints/serious_20261116/iteration_000020.pt --reference checkpoints/baselines/untrained_seed0/model.pt --output evaluation_results/serious_20_untrained --simulations 25 --max-plies 512 --device cpu

# Later improvement over the early serious checkpoint
.\.venv-cuda\Scripts\python.exe -m p4_training evaluate --candidate checkpoints/serious_20261116/iteration_000100.pt --reference checkpoints/serious_20261116/iteration_000010.pt --output evaluation_results/serious_100_10 --simulations 25 --max-plies 512 --device cpu

# Compare late candidates; iteration 200 is not automatically best
.\.venv-cuda\Scripts\python.exe -m p4_training evaluate --candidate checkpoints/serious_20261116/iteration_000200.pt --reference checkpoints/serious_20261116/iteration_000100.pt --output evaluation_results/serious_200_100 --simulations 25 --max-plies 512 --device cpu

.\.venv-cuda\Scripts\python.exe -m p4_training verify-evaluation --output evaluation_results/serious_200_100
Get-Content evaluation_results/serious_200_100/summary.json -Raw
```

Shortlist using complete W/L/draw/truncation evidence across frozen opponents.
Confirm the shortlist on a separate preselected opening suite, not seeds chosen
after inspecting results. Create this file once, keep it unchanged and add
`--openings evaluation_results/serious_holdout24.json` to each final comparison:

```powershell
.\.venv-cuda\Scripts\python.exe -m p4_training make-openings --output evaluation_results/serious_holdout24.json --count 24 --seed 20261116
```

That gives 48 games per final comparison. High truncation leaves the comparison
unresolved; do not count it as a draw or invent Elo/independent-game confidence.
If evidence is inconclusive, retain that conclusion and collect more resolved,
varied paired evidence. Existing evaluation tooling already exports everything
needed; no duplicate selection framework was added.

## What to bring back and later demo configuration

Send `metrics.jsonl`, `run.json`, the terminal's final output/errors and the list
of snapshots (names/sizes). Send complete evaluation directories: manifest,
opening suite, games JSONL/CSV and summary. Keep all snapshots and full resume
state locally; selected `.pt` files will be needed for further comparisons.
Also keep the code revision/environment used to run training. Code hashes record
identity but cannot reconstruct missing dirty source; preserve this checkout or
commit/archive it before the run (no commit is performed by these instructions).

After evidence selects a checkpoint, use its immutable path and hash. Integration's
`configured_checkpoint()` in `integration/ai_player.py` reads
`CHATURANGZERO_CHECKPOINT`; `MachinePlayer` loads it on CPU through P2 and P3.
The server also reads `CHATURANGZERO_SIMULATIONS` (default 32). For example, only
**after selection**, set the winning snapshot path in the launch terminal:

```powershell
$env:CHATURANGZERO_CHECKPOINT = 'checkpoints/serious_20261116/iteration_000100.pt'
$env:CHATURANGZERO_SIMULATIONS = '50'
.\.venv\Scripts\python.exe run_chaturangzero.py
```

Iteration 100 is an example, not a selected model. Restart the backend after
changing the environment. No UI or demo checkpoint has been switched here.
