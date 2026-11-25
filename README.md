# ChaturangZero

ChaturangZero is an AlphaZero-style learning and search system for Chaturanga. It contains a rule-complete Python game core, a 13x8x8 board representation, a residual policy-value network, MCTS/PUCT search, self-play training, reproducible benchmarks/evaluation, and a React Human-vs-Human / Human-vs-Machine client.

## Repository structure

- `shared/game_core/` - authoritative Chaturanga rules and HTTP game API
- `p1_representation/` - state/action encoding and symmetry transforms
- `p2_neural_network/` - policy-value residual network and evaluator
- `p3_mcts/` - MCTS/PUCT search
- `p4_training/` - self-play, replay, training, evaluation, workers and experiment recording
- `integration/` - machine player / model integration
- `shared/ui/` - React/Vite client
- `benchmarks/`, `benchmark_results/` - reproducible performance workloads and results
- `evaluation_results/` - recorded evaluation artifacts
- `docs/PROJECT_LOGBOOK.md` - concise engineering history and measured changes

## Quick start

Requirements: Python 3, Node.js/npm, and Git. CPU-only execution is sufficient for the game and tests.

```powershell
git clone https://github.com/Sidh-Virmani/ChaturangZero.git
cd ChaturangZero

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd shared\ui
npm install
cd ..\..

python run_chaturangzero.py
```

The launcher starts the Python game-core API on `http://127.0.0.1:8000` and the Vite UI on `http://127.0.0.1:5173`.

## Human vs Machine

Machine mode requires a local `.pt` checkpoint. Set its path before launching:

```powershell
$env:CHATURANGZERO_CHECKPOINT = 'checkpoints\path\to\model.pt'
$env:CHATURANGZERO_SIMULATIONS = '50'
python run_chaturangzero.py
```

The submitted serious run used the iteration-200 checkpoint produced by the training command below.

## Tests

```powershell
pytest -q
```

The final project state passed 187 CPU tests with one CUDA-only skip; the CUDA environment passed 188 tests.

## Reproducible performance benchmark

```powershell
python -m benchmarks.baseline
```

Committed benchmark artifacts are in `benchmark_results/`. The principal measured search optimization reduced the fixed 25-simulation workload from 3.965 s to 0.137 s and increased measured throughput from 6.31 to 182.06 simulations/s.

## CUDA environment

A separate CUDA dependency file is provided:

```powershell
python -m venv .venv-cuda
.\.venv-cuda\Scripts\Activate.ps1
pip install -r requirements-cuda.txt
```

CUDA is used for batched optimizer updates. Self-play search remains on CPU because the measured single-state search latency was lower there.

## Reproduce the serious training configuration

```powershell
.\.venv-cuda\Scripts\python.exe -u -m p4_training train `
  --output checkpoints/serious_20261116 `
  --iterations 200 `
  --games-per-iteration 6 `
  --self-play-workers 6 `
  --worker-threads 1 `
  --cpu-threads 8 `
  --self-play-device cpu `
  --device cuda `
  --simulations 50 `
  --max-plies 512 `
  --batch-size 128 `
  --training-passes 1 `
  --replay-capacity 20000 `
  --snapshot-interval 10 `
  --seed 20261115
```

Resume safely to the same total iteration target with:

```powershell
.\.venv-cuda\Scripts\python.exe -u -m p4_training resume `
  --output checkpoints/serious_20261116 `
  --until 200 `
  --device cuda
```

The submitted run completed 200 iterations, 1,195 self-play games and 76,611 generated training examples in about 150.6 minutes wall-clock on the development laptop.

## Evaluation

The evaluation layer uses 12 fixed openings with both colour assignments, deterministic search, per-game JSONL/CSV records, checkpoint hashes and replay verification.

Example:

```powershell
python -m p4_training evaluate `
  --candidate <candidate.pt> `
  --reference <reference.pt> `
  --output evaluation_results/example `
  --simulations 25 `
  --max-plies 512 `
  --device cpu

python -m p4_training verify-evaluation --output evaluation_results/example
```

## Notes

- The third occurrence of the same board and side-to-move is treated as a project anti-cycle draw rule; this is an engineering rule for self-play, not a claim about historical Chaturanga rules.
- Training uses optional 180-degree rotation plus colour-swap symmetry augmentation.
- `docs/PROJECT_LOGBOOK.md` records the motivation and measured outcome of the major post-midsemester changes.
