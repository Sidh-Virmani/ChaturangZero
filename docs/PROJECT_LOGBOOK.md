# ChaturangZero — Project Logbook

This is the short human-readable record of meaningful project improvements.

Purpose: before a presentation, reading this file should quickly answer:

- What did we change?
- Why did we change it?
- Did it actually improve anything?

Keep entries short. Git history contains implementation detail; this file contains the story.

---

## Baseline — Before Optimization

### Stable AlphaZero-style pipeline

The project currently has a complete working path:

Game Core → Representation → Policy-Value Network → MCTS → Self-Play / Training → Machine Player.

Human vs Human and Human vs Machine both work, and the full test suite passes.

**Baseline tests:** 92 passed.

---

### Existing model

Current neural network:

- 13 × 8 × 8 input
- 64 channels
- 4 residual blocks
- 4096-action policy head
- value head in [-1, 1]
- ~840k trainable parameters

The model size is currently considered sufficient; search/data quality are larger limitations.

---

### Existing training

Training already supports:

- self-play
- replay buffer
- policy/value optimization
- resumable training
- model checkpoints
- Easy / Medium / Hard milestone checkpoints

The old benchmark contains very little real training data, so the milestone names do **not** currently prove actual playing strength.

---

### Existing benchmark data

Old benchmark training produced approximately:

- 4 completed games
- 1,212 replay positions
- 1,082 unique positions

The third benchmark iteration produced two truncated games and zero new examples, but still performed training updates on old replay data.

Therefore `hard.pt` is newer, but is not yet proven stronger than `medium.pt`.

---

### Main performance problem found

Profiling showed that MCTS spends most of its time repeatedly doing game-rule work rather than running the neural network.

A 25-simulation search took approximately **4.1 seconds** on the audited machine.

Major causes:

- repeated legal move generation
- repeated terminal checks
- eager creation of child states
- validation of moves that MCTS already knows are legal

This is the first major optimization target.

---

### Neural network is not currently the main bottleneck

Approximate measured raw network inference:

- batch size 1: ~0.70 ms/state
- batch size 64: ~0.10 ms/state

State encoding itself was only around ~0.006 ms.

Therefore optimizing representation code is currently low priority.

---

### Cyclic/truncated games

The game currently has no explicit draw/repetition outcome.

Long cyclic games can continue until the maximum ply limit and then contribute **zero training examples**.

This wastes both MCTS compute and training-data generation time.

Repetition/draw handling must be resolved before serious long training.

---

### Symmetry is implemented but unused

P1 already supports a valid:

180° rotation + colour swap

but P4 training currently does not use it for augmentation.

Existing checkpoints showed large disagreement between equivalent symmetric positions, so enabling symmetry augmentation is a high-value data-quality improvement.

---

### Evaluation is not yet trustworthy

Arena games currently use deterministic search and repeatedly start from the exact same initial board.

Therefore running many games often repeats essentially the same two trajectories.

Before claiming that one checkpoint is stronger than another, arena evaluation must use a varied fixed opening suite with paired colour-swapped matches.

---

### Training throughput limitation

Self-play currently generates games sequentially and neural-network leaf evaluations happen one state at a time.

This underuses:

- CPU cores
- P2 batch inference
- available RTX 4060 GPU

Parallel self-play and batched inference will be investigated after basic MCTS optimization.

---

### GPU status

The audit found CPU-only PyTorch.

The laptop has an RTX 4060 Laptop GPU.

CUDA should later be enabled primarily for:

- training
- larger batches
- batched inference

GPU support alone will not fix slow MCTS because the current dominant bottleneck is game-rule computation.

---

# Planned Improvement Order

1. Add reproducible performance benchmark.
2. Remove repeated game-core work inside MCTS.
3. Add lazy/trusted child transitions where safe.
4. Handle repetition/draw/long cyclic games.
5. Enable symmetry training augmentation.
6. Fix training updates when no new data is generated.
7. Build proper paired arena evaluation.
8. Export durable training/evaluation metrics.
9. Parallelize self-play.
10. Investigate batched inference.
11. Enable/test CUDA.
12. Run small training pilots.
13. Compare MCTS simulation budgets.
14. Start sustained multi-day training.
15. Evaluate checkpoints and select the genuinely strongest model.
16. Add secondary optimizations only if measurements justify them.

---

# New Entries

Add new work BELOW this point in chronological order.

Use this format:

## YYYY-MM-DD — Short change name

**Changed:** One short explanation of what was modified.

**Why:** Why the change matters.

**Result:** Before/after measurement, observed effect, or `Not measured yet`.

---

<!-- Example

## 2026-11-03 — Lazy MCTS child creation

**Changed:** MCTS now creates child GameStates only when a branch is actually explored instead of constructing every legal child immediately.

**Why:** Most eagerly created children were never searched and were wasting game-core computation.

**Result:** 25-simulation search improved from 4.10 s → 1.35 s.

-->

## 2026-11-05 — Baseline Performance Benchmark

**Changed:** Added a reproducible benchmark using fixed initial, midgame and endgame positions plus the existing `hard.pt` checkpoint.

**Why:** Future search and training optimizations now have one stable workload and machine-readable baseline for before/after comparison.

**Result:** Legal moves averaged **1.60 ms/state**; raw network inference was **0.77 ms/state** at batch 1 and **0.098 ms/state** at batch 64; a 25-simulation MCTS search took **3.97 s** at **6.31 simulations/s**; CPU training reached **37.67 steps/s** for batch 64.

**Conclusion:** State encoding is negligible at **0.0049 ms/state**. Search/game-rule overhead is much larger than raw neural-network inference and remains the first optimization target.

## 2026-11-05 — MCTS expansion optimization (Phase 1A)

**Changed:** MCTS stores lightweight legal edges and creates child states only when selected. Core-owned trusted transitions avoid revalidating known legal moves; each node shares one legal-move calculation with its terminal check.
**Why:** Avoids constructing and validating branches that search never visits while preserving PUCT and public move validation.
**Result:** Same checkpoint/workload: 25-simulation search **3.965 → 0.247 s** (16.1×), throughput **6.31 → 101.34 simulations/s**, short self-play **1.83 → 19.09 plies/s**. Unchanged raw NN and core operations measured slower in this run (legal moves **1.595 → 2.706 ms**, evaluator **2.394 → 5.108 ms**); results are recorded without attributing those differences to this optimization.

## 2026-11-05 — Reused legal moves during MCTS evaluation (Phase 1B)

**Changed:** MCTS passes its core-generated legal moves to an explicit trusted evaluator method; ordinary single and batch evaluation keep generating their own masks.
**Why:** Each evaluated search state previously generated legal moves twice.
**Result:** Profiled legal generations fell **52 → 26** per 25-simulation search. Same benchmark/checkpoint: search **0.247 → 0.137 s**, throughput **101.34 → 182.06 simulations/s**, short self-play **19.09 → 26.79 plies/s**. Ordinary evaluator batch 1 stayed **5.108 → 5.100 ms/state**. All **106 tests passed**; candidate-state construction/validation inside legal generation remains the largest profile cost.

## 2026-11-10 — Repetition / draw handling (Phase 2)

**Changed:** Added an automatic draw on the third occurrence of the same board and player to move, tracked immutably by the core. This is a documented project anti-cycle rule; existing Chaturanga wins keep precedence.
**Why:** Cyclic games previously reached the training cap and lost all their examples. Draws now finish normally with value 0, and arena/server/training metrics distinguish them from wins and truncations.
**Result:** A fixed knight cycle now draws at ply 8 and retains 8 zero-value examples; a cap at ply 7 still truncates. Standard benchmark: 25-simulation search **0.137 → 0.164 s**, **182.06 → 152.54 simulations/s**, short self-play **26.79 → 20.06 plies/s**. Raw NN also slowed **1.233 → 1.736 ms/state**, so the measured regression cannot all be attributed to history tracking.
**Tests:** 116 passed; existing win rules, promotion, lazy search, server reset, draw targets and replay/resume remain covered.

## 2026-11-11 — Training symmetry augmentation (Phase 3)

**Changed:** Each sampled example can use its 180° rotated, colour-swapped board and matching move targets (50% by default). The value stays the same and replay stores only the original.
**Why:** The network can learn from both versions of a position without playing extra games. The saved sampling RNG makes the choice reproducible on resume.
**Result:** Forced-augmentation smoke training and resume each completed one game and one finite update. The frozen `hard.pt` still shows mean value gap 0.473 and policy L1 gap 0.766 across three symmetric position pairs; this is a baseline, not a strength result.

## 2026-11-11 — Data-scaled training updates (Phase 4)

**Changed** Normal updates now equal `ceil(new examples × training passes / batch size)`, with one pass by default. Fixed-step replay fitting requires an explicit setting, and older checkpoints migrate to the scaled schedule.
**Why** An iteration with no new positions previously continued fitting old replay and could report falling loss without new self-play information.
**Result** A controlled run generated one example and performed two finite updates at three passes/batch two. After resume, a truncated game generated zero examples and performed zero updates despite populated replay, preserving all model weights.
**Tests** 150 passed. The existing benchmark state also loads with the migrated schedule and a zero-update budget for zero new examples.

## 2026-11-14 — Paired evaluation and durable experiment records

**Changed:** Added a frozen 12-opening suite with both colour assignments, replay-verifiable game records, checkpoint/code fingerprints and JSON/CSV results. Training now exports iteration metrics, timings and configuration beside resume state.
**Why:** Repeating deterministic games from one board overstated the amount of evidence, while training history was difficult to inspect outside checkpoints.
**Result:** Three preliminary six-game comparisons of Hard against Easy, Medium and a fixed untrained model produced 11 repetition draws and 7 truncations, with no decisive games; this does not establish a strength ranking. All 18 exported games passed core replay verification, and smoke/resume logs preserved the zero-new-data/zero-update rule.
**Tests:** 165 passed. Existing benchmark checkpoint hashes are unchanged; no long training run was started.

## 2026-11-15 — Measured parallel self-play and CUDA training

**Changed:** Added seeded CPU self-play workers, thread controls, failure cleanup and permanent iteration snapshots; enabled CUDA in a separate pinned environment.
**Why:** Independent games can use more CPU cores while batched network updates benefit from the RTX 4060. One thread per worker avoided oversubscription.
**Result:** Matched four-game workload improved **71.1 → 26.4 s** with four workers (2.69×); six workers beat four by another 21% on six games. CUDA batch-128 updates were **42.0 → 6.0 ms** versus CPU on the same build. A resumed 12-game pilot at 50 simulations retained **2,623 examples**, approximately **295 games/hour / 64,500 examples/hour** over its short compute window; these are extrapolations, not sustained-run guarantees.
**Decision:** Recommend six CPU workers/one thread each and CUDA batch-128 training. Fifty simulations supplied decisive targets where the small frozen-model 25-simulation sample only drew; 100 cost more and truncated more. Tiny pilot evaluation established no strength gain. CPU tests: **178 passed, 1 skipped**; CUDA environment: **179 passed**. No serious run started; existing benchmark hashes are unchanged.

## 2026-11-16 — Unattended training safeguards

**Changed:** Training saves a restart point before the first game, rejects concurrent CLI writers to one directory, and prints wall-time/ETA progress. Resume can stop at an exact total with `--until 200` and gives a clear Ctrl+C recovery message.
**Why:** Interrupting iteration 1 previously had no saved restart point, and resuming with additional iterations could accidentally extend the planned run.
**Result:** Tiny tests recover initial weights after first-game interruption, reject a second writer, and verify that repeating an already-completed target performs no updates. The serious run was not launched; the operational runbook is `docs/SERIOUS_TRAINING.md`.
**Tests:** Full CPU suite **184 passed, 1 skipped**; CUDA environment **185 passed**. Existing benchmark checkpoint hashes are unchanged.
