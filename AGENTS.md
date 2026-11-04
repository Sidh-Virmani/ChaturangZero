# ChaturangZero — Agent Instructions

## Project Goal

ChaturangZero is an AlphaZero-inspired Chaturanga system.

The stable pipeline is:

Game Core  
→ P1 State / Action Representation  
→ P2 Policy-Value Neural Network  
→ P3 Monte Carlo Tree Search  
→ P4 Self-Play / Training  
→ Integration / UI

The priority from this point onward is **AI quality, search speed, training throughput, evaluation quality and reliability**.

UI redesign is not a priority.

The existing green UI on `main` should remain stable unless a functional change genuinely requires UI work.

Do not merge or base new work on the experimental purple / Three.js UI branch.

---

# Important Existing Architecture

## Game Core

`shared/game_core`

Owns all Chaturanga rules.

It handles:

- legal move generation
- move application
- check
- checkmate
- stalemate
- bare Raja
- Padati → Mantri promotion

Game rules must remain centralized here.

Do not duplicate rules inside MCTS, P1, training or integration.

---

## P1 — Representation

`p1_representation`

Current representation:

- 13 × 8 × 8 state tensor
- 6 White piece planes
- 6 Black piece planes
- 1 side-to-move plane
- 4096 possible action IDs using `from_square * 64 + to_square`
- legal-action masks derived from `game_core`

A valid colour-swap + 180° rotational symmetry already exists and is tested.

---

## P2 — Policy-Value Network

`p2_neural_network`

Current model:

- 64 channels
- 4 residual blocks
- policy head → 4096 logits
- value head → value in [-1, 1]
- approximately 840k parameters

Value is always from the **side-to-move perspective**.

Do not change this convention.

The current model size is adequate for now. Do not enlarge the network until data/search/evaluation limitations have been addressed.

---

## P3 — MCTS

`p3_mcts`

Current search includes:

- policy/value-guided search
- PUCT selection
- Dirichlet noise during self-play
- visit-based policy targets
- temperature
- value backpropagation
- subtree reuse

The major current performance problem is NOT state encoding or raw neural-network inference.

The current bottleneck is repeated game-core work during search:

- repeated legal move generation
- repeated terminal checks
- public validation of already-known legal moves
- eager child GameState creation
- eager MCTS node creation

Preserve public move validation for external callers.

Optimized MCTS may use a trusted internal transition path only when the move came directly from `game_core.legal_moves()`.

---

## P4 — Self-Play / Training

`p4_training`

Current training supports:

- MCTS self-play
- replay buffer
- policy/value training
- checkpoints
- complete resume state
- easy / medium / hard milestone snapshots
- arena comparison

Resume state already stores model, optimizer, replay data, history and RNG states.

Never overwrite existing checkpoints unless explicitly requested.

Always create a new output directory for major training experiments.

---

# Human-Readable Development Log

Every meaningful optimization, feature, training change or evaluation change MUST also be recorded in:

`docs/PROJECT_LOGBOOK.md`

The logbook is intentionally simpler than Git history.

For each meaningful change record:

- date
- short title
- what changed
- why it matters
- measured result if available

Keep each entry approximately 2–5 lines.

Use simple English.

Do not paste implementation details or large code descriptions into the logbook.

Example:

### MCTS lazy child creation
Changed MCTS so child GameStates are created only when the branch is actually explored instead of creating every legal child immediately.  
Result: 25-simulation search improved from X sec → Y sec.

This file is intended to be readable immediately before a project presentation.

---

# Development Rules

1. Measure before optimizing.

2. Run tests after meaningful changes.

3. Do not perform giant rewrites.

4. Preserve stable interfaces between:
   - game core
   - P1
   - P2
   - P3
   - P4
   - integration

5. Do not optimize P1 encoding unless profiling changes significantly; it is currently negligible.

6. Do not optimize replay sampling unless it becomes measurable at larger replay sizes.

7. Do not increase model size merely to make the architecture look more sophisticated.

8. Do not claim a newer checkpoint is stronger based only on:
   - lower loss
   - later iteration number
   - file name such as `hard.pt`

Playing strength must be demonstrated through evaluation.

9. Never silently treat truncated games as wins or losses.

10. Do not start a major multi-day training run until the P0 training/search issues below are addressed.

11. New performance changes should preserve correctness and add/update tests.

12. Do not modify visual theme or add UI polish unless explicitly requested.

---

# Working Order

The following is the intended order of work.

The order may change if measurements reveal a different bottleneck, but deviations should be justified.

---

## PHASE 0 — Freeze and Measure the Current Baseline

Before modifying performance-sensitive code:

Create reproducible benchmark tooling for:

- legal move generation
- MCTS simulations / second
- time for fixed-simulation search
- evaluator throughput
- self-play plies / second
- training batches / second

Preserve the current benchmark checkpoints.

Do not overwrite:

`checkpoints/benchmark/easy.pt`  
`checkpoints/benchmark/medium.pt`  
`checkpoints/benchmark/hard.pt`

Record baseline measurements in `docs/PROJECT_LOGBOOK.md`.

Current audit reference:

- approximately 4.1 sec for a real 25-simulation MCTS search
- legal move generation is expensive relative to raw network inference
- network batch evaluation becomes much more efficient with larger batches

---

# PHASE 1 — Optimize MCTS / Game-Core Interaction

This is currently the highest-value performance work.

Investigate and implement carefully:

### 1. Remove repeated legal-move computation

A single node expansion should not regenerate identical legal information through multiple paths.

### 2. Avoid eager child construction

Do not construct complete GameStates/MCTSNodes for every legal action if most branches will never be explored.

Prefer lazy creation where practical.

### 3. Trusted internal move transition

MCTS already receives legal moves from `game_core`.

Avoid repeating full external validation for those already-known legal moves.

The transformation itself must still remain owned by game_core.

### 4. Avoid repeated terminal calculations

Cache or structure node creation so terminal status is not unnecessarily recalculated.

After each optimization:

- run tests
- run performance benchmark
- record before/after result in `PROJECT_LOGBOOK.md`

Target:

reduce 25-simulation search substantially from the current ~4 second baseline.

Do not sacrifice correctness for speed.

---

# PHASE 2 — Repetition / Draw / Long-Game Handling

Current cyclic games can run until the 512-ply limit and then all generated examples are discarded.

This wastes large amounts of compute.

Implement an explicit and documented policy centrally in game_core.

Do not invent separate training-only rules.

Investigate:

- repetition detection
- project-appropriate draw handling
- terminal outcome representation
- effect on MCTS
- effect on self-play
- effect on arena evaluation

Add tests for cyclic positions and repetition behavior.

Record the chosen rule clearly in the logbook.

---

# PHASE 3 — Use Existing Symmetry for Training

P1 already has valid tested symmetry support.

P4 currently does not use it.

Introduce training augmentation using:

- original example OR
- 180° rotated + colour-swapped equivalent

Transform policy targets consistently.

The value target should remain logically unchanged because perspective transforms with the side to move.

Add tests ensuring:

- state transformation is correct
- policy transformation is correct
- value target remains correct

Record the change in the logbook.

---

# PHASE 4 — Fix Training Update Scheduling

Current training performs a fixed number of optimizer updates even when an iteration generated no new examples.

This previously caused a checkpoint to perform additional training on old replay data despite generating zero new positions.

Change the schedule so optimizer work is related to generated data.

At minimum:

- zero new examples should normally mean zero normal training updates
- replay-only training should require an explicit mode

Prefer a configurable relationship between:

new examples  
→ batch size  
→ effective replay/training passes  
→ update count

Record the chosen rule.

---

# PHASE 5 — Build Credible Evaluation

Current deterministic arena games repeatedly start from the identical initial position, so many matches reproduce the same trajectories.

Replace this with reproducible paired evaluation.

Use a fixed opening-position suite.

For each opening:

Candidate as White  
Candidate as Black

Record per-game:

- candidate/reference checkpoint
- opening ID
- colour assignment
- result
- terminal reason
- plies
- search simulations
- relevant configuration

Aggregate:

- wins
- losses
- draws/truncations
- score
- average plies
- confidence interval where appropriate

Support JSON and/or CSV export.

Useful baselines:

- untrained network with fixed seed
- frozen early checkpoint
- previous selected checkpoint
- optionally a simple deterministic baseline

Do not introduce Elo ratings until there are enough meaningful independent games.

---

# PHASE 6 — Training Metrics and Reproducibility

Training should emit durable metrics outside `training_state.pt`.

Prefer CSV and/or JSONL.

Useful metrics include:

- iteration
- completed games
- truncated games
- generated examples
- unique/repeated states where practical
- replay size
- average plies
- policy loss
- value loss
- total loss
- learning rate
- training updates
- elapsed time
- examples/hour
- plies/second
- checkpoint path
- configuration
- Git commit/revision where practical

Checkpoint metadata should eventually include enough information to reproduce a run.

---

# PHASE 7 — Increase Self-Play Throughput

Once serial MCTS is substantially faster and correctness remains stable:

## Parallel self-play

Introduce a small number of self-play workers.

Start conservatively, for example around 4 workers.

Avoid CPU thread oversubscription.

Measure scaling before increasing worker count.

Preserve reproducibility as far as practical using explicit seeds.

---

## Batched inference

P2 already supports batch evaluation.

Investigate batching leaf evaluations across multiple simultaneous games/searches.

Raw network inference becomes significantly more efficient with batches.

This is more invasive than simple parallel self-play, so implement only after the earlier MCTS optimization is stable.

---

# PHASE 8 — CUDA

The development laptop has an RTX 4060 Laptop GPU, but the audit found CPU-only PyTorch.

Do not assume CUDA automatically speeds up MCTS.

CUDA should first be used where it clearly helps:

- network training
- larger training batches
- batched inference

After installing a CUDA-capable PyTorch build:

- verify CUDA availability
- verify forward/backward correctness
- verify checkpoint portability
- benchmark CPU vs GPU
- record results

Do not change environment blindly without preserving reproducibility/setup instructions.

---

# PHASE 9 — Pilot Training

Do NOT immediately launch the longest possible training run.

First run a small pilot after the above high-priority fixes.

Example scale:

- 4–8 self-play games
- current 64×4 model
- initially around 25 simulations

Measure:

- simulations/sec
- completed games/hour
- plies/hour
- truncation/repetition rate
- unique states
- self-play time
- training time
- generated examples
- losses
- arena result versus frozen baseline

Use these measurements to choose the serious run configuration.

---

# PHASE 10 — Search-Budget Experiment

Compare practical self-play/search budgets such as:

- 25 simulations
- 50 simulations
- possibly 100 simulations

Do not assume more simulations are automatically better.

Compare:

- runtime
- self-play throughput
- completion rate
- policy quality/entropy if useful
- arena strength

Choose the best trade-off for available hardware.

---

# PHASE 11 — Long Training Run

Once the pipeline is optimized and measurable, begin sustained training.

The machine may run for many hours across several days.

Requirements:

- new output directory
- resumable `training_state.pt`
- frequent safe checkpointing
- durable metrics
- no overwrite of old benchmarks
- evaluation checkpoints at meaningful intervals
- recovery after interruption
- clear configuration recording

Possible starting direction after benchmarking:

- current 64-channel / 4-block model
- multiple self-play workers if stable
- 8–16 games per iteration
- selected MCTS simulation budget from experiments
- GPU training if validated
- larger training batch if GPU benefits from it
- periodic arena evaluation

Do not blindly tune hyperparameters during the same run.

---

# PHASE 12 — Model Selection

Do not automatically deploy `latest_model.pt`.

Compare checkpoints.

A final model should demonstrate measurable improvement over:

- untrained baseline
- at least one early checkpoint
- previous frozen checkpoint where appropriate

Select the strongest checkpoint based on evaluation, not iteration number.

Only after this should labels such as Easy / Medium / Hard imply actual playing strength.

---

# PHASE 13 — Optional Advanced Improvements

Only after search, evaluation and training are healthy consider:

- MCTS evaluation cache / transposition table
- learning-rate schedule
- better replay sampling
- candidate-vs-best automatic promotion
- improved difficulty system
- model metadata/checksums
- stronger behavioral stress testing
- larger model
- alternative policy head
- resignation

These are secondary.

Do not allow optional experiments to delay real training.

---

# Tests

The baseline before optimization is:

`92 passed`

All changes should preserve or increase this number unless tests are intentionally replaced with stronger equivalents.

Important future tests should include:

- random legal-game stress tests
- every generated legal move produces valid state
- repetition/draw handling
- lazy MCTS correctness
- optimized transition equivalence
- symmetry augmentation
- opening-suite evaluation
- corrupted/incompatible checkpoint behavior
- parallel self-play reproducibility/sanity

---

# Presentation / Documentation Requirement

The project will be presented orally.

Technical improvements can become difficult to remember after many small commits.

Therefore every meaningful change must leave behind a concise explanation in:

`docs/PROJECT_LOGBOOK.md`

That document should make it possible for a team member to understand the project's improvements in approximately 5–10 minutes before a presentation.

Do not turn the logbook into another technical report.

Keep it concise, factual and measurable.

---

# Definition of Success

The final project should be able to demonstrate statements such as:

- MCTS search became X times faster after optimization.
- Self-play throughput increased from X to Y.
- Cyclic/truncated games decreased significantly.
- Symmetry augmentation improved data efficiency/consistency.
- Training produced substantially more useful positions.
- Later checkpoints beat frozen earlier checkpoints in paired arena evaluation.
- The final model demonstrably learned rather than merely reporting lower loss.

The objective is not to imitate DeepMind-scale AlphaZero.

The objective is to build a technically correct, measurable, well-engineered student implementation whose improvements can be demonstrated and defended.