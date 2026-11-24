# Final demo observability — 2026-11-24

Launch from the repository root:

```powershell
$env:CHATURANGZERO_CHECKPOINT = 'checkpoints/serious_20261116/iteration_000200.pt'
$env:CHATURANGZERO_SIMULATIONS = '50'
.\.venv\Scripts\python.exe run_chaturangzero.py
```

Choose Human vs Machine. The existing green board/client layout remains in place.
No model, search policy, game rules or training data was changed. CPU inference
uses the actual iteration-200 checkpoint; training remains separate.

## What the panel means

- **Model path / iteration / short SHA:** the loaded file, plus provenance read
  only when saved metadata matches its complete SHA-256 fingerprint. Snapshot
  iteration must also match the metadata row. We never infer training facts from
  `iteration_000200.pt` alone. The complete hash is available as the SHA tooltip.
- **Engine line:** actual runtime simulation budget, PUCT constant, inference
  device. These are different from the saved self-play training settings.
- **Training provenance disclosure:** reads matching `metrics.jsonl` audit data,
  or matching `run.json` latest-model data when available. Shows self-play budget,
  workers, replay capacity, symmetry probability, training device and network
  dimensions. Missing/unreadable/unmatched metadata leaves the model playable
  and explicitly reports provenance unavailable. Full replay/resume files are
  never loaded for this UI feature.
- **Pipeline:** Board / Legal Actions → Policy + Value Networks → MCTS / PUCT →
  Selected Action. Both network heads are evaluated in one forward pass, so they
  share one stage rather than pretending they execute independently in sequence.
- **Live count and meter:** existing P3 `on_simulation` callback, published through
  the existing HTTP status endpoint. Poll interval is 75 ms. Fast network stages
  can complete between polls; no delays or fabricated intermediate counts were
  added to the backend. Existing short final-result display remains unchanged.
- **Search time / simulations per second:** `perf_counter` around the real
  `MCTS.search` call, including its root evaluation and callback overhead. Final
  rate is configured simulations divided by measured search seconds. It excludes
  the preliminary root policy/value preview, HTTP latency and UI animation; the
  API also records the wider pipeline elapsed time.
- **Candidates:** top three root children ranked by visits. Each shows the
  network-derived prior, actual visit count and share of all root visits. Visit
  share is not a calibrated probability of winning or statistical confidence.
- **Value:** the original side-to-move network estimate converted to White's
  perspective. Left means Black advantage, right White. It is labelled as the
  position before the machine move, not a value of the new board or an accuracy.
- **Repetition:** one plus exact prior occurrences of the current board and side
  to move in the core's immutable history. The core still enforces the draw;
  this panel only displays the occurrence and third-occurrence rule.

Engine/provenance/repetition are machine-mode fields. Human vs Human retains its
existing flow. No historical performance badge, Elo, win-rate claim or strength
label was added. The best Phase 1B benchmark is a different workload and should
not be confused with current-move timing. Existing untracked formal-evaluation
artifacts were left outside this change and commit.

## Suggested one-minute recording

1. Enter Human vs Machine; pause on iteration 200, SHA and 50-simulation engine line.
2. Open Training provenance briefly: six workers, CUDA, replay and symmetry.
3. Close it and make a legal White move; show the board updating immediately.
4. Capture the live MCTS meter/count while the machine replies. Counts can skip
   between polls; this is actual computation, not a staged animation.
5. Pause on the three candidates: explain prior versus searched visit share.
6. Point out measured search seconds, sim/s and selected action, then its board move.
7. Show the White-perspective value bar and the repetition-rule indicator.

Do not expect a repetition draw to occur within a one-minute machine game; the
indicator truthfully remains 1/3 unless the exact position actually repeats.

## Validation

`npm run build` passes (TypeScript and Vite); no frontend test runner is configured
and no dependency was introduced. New Python tests cover matching hashes and
iterations, unavailable/corrupt metadata, current repetition occurrences, timer
relationships, engine fields, legal moves and Human-vs-Human field isolation.

The launched app's HTTP flow was tested with the real iteration-200 checkpoint
and 50 simulations: human e2→e3, live status 0/16/41, final 50 visits, selected
a7→a6 and White to move. Measured search 0.212 s / 236 sim/s in that check (not
a performance claim or a constant). Reset to Human vs Human removed engine data.

Interactive visual inspection could not be completed: the browser access host
was unavailable, and review rejected inspecting an unrelated private Brave window.
No unrelated page was inspected. Before recording, manually check the panel at
your intended resolution and collapse Training provenance after showing it.

Full regression results: original CPU environment **187 passed, 1 skipped in
57.36s**; CUDA environment **188 passed in 70.53s**. The skipped CPU-environment
test requires CUDA. No long training or formal evaluation was started.
