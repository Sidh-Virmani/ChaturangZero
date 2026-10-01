# Human vs Machine integration

The game page now uses the existing Python AI stack. White remains the human side. The browser requests White's legal moves and submits a White move through `GameSession`, which still uses `game_core` for legality. If the game continues, the browser asks `/api/game/ai/move` for Black's reply. The server loads a P4 `latest_model.pt` checkpoint with P2, evaluates the core state through P1, searches with P3 MCTS, and applies the selected core `Move` through the same `GameSession`. The browser does not contain move rules or model code.

The Machine Analysis console displays data from that exact Black search: P1's legal-action count and fixed action-space size; P2's masked legal policy priors and side-to-move value; P3's simulation count, PUCT constant, root visit count, child visit counts, and selected action. “Selection → Expansion → Evaluation → Backup” names the stages P3 performs on each simulation; it is an algorithm outline, not a fabricated per-simulation log. The Value shown is from **Black's perspective**, because Black was to move when that state was evaluated. The interface waits for a complete search response before showing measured values.

From the repository root, start the app with `start-chaturangzero.bat`. The launcher prefers `.venv\Scripts\python.exe` when present. If no checkpoint is specified, the server uses `checkpoints\p4_smoke\latest_model.pt` when that file exists. This is a tiny, one-update integration checkpoint; its playing strength is unproven and the UI labels it accordingly. If no checkpoint exists, machine moves are unavailable and the page explains why.

To create a short **pipeline smoke run** that completes one forced near-terminal game and makes one network update:

```powershell
.\.venv\Scripts\python.exe -m p4_training smoke --output checkpoints\my_smoke_run
```

It saves `checkpoints\my_smoke_run\latest_model.pt` for play and `training_state.pt` for resuming. For a small run starting from the normal initial position, use:

```powershell
.\.venv\Scripts\python.exe -m p4_training train --output checkpoints\run1 --iterations 1 --games-per-iteration 1 --simulations 4 --training-steps 2 --batch-size 8 --channels 16 --blocks 1
```

The normal-position run can still take time because search is repeated for each move. P4 only trains from **completed** games; a capped, unfinished game does not become training data. Continue that run for one additional iteration with:

```powershell
.\.venv\Scripts\python.exe -m p4_training resume --output checkpoints\run1 --iterations 1
```

To play against its checkpoint, stop the app and start it again in PowerShell with:

```powershell
$env:CHATURANGZERO_CHECKPOINT = "checkpoints\run1\latest_model.pt"
.\start-chaturangzero.bat
```

`CHATURANGZERO_SIMULATIONS` optionally changes the number of MCTS simulations per machine move; the default is 32. A checkpoint is loaded when Human vs Machine starts. Restart the app after changing the checkpoint. The machine chooses the highest root visit count (`temperature=0`) without self-play exploration noise, so the same checkpoint and position produce a stable choice. More self-play, evaluation, and checkpoint selection are needed before claiming playing strength.
