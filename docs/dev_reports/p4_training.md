# P4 training: teaching ChaturangZero from its own games

## 1. Why P4 exists

The earlier folders each solve one part of the problem. `shared/game_core` knows which moves are legal and when a game has ended. P1 turns a position and a move into fixed numbers. P2 contains a neural network that can score moves and estimate a position. P3 searches ahead with that network.

An untrained P2 model has randomly chosen weights. P4 supplies the experience that can change those weights:

```text
current model
    ↓
P3 searches while the model plays against itself
    ↓
save what the search preferred at each position
    ↓
game_core declares the winner
    ↓
teach P2: "prefer these searched moves; this player eventually won/lost"
    ↓
save the updated model and repeat
```

This is an AlphaZero-style loop scaled for this project. It does not contain a second Chaturanga rule engine, move numbering scheme, network, or search algorithm.

## 2. The existing pieces P4 connects

P4 starts from a `shared.game_core.GameState`. It asks `game_core.terminal_status()` whether the game has ended and uses `game_core.apply_move()` to play the move that P3 selected. The final winner and each player's reward come from the core.

P1's `encode_state(state)` turns the position into 13 numerical 8×8 sheets. Its action space has 4096 slots, and `legal_action_mask(state)` identifies the currently legal slots.

P2's `PolicyValueNetwork` produces raw move scores and a value from the player-to-move's viewpoint. P2's `NetworkEvaluator` converts its scores to legal probabilities for P3. During training, P4 uses P2's `policy_value_loss()` and PyTorch's normal gradient and optimizer steps.

P3's `MCTS.search(state)` returns three things: a 4096-entry policy based on search visits, the selected legal `Move`, and that move's action ID. `MCTS.step(move)` moves its retained search tree forward after play.

P4 orchestrates these public interfaces. The dependencies point in one direction: P4 calls P3, P2, P1, and the core. None of those folders needs to know how P4 stores or trains on games.

## 3. One self-play game, move by move

`play_game(evaluator, config)` starts at `initial_state()` for normal training. It creates one P3 `MCTS` object for that game, enables P3's root exploration noise, and runs the requested number of search simulations per move.

At each turn:

```text
current GameState
    ↓
game_core: is the game over?
    ↓ no
P3 MCTS.search(state)
    ↓
store P1-encoded state and P3's searched move distribution
    ↓
choose a move and apply it through game_core
    ↓
MCTS.step(move) retains the useful child search tree
```

**Exploration noise** is a small random change to search's starting preferences. It encourages self-play to try more than one opening. This is P3's existing Dirichlet-noise feature; P4 only switches it on for self-play.

P3 returns a probability-like policy made from the number of visits each move received. During the configured opening plies, P4 accepts P3's sampled move. Later, P4 selects the highest-probability move from that same P3 result. The stored policy remains P3's full visit-based result, even when the played move is chosen greedily. That gives training more information than storing only the final move.

Normal games start with White in the standard initial position. The smoke test uses a small valid `GameState` where every legal reply captures Black's final non-Raja piece, so a complete game and one update can be checked quickly. It still passes through the real core, P1, P2, and P3.

## 4. What one training example contains

Think of a training example as a note taken at one turn:

> “Here is the board I saw. Here are the moves the search liked. After the game finished, did the player whose turn it was here win?”

`TrainingExample` stores:

1. **State:** the 13×8×8 P1-encoded board. Its zeros and ones are stored as small one-byte numbers in replay to save memory. It is converted back to floating-point numbers before P2 reads it.
2. **Policy target:** the P3 visit distribution in P1's action order. Most of the 4096 entries are zero, so replay stores only the action IDs with positive weight and their probabilities. For example, if action 528 (`a2 → a3`) received 20% of visits, it stores ID `528` with probability `0.20`. `TrainingExample.policy()` reconstructs the full 4096-entry row for P2.
3. **Value target:** `+1` if the player to move in this stored position eventually won, or `-1` if that player lost. The source is `Outcome.reward(side)` from `game_core`.

The policy is stored immediately after each search. The value cannot be filled in until the whole game has a winner. P4 therefore keeps the position's `side_to_move` temporarily. When the core reports the final winner, P4 labels every saved turn. If White loses after a White turn, that example gets `-1`. The next Black-to-move example from the same eventual Black win gets `+1`.

P3's searched policy is a useful teacher because it has looked ahead beyond P2's first guess. P4 trains P2 to imitate those improved search preferences. The outcome target trains P2's value head to estimate the eventual winner from each earlier position.

There is currently no draw rule in `game_core`. If a game reaches `max_game_plies` without a core result, P4 marks it **truncated** and discards its pending examples. It does not invent a draw or claim that either player won. The truncation count is reported so a run with too many unfinished games is visible.

## 5. Replay: learning from more than the newest game

The `ReplayBuffer` holds examples from completed games up to `replay_capacity`. When it fills, it removes the oldest examples and keeps the newest ones. This prevents one long run from growing memory forever.

During an update, P4 samples a **batch**: a small group of examples sent through P2 together. It samples with replacement, so even a small newly started replay buffer can fill a batch. Sampling from recent games gives the model a mixture of positions rather than teaching it only the last position it saw.

Replay is saved inside the full training checkpoint. Resuming a run keeps its previous examples; it does not start learning from an empty buffer. The compact stored policy form saves space, while each sampled batch is expanded to P2's full 4096-action format only when needed.

## 6. How one model update works

For each configured training step, `Trainer`:

```text
samples a batch from replay
    ↓
converts states and targets into PyTorch tensors
    ↓
asks P2 for policy logits and value predictions
    ↓
uses P2 policy_value_loss() to measure both errors
    ↓
PyTorch computes gradients and AdamW updates the model
```

A **gradient** tells a parameter which direction would reduce the measured error. `AdamW` is the optimizer that uses those gradients to adjust the network's weights. P4 also clips unusually large gradients to keep an update stable.

The policy loss rewards logits that match P3's searched move distribution. The value loss rewards values closer to the actual final outcome. P4 reports average policy, value, and combined loss for the steps in each iteration, along with completed/truncated games, wins by color, examples added, replay size, and average game length.

One **iteration** means: play `games_per_iteration` games, add completed examples to replay, run `training_steps` updates if replay has examples, then save checkpoints. If this iteration produces only truncated games but replay already contains older completed games, training can still update from that replay. If replay is empty, the update count is zero.

The updated model is the source of guidance for the next iteration. Lower loss shows that the model fit its recent targets better; it does not prove stronger play by itself. P4 also provides an optional arena comparison to measure results against another checkpoint.

## 7. Checkpoints and resuming

After each iteration, P4 saves two files under the chosen output directory:

- **`latest_model.pt`** is a P2-compatible model-only checkpoint. `p2_neural_network.load_checkpoint()` can load it for inference, including later UI integration. “Latest” means most recently updated; it is not automatically certified as the strongest model.
- **`training_state.pt`** is the full restart point. It contains the model, optimizer, replay examples, finished-iteration count, metrics history, settings, and random-number-generator states. `Trainer.resume()` uses it to continue training rather than merely starting again from the latest weights.

The files are written through temporary files and replaced after successful serialization. The training checkpoint checks P1's state shape and action-space size when loading, so an incompatible representation is rejected rather than silently misread.

The resume command's `--iterations` means **additional iterations**. For example, if the checkpoint says 3 iterations are complete, `resume --iterations 2` runs iterations 4 and 5. It keeps the saved model architecture and training settings; the resume command can choose a device, such as CPU or CUDA.

The smoke command also loads its newly written `latest_model.pt` and evaluates a position with it. That verifies the model-only checkpoint can actually be used again.

## 8. Reproducibility and practical limits

For a new run, `seed` controls PyTorch's model initialization, P3's NumPy-based exploration and move sampling, and replay sampling. The full checkpoint saves these random streams. A test compares an uninterrupted two-iteration run with a one-iteration run resumed for one more iteration and gets the same weights and metrics on CPU.

CPU execution is supported. A normal PyTorch device name can be supplied for an available accelerator. Exact reproducibility across different hardware or different PyTorch versions is not promised; the verified continuation is on the same CPU environment.

Games from the standard initial position may be long, especially while the model is untrained. More simulations improve the search target but cost more time. The normal command is deliberately not started automatically. Watch `games_truncated` and `examples_added`: if many games hit the cap, raise the ply limit or investigate play before trusting the training metrics.

P1's symmetry utility is available for future augmentation experiments. P4 currently trains on actual self-play positions only; it does not silently add transformed examples.

## 9. Optional model comparison

`compare_models(candidate, reference, ...)` plays a chosen number of games with P3 search for both models. It alternates which model controls White and Black. Search uses no exploration noise and chooses the most-visited move. It reports candidate wins, reference wins, and games that reached the ply cap without a core result.

This is an optional evaluation command, separate from the training loop. It does not automatically replace or reject a checkpoint. A tiny match is noisy, so use more games and a suitable search budget before interpreting a win rate as a strength improvement.

## 10. Important public pieces

**`TrainingConfig`** says how much work one invocation will do: games, simulations per move, training steps, replay size, learning rate, device, seed, and checkpoint directory. `TrainingConfig.smoke(path)` gives a small configuration for the quick end-to-end check.

**`play_game(evaluator, config, start_state=None)`** plays one self-play game using a P2 `NetworkEvaluator` and P3 MCTS. It returns a `SelfPlayResult` with completed-game examples and the core outcome, or an empty example list marked truncated. Normal training omits `start_state` and begins at the core's initial position.

**`TrainingExample`** is one encoded position plus its sparse search-policy target and final value target. Its `policy()` method expands the sparse policy to P1's full action-space size when training needs it.

**`ReplayBuffer`** holds recent completed-game examples, samples batches, and is saved and restored with the trainer.

**`Trainer(config)`** owns the current P2 model, evaluator, AdamW optimizer, replay, iteration count, and metrics. `trainer.run()` performs the requested additional iterations. `Trainer.resume(directory, iterations, device)` reconstructs a full saved run. `trainer.save()` writes both checkpoint types.

**`compare_models()`** runs the optional arena between two P2 models without changing either one's weights.

## 11. Exact commands

Run from the repository root in PowerShell. The smoke run is short and uses a forced near-terminal core position:

```powershell
.\.venv\Scripts\python.exe -m p4_training smoke --output checkpoints\my_smoke_run
```

The verified run completed one game in one ply, made one network update, wrote both checkpoints, and reloaded the model. Use a fresh output directory for each new smoke or training run. To start normal training from the standard initial position:

```powershell
.\.venv\Scripts\python.exe -m p4_training train --output checkpoints\run1 --iterations 1 --games-per-iteration 2 --simulations 25 --training-steps 50
```

That command can take substantial time because it searches on every move. The defaults are 512 maximum plies, a 64-channel/four-block P2 model, batch size 64, replay capacity 20,000, CPU, and seed 0. Use `--max-plies`, `--batch-size`, `--channels`, `--blocks`, `--opening-plies`, and the other CLI flags to adjust a new run.

To continue the saved run for two **more** iterations:

```powershell
.\.venv\Scripts\python.exe -m p4_training resume --output checkpoints\run1 --iterations 2
```

For the final demo, one command can start a normal run or resume it after an interruption, save three permanent strength milestones, and stop at the Hard iteration:

```powershell
.\.venv\Scripts\python.exe -m p4_training train-levels --output checkpoints\final_models --easy 10 --medium 30 --hard 60
```

The first invocation creates a normal model using the same defaults as `train`. Later invocations of the exact command find `training_state.pt` and restore the same model, optimizer, replay examples, random streams, and saved training settings. They continue from the next unfinished iteration rather than starting over. On resume, options such as games, simulations, batch size, and architecture come from `training_state.pt`; `--device` may select the device used for the resumed process.

At iterations 10, 30, and 60, the command copies the exact current P2 model to `easy.pt`, `medium.pt`, and `hard.pt`. It opens milestone names in no-overwrite mode, so an existing level file is kept. If a milestone iteration passed without its file, the command stops with an error because the exact earlier model can no longer be reconstructed. `latest_model.pt` and `training_state.pt` are still refreshed after every iteration, and the command stops automatically after iteration 60.

Each completed iteration prints its metrics plus a line such as:

```text
iteration=12/60 elapsed=00:18:42 eta=14:57:36
```

The remaining-time estimate uses the average duration of iterations completed during the current invocation. It becomes more useful after several iterations. The normal defaults remain 2 games per iteration, 25 MCTS simulations per move, a 512-ply game cap, 50 network updates per iteration, batch size 64, and CPU. Use `--device cuda` only on a machine where PyTorch reports CUDA support.

To compare two model checkpoints after training:

```powershell
.\.venv\Scripts\python.exe -m p4_training arena --candidate checkpoints\run1\latest_model.pt --reference checkpoints\baseline\latest_model.pt --games 10 --simulations 25
```

`baseline` in that example must be a checkpoint you already saved. Each training iteration prints one JSON line of metrics and the paths to the model and resume checkpoints.

## 12. What the tests prove

The P4 tests use the real P3/P2/P1/core stack to play a complete short game and inspect its example. They check the encoded board, legal action IDs, normalized search policy, and final reward. Another test supplies a two-move game ending in a Black win and checks that the preceding White-to-move example gets `-1` while the Black-to-move example gets `+1`.

Other tests confirm that an unfinished game is discarded rather than labeled with an invented result, replay drops its oldest entries at capacity, a training step changes model parameters, both checkpoints are usable, and resume preserves replay, optimizer, random streams, and iteration history. The optional arena test verifies that candidate and reference swap colors.

The full project regression suite also covers P1, P2, P3, the game core, and its API:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Result after adding final-demo milestone training: **90 passed**.

The resume CLI was also run in a separate process against the smoke checkpoint. It advanced to iteration 2, retained the first game's replay example, and made another update. Its new game reached the smoke configuration's four-ply cap, so it correctly added zero new examples and reported one truncated game.

## 13. File map

**`p4_training/config.py`** holds the validated run settings and the small smoke configuration.

**`p4_training/self_play.py`** calls P3 and the game core to generate games, then attaches core rewards to the positions seen during completed games.

**`p4_training/data.py`** stores compact examples and manages the bounded replay buffer.

**`p4_training/trainer.py`** connects self-play, replay sampling, P2 updates, metrics, and full checkpoint/resume state.

**`p4_training/levels.py`** runs one continuous trainer up to Easy, Medium, and Hard iteration milestones, preserves each level checkpoint, and calculates per-iteration timing estimates.

**`p4_training/arena.py`** contains the optional model-versus-model comparison.

**`p4_training/__main__.py`** supplies the `smoke`, `train`, `train-levels`, `resume`, and `arena` commands. **`p4_training/__init__.py`** exposes the small Python API.

**`tests/test_training_pipeline.py`** also checks continuous milestone training, exact Hard/latest equivalence, resume, milestone validation, and no-overwrite behavior. The existing P1/P2/P3/core tests remain unchanged.

## 14. How a trained checkpoint can later reach the UI

The UI work is a separate step. Its Python server can eventually load `checkpoints/run1/latest_model.pt` with P2's `load_checkpoint()`, wrap the model in `NetworkEvaluator`, and give that evaluator to P3 `MCTS`. When the AI must move, the server asks P3 to search its current core `GameState`, then applies the selected move through `game_core`. The browser only displays the resulting state.

```text
trained P4 checkpoint
    ↓ load_checkpoint()
P2 NetworkEvaluator
    ↓
P3 MCTS.search(current core GameState)
    ↓ selected Move
game_core.apply_move()
    ↓
UI displays the updated game
```

The current UI/server has not been changed by P4. Choose a checkpoint for play using game results, not merely a low training loss. `latest_model.pt` is ready for loading, while `training_state.pt` is for continuing training.
