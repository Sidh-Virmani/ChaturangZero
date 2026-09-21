# P2 neural network: turning a position into move advice and a position estimate

## 1. What problem does P2 solve?

P1 translates a Chaturanga position into numbers. Its `encode_state()` function gives us 13 board-shaped sheets containing zeros and ones. Those sheets accurately describe the pieces and whose turn it is, but they do not contain an opinion about the position.

P2 adds the part that can learn an opinion.

```text
P1's 13 numerical 8x8 sheets
                 |
                 v
         P2 neural network
           /           \
          v             v
   policy output     value output
  "which moves?"   "how promising?"
```

The **policy** gives a score for every action ID defined by P1. It answers, “Which moves look worth considering?” The **value** is one number that answers, “From the player-to-move's point of view, how promising is this position?”

At first the network has random parameters, so its answers are not good Chaturanga advice. Later P4 can improve those parameters using self-play examples. P3 can already call the same interface while it builds and tests MCTS.

P2 never decides whether a Chaturanga move is legal. It relies on P1's `legal_action_mask()`, which in turn relies on `game_core.legal_moves()`.

## 2. Input to the network

P1 describes one position as 13 separate 8×8 number grids:

```text
12 grids: locations of each side and piece kind
 1 grid:  whose turn it is
```

PyTorch calls a collection of regularly arranged numbers a **tensor**. For one position, P1's shape is `(13, 8, 8)`: 13 sheets, 8 rows, and 8 columns. Neural-network training and efficient evaluation normally process several positions together. This group is called a **batch**.

The model therefore receives:

```text
(number of positions, 13 sheets, 8 rows, 8 columns)
```

In tensor notation this is `(batch, 13, 8, 8)`. A batch of 32 positions has shape `(32, 13, 8, 8)`. A single position still needs a batch dimension and therefore has shape `(1, 13, 8, 8)`.

`PolicyValueNetwork.forward()` accepts a floating-point PyTorch tensor in this format. P4 will use this direct tensor interface while training. P3 does not have to make tensors: `NetworkEvaluator.evaluate(state)` accepts an ordinary `game_core.GameState`, calls P1's `encode_state()`, and handles the conversion.

## 3. The main network body

### What the shared body is trying to learn

The first part of the network looks at the board and builds useful internal features. A **feature** is a pattern that may help with many decisions: an attacked Raja, an open Ratha line, nearby defenders, or a Padati approaching promotion. Nobody hard-codes those examples. Training adjusts the network so useful patterns emerge.

The default model starts with 64 feature sheets and then passes them through four residual blocks:

```text
13 P1 input sheets
        |
3x3 convolution -> normalization -> ReLU
        |
64 learned feature sheets
        |
residual block 1
        |
residual block 2
        |
residual block 3
        |
residual block 4
        |
shared board understanding
        |
        +-------------------+
        |                   |
        v                   v
   policy head          value head
```

### Convolution: a small window that scans the board

A **convolution** applies the same small detector at every board location. Our main convolutions look at a 3×3 neighbourhood. The detector first examines the area around `a1`, then around `b1`, and so on. Because the same detector is reused across the board, a pattern learned near one square can also be recognized elsewhere.

This fits a board game well. Relationships between nearby pieces matter, and repeating one detector is much more efficient than learning unrelated rules for every square. Padding keeps the board at 8×8 after each convolution; edge squares do not disappear.

### ReLU: keep useful positive signals

After a convolution, the network uses **ReLU**, a simple rule that changes negative numbers to zero and leaves positive numbers alone. This lets several layers combine patterns in ways that a single straight-line calculation cannot.

### Batch normalization: keep feature scales manageable

**Batch normalization** learns a stable scale and offset for each feature sheet. During training, it also tracks running statistics. This generally makes a residual network easier to optimize because values are less likely to grow or shrink unpredictably between layers.

During evaluation, `NetworkEvaluator` temporarily puts the model in evaluation mode. That uses the saved running statistics and prevents an MCTS query from changing them. It restores the model's previous training/evaluation mode afterward.

### Residual blocks: learn a useful correction

Each `ResidualBlock` performs two 3×3 convolution steps. It then adds the block's original input back to the result:

```text
input features ----------------------+
      |                              |
      v                              |
convolution -> ReLU -> convolution   |
      |                              |
      +------------------------------+
                     add -> ReLU -> output
```

This shortcut is the **residual connection**. Instead of forcing every block to rebuild all previous knowledge, a block can learn a correction: “keep these features, but add this new relationship.” The shortcut also gives gradients a clearer route through the model during training.

Four blocks with 64 features are large enough to form several levels of spatial patterns, while remaining practical for a student project on CPU or a consumer GPU. The default network has **840,583 trainable parameters**. A parameter is one adjustable number learned during training. This is far smaller than a large production AlphaZero model, but it is a real residual policy/value network rather than a one-layer demonstration.

`ModelConfig` lets experiments change the width, block count, policy-head width, and value hidden size without editing the model code. Checkpoints save this configuration with the learned parameters.

## 4. The policy head

### What policy means

A **policy** is a set of preferences over possible actions. Imagine a tiny game with four action IDs. The network might initially output these raw scores:

```text
action ID:       0     1     2     3
raw score:     -0.2   1.4   0.6  -0.1
```

These raw scores are called **logits**. A logit is not yet a probability. It can be negative, does not have to add to one, and is best understood as “more is preferred.”

The real policy head returns one logit for each P1 action ID. P1 defines `ACTION_SPACE_SIZE = 4096`, so the output for a batch is shaped `(batch, 4096)`. Entry `policy_logits[b, 528]` always refers to P1 action 528, which is `a2 -> a3`. P2 does not create a second action numbering system.

### How the head is built

The policy head compresses the 64 shared feature sheets into two policy-specific 8×8 sheets using a 1×1 convolution. It then flattens those 128 numbers into one row and uses a fully connected layer to produce 4096 logits.

A **1×1 convolution** combines the feature sheets at each individual square without mixing neighbouring squares again. A **fully connected layer** allows every resulting feature to contribute to every action score. This is useful because an action contains both a starting and destination square.

### Turning logits into legal probabilities

The trainable model returns logits. This is intentional: PyTorch loss calculations are more numerically stable when they receive raw logits.

P3 needs probabilities. `NetworkEvaluator` performs these steps:

```text
4096 network logits
        |
P1 legal_action_mask(state)
        |
illegal slots are removed from consideration
        |
softmax over legal slots only
        |
4096 probabilities: illegal = 0, legal entries sum to 1
```

**Softmax** converts scores into nonnegative shares that add to one. In the tiny example, if only actions 1 and 3 are legal, actions 0 and 2 receive exactly zero. Softmax divides all probability between 1 and 3 according to their logits.

For a nonterminal Chaturanga position, `Evaluation.policy` is a NumPy array with 4096 entries. Illegal actions are exactly `0.0`; legal entries are positive and sum to approximately `1.0`. For a terminal state with no legal moves, all entries are zero. The core still owns the legality decision.

## 5. The value head

The policy head compares moves. The **value head** summarizes the position with one number.

The convention is exact and must be shared by P2, P3, and P4:

> The value is from the perspective of the player whose turn it is in the input `GameState`.

The final `tanh` function restricts it to `[-1, 1]`:

- `+1` means the network expects the current player to win.
- `0` means the network sees the position as roughly balanced or uncertain.
- `-1` means the network expects the current player to lose.

For example, if Black is to move and the model returns `+0.7`, that is favorable for Black. If White is to move and the model returns `-0.7`, that is unfavorable for White. The sign is never permanently tied to White or Black.

This convention fits MCTS because turns alternate down the tree. When P3 carries a child value back to its parent, it will normally negate the sign once per change of player. That sign handling belongs to P3; P2 always reports the leaf position from its own side-to-move perspective.

The value head compresses the shared features to one 8×8 sheet, flattens its 64 cells, passes them through 128 learned hidden values, and produces the final scalar. It can use evidence from the entire board while sharing the expensive feature extraction with the policy head.

An untrained network's value is only a bounded random estimate. The range is guaranteed by the architecture; accuracy must come from future training.

## 6. One complete forward pass

Consider the initial position:

```text
1. game_core provides initial_state().
             |
2. P1 encode_state() creates 13 numerical 8x8 sheets.
             |
3. P2's stem and four residual blocks build 64 feature sheets.
             |
             +------------------------------+
             |                              |
4a. policy head creates              4b. value head creates
    4096 raw logits.                     one number in [-1, 1].
             |                              |
5. NetworkEvaluator asks P1 for the legal-action mask.
             |
6. It converts only legal logits into probabilities.
             |
7. P3 receives Evaluation(policy, value).
```

The policy may say that action 528 (`a2 -> a3`) has probability 0.06 and another legal action has 0.10. Every illegal action is zero. The value may be `+0.03`, meaning the untrained network currently sees the position as nearly neutral for White, who is to move. P3 can use these numbers as search guidance; it must still use `game_core` to generate and apply moves.

For several leaf positions, P3 may call `evaluate_batch(states)`. P1 encodes each state, PyTorch processes them together, and results are returned in the same order.

## 7. Important public classes and functions

### `PolicyValueNetwork(config=None)`

**In simple words:** This is the learnable machine. It turns batches of P1-encoded positions into raw move scores and position values.

**Input:** A floating-point PyTorch tensor shaped `(batch, 13, 8, 8)`.

**Output:** `(policy_logits, values)`, shaped `(batch, 4096)` and `(batch,)`.

**Who uses it later:** P4 uses this direct interface for forward passes, losses, and backpropagation. `NetworkEvaluator` owns the normal P3 inference path.

**Example:** `logits, values = model(encoded_batch)`.

### `NetworkEvaluator(model, device="cpu")`

**In simple words:** This wraps the raw model in a safe game-facing interface. It encodes core states, disables gradient bookkeeping, masks illegal actions, and returns probabilities.

**Input:** The model and a PyTorch device such as `"cpu"` or `"cuda"`.

**Output:** An evaluator object. Construction moves the model to that device.

**Who uses it later:** P3.

**Example:** `evaluator = NetworkEvaluator(model, "cpu")`.

### `NetworkEvaluator.evaluate(state)`

**In simple words:** Evaluate one normal `GameState` without knowing PyTorch internals.

**Input:** One `shared.game_core.GameState`.

**Output:** An `Evaluation` with a full 4096-entry legal probability array and one Python `float` value.

**Who uses it later:** P3 when expanding one leaf.

**Example:** `result = evaluator.evaluate(state)`; then `result.policy[action_id]` and `result.value`.

### `NetworkEvaluator.evaluate_batch(states)`

**In simple words:** Evaluate several core states together for better hardware use.

**Input:** A sequence of `GameState` objects.

**Output:** A tuple of `Evaluation` objects in the same order. An empty sequence returns an empty tuple.

**Who uses it later:** P3 if it adds batched leaf evaluation.

### `policy_value_loss(logits, values, target_policy, target_values)`

**In simple words:** Measure how far the model's two answers are from training targets.

**Input:** Raw policy logits, predicted values, full 4096-entry target policies, and target values from the current-player perspective.

**Output:** `PolicyValueLoss(total, policy, value)`. `total` is the sum used for backpropagation; the two parts are available for logging.

**Who uses it later:** P4's training loop.

### `save_checkpoint(model, path)` and `load_checkpoint(path, device="cpu")`

**In simple words:** Save the model's learned numbers and architecture, then reconstruct it later.

**Input/output:** Saving accepts a model and file path. Loading accepts the path and desired device and returns an evaluation-mode `PolicyValueNetwork`.

**Who uses it later:** P4 for model checkpoints, and P3 or an application when loading a trained model.

## 8. Important design decisions

### Four residual blocks and 64 feature sheets

We chose a default model with four residual blocks, 64 shared channels, and 840,583 trainable parameters. It matches the proposal's lower practical range and is small enough for student hardware. A much larger AlphaZero model would slow self-play and experiments. A shallow fully connected model would throw away the useful spatial structure of an 8×8 board.

### One shared body and two heads

Policy and value both need to understand the same position. Sharing the convolutional body avoids calculating two separate board representations. Separate final heads let one specialize in move preference and the other in overall outcome.

### Raw logits from the model, legal probabilities from the evaluator

Training works best with logits, while P3 wants usable priors. Keeping both interfaces explicit avoids hiding softmax inside the trainable model. `NetworkEvaluator` masks with P1 and then applies softmax, so an illegal move can never gain a playable probability merely because its raw logit is high.

An alternative was to make P3 mask logits itself. The chosen evaluator reduces duplicated boundary code and gives the teammate a direct, rule-safe result while still keeping legality in P1.

### Current-player value perspective

The value follows the project proposal: positive means good for the player to move. A fixed-White perspective was possible, but it would force every search and training caller to inspect colors and convert signs. Current-player perspective matches alternating MCTS nodes naturally, provided P3 negates values while backing them through each ply.

### Full P1 action space

The policy has exactly `ACTION_SPACE_SIZE` outputs. A smaller custom move format could reduce the last layer, but it would duplicate or replace P1's established action contract and create integration risk.

### Normal PyTorch device handling

The model works on CPU. `model.to(device)` and `NetworkEvaluator(..., device)` also support a normal PyTorch accelerator when available. No CUDA-only code is present. Checkpoint loading uses CPU by default so a model saved elsewhere remains usable on a CPU machine.

## 9. Training concepts implemented in P2

P2 implements the model and its loss calculation. It does not implement replay buffers, game generation, epochs, optimizers, or self-play orchestration.

The **policy loss** compares the model's logits with a target distribution, such as future MCTS visit proportions. If MCTS strongly preferred one legal action, the loss pushes that action's logit upward relative to others. It uses soft-target cross entropy, so the target may spread weight across several actions.

The **value loss** is mean squared error. It measures the squared distance between the predicted value and the final outcome target. A future example with value target `+1` says that the player who was to move in that stored state eventually won; `-1` says that player lost.

The total returned loss is:

```text
total loss = policy loss + value loss
```

Calling `total.backward()` makes PyTorch calculate **gradients**: for each trainable parameter, which direction would reduce the error? P4 will later choose an optimizer and use those gradients to update the parameters. Weight decay/L2 regularization belongs in that future optimizer configuration rather than being silently embedded in this helper.

A saved model file is called a **checkpoint**. P2's checkpoint stores `ModelConfig` and model parameters. Optimizer state, training counters, replay data, and best-model promotion belong to P4's larger training checkpoint if P4 needs them.

## 10. P2 to P3 MCTS contract

P3 should use this interface:

```python
from p2_neural_network import NetworkEvaluator, PolicyValueNetwork, load_checkpoint

model = load_checkpoint(path)          # or PolicyValueNetwork() while untrained
evaluator = NetworkEvaluator(model, device="cpu")
evaluation = evaluator.evaluate(state)
```

The exact contract is:

- Pass `evaluate()` one `shared.game_core.GameState`.
- Receive `evaluation.policy`, a NumPy `float32` array shaped `(4096,)`.
- Its index order is exactly P1's action order. Use P1 `move_to_action(move)` or `action_to_move(index)`.
- Illegal entries are exactly zero. Legal entries are positive and sum to approximately one.
- For terminal states with no legal moves, the entire policy is zero. P3 should normally detect terminal states through `game_core` and use the exact result rather than asking the network to estimate one.
- Receive `evaluation.value`, a Python float in `[-1, 1]` from the input state's **side-to-move perspective**.
- When backing a child value up to its parent player, P3 normally negates it once per ply.
- `evaluate_batch(states)` returns results in input order and may be used for batched leaves.
- Evaluation uses `torch.inference_mode()`, does not calculate gradients, does not update parameters or normalization statistics, and restores the model's prior train/eval mode.
- `NetworkEvaluator` moves its model to the selected device when constructed. Do not simultaneously train and search with the same model object from unsynchronized threads.

P3 must continue to use `game_core` to generate/apply moves. The probabilities are guidance, not permission.

## 11. P2 to P4 training contract

P4 can rely on these facts:

- `PolicyValueNetwork` consumes floating tensors shaped `(batch, *STATE_SHAPE)`.
- It returns raw logits shaped `(batch, ACTION_SPACE_SIZE)` and values shaped `(batch,)`.
- Target policies should use the same full action order and shape as P1. Illegal entries should be zero; each target row should normally sum to one.
- Target values must use the current-player convention in `[-1, 1]`.
- `policy_value_loss()` returns differentiable total, policy, and value losses.
- Normal `model.train()`, `loss.total.backward()`, and optimizer steps work.
- `save_checkpoint()` and `load_checkpoint()` preserve architecture and predictions. P4 may wrap these with additional training-state storage later.
- P1 symmetry may augment an encoded state and policy, but P4 must transform state, action targets, and outcome perspective consistently. P2 does not perform augmentation automatically.

## 12. Tests: what are we proving?

The forward tests encode real initial and midgame `GameState` values through P1. They verify one and several positions produce exactly 4096 logits per position and one value per position. They also verify all outputs are finite, values remain inside `[-1, 1]`, and model evaluation is repeatable.

The gradient test calculates policy and value losses, calls `backward()`, and checks that every trainable branch receives finite gradients. This proves the architecture is connected for future training rather than merely producing numbers.

The evaluator tests compare its nonzero policy entries with P1's actual legal-action mask. They prove illegal actions receive zero, legal probabilities sum to one, initial and midgame batches preserve order, terminal policy is all zero, input states remain unchanged, and the caller's model mode is restored.

The checkpoint test saves a model, loads it on CPU, and verifies its policy logits and values exactly match the original predictions. Shape and type error tests catch malformed network inputs.

The complete regression run also includes every existing P1 representation, game-core, and game API test:

```text
.venv\Scripts\python.exe -m pytest tests -q
```

Result: **67 passed** in **4.28 seconds** on the CPU environment used for this implementation.

## 13. File map

**`p2_neural_network/model.py`** defines `ModelConfig`, the residual block, and `PolicyValueNetwork`. This is the trainable architecture.

**`p2_neural_network/inference.py`** defines `NetworkEvaluator` and `Evaluation`. This is the clean P3-facing boundary that encodes states and masks policies.

**`p2_neural_network/training.py`** contains only the policy/value loss calculation that naturally belongs with the model. It does not run a training loop.

**`p2_neural_network/checkpoint.py`** saves and restores model configuration and learned parameters.

**`p2_neural_network/__init__.py`** exposes the small public P2 API without requiring callers to know the internal file layout.

**`tests/test_neural_network.py`** verifies shapes, values, batching, gradients, masking, CPU inference, and checkpoint behavior. **`requirements.txt`** now declares PyTorch.

## 14. Final big-picture summary

```text
shared/game_core
    owns pieces, positions, legal moves, and exact results
             |
             v
p1_representation
    turns GameState into 13 number sheets
    gives every Move one of 4096 action IDs
    supplies the legal-action mask
             |
             v
p2_neural_network  <-- completed here
    learns spatial board features
    returns move scores and a current-player value
             |
             v
p3_mcts
    can use P2 as guidance while game_core controls legality
    (developed separately; not implemented by P2)
             |
             v
p4 training and self-play
    can eventually create targets and improve P2's parameters
    (not implemented yet)
```

P2 is now a complete network component and integration boundary. It is not yet a trained Chaturanga player. Its usefulness will grow when P3 searches with it and P4 supplies learning examples.
