# ChaturangZero Architecture & Team Plan

> Original design proposal. The implemented system, setup instructions, and
> measured results are documented in [README.md](README.md),
> [docs/MODEL_CARD.md](docs/MODEL_CARD.md), and
> [docs/COMPLETION_AUDIT.md](docs/COMPLETION_AUDIT.md).

This document explains the project structure in plain language so that any teammate — or any coding assistant such as Codex or Claude — can understand what each module is responsible for, how the modules connect, and what should be built first.

## 1. High-level idea

ChaturangZero is a small AlphaZero-style Chaturanga engine.

The final system should work like this:

```text
Chaturanga Game Core
        ↓
State / Action Representation
        ↓
Policy-Value Neural Network
        ↓
Monte Carlo Tree Search (MCTS)
        ↓
Move chosen
        ↓
Self-play games
        ↓
Replay Buffer
        ↓
Training
        ↓
Stronger Neural Network
        ↺
```

The important idea is:

- The **Game Core** knows the rules.
- The **Representation module** converts game positions and moves into numbers the AI can learn from.
- The **Neural Network** gives an informed guess about which moves look good and whether a position looks winning or losing.
- **MCTS** searches ahead using those neural-network guesses.
- The **Training system** makes the engine play itself, saves those games, and trains the neural network from them.

---

## 2. Shared foundation: Game Core + basic UI

The Game Core should NOT permanently belong to one person.

All four members depend on it, so it should be treated as a shared foundation and reviewed by everyone.

The shared Game Core should eventually provide:

```text
legal_moves(state)
apply_move(state, move)
is_terminal(state)
result(state)
clone/copy_state(state)
```

It must handle:

- board state
- piece movement
- legal move generation
- turn switching
- check/checkmate or whatever exact terminal rules the team freezes
- draw conditions
- promotion rules
- win/loss detection

A very basic UI should also be built early.

The first UI only needs to:

- draw the 8×8 board
- show pieces
- allow clicking/selecting a piece
- highlight legal moves
- apply moves through the Game Core
- switch turns
- show game-over state

This UI is primarily a **visual debugger**. It should not contain its own separate rules logic.

The polished UI comes much later.

---

# 3. Final four-person AI split

## Person 1 — Representation & Data Efficiency

This person's job is to decide how a Chaturanga position and move are represented for the AI.

They sit between the raw game engine and the neural network.

### Main responsibilities

#### State encoding

Convert a board position into a tensor.

Example:

```text
Chaturanga board
      ↓
multi-channel 8×8 tensor
```

The proposal suggests roughly 13 channels, for example piece-location planes plus turn information.

#### Action encoding

Convert a move into an integer the policy network understands.

The current proposal uses:

```text
64 possible FROM squares
×
64 possible TO squares
=
4096 possible action indices
```

So Person 1 owns functions conceptually like:

```text
encode_state(state)
encode_action(move)
decode_action(index)
legal_action_mask(state)
```

#### Legal-action masking

The neural network may output scores for all 4096 possible from/to combinations.

Most of those will be illegal in any given position.

Person 1 provides a mask so the AI only considers legal moves.

#### Canonical representation

Instead of forcing the network to learn separate versions of equivalent situations for both sides, positions can potentially be encoded from the current player's perspective.

The exact approach should be tested rather than assumed.

#### Data augmentation

If a board transformation such as a left-right reflection preserves the frozen Chaturanga rules, the same self-play position may be transformed into another valid training example.

This can improve data efficiency without generating another entire self-play game.

### AI/ML experiments Person 1 can own

Compare different representation choices and measure whether they improve learning.

Examples:

```text
basic piece planes
vs
current-player canonical encoding
vs
canonical encoding + symmetry augmentation
```

Measure things such as:

- training convergence
- policy loss
- value loss
- playing strength
- win rate against fixed baselines

This makes Person 1 a genuine ML/data role rather than only a rule-engine role.

---

## Person 2 — Policy-Value Neural Network

This person owns the learned model.

The neural network receives an encoded board and produces two outputs.

### Policy output

The policy tells us:

> Which legal moves currently look promising?

Conceptually:

```text
Board
 ↓
Neural Network
 ↓
move A: 0.45
move B: 0.30
move C: 0.15
...
```

It is not the final decision. MCTS will use these probabilities as guidance.

### Value output

The value tells us:

> How good does this position look for the current player?

Typical range:

```text
-1 = likely losing
 0 = roughly equal
+1 = likely winning
```

### Main responsibilities

- PyTorch model
- convolutional / residual backbone
- policy head
- value head
- forward pass
- loss computation
- optimizer
- model save/load
- CPU/GPU support

The proposal suggests a lightweight residual network rather than a huge AlphaZero network.

### Experiments Person 2 can own

Compare model sizes such as:

```text
2 residual blocks / 32 channels
4 residual blocks / 64 channels
6 residual blocks / 64 channels
```

Compare:

- training speed
- inference speed
- policy/value losses
- actual playing strength

---

## Person 3 — Monte Carlo Tree Search (MCTS)

This person owns the search algorithm.

The Neural Network gives a quick informed guess.

MCTS performs actual lookahead.

A useful mental model is:

```text
Neural Network = intuition
MCTS           = search / calculation
```

### How MCTS works at a high level

For the current board:

1. Start at the root.
2. Select a promising child.
3. Continue down the tree.
4. Expand a new position.
5. Ask the Neural Network for:
   - policy probabilities
   - value estimate
6. Backpropagate the result through the visited nodes.
7. Repeat many times.

The proposal suggests roughly 30–50 simulations per move during training.

After search, moves with more visits become more likely to be chosen.

### Main responsibilities

- MCTS node structure
- child nodes
- visit counts
- prior probabilities
- Q/value statistics
- selection
- expansion
- backpropagation
- PUCT score
- root exploration noise
- temperature-based move selection

### Experiments Person 3 can own

Compare:

```text
25 vs 50 vs 100 simulations
different PUCT constants
different temperature values
with/without exploration noise
```

Measure both playing strength and inference time.

---

## Person 4 — Self-Play, Training & Evaluation

This person owns the reinforcement-learning loop.

This module connects the entire system and repeatedly improves the model.

### Self-play

The current model plays games against itself.

During every move, MCTS produces an improved move distribution.

For each position, store approximately:

```text
(state, MCTS_policy, final_game_result)
```

These examples become the training data.

There is no external Chaturanga dataset required for the basic AlphaZero-style plan.

### Replay buffer

Stores recent self-play training examples.

The trainer samples mini-batches from this buffer instead of training only on the newest game.

### Training loop

Conceptually:

```text
current model
     ↓
self-play games
     ↓
replay buffer
     ↓
sample batches
     ↓
train model
     ↓
new candidate model
```

### Candidate-vs-best evaluation

A newer checkpoint should not automatically become the best model.

A stronger setup is:

```text
Best model
    ↓
generate/train candidate
    ↓
Candidate vs Best arena
    ↓
candidate clearly stronger?
   /   \
 yes    no
  ↓      ↓
promote  reject/continue training
```

### Main responsibilities

- self-play workers
- replay buffer
- batch sampling
- training orchestration
- checkpoint management
- candidate-vs-best evaluation
- random/baseline opponents
- win-rate metrics
- learning curves
- optional Elo-style rating

---

# 4. Why NN and MCTS are separate

They solve different problems.

## Neural Network

Looks at one position and gives a fast learned estimate:

```text
"These moves look promising."
"This position looks good/bad."
```

It learns patterns from previous self-play.

## MCTS

Uses those estimates to search possible future move sequences before making the current move.

So:

```text
NN = learned intuition
MCTS = lookahead using that intuition
```

They improve each other:

```text
better NN
  ↓
better MCTS searches
  ↓
better self-play targets
  ↓
better NN
```

---

# 5. Recommended repository structure

```text
ChaturangZero/
│
├── shared/
│   ├── game_core/
│   └── ui/
│
├── p1_representation/
├── p2_neural_network/
├── p3_mcts/
├── p4_training/
│
├── integration/
├── tests/
├── docs/
├── data/
└── checkpoints/
```

## Folder ownership

```text
shared/game_core/
    Shared by all four.
    Exact game rules and legal-move implementation.

shared/ui/
    Shared.
    Basic board first, polished UI later.

p1_representation/
    State encoding, action encoding, legal-action masks,
    canonical representation, data augmentation.

p2_neural_network/
    Policy-value PyTorch model and model experiments.

p3_mcts/
    MCTS / PUCT search.

p4_training/
    Self-play, replay buffer, training loop,
    checkpoints and evaluation.

integration/
    Code that wires all modules together.

tests/
    Automated tests across modules.

docs/
    Architecture, rules and experiment documentation.

data/
    Generated self-play data. Usually ignored by Git.

checkpoints/
    Trained model weights. Usually ignored by Git.
```

---

# 6. Development order

Do NOT attempt to build AlphaZero all at once.

Use this order:

```text
1. Freeze exact Chaturanga rules
        ↓
2. Shared Game Core
        ↓
3. Basic human-vs-human visual board
        ↓
4. Automated Game Core tests
        ↓
5. Random-vs-random full games
        ↓
6. P1 representation functions
        ↓
7. P2 neural-network forward pass
        ↓
8. P3 MCTS with an untrained network
        ↓
9. P4 one complete self-play game
        ↓
10. Replay buffer + one training step
        ↓
11. Full self-play/train loop
        ↓
12. Candidate-vs-best evaluation
        ↓
13. Run experiments and improve strength
        ↓
14. Polish final UI
```

---

# 7. Important interfaces to agree on

Before the four people code independently, agree on the conceptual API.

The exact syntax can change later, but the responsibilities should be stable.

## Game Core

```text
legal_moves(state)
apply_move(state, move)
is_terminal(state)
result(state)
clone_state(state)
```

## Representation

```text
encode_state(state)
encode_action(move)
decode_action(action_index)
legal_action_mask(state)
```

## Neural Network

```text
predict(encoded_state)
    → policy_logits, value
```

## MCTS

```text
search(state, model)
    → improved_policy
```

## Self-play

```text
play_game(model)
    → list of (state, improved_policy, result)
```

These interfaces allow the four people to work independently without tightly coupling their internal code.

---

# 8. First milestone

The first milestone is NOT "make a strong AI".

It is:

> Two humans can play a complete legal Chaturanga game through the basic UI, with the UI using the Game Core for every move.

Then:

> Two random agents can complete many legal games without crashes or impossible states.

Only after those are reliable should serious AI training begin.

A bug in the rules engine would cause the AI to learn the wrong game.

---

# 9. What we are deliberately NOT adding yet

Do not add unrelated techniques just to make the project sound more advanced.

Avoid, unless there is later a strong experimental reason:

- Transformers
- genetic algorithms
- Q-learning as a second unrelated agent
- handcrafted chess evaluation functions
- external expert-game datasets
- complicated distributed infrastructure

The project's strength should come from implementing and evaluating the AlphaZero-style loop well.

---

# 10. Final goal

A successful final project should demonstrate:

1. Correct Chaturanga rules.
2. A working policy-value network.
3. Neural-guided MCTS.
4. Self-play reinforcement learning.
5. Measurable improvement across checkpoints.
6. Strong performance against random / simple baselines.
7. Candidate-vs-best model evaluation.
8. A polished human-playable interface.

The important academic result is not "we made an unbeatable engine".

It is:

> We built a complete AlphaZero-style learning system for Chaturanga and demonstrated quantitatively that its playing strength improves through self-play.
