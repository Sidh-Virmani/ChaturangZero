# P3 MCTS Guide

## What Has Been Completed

- A complete Monte Carlo Tree Search implementation for ChaturangZero.
- PUCT-based move selection.
- Neural-network policy and value evaluation.
- Expansion of legal moves only.
- Correct value-sign changes between White and Black turns.
- Exact handling of terminal positions such as checkmate and bare king.
- Dirichlet noise for self-play exploration.
- Temperature-based move selection.
- Search-tree reuse after a move is played.
- Input and configuration validation.
- Unit and integration tests.

## Package Files

- `config.py`
  - Contains `MCTSConfig`.
  - Controls simulation count, exploration, noise, and temperature.

- `node.py`
  - Contains `MCTSNode`.
  - Stores a game state, children, visit count, prior, and accumulated value.
  - Calculates PUCT scores.
  - Expands legal child positions.

- `mcts.py`
  - Contains `MCTS`.
  - Runs selection, evaluation, expansion, and backpropagation.
  - Produces the final move probability distribution.
  - Selects and returns a move.

- `__init__.py`
  - Exports `MCTS`, `MCTSConfig`, and `MCTSNode`.

- `tests/test_mcts.py`
  - Tests configuration validation.
  - Tests PUCT selection.
  - Tests value-sign inversion.
  - Tests terminal positions.
  - Tests Dirichlet noise and temperature.
  - Tests legal policy output.
  - Tests the real neural-network integration.

## How MCTS Works

For every move, MCTS repeats the following process:

1. **Selection**
   - Start at the current board position.
   - Use the PUCT score to choose promising moves.
   - Continue until an unexplored or terminal position is reached.

2. **Evaluation**
   - If the position is terminal, use the exact game result.
   - Otherwise, ask the neural network for:
     - A policy: which legal moves look promising.
     - A value: how good the position is for the current player.

3. **Expansion**
   - Add every legal move as a child node.
   - Assign the neural-network policy probability as each child's prior.

4. **Backpropagation**
   - Send the evaluated value back through the visited nodes.
   - Reverse the sign at every turn because the players alternate.

5. **Move selection**
   - Convert root visit counts into a 4096-entry policy vector.
   - Select the final action using the configured temperature.

## Important Configuration Options

- `num_simulations`
  - Number of search rollouts per move.
  - More simulations normally improve play but take longer.

- `c_puct`
  - Balances known strong moves against less-explored moves.

- `temperature`
  - `0`: always choose the most-visited move.
  - Greater than `0`: allow sampling from the visit distribution.

- `add_noise`
  - `True`: useful during self-play training.
  - `False`: recommended for evaluation and human-vs-AI games.

- `dirichlet_alpha` and `dirichlet_epsilon`
  - Control root exploration noise during self-play.

Example:

```python
from p3_mcts import MCTSConfig

config = MCTSConfig(
    num_simulations=50,
    c_puct=1.5,
    temperature=0.0,
    add_noise=False,
)
```

## Basic Python Usage

```python
from p2_neural_network import NetworkEvaluator, PolicyValueNetwork
from p3_mcts import MCTS, MCTSConfig
from shared.game_core import initial_state

model = PolicyValueNetwork()
evaluator = NetworkEvaluator(model, device="cpu")
mcts = MCTS(
    evaluator,
    MCTSConfig(num_simulations=50, temperature=0.0),
)

state = initial_state()
policy, move, action = mcts.search(state)

print(move.uci)
print(action)
print(policy.sum())
```

The search returns:

- `policy`: a normalized NumPy array with shape `(4096,)`.
- `move`: the selected legal `Move`.
- `action`: the selected integer action ID.

## Current UI Status

- The React UI is connected to the Python game core.
- The UI currently supports human-vs-human play.
- The HTTP game server currently applies only the move sent by the user.
- The server does not currently create or call an `MCTS` instance.
- The server always returns `isAiThinking: false`.
- Therefore, adding the MCTS package alone does not visibly change the UI.

## How to Connect the UI and MCTS

### 1. Load the AI in the Python server

- Update `shared/game_core/server.py` or create a separate AI service.
- Import:
  - `PolicyValueNetwork`
  - `NetworkEvaluator`
  - `MCTS`
  - `MCTSConfig`
- Load the best trained checkpoint if one exists.
- Create one evaluator and MCTS instance when the server starts.
- Do not reload the model for every move.

### 2. Add an AI-move operation

- Add a method such as `GameSession.apply_ai_move()`.
- Inside that method:
  - Read the current `GameState`.
  - Call `mcts.search(state)`.
  - Apply the returned move with the game core.
  - Add the move to the move history.
  - Call `mcts.step(move)` so the tree can be reused.
- After later human moves, also call `mcts.step(human_move)` when possible.
- If a retained tree does not match the board, the next search safely starts a new tree.

Example logic:

```python
policy, move, action = mcts.search(self.game.state)
self.game.apply(move)
mcts.step(action)
```

### 3. Expose an API endpoint

- Add an endpoint such as:
  - `POST /api/game/ai-move`
- The endpoint should:
  - Reject requests when the game is over.
  - Run the MCTS search.
  - Apply the chosen move.
  - Return the updated game state.

### 4. Update the React adapter

- Add a method to `GameAdapter`, for example:

```typescript
requestAiMove(): Promise<BoardState>
```

- Implement it in `httpGameAdapter.ts` by calling:

```text
POST /api/game/ai-move
```

### 5. Trigger the AI after a human move

- After the human move succeeds:
  - Check whether the game is still active.
  - Show the AI-thinking state.
  - Request the AI move.
  - Replace the UI state with the server response.
  - Hide the AI-thinking state.

### 6. Add player and difficulty controls

- Allow the user to choose White or Black.
- Run MCTS only when it is the AI side's turn.
- Suggested simulation budgets:
  - Easy: 10 to 20 simulations.
  - Medium: 50 simulations.
  - Hard: 100 or more simulations.

### 7. Reset MCTS with the game

- When starting a new game:
  - Reset the game-core state.
  - Discard the old MCTS tree.
  - Create a fresh MCTS instance or clear its root.

## Important UI Integration Notes

- Run MCTS outside the browser; the Python server should run it.
- Do not implement Chaturanga rules again in React.
- Disable board interaction while the AI is thinking.
- Handle server errors so the board does not become stuck.
- Use `temperature=0.0` and `add_noise=False` for human-vs-AI play.
- Use `add_noise=True` only for self-play training.
- An untrained neural network will make weak or random-looking moves.
- Load a trained checkpoint to make the visible AI meaningfully stronger.

## How to Test the Implementation

From the project root, run the MCTS tests:

```powershell
python -m pytest tests\test_mcts.py -v
```

Run the complete project test suite:

```powershell
python -m pytest -q
```

Expected full-suite result at the time this guide was created:

```text
76 passed
```

## Recommended Next Step

- Connect MCTS to a new server-side AI-move endpoint.
- Add a Human vs AI option to the UI.
- First verify the connection with the untrained model.
- Then load a trained checkpoint for stronger play.
