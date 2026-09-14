# ChaturangZero 🏺♟️

An **AlphaZero-style AI engine** for **Chaturanga** — the ancient Indian predecessor to chess, originating around 600 CE.

> *"We built a complete AlphaZero-style learning system for Chaturanga and demonstrated quantitatively that its playing strength improves through self-play."*

---

## What is Chaturanga?

Chaturanga is the 6th-century Indian board game that gave rise to modern chess.  
It is played on an 8×8 board with 6 piece types:

| Piece | Sanskrit | Moves Like |
|-------|----------|-----------|
| Raja | राज | King — 1 step any direction |
| Mantri | मंत्री | Ferz — 1 step *diagonally* (not like modern Queen!) |
| Ratha | रथ | Rook — slides orthogonally |
| **Gaja** | गज | **Jumps exactly 2 diagonally** (not bishop!) |
| Ashva | अश्व | Knight — L-shape |
| Padati | पदाति | Pawn — 1 forward, captures diagonally |

> The **Gaja** and **Mantri** are the key differences from modern chess.

---

## Architecture

```
React + TypeScript UI
        ↓
FastAPI Integration Layer (integration/api.py)
        ↓
Python Game Core (shared/game_core/)
        ↓
MCTS Search (p3_mcts/)   ←←   Neural Network (p2_neural_network/)
        ↓
Self-Play RL Training (p4_training/)
        ↑────────────────┘  (loop)
```

### AI Pipeline

```
Self-play game
    ↓
(state, MCTS policy, result) tuples
    ↓
Replay Buffer
    ↓
Train PyTorch ResNet
    ↓
Candidate vs Best Arena
    ↓
Promote if win rate ≥ 55%
    ↓
Track Elo rating
```

---

## Quick Start — Run Locally

### 1. Install dependencies

```bash
# Python (from repo root)
pip install -r requirements.txt

# Node (for the UI)
cd shared/ui
npm install
cd ../..
```

### 2. Start the React UI (mock mode — no Python needed)

```bash
cd shared/ui
npm run dev
```

Open: **http://localhost:5173**

### 3. Start the Python backend (real engine + AI)

```bash
# From repo root
uvicorn integration.api:app --reload --port 8000
```

### 4. Run both at once (Windows)

```bash
start-chaturangzero.bat
```

---

## Run Tests

```bash
# From repo root
pytest tests/ -v
```

---

## Train the AI

### Quick demo (CPU, ~5-10 minutes):

```bash
python -m p4_training.trainer --iterations 5 --games-per-iter 3 --sims 25
```

### Full training (GPU recommended):

```bash
python -m p4_training.trainer --iterations 50 --games-per-iter 25 --sims 100 --device cuda
```

Training outputs:
- `checkpoints/best_model.pt` — the current best model
- `checkpoints/elo_history.json` — Elo rating over time
- `checkpoints/elo_curve.png` — learning curve plot
- `checkpoints/opening_book.json` — learned opening preferences

---

## Project Structure

```
ChaturangZero/
├── shared/
│   ├── game_core/          # Python rules engine (THE source of truth)
│   │   ├── state.py        # Data classes: Piece, Square, Move, ChaturangaState
│   │   ├── rules.py        # Movement rules for all 6 piece types
│   │   └── engine.py       # Public API: legal_moves, apply_move, etc.
│   └── ui/                 # React + TypeScript frontend
│       └── src/
│           ├── components/ # Board, Sidebar, AIInsights, ModeSelector
│           └── services/   # Mock adapter + HTTP adapter
├── p1_representation/      # State/action encoding for the neural network
├── p2_neural_network/      # PyTorch ResNet (policy + value heads)
├── p3_mcts/                # Monte Carlo Tree Search (PUCT)
├── p4_training/            # Self-play, replay buffer, training loop, Elo
├── integration/            # FastAPI backend connecting UI ↔ Python AI
├── tests/                  # Pytest test suite
├── docs/                   # Documentation
├── checkpoints/            # Saved model weights (gitignored)
└── data/                   # Self-play data (gitignored)
```

---

## What Makes This Unique

| Feature | Description |
|---------|-------------|
| 🏺 **Authentic Chaturanga** | Correct ancient rules (Gaja jumps 2 diagonally, Mantri only moves diagonally) |
| 🧠 **AI Transparency** | The UI shows the top-3 moves the AI considered with visit % and value |
| 📈 **Elo Tracker** | Charts how the AI's strength improves across training iterations |
| 📖 **Opening Book** | Records which openings the AI naturally learns to prefer |
| 🎮 **Human vs AI mode** | Play against the trained AI at Easy / Medium / Hard difficulty |
| 🔬 **Educational** | Every AI decision is explained — not a black box |

---

## Team Contributions

| Person | Module |
|--------|--------|
| Anji | UI/UX (`shared/ui/`) + Full project completion |
| Person 2 | `p2_neural_network/` |
| Person 3 | `p3_mcts/` |
| Person 4 | `p4_training/` |

---

## License

MIT License — see `LICENSE` for details.
