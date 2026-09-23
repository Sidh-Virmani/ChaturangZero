"""
integration/api.py
==================
FastAPI backend that connects the React UI to the Python game engine and AI.

Endpoints
---------
GET  /health          — Server status
GET  /state           — Current board state (UI format)
GET  /legal_moves/{sq} — Legal moves from a square
POST /move            — Apply a human move
POST /ai_move         — Let the AI (MCTS) make a move
POST /reset           — Start a new game
GET  /elo             — Current Elo history
GET  /opening_book    — Top learned openings
WS   /ws/training     — WebSocket for live training metrics (future)

Run with:
    cd ChaturangZero-main
    uvicorn integration.api:app --reload --port 8000
"""

from __future__ import annotations

import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Square, initial_state

# ── Try to load the neural network (optional — API still works without it) ──
_model = None
_TORCH_AVAILABLE = False
try:
    import torch
    from p2_neural_network.model import load_checkpoint
    _best_ckpt = pathlib.Path("checkpoints/best_model.pt")
    if _best_ckpt.exists():
        _model = load_checkpoint(str(_best_ckpt), device="cpu")
        print(f"[api] Loaded model from {_best_ckpt}")
    _TORCH_AVAILABLE = True
except Exception as exc:
    print(f"[api] No model loaded ({exc}). AI will use random moves.")

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ChaturangZero API",
    description="AlphaZero-style Chaturanga engine API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, restrict to your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton game session ───────────────────────────────────────────────────
# (For multi-user support later: use sessions / Redis)

engine: ChaturangaEngine = ChaturangaEngine()
game_state: ChaturangaState = engine.new_game()


# ── Request / response models ─────────────────────────────────────────────────

class MoveRequest(BaseModel):
    from_sq: str   # e.g. "e2"
    to_sq:   str   # e.g. "e4"


class AIMoveRequest(BaseModel):
    num_simulations: int  = 50    # MCTS budget — maps to UI difficulty
    rule_variant:    str  = "standard"   # R1/R2/R3/R4


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "torch_available": _TORCH_AVAILABLE,
    }


@app.get("/state")
def get_state():
    """Return the current board state in the format expected by the React UI."""
    return engine.state_to_api_dict(game_state)


@app.get("/legal_moves/{square}")
def get_legal_moves(square: str):
    """Return all legal destination squares for the piece on `square`."""
    try:
        sq = Square.from_label(square)
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail=f"Invalid square: {square!r}")

    moves = engine.legal_moves_from(game_state, sq)
    return [
        {
            "from":      m.from_sq.label(),
            "to":        m.to_sq.label(),
            "isCapture": m.captured is not None,
        }
        for m in moves
    ]


@app.post("/move")
def make_move(req: MoveRequest):
    """Apply a human move and return the new board state."""
    global game_state

    try:
        from_sq = Square.from_label(req.from_sq)
        to_sq   = Square.from_label(req.to_sq)
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="Invalid square label.")

    legal = engine.legal_moves_from(game_state, from_sq)
    move  = next((m for m in legal if m.to_sq == to_sq), None)
    if move is None:
        raise HTTPException(status_code=422, detail="That move is not legal.")

    game_state = engine.apply_move(game_state, move)
    return engine.state_to_api_dict(game_state)


@app.post("/ai_move")
def ai_move(req: AIMoveRequest):
    """
    Let the AI choose and apply a move.
    Returns new board state + AI Insights + Explainability (ChaturangZero-X).
    """
    global game_state, engine

    if engine.is_terminal(game_state):
        raise HTTPException(status_code=409, detail="Game is already over.")

    # Apply rule variant if requested
    try:
        from shared.game_core.rule_variants import RuleVariant, RuleConfig
        variant = RuleVariant(req.rule_variant)
        current_engine = ChaturangaEngine(rule_config=RuleConfig.from_variant(variant))
    except Exception:
        current_engine = engine

    from p3_mcts.search import MCTSSearch
    mcts = MCTSSearch(
        current_engine,
        model=_model,
        num_simulations=req.num_simulations,
        c_uncertainty=0.5,
        temperature=0.0,
        device="cpu",
    )

    # Run search once — get policy + top moves
    policy, top_moves_raw = mcts.search(game_state)
    value_std = mcts._last_root_uncertainty

    # Best move via tactical verifier (built into best_move)
    from p1_representation.encoder import StateEncoder
    legal = engine.legal_moves(game_state)
    if not legal:
        raise HTTPException(status_code=500, detail="No legal moves.")

    legal_indices = {StateEncoder.encode_action(m): m for m in legal}
    best_idx  = max(legal_indices, key=lambda i: policy[i])
    move      = legal_indices[best_idx]

    # Apply move
    prev_state = game_state
    game_state = engine.apply_move(game_state, move)
    state_dict = engine.state_to_api_dict(game_state)

    # Build top moves for UI
    top_moves = []
    for action_idx, visit_frac, q_val in top_moves_raw:
        try:
            fsq, tsq = StateEncoder.decode_action(action_idx)
            top_moves.append({
                "from":          fsq.label(),
                "to":            tsq.label(),
                "visitFraction": round(visit_frac, 3),
                "qValue":        round(q_val, 3),
            })
        except Exception:
            pass

    # Build explanation (Research Contribution #9)
    explanation_dict = None
    try:
        from p2_neural_network.explainer import MoveExplainer
        explainer   = MoveExplainer(engine)
        explanation = explainer.explain(
            prev_state, move, top_moves_raw, value_std=value_std
        )
        explanation_dict = explanation.to_dict()
    except Exception as exc:
        explanation_dict = {"error": str(exc)}

    state_dict["aiInsights"] = {
        "chosenMove":  {"from": move.from_sq.label(), "to": move.to_sq.label()},
        "topMoves":    top_moves,
        "numSims":     req.num_simulations,
        "uncertainty": round(value_std, 4),
        "explanation": explanation_dict,
    }
    return state_dict


@app.post("/reset")
def reset_game():
    """Reset to the starting position."""
    global game_state
    game_state = engine.new_game()
    return engine.state_to_api_dict(game_state)


@app.get("/elo")
def get_elo():
    """Return the Elo rating history."""
    elo_path = pathlib.Path("checkpoints/elo_history.json")
    if not elo_path.exists():
        return {"history": [], "currentElo": 1000}
    import json
    with open(elo_path) as f:
        history = json.load(f)
    current = history[-1]["elo"] if history else 1000
    return {"history": history, "currentElo": current}


@app.get("/opening_book")
def get_opening_book():
    """Return the top learned opening sequences."""
    book_path = pathlib.Path("checkpoints/opening_book.json")
    if not book_path.exists():
        return {"openings": [], "depth": 6}
    import json
    with open(book_path) as f:
        return json.load(f)


@app.get("/rule_variants")
def get_rule_variants():
    """Return all available historical rule variants."""
    return {
        "variants": [
            {
                "id":          "standard",
                "name":        "Standard (R1)",
                "description": "Checkmate wins, stalemate draws",
            },
            {
                "id":          "capture_raja",
                "name":        "Capture-Raja (R2)",
                "description": "Capturing the Raja wins immediately",
            },
            {
                "id":          "bare_raja",
                "name":        "Bare-Raja (R3)",
                "description": "Leaving opponent with only Raja = win",
            },
            {
                "id":          "no_stalemate",
                "name":        "No-Stalemate (R4)",
                "description": "Stalemate is a loss, not a draw",
            },
        ]
    }


class MovesRequest(BaseModel):
    moves: list[str]  # e.g. ["e2e4", "e7e5"]

@app.post("/infer_rules")
def infer_rules(req: MovesRequest):
    """
    Given a sequence of moves, infer the most likely historical rule variant.
    (Research Contribution A)
    """
    from p6_rule_inference.bayesian_classifier import BayesianRuleClassifier
    from shared.game_core.rule_variants import RuleVariant
    
    # We create a new classifier per request for now, or you could make it global
    classifier = BayesianRuleClassifier()
    # In a real app we'd load 4 separate trained models, here we reuse _model
    for v in ["standard", "capture_raja", "bare_raja", "no_stalemate"]:
        classifier.add_model(RuleVariant(v), _model)
        
    try:
        probs = classifier.infer_rules(req.moves)
        return {"posterior_probabilities": probs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategy_dna")
def get_strategy_dna(req: MovesRequest):
    """
    Extract the Strategy DNA fingerprint from a sequence of moves.
    (Research Contribution C)
    """
    from p6_rule_inference.strategy_dna import extract_game_dna
    try:
        dna = extract_game_dna(req.moves, engine)
        return {"dna_fingerprint": dna.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game_quality")
def optimize_game_quality():
    """
    Run a mini-batch of the Neural Game Quality Optimizer.
    (Research Contribution D)
    """
    from p6_rule_inference.game_quality_optimizer import GameQualityOptimizer
    try:
        optimizer = GameQualityOptimizer(model=_model, num_sims=5)
        # Using a small number of games for API speed
        best_report = optimizer.optimize(n_games_per_config=2, verbose=False)
        leaderboard = optimizer.get_leaderboard()
        
        return {
            "best_variant": best_report.rule_variant,
            "best_quality_score": best_report.quality_score,
            "leaderboard": [{"variant": v, "score": s} for v, s in leaderboard]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


