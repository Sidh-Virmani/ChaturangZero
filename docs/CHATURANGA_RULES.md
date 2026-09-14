# Chaturanga Rules — Frozen Specification

This document defines the **exact rules** used by ChaturangZero.
The game core implements these rules precisely.
The neural network learns to play *this* game — any ambiguity in rules
would cause the AI to learn the wrong thing.

---

## Board

Standard 8×8 grid.  
Files: a–h (left to right from White's perspective).  
Ranks: 1–8 (White starts at ranks 1–2, Black at ranks 7–8).

---

## Pieces and Movement

### Raja (राज — King)
- Moves **1 step** in any of the 8 directions (orthogonal or diagonal).
- Cannot move into check.
- **No castling** (Chaturanga predates castling).

### Mantri (मंत्री — Minister / Ferz)
- Moves **exactly 1 step diagonally** (4 possible squares).
- This is the *Ferz* (not the modern Queen!).
- The Mantri is the weakest major piece — much weaker than a modern Queen.

### Ratha (रथ — Chariot)
- Slides any number of squares **orthogonally** (like a modern Rook).
- Blocked by any piece; can capture the first enemy piece it reaches.

### Gaja (गज — Elephant) ⚠️ KEY DIFFERENCE
- Jumps **exactly 2 squares diagonally** (like a 2-square diagonal leap).
- It does **NOT** slide like a modern Bishop.
- It **ignores** any piece on the intermediate square (it jumps over).
- It can capture on the landing square if occupied by an enemy.
- From the centre, a Gaja reaches 4 squares: (±2, ±2).
- This is significantly weaker than a modern Bishop.

### Ashva (अश्व — Horse)
- Moves in an **L-shape**: 2 squares in one direction + 1 square perpendicular.
- Identical to the modern Knight.
- Jumps over pieces.

### Padati (पदाति — Soldier / Pawn)
- Moves **1 step forward** (White moves toward rank 8; Black toward rank 1).
- Captures **1 step diagonally forward**.
- **No double-step** from the starting rank (unlike modern chess).
- **No en passant**.
- **Promotion**: Upon reaching the opponent's back rank (rank 8 for White, rank 1 for Black), the Padati promotes to **Mantri** (not Queen — Mantri is the only promotion option).

---

## Check and Checkmate

- A player is **in check** when their Raja is attacked by an enemy piece.
- A player **in check** must resolve it on their next move.
- If a player has no legal moves and is **in check** → **Checkmate** (game over).
- The player who delivers checkmate **wins**.

## Stalemate

- If a player has no legal moves and is **not in check** → **Stalemate** → **Draw**.

## 50-Move Rule

- If 100 half-moves (50 full moves) pass without a capture or Padati move → **Draw**.
- This prevents infinite games during training.

---

## Starting Position

```
  a   b   c   d   e   f   g   h
8 [R] [A] [G] [M] [K] [G] [A] [R]   Black back rank
7 [P] [P] [P] [P] [P] [P] [P] [P]   Black Padati
6  .   .   .   .   .   .   .   .
5  .   .   .   .   .   .   .   .
4  .   .   .   .   .   .   .   .
3  .   .   .   .   .   .   .   .
2 [p] [p] [p] [p] [p] [p] [p] [p]   White Padati
1 [r] [a] [g] [m] [k] [g] [a] [r]   White back rank

K/k = Raja   M/m = Mantri   R/r = Ratha   G/g = Gaja   A/a = Ashva   P/p = Padati
```

White always moves first.

---

## What We Are NOT Using

To keep the AI well-defined and the codebase clean:

- ❌ No castling
- ❌ No en passant  
- ❌ No multiple promotion choices (Mantri only)
- ❌ No Chaturanga-specific 4-player variant
- ❌ No dice-based move selection (the Chaturanga variant we implement is the 2-player deterministic version)

---

## Why These Rules Matter for the AI

The neural network learns *only from self-play using these rules*.  
A bug in movement logic = the AI learns the wrong game.  
This is why `shared/game_core/rules.py` is thoroughly tested in `tests/test_game_core.py`.
