# Phase 2: repetition draws

The repository's rules specify checkmate, bare Raja and stalemate wins. They
did not specify repetition. We adopt an explicit **project engineering rule**:
the third occurrence of exactly the same pieces on the same squares, with
the same player to move, automatically ends the game as a draw. This is not
a claim of a canonical historical Chaturanga rule. There is no player claim
button, no modern-chess fifty-move rule and no core move limit.

For example, White's Ashva moves b1–a3, Black's b8–a6, then both return.
The initial position has now appeared twice. Repeating that four-ply cycle
once more produces the third occurrence, so the game ends at ply 8.

`GameState.position_history` is an immutable tuple of prior `(board, side)`
pairs. The current position counts once implicitly. Boards are exact tuples,
not lossy hashes; ply count and last move do not enter the repetition key.
A directly constructed state starts a fresh history. Both validated moves
and trusted MCTS transitions append the parent position. Candidate states
used only to test Raja safety do not need history. Branches share immutable
board values and cannot change each other's histories. Reset starts fresh.
Existing checkmate, stalemate and bare-Raja wins take precedence over a draw.

`Outcome.draw` identifies a terminal result with no winner; `reward()` returns
0 for either side. Core legal moves/masks are empty at a draw. MCTS terminal
nodes stop expansion and back up zero. Self-play already uses core rewards,
so it retains draw examples with zero targets without a separate P4 rule.
Arena now reports `draws` separately from wins and `truncated`. The server
reports Draw through the existing result-modal contract.

A max-ply cap remains a resource limit: unfinished games are still truncated
and their examples discarded. Nonrepeating long games can still hit that cap.

The existing 13-plane network input and checkpoints remain compatible. That
input does **not** describe history, so positions with different repetition
contexts look identical to P2. Core/MCTS knows the exact context and overrides
terminal values; the learned nonterminal value is an estimate without that
context. A history-aware network would require a separately planned format
change. The existing P1 state rotation now also rotates/swaps prior boards
to preserve rule equivalence; no training augmentation was added.
