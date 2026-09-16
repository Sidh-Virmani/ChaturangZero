# P1 representation: the bridge between Chaturanga and numbers

## 1. What problem does P1 solve?

`shared/game_core` knows the game. Its `GameState` says which pieces occupy which squares and whose turn it is. Its `Move` says where a piece starts and where it goes. Its `legal_moves(state)` function knows the rules and returns only moves that are allowed.

A future neural network cannot read Python `Piece` and `Move` objects directly. It needs numbers in a predictable arrangement: the same kind of information must always appear in the same place. P1 translates between these two worlds:

```text
Game world                         P1 bridge                      AI world
GameState with Piece objects  ->  encode_state()           ->  13 boards of numbers
Move(from_square, to_square)  ->  move_to_action()         ->  one action number
GameState                    ->  legal_action_mask()      ->  allowed action numbers
```

P1 does not play Chaturanga or decide what is legal. It translates information supplied by `game_core` so later AI components can use it.

## 2. The big picture

```text
                           GameState
                          /         \
                         /           \
            encode_state(state)    game_core.legal_moves(state)
                    |                       |
          13 numbered 8x8 boards       legal Move objects
                                            |
                                     move_to_action(move)
                                            |
                                        action IDs
                                            |
                          legal_action_mask(state) marks these IDs with 1
```

`encode_state()` looks at the pieces and turn in a `GameState`. `game_core.legal_moves()` independently supplies the allowed `Move` objects. `move_to_action()` turns each move into an integer ID. `legal_action_mask()` puts a `1` at each allowed ID and leaves every other ID at `0`.

The state numbers and action numbers use fixed formats. This matters because a future network must learn that, for example, the White Raja is always in the same part of its input and `a2` to `a3` always has the same action ID.

## 3. State encoding, step by step

### Start with `GameState`

In this project, `GameState` comes from `shared.game_core`. Its `board` is a tuple with 64 entries, one per square. An entry is either `None` (empty) or a `Piece` with a side and a kind, such as White Padati. The state also stores `side_to_move`, `ply_count`, and `last_move`.

The board tuple follows `a1, b1, ..., h1, a2, ..., h8`. In other words, `a1` is entry 0, `b1` is entry 1, `a2` is entry 8, and `h8` is entry 63. P1 reads this tuple; it does not maintain a second board with its own rules.

### Make simple number grids

An **8×8 grid** is a board-shaped table of numbers. A **plane** (also called a **channel**) is one such grid devoted to a single question. For example, the White Raja plane asks, “Where is a White Raja?” A square gets `1` if the answer is yes and `0` otherwise. Because different kinds of pieces need different questions, we stack several grids together, like 13 transparent board-shaped sheets.

The first 12 sheets describe pieces:

| Sheet number | Question answered by a `1` |
| --- | --- |
| 0 | Is there a White Raja here? |
| 1 | Is there a White Mantri here? |
| 2 | Is there a White Gaja here? |
| 3 | Is there a White Ashva here? |
| 4 | Is there a White Ratha here? |
| 5 | Is there a White Padati here? |
| 6 | Is there a Black Raja here? |
| 7 | Is there a Black Mantri here? |
| 8 | Is there a Black Gaja here? |
| 9 | Is there a Black Ashva here? |
| 10 | Is there a Black Ratha here? |
| 11 | Is there a Black Padati here? |

The final sheet, number 12, answers a different question: “Is it White's turn?” Every cell on this sheet is `1` for White to move or `0` for Black to move. A future network sees the whole stack at once. Without this sheet, two positions with identical pieces but different players to move would look identical, although their legal moves and prospects can differ.

`PLANE_NAMES` publishes these 13 names in order. `STATE_SHAPE` is `(13, 8, 8)`: **13 sheets, each with 8 rows and 8 columns**. People sometimes call this whole stack a **tensor**; here it is simply a NumPy array of numbers with three dimensions.

### Locate a square on a sheet

Rows follow ranks from 1 to 8, and columns follow files from a to h. Both counts start at zero inside an array:

```text
square a1 -> row 0, column 0
square e4 -> row 3, column 4
square h8 -> row 7, column 7
```

For example, `e4` is square index 28 in `game_core`: three full ranks contain 24 squares, then file `e` is four columns along. P1 divides 28 by 8 to get row 3 and remainder 4 to get column 4. The board is always viewed from the same direction; Black positions are not silently flipped.

Suppose a White Padati has moved from `a2` to `a3`. `encode_state(state)` writes `1` at **sheet 5, row 2, column 0**. The old `a2` cell on that sheet is `0`. A White Raja on `e1` appears at **sheet 0, row 0, column 4**. Every other piece goes onto its own sheet in the same way.

### Finish the array

```text
GameState
   -> inspect each entry of its 64-square board
   -> place a 1 on the matching piece sheet and square
   -> fill the turn sheet with 1s or 0s
   -> return a new (13, 8, 8) NumPy array
```

NumPy's **dtype** tells us what kind of number each cell stores. Here it is `float32`, a standard 32-bit decimal-number format suitable for later neural-network input. The values happen to be only `0.0` and `1.0`. `encode_state()` makes a fresh array each time and never changes the `GameState`.

`ply_count` (how many half-moves have occurred) and `last_move` are not included. Under the current `game_core` rules, the board and side to move determine legal moves and terminal results; move history does not change them. Two states that differ only in those bookkeeping fields therefore receive the same encoding. If future rules start depending on history, this format must be extended before training with those rules.

## 4. Move to action encoding, step by step

A future policy network will give one score to each possible action. A **policy** is simply its set of move preferences. It is easier to store those scores in a numbered list than to make the network output Python `Move` objects. An **action space** is that complete numbered list of possible move slots.

Our core `Move` contains exactly two square numbers: where the piece starts and where it lands. A promotion needs no extra choice: `game_core` automatically makes a Padati a Mantri on the far rank. Captures also need no extra label: the destination square tells the core what was captured. These facts let one source/destination pair represent every move our current rules allow.

There are 64 possible starting squares and 64 possible destination squares, so `ACTION_SPACE_SIZE` is **4096**. Some slots describe impossible moves, including moving a piece to the same square. Keeping all pairs gives us one simple fixed numbering scheme; the legal mask in the next section filters them out.

The idea is to give each starting square its own block of 64 IDs. Within that block, the destination square selects one ID. In code, `move_to_action(move)` computes:

```text
action ID = (starting square index × 64) + destination square index
```

Concrete examples:

- `a1 -> b1`: `a1` is 0 and `b1` is 1, so `0 × 64 + 1 = 1`.
- `b1 -> c3`: `b1` is 1 and `c3` is 18, so `1 × 64 + 18 = 82`.
- `a2 -> a3`: `a2` is 8 and `a3` is 16, so `8 × 64 + 16 = 528`.
- `h8 -> a1`: `h8` is 63 and `a1` is 0, so `63 × 64 + 0 = 4032`.

The same `Move` always receives the same ID. Different source/destination pairs cannot collide: different starting squares occupy different blocks, and destinations are unique within a block.

`action_to_move(action)` goes backward. For ID 528, it finds that 8 complete blocks of 64 fit, with 16 left over. It returns the core `Move` from square 8 (`a2`) to square 16 (`a3`). This reverse translation does **not** say whether the move is legal in a particular position.

## 5. Legal action mask, step by step

The 4096 slots do not mean 4096 moves are available on every turn. At the start of the game, `game_core` reports only 16 legal moves. A **mask** is a row of switches showing which action slots are allowed now. In a tiny imaginary seven-action game, legal IDs 1, 3, and 6 would give:

```text
action ID:  0 1 2 3 4 5 6
mask:       0 1 0 1 0 0 1
```

Our `legal_action_mask(state)` does the same thing at full size:

```text
GameState
   -> game_core.legal_moves(state) decides what is allowed
   -> move_to_action(move) finds each legal move's ID
   -> a 4096-entry mask has 1 at those IDs and 0 everywhere else
```

For the initial position, `a2 -> a3` has ID 528 and its mask entry is `1`. `a2 -> a4` is a forbidden Padati double-step, so its entry is `0`. P1 does not contain a Padati movement rule; it learns that answer by asking `game_core` for legal moves. Likewise, a move exposing one's Raja, a move after the game ends, and every unused action slot remain `0` because the core did not return them.

The mask is a new NumPy array with 4096 entries. Its dtype is `uint8`, meaning each entry is a small nonnegative whole number stored in one byte. Here the only values are `0` and `1`. A future policy consumer can use the mask to ignore scores for moves that cannot be played.

## 6. Symmetry and data augmentation

**Data augmentation** means creating another useful training example from an existing one without playing another game. If a board transformation preserves the game's rules, a position and its transformed copy can teach the same idea. But a position, its move labels, and its result must all be transformed consistently.

We checked the actual rules and initial piece arrangement. A simple left-to-right reflection moves the initial Raja and Mantri to different starting files. Swapping colors without rotating would also reverse which way Padatis should advance. We therefore implement just one combined transformation: **rotate the board 180 degrees and exchange White and Black labels**.

Examples of that transformation:

```text
a1 becomes h8; a2 becomes h7.
White becomes Black, and Black becomes White.
White to move becomes Black to move.
Move a2 -> a3 becomes h7 -> h6.
```

`rotate_state(state)` creates the transformed core state, including a transformed `last_move`; `encode_state(rotate_state(state))` then gives the transformed numerical board. `rotate_action(action)` transforms the corresponding action ID. If a later training example assigns a probability to action 528 (`a2 -> a3`), its rotated example must assign that probability to the ID for `h7 -> h6`.

The winner's color also changes: a White win in the original game becomes a Black win in the rotated one. If a later training value is written from the *current player's* point of view, its number stays the same because both the player and winner swap colors. P1 provides the state and action transformations; later training code remains responsible for keeping its chosen result format consistent.

The transformation maps the **piece layout** of the initial position back to itself, but exchanges the side to move. Thus the transformed initial state has Black to move and is not the project's literal starting state. It is still a valid position under the core's rules. Tests check the stronger property needed for augmentation: rotating legal moves gives legal moves in the rotated position, and rotating after a move gives the same state as moving after rotation. They also check promotion and a terminal winner. This is an optional tool for later training experiments, not a change to how an actual game starts.

## 7. How everything connects

Imagine the normal initial `GameState`, where White is to move:

```text
1. encode_state(state)
      -> 13 numbered boards for a future network to read.

2. game_core.legal_moves(state)
      -> 16 legal Move objects, including a2 -> a3.

3. move_to_action(Move.from_names("a2", "a3"))
      -> action ID 528.

4. legal_action_mask(state)
      -> 4096 switches, with switch 528 on.

5. A future policy output may contain 4096 scores.
      -> the mask says which scores may be considered;
         action_to_move(528) can recover the core Move.
```

P2 can later build a network that reads the 13-board array and emits scores in this 4096-slot order. P3 can later use the mask and action IDs while searching, then return a `Move` to `game_core` for actual rule-checked application. Neither stage needs to invent a second Chaturanga rulebook.

## 8. Important functions and constants to know

**`encode_state(state)`** takes a core `GameState` and returns a fresh `float32` NumPy array shaped `(13, 8, 8)`. In simple words, it draws the 13 numbered boards described above. P2 will use its output as network input. For example, after `a2 -> a3`, the White Padati sheet has a `1` at `a3`.

**`move_to_action(move)`** takes a core `Move` and returns an integer from 0 to 4095. It labels a source/destination pair; it does not check legality. P2 and P3 can use the label to locate the move's score. For example, `Move.from_names("a2", "a3")` becomes 528.

**`action_to_move(action)`** takes an action ID and returns a core `Move`. P3 can use it to turn a selected ID back into a move. For example, 528 becomes `a2 -> a3`. The caller must still consult the mask or core before treating it as playable.

**`legal_action_mask(state)`** takes a core `GameState` and returns a fresh, 4096-entry `uint8` NumPy array. P2 or P3 can use its `1` entries to identify legal policy slots. It gets every legal move from `game_core.legal_moves(state)`.

**`rotate_state(state)`** takes a core `GameState` and returns a new position rotated 180 degrees with side labels exchanged. **`rotate_action(action)`** takes one action ID and returns its matching rotated ID. Later training code can use the pair to transform an example consistently; neither function changes its input.

**`STATE_SHAPE`, `PLANE_NAMES`, and `ACTION_SPACE_SIZE`** publish the fixed formats so callers need not guess them: `(13, 8, 8)`, the 13 sheet names in order, and 4096 action slots.

All of these are imported from `p1_representation`.

## 9. Important design decisions

| Choice | Why we chose it | Alternative and why we did not use it |
| --- | --- | --- |
| Keep rank 1 in row 0 for both sides | A square always appears at the same array location. | Turning every Black position around would make square positions depend on the turn and complicate action alignment. |
| Use one sheet per side and piece kind | A `1` has one clear meaning, and all six current piece kinds are covered for both sides. | A single board of piece codes would be compact, but numeric codes could misleadingly look ordered to a network. |
| Include a turn sheet | The same pieces can have different legal moves depending on who moves next. | Leaving out turn would make those two states indistinguishable to P2. |
| Use 64 × 64 source/destination slots | It covers every core `Move`, including compulsory promotion, with a simple reversible formula. | Special chess-style promotion choices or direction-based move codes are unnecessary under our rules. |
| Ask `game_core` for legal moves | There is one authoritative rules engine. | Reimplementing movement in P1 could disagree with the core about check, promotion, or terminal states. |
| Omit `ply_count` and `last_move` from the network input | Current legal and terminal rules do not read them. | Including them would distinguish positions whose playable situation is currently the same. |
| Offer only 180-degree rotation plus color exchange | This combination preserves movement directions and legal transitions. | Other flips do not preserve the project's starting arrangement or side-relative movement in the same way. |

## 10. The maths we actually use

The square numbering comes from the core: each rank has eight squares. To find an array cell, divide the square index by 8. The whole-number result is the row; the remainder is the column. Thus `e4` is index 28, giving row 3 and column 4. This is the same as saying “rank 4, file e,” with zero-based array positions.

An action ID reserves 64 consecutive numbers for each source square. Multiply the source index by 64 to reach its block; add the destination index to find the move. For `a2 -> a3`, that is `8 × 64 + 16 = 528`. To reverse ID 528, divide by 64: the whole-number result 8 is `a2`, and remainder 16 is `a3`.

For the 180-degree rotation, square index 0 (`a1`) goes to 63 (`h8`), index 1 goes to 62, and so on. The rotated square index is therefore `63 - original index`. Both squares of a move are transformed this way, while `rotate_state` also swaps the piece and turn colors. These are coordinate calculations, not Chaturanga movement rules.

## 11. Tests: what are we actually proving?

The state tests inspect the initial board sheet by sheet. For example, they check that `e1` appears on the White Raja sheet and that all eight White Padatis appear on their expected squares. They also play several legal moves through `game_core`, then check that pieces appear at their new cells, old cells are empty, the turn sheet changes, repeated encoding is identical, and the original state stays unchanged.

The action tests check concrete IDs such as 528 for `a2 -> a3`, then try **all 4096 square pairs**. Each pair must receive a different ID, and decoding must recover the original `Move`. Other tests reject out-of-range IDs and squares. Tests for Raja, Mantri, Gaja, Ashva, Ratha, Padati, captures, and both colors' promotions show that the same source/destination scheme covers actual core moves.

The mask tests compare every `1` against `game_core.legal_moves()` in the initial position and several positions after play. They check known illegal moves stay `0`, terminal positions have no allowed IDs, and asking twice makes independent arrays.

The symmetry tests rotate every action ID twice and recover the original. They compare rotated numerical piece sheets with the expected reversed-color sheets, compare complete legal-move sets before and after rotation, and check that rotating and applying a move commute. All six piece kinds, promotion, and terminal winners are checked.

Finally, the existing game core and API tests run as regression checks. Command used from the repository root:

```text
.venv\Scripts\python.exe -m pytest tests\test_state_representation.py tests\test_action_representation.py tests\test_representation_symmetry.py tests\test_game_core.py tests\test_game_api.py -q
```

Result: **57 passed**.

## 12. File map

**`p1_representation/state.py`** contains the 13-sheet board encoder and its layout names and shape. It reads `GameState` but knows no movement rules.

**`p1_representation/actions.py`** contains the action numbering, reverse translation, and legal mask. Its only source of legal moves is `shared.game_core.legal_moves`.

**`p1_representation/symmetry.py`** contains the one supported board-and-color transformation and matching action transformation. It creates new values rather than changing game states.

**`p1_representation/__init__.py`** is the small public doorway: callers can import the functions and constants above directly from `p1_representation`.

**`tests/test_state_representation.py`, `tests/test_action_representation.py`, and `tests/test_representation_symmetry.py`** check the three parts separately and together. **`requirements.txt`** declares NumPy for the arrays and pytest for the tests.

## 13. What P2 and P3 can now use

P2 can ask P1, “Turn this `GameState` into my input,” with `encode_state(state)`. It knows the input size from `STATE_SHAPE` and can associate its 4096 policy outputs with action IDs using `ACTION_SPACE_SIZE` and `move_to_action(move)`. It can use `legal_action_mask(state)` to distinguish playable outputs from the rest.

P3 can ask, “Which policy slot represents this core `Move`?” with `move_to_action(move)`, or “Which core `Move` does this selected slot mean?” with `action_to_move(action)`. It still asks `game_core` to apply moves and decide legality. If later training code tries the supported augmentation, it must rotate both the state and its action labels with `rotate_state` and `rotate_action`.
