# ChaturangZero UI Checklist

The UI is a visual client for the Chaturanga engine. It must never become a second rules engine.

**Source of truth:** `shared/game_core/`

The UI may display legal moves, game state, turn, result, etc., but these values should ultimately come from the game core through a thin integration/API layer.

---

### App shell
- [ ] Clean desktop-first page layout
- [ ] Header with project name
- [ ] Main game area with board + side panel
- [ ] Consistent spacing, typography and CSS variables
- [ ] Responsive behaviour for normal laptop widths

### Board
- [ ] Render an 8×8 board with alternating square colours
- [ ] File/rank coordinate labels
- [ ] Render all Chaturanga pieces from supplied state data
- [ ] Correct initial-position rendering
- [ ] Selected-square state
- [ ] Legal destination highlights
- [ ] Capture destination highlight
- [ ] Last-move from/to highlight
- [ ] Hover state for selectable pieces
- [ ] Prevent UI interaction with squares that are not currently selectable
- [ ] Board orientation designed so it can support flipping later

### Game interaction
- [ ] Click piece → select it
- [ ] Click legal destination → request move
- [ ] Click selected piece again / outside valid target → clear selection
- [ ] Turn indicator
- [ ] Simple game-status text: playing / check / terminal
- [ ] New Game / Reset control
- [ ] Basic end-of-game result panel
- [ ] Move history panel structure

### Integration readiness
- [ ] UI state types defined clearly in TypeScript
- [ ] One adapter/service layer between React components and game logic
- [ ] Components never implement Chaturanga legality themselves
- [ ] Temporary mock adapter only for visual development
- [ ] Loading / disabled state while waiting for engine response
- [ ] Error message state if the engine/API call fails
- [ ] Placeholder `AI thinking...` state for future MCTS integration


- [ ] Drag-and-drop movement in addition to click-to-move
- [ ] Smooth piece movement / capture animations
- [ ] Flip Board control
- [ ] Human vs Human / Human vs AI mode selector
- [ ] AI difficulty selector mapped to future MCTS simulation budgets
- [ ] Captured-pieces display
- [ ] Resign / offer-draw controls if supported by the frozen rules
- [ ] Optional player clocks
- [ ] Better move-history notation and scroll behaviour
- [ ] Sound effects with an on/off setting
- [ ] Theme/settings panel
- [ ] Keyboard accessibility and focus states
- [ ] Stronger small-screen/mobile layout
- [ ] Final end-game modal and rematch flow

---

## UI architecture rule

Target architecture:

```text
React + TypeScript UI
        ↓
UI adapter / API client
        ↓
thin integration API
        ↓
Python Chaturanga Game Core
        ↓
later: MCTS + Neural Network
```

The React frontend should **not** know how a Gaja, Mantri, Raja, etc. is legally allowed to move.

For example, the UI should eventually ask the backend:

```text
"Give me legal destinations for this selected square."
```

rather than calculating those moves in React.

That keeps the rules correct in exactly one place and makes the future AlphaZero integration much easier.
