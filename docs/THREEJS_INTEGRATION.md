# Integrating the atelier with main

Base: `origin/main` at `468ce5d1725bbfc3b9bb88a5cfdf2c4da668b144`.
Branch: `codex/threejs-main-integration`.
The separate `codex/complete-chaturangzero` prototype was published unchanged.

## Review boundary

The rules engine, P1 representation, P2 architecture/checkpoint serialization,
P3 MCTS, and P4 trainer are retained. `App.tsx` swaps the displayed board for
`BoardExperience`; the original `Board` remains the 2D and keyboard view.
Machine search progress, analysis, result dialogs, reset and flip still use
the existing state and adapter.

New files are the scene/sculptures, 3D wrapper, educational UI, scoped palette,
neural types, inspector service, regression tests, and documentation. Dependencies
add pinned Three.js/types and local Fontsource fonts. `npm ci` is required after
switching from the old UI dependencies.

The API adds optional `BoardState.positionKey` and
`GameAdapter.getNetworkInspection(signal?)`. Custom adapters, if introduced later,
must implement the new method or the educational capability should be made optional.
No existing route changes its move or analysis format. The network route returns:

```text
GET /api/game/network
  positionKey, checkpoint, smokeCheckpoint, sideToMove,
  layers[{id, label, operation, explanation, shape, channels,
          channelNames, minimum, maximum}],
  policy[{from, to, actionId, prior}], value,
  legalActionCount, parameterCount
```

Each `channels` entry contains row-major raw values, in P1 physical coordinates.
Most layers are channel × 8 × 8; the outcome is a single scalar. Policy spatial
cells sum legal from–to priors over destinations. No MCTS counts are fabricated.
No checkpoint yields only the input layer and a null value/checkpoint. Inspection
while a machine search is active returns 503; the UI refreshes after search.

The session captures state/revision under its lock, releases it for inference,
and checks revision again afterwards. A MachinePlayer inference lock serializes
inspection/search without holding the session lock during inference. A reset
can still cancel a search; stale results do not replace the current snapshot.

The scene preserves moving meshes using the supplied last move rather than
position-based IDs. All move legality and terminal handling remain in Python.
WebGL failure returns to the existing 2D board. The interface includes a
keyboard activation grid and honours reduced motion.

## Before merging

Fetch main and review any new changes since the base commit. Rebase/merge this
branch using the team's normal workflow, resolve changes to App/API adapters,
then run `python -m pytest -q` and `npm run build` in `shared/ui`. Test both game
modes, selection/capture, reset during search, flip, 2D fallback, checkpoint-free
encoding, and a real checkpoint with its configured layer count.

Keep local checkpoint binaries, node_modules, dist, and verification/runtime
artifacts out of the commit. A smoke checkpoint is adequate for a pipeline demo,
but playing strength requires a separate evaluated training run.

## Resource and performance choices

The lazy Three.js module keeps the home page and 2D client small. Piece geometry
is created once per side/kind and merged by material. Activation cells are GPU
instances. Rendering caps pixel ratio at 1.75; shadows use a 2,048-pixel map.
Camera/layer movement uses elapsed time, and hidden tabs suspend their render loop.
Scene teardown releases geometry, textures, environment maps, shadows, controls,
listeners, observers, labels, and the renderer. Development-only canvas attributes
report observed frames, draw calls and triangles for local QA; no telemetry is sent.

The UI retains a clear distinction between measured activations and illustrative
animation. It visualizes selected/aggregated channel maps, not every internal
convolution operation or an explanation derived from attribution methods.
