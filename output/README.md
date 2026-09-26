# Runtime and release output

`server/api_server.py` exposes the Python rules and AI through a local HTTP API.
It also serves the production bundle in `shared/ui/dist`.

Run the whole app from the project root with `python run.py --open`, or use
`start-chaturangzero.bat`. See the root README for first-time setup.

## API

Requests use a tab-specific `X-Session-ID`. State-changing UI requests include
the last received `revision`; HTTP 409 means the position changed or the move
is illegal. All move legality comes from `shared/game_core`.

| Endpoint | Request / result |
|---|---|
| `GET /api/health` | Application identity and loaded model metadata |
| `GET /api/model` | Architecture, parameter count, training provenance |
| `GET /api/state` | Pieces, turn, history, legal origins, status, revision, scenario |
| `GET /api/legal-moves?square=e2` | Engine-verified destinations |
| `POST /api/move` | `{from, to, revision}` -> updated state |
| `POST /api/reset` | `{scenario: "opening" | "mate" | "endgame"}` -> new game |
| `POST /api/undo` | `{plies: 1 | 2, revision}` -> restored state |
| `POST /api/analyze` | `{simulations: 1..256, revision}` -> state + real analysis |
| `POST /api/ai-move` | Same analysis request, then applies the selected move |
| `GET /api/training` | Training-job status and saved measurements |
| `POST /api/training/start` | Start a bounded local demonstration cycle |

Analysis includes the analyzed ply and side, per-move visits/priors/Q values,
search duration/depth, the actual visited tree, principal variation, encoded
input, and normalized mean-absolute activation maps. The UI labels old analysis
as “Last decision” and overlays board paths only when its position matches.

`runtime/` holds local process logs. `releases/` holds generated delivery
archives; both are excluded from version control.
