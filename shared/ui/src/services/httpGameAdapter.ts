import type { Analysis, BoardState, LegalMove, ModelInfo, SquareId, TrainingState } from '../types/game'

/**
 * The real adapter: every question about the rules goes to the Python
 * game core over HTTP. This file contains no Chaturanga logic at all —
 * that is the whole point. If you ever find yourself wanting to work out
 * "can this piece move there?" here, it belongs in shared/game_core/
 * instead.
 *
 * Requests go to a relative /api path, which Vite proxies to the Python
 * server in development (see vite.config.ts). That keeps the port number
 * in exactly one place and avoids CORS entirely.
 */

const API_BASE = '/api'
const session = sessionStorage.getItem('cz-session') || crypto.randomUUID()
sessionStorage.setItem('cz-session', session)

/** Surfaced to the user in App.tsx's error banner. */
class EngineError extends Error {}

const ENGINE_UNREACHABLE =
  'Cannot reach the game engine. Is the Python server running? Start everything with start-chaturangzero.bat, or run: python run.py'

/**
 * Gateway statuses mean the Vite dev proxy is up but could not reach the
 * Python server behind it. This is by far the most common failure here
 * (someone started the UI but not the engine), and the browser reports
 * it as a normal HTTP response rather than a network error — so it has
 * to be caught explicitly or the user just sees "HTTP 502".
 */
const UPSTREAM_DOWN = new Set([502, 503, 504])

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response

  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', 'X-Session-ID': session },
      signal: AbortSignal.timeout(90000),
      ...init,
    })
  } catch {
    // fetch only rejects on network-level failures, which in practice
    // means nothing is listening at all.
    throw new EngineError(ENGINE_UNREACHABLE)
  }

  if (UPSTREAM_DOWN.has(response.status)) {
    throw new EngineError(ENGINE_UNREACHABLE)
  }

  if (!response.ok) {
    // The server sends { "error": "..." } for anything it rejects, e.g.
    // an illegal move. Prefer that message over a bare status code.
    const detail = await response.json().catch(() => null)
    throw new EngineError(detail?.error ?? `Engine returned HTTP ${response.status}`)
  }

  return response.json() as Promise<T>
}

const post = <T,>(path: string, body = {}) => request<T>(path, { method: 'POST', body: JSON.stringify(body) })
export const api = {
  state: () => request<BoardState>('/state'),
  model: () => request<ModelInfo>('/model'),
  legal: (square: SquareId) => request<LegalMove[]>(`/legal-moves?square=${square}`),
  move: (from: SquareId, to: SquareId, revision: number) => post<BoardState>('/move', { from, to, revision }),
  reset: (scenario = 'opening') => post<BoardState>('/reset', { scenario }),
  undo: (plies: number, revision: number) => post<BoardState>('/undo', { plies, revision }),
  analyze: (simulations: number, revision: number, play = false) => post<{ state: BoardState; analysis: Analysis }>(play ? '/ai-move' : '/analyze', { simulations, revision }),
  training: () => request<TrainingState>('/training'),
  train: () => post<TrainingState>('/training/start'),
}
