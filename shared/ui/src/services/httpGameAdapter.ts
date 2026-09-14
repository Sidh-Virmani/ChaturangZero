/**
 * httpGameAdapter.ts
 * ==================
 * Real HTTP adapter that connects to the FastAPI backend (integration/api.py).
 *
 * The adapter implements the same GameAdapter interface as the mock,
 * so all React components work unchanged.
 *
 * Backend must be running at:  http://localhost:8000
 * Start it with:
 *   cd ChaturangZero-main
 *   uvicorn integration.api:app --reload --port 8000
 */

import type { GameAdapter } from './gameAdapter'
import type { BoardState, LegalMove, SquareId } from '../types/game'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(
      typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail),
    )
  }
  return res.json() as Promise<T>
}

export class HttpGameAdapter implements GameAdapter {
  async getState(): Promise<BoardState> {
    return apiRequest<BoardState>('/state')
  }

  async getLegalMoves(square: SquareId): Promise<LegalMove[]> {
    return apiRequest<LegalMove[]>(`/legal_moves/${square}`)
  }

  async requestMove(move: Pick<LegalMove, 'from' | 'to'>): Promise<BoardState> {
    return apiRequest<BoardState>('/move', {
      method: 'POST',
      body: JSON.stringify({ from_sq: move.from, to_sq: move.to }),
    })
  }

  async requestAiMove(numSimulations = 50): Promise<BoardState> {
    return apiRequest<BoardState>('/ai_move', {
      method: 'POST',
      body: JSON.stringify({ num_simulations: numSimulations }),
    })
  }

  async resetGame(): Promise<BoardState> {
    return apiRequest<BoardState>('/reset', { method: 'POST' })
  }
}

export const httpGameAdapter: GameAdapter = new HttpGameAdapter()
