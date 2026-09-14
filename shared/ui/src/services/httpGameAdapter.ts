import type { GameAdapter } from './gameAdapter'
import type { BoardState, LegalMove, SquareId } from '../types/game'

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...init?.headers,
      },
    })
  } catch {
    throw new Error('Cannot reach the Chaturanga game core. Start the app with start-chaturangzero.bat.')
  }

  const payload = (await response.json().catch(() => null)) as { error?: string } | null
  if (!response.ok) {
    throw new Error(payload?.error ?? `Game core rejected the request (${response.status}).`)
  }
  return payload as T
}

export class HttpGameAdapter implements GameAdapter {
  getState(): Promise<BoardState> {
    return apiRequest<BoardState>('/api/game')
  }

  getLegalMoves(square: SquareId): Promise<LegalMove[]> {
    return apiRequest<LegalMove[]>(`/api/game/legal-moves?from=${encodeURIComponent(square)}`)
  }

  requestMove(move: Pick<LegalMove, 'from' | 'to'>): Promise<BoardState> {
    return apiRequest<BoardState>('/api/game/move', {
      method: 'POST',
      body: JSON.stringify(move),
    })
  }

  resetGame(): Promise<BoardState> {
    return apiRequest<BoardState>('/api/game/reset', {
      method: 'POST',
      body: '{}',
    })
  }
}

export const gameAdapter: GameAdapter = new HttpGameAdapter()

