import type { BoardState, LegalMove, SquareId } from '../types/game'

/**
 * The only boundary the React application uses to communicate with game logic.
 * A future HTTP adapter can implement this interface and delegate every method
 * to the Python game core without changing board components.
 */
export interface GameAdapter {
  getState(): Promise<BoardState>
  getLegalMoves(square: SquareId): Promise<LegalMove[]>
  requestMove(move: Pick<LegalMove, 'from' | 'to'>): Promise<BoardState>
  resetGame(): Promise<BoardState>
}
