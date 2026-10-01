import type { BoardState, GameMode, LegalMove, MachineProgress, SquareId } from '../types/game'

/**
 * The only boundary the React application uses to communicate with game logic.
 * The HTTP implementation delegates every operation to the authoritative
 * Python core; React components never calculate or apply moves themselves.
 */
export interface GameAdapter {
  getState(): Promise<BoardState>
  getLegalMoves(square: SquareId): Promise<LegalMove[]>
  requestMove(move: Pick<LegalMove, 'from' | 'to'>): Promise<BoardState>
  resetGame(mode: GameMode): Promise<BoardState>
  requestMachineMove(): Promise<BoardState>
  getMachineProgress(): Promise<MachineProgress>
}
