import { GameControls } from './GameControls'
import { GameResult } from './GameResult'
import { GameStatus } from './GameStatus'
import { MoveHistory } from './MoveHistory'
import type { BoardState } from '../types/game'

interface GameSidebarProps {
  state: BoardState
  disabled: boolean
  onReset: () => void
}

export function GameSidebar({ state, disabled, onReset }: GameSidebarProps) {
  return (
    <aside className="sidebar" aria-label="Game information">
      <GameStatus state={state} />
      <GameControls disabled={disabled} onReset={onReset} />
      <MoveHistory moves={state.moveHistory} />
      <GameResult status={state.status} />
    </aside>
  )
}
