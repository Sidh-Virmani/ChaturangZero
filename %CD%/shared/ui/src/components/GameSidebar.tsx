import { GameControls } from './GameControls'
import { GameResult } from './GameResult'
import { GameStatus } from './GameStatus'
import { MoveHistory } from './MoveHistory'
import { PieceLegend } from './PieceLegend'
import type { BoardState } from '../types/game'
import type { Difficulty, PlayMode } from '../lib/prefs'

interface GameSidebarProps {
  state: BoardState
  disabled: boolean
  playMode: PlayMode
  difficulty: Difficulty
  onReset: () => void
  onPlayMode: (mode: PlayMode) => void
  onDifficulty: (difficulty: Difficulty) => void
}

export function GameSidebar({
  state,
  disabled,
  playMode,
  difficulty,
  onReset,
  onPlayMode,
  onDifficulty,
}: GameSidebarProps) {
  const playModeLabel =
    playMode === 'ai'
      ? `Human vs AI · ${difficulty}`
      : 'Human vs Human'

  return (
    <aside className="sidebar" aria-label="Game information">
      <GameStatus state={state} playModeLabel={playModeLabel} />
      <GameControls
        playMode={playMode}
        difficulty={difficulty}
        onPlayMode={onPlayMode}
        onDifficulty={onDifficulty}
      />
      <MoveHistory moves={state.moveHistory} />
      <PieceLegend />
      <GameResult status={state.status} onReset={onReset} disabled={disabled} />
    </aside>
  )
}
