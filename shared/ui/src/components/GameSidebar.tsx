import { GameControls } from './GameControls'
import { GameStatus } from './GameStatus'
import { MoveHistory } from './MoveHistory'
import { MachineAnalysisPanel } from './MachineAnalysisPanel'
import type { GameMode } from '../types/game'
import type { BoardState, MachineAnalysis, MachineProgress, Side } from '../types/game'

interface GameSidebarProps {
  state: BoardState
  mode: GameMode
  orientation: Side
  disabled: boolean
  thinking: boolean
  progress: MachineProgress | null
  machineResult: MachineAnalysis | null
  onReset: () => void
  onFlip: () => void
}

export function GameSidebar({ state, mode, orientation, thinking, progress, machineResult, disabled, onReset, onFlip }: GameSidebarProps) {
  return <aside className="sidebar" aria-label="Game information">
    <div className="sidebar__header"><span className="eyebrow">MATCH CENTER</span><span className="sidebar__menu-mark" aria-hidden="true"><i /><i /><i /></span></div>
    <GameStatus state={state} mode={mode} thinking={thinking} />
    {mode === 'machine' && <MachineAnalysisPanel state={state} thinking={thinking && !machineResult} progress={progress} result={machineResult} />}
    <MoveHistory moves={state.moveHistory} />
    <GameControls disabled={disabled} orientation={orientation} onReset={onReset} onFlip={onFlip} />
  </aside>
}
