import { GameControls } from './GameControls'
import { GameStatus } from './GameStatus'
import { MoveHistory } from './MoveHistory'
import type { GameMode } from '../App'
import type { BoardState, Side } from '../types/game'

interface GameSidebarProps {
  state: BoardState
  mode: GameMode
  orientation: Side
  disabled: boolean
  onReset: () => void
  onFlip: () => void
  onHome: () => void
}

export function GameSidebar({ state, mode, orientation, disabled, onReset, onFlip, onHome }: GameSidebarProps) {
  return <aside className="sidebar" aria-label="Game information">
    <div className="sidebar__header"><span className="eyebrow">MATCH CENTER</span><span className="sidebar__menu-mark" aria-hidden="true"><i /><i /><i /></span></div>
    <GameStatus state={state} mode={mode} />
    <MoveHistory moves={state.moveHistory} />
    {mode === 'machine' && <section className="ai-panel" aria-labelledby="ai-heading"><div className="ai-panel__heading"><span className="ai-panel__symbol" aria-hidden="true">✧</span><h2 id="ai-heading">Machine analysis</h2></div><p>Search details and candidate moves will appear here once a trained checkpoint and an AI move endpoint are connected.</p></section>}
    <GameControls disabled={disabled} orientation={orientation} onReset={onReset} onFlip={onFlip} onHome={onHome} />
  </aside>
}
