import type { GameMode } from '../App'
import type { BoardState } from '../types/game'

interface GameStatusProps { state: BoardState; mode: GameMode }

export function GameStatus({ state, mode }: GameStatusProps) {
  const turnName = state.currentTurn === 'white' ? 'Ivory' : 'Onyx'
  const isTerminal = state.status.code === 'terminal'
  return <section className="status-card" aria-labelledby="game-status-heading">
    <div className="status-card__label"><span>GAME STATUS</span><span className="status-card__live"><i /> {isTerminal ? 'FINISHED' : mode === 'machine' ? 'PREVIEW' : 'LIVE'}</span></div>
    <div className="turn-display"><span className={`turn-piece turn-piece--${state.currentTurn}`} aria-hidden="true" /><div><h2 id="game-status-heading">{isTerminal ? 'Game complete' : mode === 'machine' ? 'Machine play unavailable' : `${turnName} to move`}</h2><p>{mode === 'machine' && !isTerminal ? 'A trained model and AI game endpoint are needed before this mode can play.' : state.status.detail}</p></div></div>
    {mode === 'human' && !isTerminal && <div className="status-card__hint">Select a piece to see its legal moves.</div>}
  </section>
}
