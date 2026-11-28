import type { BoardState } from '../types/game'

interface GameStatusProps {
  state: BoardState
  playModeLabel: string
}

export function GameStatus({ state, playModeLabel }: GameStatusProps) {
  const turnName = state.currentTurn === 'white' ? 'White (Ivory)' : 'Black (Onyx)'
  const thinking = state.isAiThinking

  return (
    <section className="panel status-card" aria-labelledby="game-status-heading">
      <div className="turn-display">
        <span className={`turn-piece turn-piece--${state.currentTurn}`} aria-hidden="true" />
        <div>
          <h2 id="game-status-heading">
            {state.status.code === 'terminal' ? state.status.label : `${turnName} to move`}
          </h2>
          <p>{playModeLabel}</p>
        </div>
      </div>
      <div className={`engine-state ${thinking ? 'engine-state--thinking' : ''}`}>
        <span className="engine-state__dot" />
        {thinking ? 'AI thinking…' : state.status.detail}
      </div>
    </section>
  )
}
