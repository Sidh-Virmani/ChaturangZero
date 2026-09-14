import type { BoardState } from '../types/game'

interface GameStatusProps {
  state: BoardState
}

export function GameStatus({ state }: GameStatusProps) {
  const turnName = state.currentTurn === 'white' ? 'Ivory' : 'Onyx'

  return (
    <section className="panel status-card" aria-labelledby="game-status-heading">
      <div className="turn-display">
        <span className={`turn-piece turn-piece--${state.currentTurn}`} aria-hidden="true" />
        <div>
          <h2 id="game-status-heading">{turnName} to move</h2>
          <p>{state.status.label}</p>
        </div>
      </div>
      <div className={`engine-state ${state.isAiThinking ? 'engine-state--thinking' : ''}`}>
        <span className="engine-state__dot" />
        {state.isAiThinking ? 'AI thinking...' : state.status.detail}
      </div>
    </section>
  )
}
