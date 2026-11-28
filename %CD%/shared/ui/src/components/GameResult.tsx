import type { GameStatus } from '../types/game'

interface GameResultProps {
  status: GameStatus
  onReset: () => void
  disabled: boolean
}

export function GameResult({ status, onReset, disabled }: GameResultProps) {
  const complete = status.code === 'terminal'

  return (
    <section
      className={`panel result ${complete ? 'result--complete' : ''}`}
      aria-labelledby="result-heading"
    >
      <div className="result__copy">
        <span className="result__label">{complete ? 'Game over' : 'Result'}</span>
        <strong id="result-heading">
          {complete ? status.result ?? status.label : 'In progress'}
        </strong>
        {complete && <p>{status.detail}</p>}
      </div>
      <button
        className={`button ${complete ? 'button--primary button--wide' : 'button--ghost'}`}
        type="button"
        onClick={onReset}
        disabled={disabled}
      >
        New game
      </button>
    </section>
  )
}
