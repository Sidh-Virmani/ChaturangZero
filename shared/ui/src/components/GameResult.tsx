import type { GameStatus } from '../types/game'

interface GameResultProps {
  status: GameStatus
}

export function GameResult({ status }: GameResultProps) {
  return (
    <section
      className={`panel result ${status.code === 'terminal' ? 'result--complete' : ''}`}
      aria-labelledby="result-heading"
    >
      <span className="result__label">Result</span>
      <strong id="result-heading">
        {status.code === 'terminal' ? status.result ?? 'Game complete' : 'Pending'}
      </strong>
    </section>
  )
}
