import type { Move } from '../types/game'

interface MoveHistoryProps {
  moves: Move[]
}

export function MoveHistory({ moves }: MoveHistoryProps) {
  const rounds: Array<{ number: number; white?: Move; black?: Move }> = []

  moves.forEach((move) => {
    const index = Math.floor((move.ply - 1) / 2)
    rounds[index] ??= { number: index + 1 }
    rounds[index][move.side] = move
  })

  return (
    <section className="panel history" aria-labelledby="history-heading">
      <div className="history__heading">
        <h2 id="history-heading">Move history</h2>
        <span className="move-count">{moves.length} moves</span>
      </div>
      {rounds.length === 0 ? (
        <div className="empty-state">
          <span className="empty-state__mark" aria-hidden="true"><i /><i /></span>
          <p>No moves yet</p>
        </div>
      ) : (
        <ol className="move-list">
          {rounds.map((round) => (
            <li key={round.number}>
              <span className="move-number">{round.number}.</span>
              <span>{round.white?.notation ?? '—'}</span>
              <span>{round.black?.notation ?? '—'}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
