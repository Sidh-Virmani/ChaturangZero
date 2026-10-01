import type { Move } from '../types/game'

interface MoveHistoryProps { moves: Move[] }

export function MoveHistory({ moves }: MoveHistoryProps) {
  const rounds: Array<{ number: number; white?: Move; black?: Move }> = []
  moves.forEach((move) => {
    const index = Math.floor((move.ply - 1) / 2)
    rounds[index] ??= { number: index + 1 }
    rounds[index][move.side] = move
  })
  return <section className="history" aria-labelledby="history-heading">
    <div className="history__heading"><div><span className="eyebrow">THE STORY SO FAR</span><h2 id="history-heading">Move history</h2></div><span className="move-count">{moves.length} {moves.length === 1 ? 'move' : 'moves'}</span></div>
    <div className="history__columns" aria-hidden="true"><span>NO.</span><span>IVORY</span><span>ONYX</span></div>
    {rounds.length === 0 ? <div className="empty-state"><span className="empty-state__mark" aria-hidden="true">◇</span><p>No moves yet</p><small>The first move begins the story.</small></div> : <ol className="move-list" aria-live="polite">{rounds.map((round) => <li key={round.number}><span className="move-number">{round.number}.</span><span>{round.white?.notation ?? '—'}</span><span>{round.black?.notation ?? '—'}</span></li>)}</ol>}
  </section>
}
