import { useEffect, useRef } from 'react'
import type { Move } from '../types/game'
import { PIECE_LABELS } from '../lib/capturedPieces'

interface MoveHistoryProps {
  moves: Move[]
}

export function MoveHistory({ moves }: MoveHistoryProps) {
  const listRef = useRef<HTMLOListElement>(null)
  const lastId = moves.at(-1)?.id
  const rounds: Array<{ number: number; white?: Move; black?: Move }> = []

  moves.forEach((move) => {
    const index = Math.floor((move.ply - 1) / 2)
    rounds[index] ??= { number: index + 1 }
    rounds[index][move.side] = move
  })

  useEffect(() => {
    const list = listRef.current
    if (!list) return
    const current = list.querySelector('[data-current="true"]')
    current?.scrollIntoView({ block: 'nearest' })
  }, [lastId])

  return (
    <section className="panel history" aria-labelledby="history-heading">
      <div className="history__heading">
        <h2 id="history-heading">Move history</h2>
        <span className="move-count">{moves.length} plies</span>
      </div>
      {rounds.length === 0 ? (
        <div className="empty-state">
          <span className="empty-state__mark" aria-hidden="true">
            <i />
            <i />
          </span>
          <p>No moves yet. Highlighted pieces are the only demo replies the mock adapter knows.</p>
        </div>
      ) : (
        <ol className="move-list" ref={listRef}>
          {rounds.map((round) => (
            <li key={round.number}>
              <span className="move-number">{round.number}.</span>
              <HistoryPly move={round.white} isCurrent={round.white?.id === lastId} />
              <HistoryPly move={round.black} isCurrent={round.black?.id === lastId} />
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}

function HistoryPly({ move, isCurrent }: { move?: Move; isCurrent: boolean }) {
  if (!move) return <span className="ply ply--empty">—</span>
  return (
    <span
      className={`ply ${isCurrent ? 'ply--current' : ''}`}
      data-current={isCurrent || undefined}
      title={`${PIECE_LABELS[move.piece]} ${move.from} to ${move.to}${move.captured ? `, captures ${PIECE_LABELS[move.captured]}` : ''}`}
    >
      {move.notation}
    </span>
  )
}
