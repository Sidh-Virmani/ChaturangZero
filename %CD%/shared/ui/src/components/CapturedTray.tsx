import { PieceGlyph } from '../assets/PieceGlyph'
import type { PieceKind, Side } from '../types/game'
import { PIECE_LABELS } from '../lib/capturedPieces'

interface CapturedTrayProps {
  side: Side
  pieces: PieceKind[]
  label: string
}

export function CapturedTray({ side, pieces, label }: CapturedTrayProps) {
  return (
    <div className="captured-tray" aria-label={label}>
      {pieces.length === 0 ? (
        <span className="captured-empty">No captures</span>
      ) : (
        pieces.map((kind, index) => (
          <span
            key={`${kind}-${index}`}
            className={`piece piece--${side} piece--mini`}
            title={`${PIECE_LABELS[kind]} captured`}
          >
            <PieceGlyph kind={kind} side={side} />
          </span>
        ))
      )}
    </div>
  )
}
