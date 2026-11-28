import type { Piece as PieceData } from '../types/game'
import { getPieceAsset } from '../assets/pieceAssets'
import { PieceGlyph } from '../assets/PieceGlyph'
import { PIECE_LABELS } from '../lib/capturedPieces'

interface PieceProps {
  piece: PieceData
  canDrag?: boolean
  onPointerDragStart?: () => void
}

export function Piece({ piece, canDrag = false, onPointerDragStart }: PieceProps) {
  const assetUrl = getPieceAsset(piece)
  const label = `${piece.side === 'white' ? 'White' : 'Black'} ${PIECE_LABELS[piece.kind]}`

  return (
    <span
      className={`piece piece--${piece.side} piece--${piece.kind} ${canDrag ? 'piece--draggable' : ''}`}
      aria-label={label}
      onPointerDown={(event) => {
        if (!canDrag || event.button !== 0) return
        event.preventDefault()
        onPointerDragStart?.()
      }}
    >
      {assetUrl ? (
        <img className="piece__asset" src={assetUrl} alt="" draggable={false} />
      ) : (
        <PieceGlyph kind={piece.kind} side={piece.side} />
      )}
    </span>
  )
}
