import type { Piece as PieceData } from '../types/game'
import { getPieceAsset } from '../assets/pieceAssets'

const pieceNames: Record<PieceData['kind'], string> = {
  raja: 'Raja',
  mantri: 'Mantri',
  ratha: 'Ratha',
  gaja: 'Gaja',
  ashva: 'Ashva',
  padati: 'Padati',
}

interface PieceProps {
  piece: PieceData
}

export function Piece({ piece }: PieceProps) {
  const assetUrl = getPieceAsset(piece)

  return (
    <span
      className={`piece piece--${piece.side} piece--${piece.kind}`}
      aria-label={`${piece.side} ${pieceNames[piece.kind]}`}
    >
      {assetUrl ? (
        <img className="piece__asset" src={assetUrl} alt="" draggable={false} />
      ) : (
        <span className="piece__placeholder" aria-hidden="true" />
      )}
    </span>
  )
}
