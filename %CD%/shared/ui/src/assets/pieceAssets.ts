import type { Piece, PieceKind, Side } from '../types/game'

type PieceAssetKey = `${Side}-${PieceKind}`

/**
 * Add imported custom piece image URLs here when the final asset set is ready.
 * Piece rendering will automatically replace the neutral fallback silhouette.
 */
const pieceAssetUrls: Partial<Record<PieceAssetKey, string>> = {}

export function getPieceAsset(piece: Pick<Piece, 'side' | 'kind'>): string | undefined {
  return pieceAssetUrls[`${piece.side}-${piece.kind}`]
}
