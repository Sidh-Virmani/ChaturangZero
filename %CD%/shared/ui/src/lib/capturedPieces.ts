import type { Piece, PieceKind, Side } from '../types/game'

export const PIECE_ORDER: PieceKind[] = [
  'raja',
  'mantri',
  'ratha',
  'gaja',
  'ashva',
  'padati',
]

export const PIECE_LABELS: Record<PieceKind, string> = {
  raja: 'Raja',
  mantri: 'Mantri',
  ratha: 'Ratha',
  gaja: 'Gaja',
  ashva: 'Ashva',
  padati: 'Padati',
}

export const PIECE_ROLES: Record<PieceKind, string> = {
  raja: 'King',
  mantri: 'Counselor',
  ratha: 'Chariot',
  gaja: 'Elephant',
  ashva: 'Horse',
  padati: 'Foot soldier',
}

const START_COUNTS: Record<PieceKind, number> = {
  raja: 1,
  mantri: 1,
  ratha: 2,
  gaja: 2,
  ashva: 2,
  padati: 8,
}

/** Pieces of `side` that are no longer on the board, vs the standard starting set. */
export function missingPieces(side: Side, pieces: Piece[]): PieceKind[] {
  const counts: Record<PieceKind, number> = {
    raja: 0,
    mantri: 0,
    ratha: 0,
    gaja: 0,
    ashva: 0,
    padati: 0,
  }
  for (const piece of pieces) {
    if (piece.side === side) counts[piece.kind] += 1
  }

  const missing: PieceKind[] = []
  for (const kind of PIECE_ORDER) {
    const gone = START_COUNTS[kind] - counts[kind]
    for (let i = 0; i < gone; i += 1) missing.push(kind)
  }
  return missing
}

/** Material taken by `side` (the opponent’s missing pieces). Display only. */
export function capturedBy(side: Side, pieces: Piece[]): PieceKind[] {
  const opponent: Side = side === 'white' ? 'black' : 'white'
  return missingPieces(opponent, pieces)
}
