export const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const
export const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const

export type File = (typeof FILES)[number]
export type Rank = (typeof RANKS)[number]
export type SquareId = `${File}${Rank}`
export type Side = 'white' | 'black'
export type PieceKind = 'raja' | 'mantri' | 'ratha' | 'gaja' | 'ashva' | 'padati'

export interface Square {
  id: SquareId
  file: File
  rank: Rank
}

export interface Piece {
  id: string
  kind: PieceKind
  side: Side
  square: SquareId
}

export interface LegalMove {
  from: SquareId
  to: SquareId
  isCapture: boolean
}

export interface Move {
  id: string
  ply: number
  side: Side
  piece: PieceKind
  from: SquareId
  to: SquareId
  captured?: PieceKind
  notation: string
}

export type GameStatusCode = 'playing' | 'check' | 'terminal'

export interface GameStatus {
  code: GameStatusCode
  label: string
  detail: string
  result?: string
}

export interface BoardState {
  pieces: Piece[]
  currentTurn: Side
  selectableSquares: SquareId[]
  lastMove: Move | null
  moveHistory: Move[]
  status: GameStatus
  isAiThinking: boolean
}
