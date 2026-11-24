export const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const
export const RANKS = [8, 7, 6, 5, 4, 3, 2, 1] as const

export type File = (typeof FILES)[number]
export type Rank = (typeof RANKS)[number]
export type SquareId = `${File}${Rank}`
export type Side = 'white' | 'black'
export type GameMode = 'human' | 'machine'
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

export interface AnalyzedAction {
  from: SquareId
  to: SquareId
  actionId: number
  prior: number
  visits?: number
}

export interface MachineAnalysis {
  searchElapsedSeconds: number
  simulationsPerSecond: number
  pipelineElapsedSeconds: number
  checkpoint: string
  smokeCheckpoint: boolean
  sideToMove: Side
  actionSpaceSize: number
  legalActionCount: number
  value: number
  simulations: number
  cPuct: number
  rootVisits: number
  policyPriors: AnalyzedAction[]
  candidates: AnalyzedAction[]
  selectedAction: AnalyzedAction
}

export interface MachineProgress {
  searchElapsedSeconds?: number
  phase: string
  events: string[]
  legalActionCount: number | null
  simulationsCompleted: number
  totalSimulations: number
  done: boolean
  error: string | null
}

export interface BoardState {
  engine?: {
    checkpoint: string
    sha256: string
    inferenceDevice: string
    simulations: number
    cPuct: number
    model: { channels: number; residual_blocks: number }
    training: {
      iteration: number
      selfPlaySimulations?: number | null
      workers?: number | null
      replayCapacity?: number | null
      symmetryProbability?: number | null
      device?: string | null
      gitCommit?: string | null
      source: string
    } | null
  } | null
  repetition?: { occurrences: number; drawAt: number } | null
  pieces: Piece[]
  currentTurn: Side
  selectableSquares: SquareId[]
  lastMove: Move | null
  moveHistory: Move[]
  status: GameStatus
  isAiThinking: boolean
  mode: GameMode
  machineReady: boolean
  checkpointLabel: string | null
  smokeCheckpoint: boolean
  analysis: MachineAnalysis | null
}
