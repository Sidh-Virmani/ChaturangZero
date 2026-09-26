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
  revision: number
  scenario: 'opening' | 'mate' | 'endgame'
  pieces: Piece[]
  currentTurn: Side
  selectableSquares: SquareId[]
  lastMove: Move | null
  moveHistory: Move[]
  status: GameStatus
  isAiThinking: boolean
}

export interface ModelInfo {
  name: string
  trained: boolean
  parameters: number
  channels: number
  blocks: number
  actions: number
  epochs?: number
  examples?: number
  trainingGames?: number
  curriculum?: boolean
  limitations: string
}

export interface Candidate {
  from: SquareId
  to: SquareId
  notation: string
  visits: number
  prior: number
  probability: number
  value: number
}

export interface SearchNode {
  move: string
  visits: number
  value: number
  prior: number
  children?: SearchNode[]
}

export interface Analysis {
  value: number
  searchValue: number
  simulations: number
  elapsedMs: number
  depth: number
  candidates: Candidate[]
  tree: SearchNode
  layers: { name: string; shape: number[]; heatmap: number[]; maxActivation: number; channels: number[][] }[]
  input: number[][][]
  inputShape: number[]
  policyHeatmap: number[]
  policyPriorHeatmap: number[]
  principalVariation: string[]
  model: ModelInfo
  positionPly: number
  positionTurn: Side
}

export interface Evaluation {
  wins: number; draws: number; losses: number; games: number; score: number
  truncated: number; scope: string; opponent: string; simulations: number; maxPlies: number
}

export interface TrainingState {
  job: { running: boolean; phase: string; detail: string; progress: number }
  evaluation?: null | {
    complete?: boolean
    model: ModelInfo
    sha256: string
    seed: number
    simulations: number
    maxPlies: number
    gamesPerComparison: number
    comparisons: (Evaluation & { name: string; paired95Interval: [number, number] })[]
  }
  metrics: null | {
    replayExamples?: number
    epochs: { step: number; loss: number; policyLoss: number; valueLoss: number }[]
    games: { plies: number; winner: string | null; truncated: boolean; source: string }[]
    runs: { id: number; round: number; promoted: boolean; bootstrap: boolean; arena: Evaluation; baseline: Evaluation; seconds: number }[]
    model: ModelInfo
    updatedAt: string
  }
}
