import type { GameAdapter } from './gameAdapter'
import {
  FILES,
  type BoardState,
  type LegalMove,
  type Piece,
  type PieceKind,
  type Side,
  type SquareId,
} from '../types/game'

const WAIT_MS = 260

const backRank: PieceKind[] = [
  'ratha',
  'ashva',
  'gaja',
  'mantri',
  'raja',
  'gaja',
  'ashva',
  'ratha',
]

function createInitialPieces(): Piece[] {
  const pieces: Piece[] = []

  for (const side of ['black', 'white'] satisfies Side[]) {
    const backRankNumber = side === 'white' ? 1 : 8
    const padatiRankNumber = side === 'white' ? 2 : 7

    FILES.forEach((file, index) => {
      pieces.push({
        id: `${side}-${backRank[index]}-${file}`,
        kind: backRank[index],
        side,
        square: `${file}${backRankNumber}` as SquareId,
      })
      pieces.push({
        id: `${side}-padati-${file}`,
        kind: 'padati',
        side,
        square: `${file}${padatiRankNumber}` as SquareId,
      })
    })
  }

  return pieces
}

// These are mock response fixtures, not a movement algorithm. They exist only
// to exercise the UI until the Python game core supplies authoritative moves.
const mockResponses: Record<Side, Partial<Record<SquareId, LegalMove[]>>> = {
  white: {
    a2: [{ from: 'a2', to: 'a3', isCapture: false }],
    b2: [{ from: 'b2', to: 'b3', isCapture: false }],
    c2: [{ from: 'c2', to: 'c3', isCapture: false }],
    c3: [{ from: 'c3', to: 'c4', isCapture: false }],
    c4: [{ from: 'c4', to: 'd5', isCapture: true }],
    d2: [{ from: 'd2', to: 'd3', isCapture: false }],
    e2: [{ from: 'e2', to: 'e3', isCapture: false }],
    f2: [{ from: 'f2', to: 'f3', isCapture: false }],
    g2: [{ from: 'g2', to: 'g3', isCapture: false }],
    h2: [{ from: 'h2', to: 'h3', isCapture: false }],
    b1: [
      { from: 'b1', to: 'a3', isCapture: false },
      { from: 'b1', to: 'c3', isCapture: false },
    ],
    g1: [
      { from: 'g1', to: 'f3', isCapture: false },
      { from: 'g1', to: 'h3', isCapture: false },
    ],
  },
  black: {
    a7: [{ from: 'a7', to: 'a6', isCapture: false }],
    b7: [{ from: 'b7', to: 'b6', isCapture: false }],
    c7: [{ from: 'c7', to: 'c6', isCapture: false }],
    d7: [{ from: 'd7', to: 'd6', isCapture: false }],
    d6: [{ from: 'd6', to: 'd5', isCapture: false }],
    e7: [{ from: 'e7', to: 'e6', isCapture: false }],
    f7: [{ from: 'f7', to: 'f6', isCapture: false }],
    g7: [{ from: 'g7', to: 'g6', isCapture: false }],
    h7: [{ from: 'h7', to: 'h6', isCapture: false }],
    b8: [
      { from: 'b8', to: 'a6', isCapture: false },
      { from: 'b8', to: 'c6', isCapture: false },
    ],
    g8: [
      { from: 'g8', to: 'f6', isCapture: false },
      { from: 'g8', to: 'h6', isCapture: false },
    ],
  },
}

const pause = () => new Promise((resolve) => window.setTimeout(resolve, WAIT_MS))

const copyState = (state: BoardState): BoardState => ({
  ...state,
  pieces: state.pieces.map((piece) => ({ ...piece })),
  selectableSquares: [...state.selectableSquares],
  moveHistory: state.moveHistory.map((move) => ({ ...move })),
  lastMove: state.lastMove ? { ...state.lastMove } : null,
  status: { ...state.status },
})

function availableSources(side: Side, pieces: Piece[]): SquareId[] {
  const occupiedBySide = new Set(
    pieces.filter((piece) => piece.side === side).map((piece) => piece.square),
  )

  return (Object.keys(mockResponses[side]) as SquareId[]).filter((square) =>
    occupiedBySide.has(square),
  )
}

function initialState(): BoardState {
  const pieces = createInitialPieces()
  return {
    pieces,
    currentTurn: 'white',
    selectableSquares: availableSources('white', pieces),
    lastMove: null,
    moveHistory: [],
    status: {
      code: 'playing',
      label: 'Game in progress',
      detail: 'Select a highlighted piece to begin.',
    },
    isAiThinking: false,
  }
}

export class MockGameAdapter implements GameAdapter {
  private state = initialState()

  /** Stub AI move — picks the first available mock move for the current side. */
  async requestAiMove(_numSimulations?: number): Promise<BoardState> {
    await pause()
    const side = this.state.currentTurn
    const sources = availableSources(side, this.state.pieces)
    for (const src of sources) {
      const moves = mockResponses[side][src] ?? []
      if (moves.length > 0) {
        return this.requestMove({ from: moves[0].from, to: moves[0].to })
      }
    }
    return copyState(this.state)
  }

  async getState(): Promise<BoardState> {
    await pause()
    return copyState(this.state)
  }

  async getLegalMoves(square: SquareId): Promise<LegalMove[]> {
    await pause()
    if (!this.state.selectableSquares.includes(square)) return []
    return (mockResponses[this.state.currentTurn][square] ?? []).map((move) => ({ ...move }))
  }

  async requestMove(request: Pick<LegalMove, 'from' | 'to'>): Promise<BoardState> {
    await pause()
    const suppliedMove = (mockResponses[this.state.currentTurn][request.from] ?? []).find(
      (move) => move.to === request.to,
    )
    const movingPiece = this.state.pieces.find(
      (piece) => piece.square === request.from && piece.side === this.state.currentTurn,
    )

    if (!suppliedMove || !movingPiece) {
      throw new Error('The mock engine rejected that move. Please select a highlighted destination.')
    }

    const capturedPiece = this.state.pieces.find((piece) => piece.square === request.to)
    const nextPieces = this.state.pieces
      .filter((piece) => piece.id !== capturedPiece?.id)
      .map((piece) =>
        piece.id === movingPiece.id ? { ...piece, square: request.to } : piece,
      )
    const ply = this.state.moveHistory.length + 1
    const move = {
      id: `mock-move-${ply}`,
      ply,
      side: movingPiece.side,
      piece: movingPiece.kind,
      from: request.from,
      to: request.to,
      captured: capturedPiece?.kind,
      notation: `${request.from}–${request.to}`,
    }
    const nextTurn: Side = this.state.currentTurn === 'white' ? 'black' : 'white'

    this.state = {
      ...this.state,
      pieces: nextPieces,
      currentTurn: nextTurn,
      selectableSquares: availableSources(nextTurn, nextPieces),
      lastMove: move,
      moveHistory: [...this.state.moveHistory, move],
      status: {
        code: 'playing',
        label: 'Game in progress',
        detail: `${nextTurn === 'white' ? 'Ivory' : 'Onyx'} to move.`,
      },
    }

    return copyState(this.state)
  }

  async resetGame(): Promise<BoardState> {
    await pause()
    this.state = initialState()
    return copyState(this.state)
  }
}

export const gameAdapter: GameAdapter = new MockGameAdapter()
