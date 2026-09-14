import { Square } from './Square'
import {
  FILES,
  RANKS,
  type BoardState,
  type LegalMove,
  type Square as SquareData,
  type SquareId,
} from '../types/game'

interface BoardProps {
  state: BoardState
  selectedSquare: SquareId | null
  legalMoves: LegalMove[]
  disabled: boolean
  onSquareClick: (square: SquareId) => void
}

export function Board({
  state,
  selectedSquare,
  legalMoves,
  disabled,
  onSquareClick,
}: BoardProps) {
  const piecesBySquare = new Map(state.pieces.map((piece) => [piece.square, piece]))
  const moveByDestination = new Map(legalMoves.map((move) => [move.to, move]))
  const lastMoveSquares = new Set(
    state.lastMove ? [state.lastMove.from, state.lastMove.to] : [],
  )

  return (
    <div className="board-frame">
      <div className="board" role="grid" aria-label="Chaturanga board" aria-busy={disabled}>
        {RANKS.flatMap((rank, rankIndex) =>
          FILES.map((file, fileIndex) => {
            const square: SquareData = {
              id: `${file}${rank}`,
              file,
              rank,
            }
            const legalMove = moveByDestination.get(square.id)
            const isSelectable = state.selectableSquares.includes(square.id)
            const isSelected = selectedSquare === square.id
            const isInteractive = isSelectable || isSelected || Boolean(legalMove)

            return (
              <Square
                key={square.id}
                square={square}
                piece={piecesBySquare.get(square.id)}
                tone={(rankIndex + fileIndex) % 2 === 0 ? 'light' : 'dark'}
                isSelected={isSelected}
                isSelectable={isSelectable}
                isLegalDestination={Boolean(legalMove)}
                isCaptureDestination={legalMove?.isCapture ?? false}
                isLastMove={lastMoveSquares.has(square.id)}
                disabled={disabled || (!isInteractive && selectedSquare === null)}
                onClick={() => onSquareClick(square.id)}
              />
            )
          }),
        )}
      </div>
      {disabled && (
        <div className="board-loading" role="status">
          <span className="spinner" />
          Syncing with engine...
        </div>
      )}
    </div>
  )
}
