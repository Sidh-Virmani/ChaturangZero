import { Square } from './Square'
import {
  FILES,
  RANKS,
  type BoardState,
  type LegalMove,
  type Side,
  type Square as SquareData,
  type SquareId,
} from '../types/game'

interface BoardProps {
  state: BoardState
  orientation: Side
  selectedSquare: SquareId | null
  legalMoves: LegalMove[]
  disabled: boolean
  busy: boolean
  busyLabel: string
  onSquareClick: (square: SquareId) => void
}

export function Board({
  state,
  orientation,
  selectedSquare,
  legalMoves,
  disabled,
  busy,
  busyLabel,
  onSquareClick,
}: BoardProps) {
  const piecesBySquare = new Map(state.pieces.map((piece) => [piece.square, piece]))
  const moveByDestination = new Map(legalMoves.map((move) => [move.to, move]))
  const lastMoveSquares = new Set(
    state.lastMove ? [state.lastMove.from, state.lastMove.to] : [],
  )
  const files = orientation === 'white' ? FILES : [...FILES].reverse()
  const ranks = orientation === 'white' ? RANKS : [...RANKS].reverse()

  return (
    <div className="board-frame">
      <div className="board" role="grid" aria-label={`Chaturanga board, ${orientation} perspective`} aria-busy={busy}>
        {ranks.flatMap((rank, rankIndex) =>
          files.map((file, fileIndex) => {
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
                isLastMoveDestination={state.lastMove?.to === square.id}
                showRank={fileIndex === 0}
                showFile={rankIndex === 7}
                disabled={disabled || (!isInteractive && selectedSquare === null)}
                onClick={() => onSquareClick(square.id)}
              />
            )
          }),
        )}
      </div>
      {busy && (
        <div className="board-loading" role="status">
          <span className="spinner" />
          {busyLabel}
        </div>
      )}
    </div>
  )
}
