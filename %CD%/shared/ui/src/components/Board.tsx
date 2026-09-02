import type { KeyboardEvent } from 'react'
import { useEffect, useRef } from 'react'
import { Square } from './Square'
import {
  type BoardState,
  type LegalMove,
  type Side,
  type Square as SquareData,
  type SquareId,
} from '../types/game'
import { displayFiles, displayRanks, squareTone } from '../lib/squares'

interface BoardProps {
  state: BoardState
  selectedSquare: SquareId | null
  focusedSquare: SquareId
  showFocus: boolean
  legalMoves: LegalMove[]
  orientation: Side
  locked: boolean
  busyLabel: string | null
  onSquareClick: (square: SquareId) => void
  onDragStart: (square: SquareId) => void
  onDropSquare: (square: SquareId) => void
  onKeyNavigate: (event: KeyboardEvent<HTMLDivElement>) => void
}

export function Board({
  state,
  selectedSquare,
  focusedSquare,
  showFocus,
  legalMoves,
  orientation,
  locked,
  busyLabel,
  onSquareClick,
  onDragStart,
  onDropSquare,
  onKeyNavigate,
}: BoardProps) {
  const files = displayFiles(orientation)
  const ranks = displayRanks(orientation)
  const boardRef = useRef<HTMLDivElement>(null)
  const piecesBySquare = new Map(state.pieces.map((piece) => [piece.square, piece]))
  const moveByDestination = new Map(legalMoves.map((move) => [move.to, move]))
  const lastMoveSquares = new Set(
    state.lastMove ? [state.lastMove.from, state.lastMove.to] : [],
  )
  const bottomRank = ranks[ranks.length - 1]
  const leftFile = files[0]

  useEffect(() => {
    const board = boardRef.current
    if (!board) return
    const active = document.activeElement
    if (active && (active === board || board.contains(active))) {
      const cell = board.querySelector('.square--focused')
      if (cell instanceof HTMLElement) cell.focus()
    }
  }, [focusedSquare])

  return (
    <div className="board-frame">
      <div
        className="board"
        role="grid"
        ref={boardRef}
        aria-label="Chaturanga board"
        aria-busy={Boolean(busyLabel)}
        tabIndex={0}
        onKeyDown={onKeyNavigate}
      >
        {ranks.flatMap((rank) =>
          files.map((file) => {
            const square: SquareData = {
              id: `${file}${rank}`,
              file,
              rank,
            }
            const legalMove = moveByDestination.get(square.id)
            const isSelectable = state.selectableSquares.includes(square.id)

            return (
              <Square
                key={square.id}
                square={square}
                piece={piecesBySquare.get(square.id)}
                tone={squareTone(file, rank)}
                isSelected={selectedSquare === square.id}
                isSelectable={isSelectable}
                isLegalDestination={Boolean(legalMove)}
                isCaptureDestination={legalMove?.isCapture ?? false}
                isLastMove={lastMoveSquares.has(square.id)}
                isFocused={showFocus && focusedSquare === square.id}
                isFileLabel={rank === bottomRank}
                isRankLabel={file === leftFile}
                locked={locked}
                onClick={() => onSquareClick(square.id)}
                onDragStart={onDragStart}
                onDropSquare={onDropSquare}
              />
            )
          }),
        )}
      </div>
      {busyLabel && (
        <div className="board-busy" role="status">
          <span className="spinner" />
          {busyLabel}
        </div>
      )}
    </div>
  )
}
