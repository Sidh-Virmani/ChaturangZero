import { Piece } from './Piece'
import type { Piece as PieceData, Square as SquareData } from '../types/game'

interface SquareProps {
  square: SquareData
  piece?: PieceData
  tone: 'light' | 'dark'
  isSelected: boolean
  isSelectable: boolean
  isLegalDestination: boolean
  isCaptureDestination: boolean
  isLastMove: boolean
  isLastMoveDestination: boolean
  showRank: boolean
  showFile: boolean
  disabled: boolean
  onClick: () => void
}

export function Square({
  square,
  piece,
  tone,
  isSelected,
  isSelectable,
  isLegalDestination,
  isCaptureDestination,
  isLastMove,
  isLastMoveDestination,
  showRank,
  showFile,
  disabled,
  onClick,
}: SquareProps) {
  const classes = [
    'square',
    `square--${tone}`,
    isSelected && 'square--selected',
    isSelectable && 'square--selectable',
    isLegalDestination && 'square--legal',
    isCaptureDestination && 'square--capture',
    isLastMove && 'square--last-move',
    isLastMoveDestination && 'square--move-arrival',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button
      type="button"
      className={classes}
      onClick={onClick}
      disabled={disabled}
      aria-label={`${square.id}${piece ? `, ${piece.side} ${piece.kind}` : ''}`}
      aria-pressed={isSelected}
    >
      {showRank && <span className="coordinate coordinate--rank">{square.rank}</span>}
      {showFile && <span className="coordinate coordinate--file">{square.file}</span>}
      {piece && <Piece piece={piece} />}
      {isLegalDestination && !isCaptureDestination && (
        <span className="move-marker" aria-hidden="true" />
      )}
      {isCaptureDestination && <span className="capture-marker" aria-hidden="true" />}
    </button>
  )
}
