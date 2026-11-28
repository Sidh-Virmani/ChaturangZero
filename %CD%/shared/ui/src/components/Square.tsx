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
  isFocused: boolean
  isFileLabel: boolean
  isRankLabel: boolean
  locked: boolean
  onClick: () => void
  onDragStart: (from: SquareData['id']) => void
  onDropSquare: (to: SquareData['id']) => void
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
  isFocused,
  isFileLabel,
  isRankLabel,
  locked,
  onClick,
  onDragStart,
  onDropSquare,
}: SquareProps) {
  const classes = [
    'square',
    `square--${tone}`,
    isSelected && 'square--selected',
    isSelectable && 'square--selectable',
    isLegalDestination && 'square--legal',
    isCaptureDestination && 'square--capture',
    isLastMove && 'square--last-move',
    isFocused && 'square--focused',
  ]
    .filter(Boolean)
    .join(' ')

  const canDrag = Boolean(piece && isSelectable && !locked)

  return (
    <div
      role="gridcell"
      className={classes}
      tabIndex={isFocused ? 0 : -1}
      onClick={onClick}
      onPointerUp={() => {
        if (locked) return
        onDropSquare(square.id)
      }}
      aria-selected={isSelected}
      aria-disabled={locked}
      aria-label={`${square.id}${piece ? `, ${piece.side} ${piece.kind}` : ', empty'}`}
    >
      {isRankLabel && <span className="coordinate coordinate--rank">{square.rank}</span>}
      {isFileLabel && <span className="coordinate coordinate--file">{square.file}</span>}
      {piece && (
        <Piece
          piece={piece}
          canDrag={canDrag}
          onPointerDragStart={() => onDragStart(square.id)}
        />
      )}
      {isLegalDestination && !isCaptureDestination && (
        <span className="move-marker" aria-hidden="true" />
      )}
      {isCaptureDestination && <span className="capture-marker" aria-hidden="true" />}
    </div>
  )
}
