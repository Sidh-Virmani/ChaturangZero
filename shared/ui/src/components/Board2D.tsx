import type { Board3DProps } from './Board3D'
import { PieceIcon } from './PieceIcon'
import type { SquareId } from '../types/game'

export function Board2D({ state, selected, legal, disabled, onSquare, flipped, analysis, policy }: Board3DProps) {
  const aligned = analysis?.positionPly === state.moveHistory.length && analysis.positionTurn === state.currentTurn
  return <div className="flat-board-wrap"><div className="flat-board" role="group" aria-label="Chaturanga board">
    {Array.from({ length: 64 }, (_, index) => {
      const file = flipped ? 7 - index % 8 : index % 8
      const rank = flipped ? Math.floor(index / 8) : 7 - Math.floor(index / 8)
      const square = `${'abcdefgh'[file]}${rank + 1}` as SquareId
      const piece = state.pieces.find(p => p.square === square)
      const move = legal.find(m => m.to === square)
      const last = state.lastMove && [state.lastMove.from, state.lastMove.to].includes(square)
      const intensity = policy && aligned ? analysis.policyHeatmap[rank * 8 + file] : 0
      return <button key={square} disabled={disabled} onClick={() => onSquare(square)}
        className={`flat-square ${(file + rank) % 2 === 0 ? 'dark' : 'light'} ${selected === square ? 'selected' : ''} ${last ? 'last' : ''}`}
        aria-label={`${square}${piece ? ` ${piece.side} ${piece.kind}` : ' empty'}${move ? ' legal destination' : ''}`}
        style={intensity ? { boxShadow: `inset 0 0 0 100px rgba(89,220,174,${Math.min(.65, intensity * 2)})` } : undefined}>
        {index % 8 === 0 && <span className="rank-label">{rank + 1}</span>}
        {index >= 56 && <span className="file-label">{'abcdefgh'[file]}</span>}
        {piece && <PieceIcon kind={piece.kind} side={piece.side} />}
        {move && <span className={move.isCapture ? 'capture-dot' : 'legal-dot'} />}
      </button>
    })}
  </div></div>
}
