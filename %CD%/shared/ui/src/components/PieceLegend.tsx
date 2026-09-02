import { PieceGlyph } from '../assets/PieceGlyph'
import { PIECE_LABELS, PIECE_ORDER, PIECE_ROLES } from '../lib/capturedPieces'

export function PieceLegend() {
  return (
    <section className="panel legend" aria-labelledby="legend-heading">
      <h2 id="legend-heading">Piece legend</h2>
      <ul className="legend-list">
        {PIECE_ORDER.map((kind) => (
          <li key={kind}>
            <span className="legend-glyphs">
              <span className="piece piece--white piece--mini">
                <PieceGlyph kind={kind} side="white" />
              </span>
              <span className="piece piece--black piece--mini">
                <PieceGlyph kind={kind} side="black" />
              </span>
            </span>
            <span>
              <strong>{PIECE_LABELS[kind]}</strong>
              <small>{PIECE_ROLES[kind]}</small>
            </span>
          </li>
        ))}
      </ul>
    </section>
  )
}
