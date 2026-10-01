import type { Piece as PieceData, PieceKind } from '../types/game'

const pieceNames: Record<PieceKind, string> = {
  raja: 'Raja', mantri: 'Mantri', ratha: 'Ratha',
  gaja: 'Gaja', ashva: 'Ashva', padati: 'Padati',
}

function PieceFigure({ kind }: { kind: PieceKind }) {
  switch (kind) {
    case 'raja': return <>
      <path d="M21 39 27 68H73L79 39 66 49 59 28 50 45 41 28 34 49Z" />
      <circle cx="21" cy="36" r="4" /><circle cx="41" cy="24" r="4" /><circle cx="59" cy="24" r="4" /><circle cx="79" cy="36" r="4" />
      <path className="piece__detail" d="M28 59H72M50 48v15M44 54h12" />
    </>
    case 'mantri': return <>
      <path d="M50 18 59 33 68 38 64 51 71 68H29L36 51 32 38 41 33Z" />
      <path className="piece__detail" d="M33 41q17 11 34 0M39 57q11 7 22 0M50 22v20" />
      <circle cx="50" cy="47" r="3" className="piece__gem" />
    </>
    case 'ratha': return <>
      <path d="M22 27h12v9h10v-9h12v9h10v-9h12v19H22ZM28 48h44l-4 20H32Z" />
      <path className="piece__detail" d="M30 45h40M40 56h20M48 58v10" />
      <circle cx="30" cy="72" r="7" /><circle cx="70" cy="72" r="7" />
      <circle cx="30" cy="72" r="2" className="piece__gem" /><circle cx="70" cy="72" r="2" className="piece__gem" />
    </>
    case 'gaja': return <>
      <path d="M31 33Q24 29 18 38q-7 14 7 24l14-5M69 33q7-4 13 5 7 14-7 24L61 57" />
      <path d="M34 28q16-12 32 0l6 16-9 25H37L28 44Z" />
      <path d="M44 49h12l-1 22q-1 11-9 9l-6-6 6-6Z" />
      <path className="piece__detail" d="M30 45q-3 9-8 11M70 45q3 9 8 11M40 42h4M58 42h4" />
      <path className="piece__tusk" d="M36 58q-6 7-1 13M64 58q6 7 1 13" />
    </>
    case 'ashva': return <>
      <path d="M34 68q6-21 2-30L29 30l8-4 5-11 7 11 12 5 8 15 10 6-3 10-19-2-3 11Z" />
      <path className="piece__detail" d="M38 33q7 3 12 12l-6 13M62 45l9 5M47 29l6 4" />
      <circle cx="57" cy="40" r="2.6" className="piece__gem" />
      <path className="piece__mane" d="M34 34 28 40l8 5-5 6 8 5-4 8" />
    </>
    case 'padati': return <>
      <path d="M39 30q11-12 22 0l3 13H36Z" />
      <circle cx="50" cy="44" r="12" />
      <path d="M37 56q-13 5-16 16h58q-3-11-16-16l-7 5H44Z" />
      <path className="piece__detail" d="M37 36h26M44 58l6 6 6-6" />
      <circle cx="68" cy="63" r="10" className="piece__shield" />
      <path className="piece__detail" d="M68 57v12M62 63h12" />
    </>
  }
}

export function Piece({ piece }: { piece: PieceData }) {
  return <span className={`piece piece--${piece.side} piece--${piece.kind}`} aria-label={`${piece.side} ${pieceNames[piece.kind]}`}>
    <svg className="piece__drawing" viewBox="0 0 100 100" aria-hidden="true" focusable="false">
      <PieceFigure kind={piece.kind} />
      <path className="piece__neck" d="M28 73h44v6H28Z" />
      <path className="piece__base" d="M22 79h56l5 8H17Z" />
      <path className="piece__detail" d="M24 82h52" />
    </svg>
  </span>
}
