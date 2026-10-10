import { useEffect, useRef, useState } from 'react'
import Board3D from './Board3D'
import type { SceneSettings } from '../three/scene'
import type { BoardState, Move, Piece, PieceKind, SquareId } from '../types/game'
import '../gallery.css'

interface Props { onClose: () => void }
interface Character {
  kind: PieceKind
  name: string
  english: string
  symbol: string
  signature: string
  rule: string
  description: string
  from: SquareId
  to: SquareId
}

const characters: Character[] = [
  { kind: 'raja', name: 'Raja', english: 'King', symbol: '♔', signature: 'Royal eclipse',
    rule: 'One square in any direction. The Raja cannot move into check. There is no castling.',
    description: 'An open lotus crown and a faceted jewel. A deliberate approach blooms into a royal halo.', from: 'd4', to: 'e5' },
  { kind: 'mantri', name: 'Mantri', english: 'Advisor', symbol: '♕', signature: 'Astral sigil',
    rule: 'Exactly one square diagonally. The Mantri has a much shorter range than a modern chess queen.',
    description: 'A jewel-set turban with fine gold ribs. A floating turn traces a luminous diagonal sigil.', from: 'd4', to: 'e5' },
  { kind: 'gaja', name: 'Gaja', english: 'Elephant', symbol: '♗', signature: 'Lotus shockwave',
    rule: 'Exactly two squares diagonally. The Gaja jumps over the intervening square.',
    description: 'Ivory tusks, engraved ears and a ceremonial harness. A grounded landing releases a lotus shockwave.', from: 'c3', to: 'e5' },
  { kind: 'ashva', name: 'Ashva', english: 'Horse', symbol: '♘', signature: 'Crescent leap',
    rule: 'An L-shaped jump: two squares in one direction and one perpendicular. Intervening pieces do not block it.',
    description: 'A carved mane and gemstone bridle. An arcing leap leaves a crescent of light at contact.', from: 'c4', to: 'e5' },
  { kind: 'ratha', name: 'Ratha', english: 'Chariot', symbol: '♖', signature: 'Chariot charge',
    rule: 'Any distance along a rank or file, with a clear path. It cannot jump over another piece.',
    description: 'Exposed gold spokes and inlaid battlements. A fast linear charge meets a bright impact crest.', from: 'e2', to: 'e5' },
  { kind: 'padati', name: 'Padati', english: 'Infantry', symbol: '♙', signature: 'Spear strike',
    rule: 'One square forward; captures one square diagonally forward. No double first move or en passant. Promotes to Mantri on the far rank.',
    description: 'A domed shield, guarded helmet and leaf-shaped spear. A short lunge ends in a precise flash.', from: 'd4', to: 'e5' },
]

// An isolated rendering fixture. This component never reads or writes the game
// adapter, and these snapshots are not claims about an AI or a live position.
function fixture(character: Character, captured: boolean, sequence: number): BoardState {
  const pieces: Piece[] = [
    { id: `gallery-${character.kind}`, kind: character.kind, side: 'white', square: captured ? character.to : character.from },
    { id: 'gallery-black-raja', kind: 'raja', side: 'black', square: 'h8' },
  ]
  if (character.kind !== 'raja') pieces.push({ id: 'gallery-white-raja', kind: 'raja', side: 'white', square: 'a1' })
  if (!captured) pieces.push({ id: 'gallery-victim', kind: 'padati', side: 'black', square: character.to })
  const move: Move | null = captured ? { id: `gallery-${character.kind}-${sequence}`, ply: 1,
    side: 'white', piece: character.kind, from: character.from, to: character.to,
    captured: 'padati', notation: `${character.from}×${character.to}` } : null
  return { positionKey: `gallery-${character.kind}-${sequence}-${captured ? 'after' : 'before'}`, pieces,
    currentTurn: captured ? 'black' : 'white', selectableSquares: [], lastMove: move,
    moveHistory: move ? [move] : [], status: { code: 'playing', label: 'Animation study', detail: 'Illustrative position' },
    isAiThinking: false, mode: 'human', machineReady: false, checkpointLabel: null, smokeCheckpoint: false, analysis: null }
}

const noop = () => {}

export default function PieceGallery({ onClose }: Props) {
  const [kind, setKind] = useState<PieceKind>('ashva')
  const character = characters.find(item => item.kind === kind)!
  const [state, setState] = useState(() => fixture(characters[3], false, 0))
  const [animating, setAnimating] = useState(false)
  const [failed, setFailed] = useState(false)
  const [resetCamera, setResetCamera] = useState(0)
  const [played, setPlayed] = useState(false)
  const dialog = useRef<HTMLDivElement>(null)
  const closeButton = useRef<HTMLButtonElement>(null)
  const close = useRef(onClose); close.current = onClose
  const frame = useRef<number | null>(null)
  const pending = useRef(false)
  const sequence = useRef(0)

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    closeButton.current?.focus()
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close.current(); return }
      if (event.key !== 'Tab' || !dialog.current) return
      const items = Array.from(dialog.current.querySelectorAll<HTMLElement>('button:not([disabled]), [href], select:not([disabled]), [tabindex="0"]'))
      const first = items[0], last = items.at(-1)
      if (!first || !last) { event.preventDefault(); dialog.current.focus(); return }
      if (event.shiftKey && (document.activeElement === first || !dialog.current.contains(document.activeElement))) {
        event.preventDefault(); last.focus()
      } else if (!event.shiftKey && (document.activeElement === last || !dialog.current.contains(document.activeElement))) {
        event.preventDefault(); first.focus()
      }
    }
    document.addEventListener('keydown', handleKey, true)
    return () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current)
      pending.current = false
      document.removeEventListener('keydown', handleKey, true)
      document.body.style.overflow = previousOverflow
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [])

  const choose = (next: Character) => {
    if (frame.current !== null) cancelAnimationFrame(frame.current)
    pending.current = false; frame.current = null
    setKind(next.kind); setState(fixture(next, false, ++sequence.current)); setPlayed(false)
    setResetCamera(value => value + 1)
  }
  const playCapture = () => {
    if (failed || animating || pending.current) return
    const run = ++sequence.current
    pending.current = true; setAnimating(true)
    setState(fixture(character, false, run))
    // Two paint boundaries let Board3D receive the reset before the contiguous
    // one-move snapshot, including when replaying the same character.
    frame.current = requestAnimationFrame(() => {
      frame.current = requestAnimationFrame(() => {
        frame.current = null; pending.current = false
        setState(fixture(character, true, run)); setPlayed(true)
      })
    })
  }
  const motionChanged = (busy: boolean) => { if (!pending.current || busy) setAnimating(busy) }
  const failure = () => {
    if (frame.current !== null) cancelAnimationFrame(frame.current)
    frame.current = null; pending.current = false
    setFailed(true); setAnimating(false)
  }
  const settings: SceneSettings = { state, orientation: 'white', selectedSquare: state.lastMove ? character.to : character.from,
    legalMoves: state.lastMove ? [] : [{ from: character.from, to: character.to, isCapture: true }],
    disabled: true, inspection: null, exploded: false, focus: 'overview', channel: -1, replay: false, resetCamera, showcase: true }

  return <div className="piece-gallery-backdrop" onPointerDown={event => { if (event.target === event.currentTarget) onClose() }}>
    <div className="piece-gallery" ref={dialog} role="dialog" aria-modal="true" aria-labelledby="piece-gallery-title" aria-describedby="piece-gallery-description" tabIndex={-1}>
      <header className="piece-gallery__header">
        <div><span className="eyebrow">THE ROYAL COLLECTION</span><h2 id="piece-gallery-title">Six characters. Six signatures.</h2><p id="piece-gallery-description">Explore the sculptures and replay their capture choreography.</p></div>
        <button ref={closeButton} type="button" className="piece-gallery__close" onClick={onClose} aria-label="Close piece collection">×</button>
      </header>
      <div className="piece-gallery__body">
        <div className="piece-gallery__stage">
          {failed ? <div className="piece-gallery__unavailable" role="status"><span aria-hidden="true">◇</span><strong>3D preview is unavailable.</strong><p>You can still explore every character and its movement rule.</p></div> : <Board3D {...settings} onSquareClick={noop} onHover={noop} onFocus={noop} onFailure={failure} onMotionChange={motionChanged} />}
          {!failed && <div className="piece-gallery__stage-label"><span>{character.signature}</span><small>{character.from} → {character.to} · {character.name} captures Padati</small></div>}
          {!failed && <button type="button" className="piece-gallery__camera" title="Reset preview camera" aria-label="Reset preview camera" onClick={() => setResetCamera(value => value + 1)}>↺</button>}
        </div>
        <aside className="piece-gallery__sidebar" aria-label="Characters and movement rules">
          <div className="piece-gallery__choices" role="group" aria-label="Choose a character">{characters.map(item => <button type="button" key={item.kind} disabled={animating} aria-pressed={kind === item.kind} onClick={() => choose(item)}><span className="piece-gallery__symbol" aria-hidden="true">{item.symbol}</span><span><strong>{item.name}</strong><small>{item.english}</small></span></button>)}</div>
          <div className="piece-gallery__character"><span className="eyebrow">{character.english.toUpperCase()}</span><h3>{character.name}</h3><p>{character.description}</p></div>
          <div className="piece-gallery__rule"><span>HOW IT MOVES</span><p>{character.rule}</p></div>
        </aside>
      </div>
      <footer className="piece-gallery__footer">
        <div><strong>Animation study · illustrative position</strong><span>Drag to orbit · Scroll or pinch to zoom</span></div>
        <button type="button" className="button button--primary piece-gallery__play" disabled={animating || failed} onClick={playCapture}><span aria-hidden="true">{animating ? '◌' : '▷'}</span> {animating ? 'Animating…' : played ? 'Replay capture' : 'Play capture'}</button>
        <span className="piece-gallery__live" role="status" aria-live="polite">{animating ? `${character.signature} animation in progress` : played ? `${character.name} capture complete` : ''}</span>
      </footer>
    </div>
  </div>
}
