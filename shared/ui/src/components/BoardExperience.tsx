import { Component, lazy, Suspense, useEffect, useRef, useState, type ReactNode } from 'react'
import { Board } from './Board'
import { gameAdapter } from '../services/httpGameAdapter'
import type { BoardState, LegalMove, Move, Side, SquareId } from '../types/game'
import type { InspectCell, NeuralInspection } from '../types/neural'

const Board3D = lazy(() => import('./Board3D'))
const PieceGallery = lazy(() => import('./PieceGallery'))
class SceneBoundary extends Component<{ onFailure: () => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  componentDidCatch() { this.props.onFailure() }
  render() { return this.state.failed ? null : this.props.children }
}
interface Props {
  state: BoardState
  orientation: Side
  selectedSquare: SquareId | null
  legalMoves: LegalMove[]
  disabled: boolean
  busy: boolean
  thinking: boolean
  onSquareClick: (square: SquareId) => void
  onMotionChange: (busy: boolean) => void
}

const format = (value: number) => Math.abs(value) > 0 && Math.abs(value) < .0001 ? value.toExponential(3) : value.toFixed(4)

export function BoardExperience(props: Props) {
  const [view, setView] = useState<'3d' | '2d'>('3d')
  const [exploded, setExploded] = useState(false)
  const [inspection, setInspection] = useState<NeuralInspection | null>(null)
  const [focus, setFocus] = useState('overview')
  const [channel, setChannel] = useState(-1)
  const [hover, setHover] = useState<InspectCell | null>(null)
  const [replay, setReplay] = useState(false)
  const [resetCamera, setResetCamera] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [failed, setFailed] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [animating, setAnimating] = useState(false)
  const [visualMove, setVisualMove] = useState<Move | null>(null)
  const [gallery, setGallery] = useState(false)
  const request = useRef(0)
  const positionKey = props.state.positionKey
  useEffect(() => {
    if (view === '2d' || failed) { setAnimating(false); props.onMotionChange(false) }
  }, [view, failed, positionKey, props.onMotionChange])
  const motionChanged = (busy: boolean, move?: Move | null) => { setAnimating(busy); setVisualMove(move ?? null); props.onMotionChange(busy) }
  useEffect(() => {
    const epoch = ++request.current
    const controller = new AbortController()
    if (!exploded || props.thinking || props.busy) {
      setLoading(false)
      return () => controller.abort()
    }
    setLoading(true); setError(null)
    void gameAdapter.getNetworkInspection(controller.signal).then(snapshot => {
      if (request.current !== epoch || controller.signal.aborted) return
      if (positionKey !== undefined && snapshot.positionKey !== positionKey) {
        setError('The position changed. Refresh to inspect the current board.'); return
      }
      setInspection(snapshot); setHover(null)
      setFocus(previous => previous === 'overview' || snapshot.layers.some(layer => layer.id === previous) ? previous : 'overview')
    }).catch(cause => {
      if (request.current === epoch && !controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Unable to inspect this position.')
    }).finally(() => { if (request.current === epoch && !controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [exploded, positionKey, props.thinking, props.busy, refresh])
  const currentInspection = inspection && (positionKey === undefined || inspection.positionKey === positionKey) ? inspection : null
  const layer = currentInspection?.layers.find(item => item.id === focus)
  const selectedChannel = layer ? layer.channels.length === 1 ? 0 : Math.min(channel, layer.channels.length - 1) : -1
  const values = layer ? selectedChannel < 0 ? layer.channels[0].map((_, index) => layer.channels.reduce((total, item) => total + Math.abs(item[index]), 0) / layer.channels.length) : layer.channels[selectedChannel] : []
  const maximum = Math.max(.000001, ...values.map(Math.abs))
  const goToLayer = (id: string) => { setFocus(id); setChannel(id === 'input' ? 0 : -1); setHover(null) }
  const fallback = () => { setFailed(true); setView('2d'); setExploded(false) }
  const toggleNetwork = () => { setView('3d'); setExploded(previous => !previous); setFocus('overview'); setHover(null) }
  const topPolicy = [...(currentInspection?.policy ?? [])].sort((a, b) => b.prior - a.prior).slice(0, 4)
  return <div className={`board-experience ${exploded ? 'board-experience--exploded' : ''}`}>
    <div className="experience-toolbar">
      <div className="view-switch" aria-label="Board view">
        <button type="button" disabled={failed} aria-pressed={view === '3d'} onClick={() => setView('3d')}>3D atelier</button>
        <button type="button" aria-pressed={view === '2d'} onClick={() => { setView('2d'); setExploded(false) }}>2D board</button>
      </div>
      <div className="experience-actions">
        <button type="button" disabled={failed} className="collection-toggle" aria-haspopup="dialog" onClick={() => setGallery(true)}><span aria-hidden="true">♘</span> Piece collection</button>
        <button type="button" className="network-toggle" disabled={failed} aria-pressed={exploded} onClick={toggleNetwork}><span aria-hidden="true">◇</span> {exploded ? 'Return to game' : 'Explore the network'}</button>
        {view === '3d' && <button type="button" className="camera-reset" aria-label="Reset camera" title="Reset camera" onClick={() => setResetCamera(value => value + 1)}>↺</button>}
      </div>
    </div>
    {failed && <p className="experience-notice" role="status">3D graphics are unavailable in this browser. You can play every move on the 2D board.</p>}
    {view === '2d' ? <Board {...props} busyLabel="Syncing with engine…" /> : <div className="three-stage-wrap">
      <SceneBoundary onFailure={fallback}><Suspense fallback={<div className="three-loading" role="status"><span className="spinner" />Preparing the atelier…</div>}>
        <Board3D {...props} paused={gallery} inspection={currentInspection} exploded={exploded} focus={focus} channel={selectedChannel} replay={replay} resetCamera={resetCamera} onHover={setHover} onFocus={goToLayer} onFailure={fallback} onMotionChange={motionChanged} />
      </Suspense></SceneBoundary>
      <div className="stage-caption" aria-hidden="true"><span>{exploded ? 'INSIDE THE NETWORK' : 'THE ROYAL ATELIER'}</span><small>{exploded ? 'One position · real tensors' : 'Ivory porcelain · onyx · champagne gold'}</small></div>
      {!exploded && <p className="stage-help">Drag to orbit <i /> Scroll to zoom <i /> Click a piece to play</p>}
      {!exploded && animating && visualMove?.captured && <div className="capture-caption" role="status"><span className="capture-caption__icon" aria-hidden="true">✦</span><span><strong>{({ ashva: 'Crescent leap', gaja: 'Lotus shockwave', ratha: 'Chariot charge', mantri: 'Astral sigil', raja: 'Royal eclipse', padati: 'Spear strike' })[visualMove.piece]}</strong><small>{visualMove.from} → {visualMove.to} · {visualMove.piece} takes {visualMove.captured}</small></span></div>}
      {exploded && currentInspection && <div className="canvas-layer-controls">
        <label>Focus layer<select aria-label="Focus network layer" value={focus} onChange={event => goToLayer(event.target.value)}><option value="overview">All layers · exploded view</option>{currentInspection.layers.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
        {layer && layer.channels.length > 1 && <label>Channel<select aria-label="3D feature channel" value={selectedChannel} onChange={event => { setChannel(Number(event.target.value)); setHover(null) }}><option value={-1}>Mean absolute activation</option>{layer.channels.map((_, index) => <option key={index} value={index}>{layer.channelNames?.[index]?.replaceAll('_', ' ') ?? `Channel ${index + 1}`}</option>)}</select></label>}
      </div>}
      {exploded && (hover || layer) && <div className="canvas-activation-card">
        <span>{hover ? `${hover.square.toUpperCase()} · ${currentInspection?.layers.find(item => item.id === hover.layerId)?.label}` : layer?.label}</span>
        {hover ? <><strong>{format(hover.value)}</strong><small>{hover.channel.replaceAll('_', ' ')} · original tensor value</small></> : <><strong>{layer?.shape.join(' × ')}</strong><small>{layer?.explanation}</small></>}
      </div>}
      {exploded && <div className="network-canvas-hint">{focus === 'overview' ? 'Choose a layer to move closer' : 'Hover or tap an activation to inspect it'}</div>}
      {(props.busy || loading || props.thinking) && <div className="stage-status" role="status"><span className="spinner" />{props.thinking ? 'Machine search in progress' : loading ? 'Reading the current position' : 'Syncing with the engine'}</div>}
    </div>}
    {exploded && <section className="educational-inspector" aria-label="Educational neural network inspector">
      <div className="inspector-heading"><div><span className="eyebrow">FOLLOW THE FORWARD PASS</span><h2>From position to possibility.</h2></div><button type="button" aria-pressed={replay} onClick={() => setReplay(value => !value)}>{replay ? 'Pause signal' : 'Animate signal'}</button></div>
      <p className="inspector-intro">Explore the exact board encoding and learned features. The light traces illustrate data flow; their speed does not measure inference time.</p>
      {error && <div className="experience-notice" role="alert">{error} <button type="button" onClick={() => setRefresh(value => value + 1)}>Refresh inspector</button></div>}
      {currentInspection ? <>
        <div className="model-provenance"><span className={`model-dot ${currentInspection.checkpoint ? 'model-dot--ready' : ''}`} /><strong>{currentInspection.checkpoint ? currentInspection.smokeCheckpoint ? 'Smoke checkpoint · demonstration only' : 'Loaded checkpoint' : 'Encoding only · no checkpoint loaded'}</strong><small>{currentInspection.checkpoint ?? 'Load a P2/P4 checkpoint to see learned layers.'}</small></div>
        <div className="layer-navigation" aria-label="Network layers"><button type="button" aria-pressed={focus === 'overview'} onClick={() => goToLayer('overview')}>All layers</button>{currentInspection.layers.map(item => <button type="button" key={item.id} aria-pressed={focus === item.id} onClick={() => goToLayer(item.id)}>{item.label}</button>)}</div>
        <div className="inspector-detail">
          <div className="layer-explanation">
            <span className="eyebrow">{layer ? layer.shape.join(' × ') : `${currentInspection.layers.length} INSPECTABLE STAGES`}</span>
            <h3>{layer?.label ?? 'A shared backbone. Two distinct heads.'}</h3>
            <p>{layer?.explanation ?? 'The stem and residual blocks extract spatial patterns. The policy head proposes legal moves; the value head estimates the outcome from the side to move. MCTS uses both to search before selecting an action.'}</p>
            {layer && <code className="layer-operation">{layer.operation}</code>}
            {layer && layer.channels.length > 1 && <label className="channel-picker">Feature channel<select value={selectedChannel} onChange={event => { setChannel(Number(event.target.value)); setHover(null) }}><option value={-1}>Mean absolute activation</option>{layer.channels.map((_, index) => <option key={index} value={index}>{layer.channelNames?.[index]?.replaceAll('_', ' ') ?? `Channel ${index + 1}`}</option>)}</select></label>}
            {layer && <p className="normalization-note">Colour and bar height use the maximum magnitude of this displayed map. Numeric readings retain the original tensor values.</p>}
          </div>
          <div className="tensor-readout">
            {layer ? <>
              <div className="tensor-grid" role="group" aria-label={`${layer.label} activation values`} style={{ gridTemplateColumns: `repeat(${values.length === 1 ? 1 : 8}, 1fr)` }}>{values.map((value, index) => {
                const square = values.length === 1 ? 'scalar' : `${String.fromCharCode(97 + index % 8)}${Math.floor(index / 8) + 1}`
                return <button type="button" key={index} style={{ background: `color-mix(in srgb, #8adcd1 ${12 + Math.abs(value) / maximum * 68}%, #1d2839)` }} aria-label={`${square}: ${format(value)}`} onFocus={() => setHover({ layerId: layer.id, square, value, channel: selectedChannel < 0 ? 'Mean absolute activation' : layer.channelNames?.[selectedChannel] ?? `Channel ${selectedChannel + 1}` })} onPointerEnter={() => setHover({ layerId: layer.id, square, value, channel: selectedChannel < 0 ? 'Mean absolute activation' : layer.channelNames?.[selectedChannel] ?? `Channel ${selectedChannel + 1}` })} onClick={() => setHover({ layerId: layer.id, square, value, channel: selectedChannel < 0 ? 'Mean absolute activation' : layer.channelNames?.[selectedChannel] ?? `Channel ${selectedChannel + 1}` })}>{values.length === 1 ? format(value) : ''}</button>
              })}</div>
              <p className="tensor-axis">{values.length > 1 ? 'a → h · rank 1 at the top · fixed P1 coordinates' : 'Side-to-move value · tanh output'}</p>
            </> : <div className="network-facts"><div><strong>13</strong><small>input planes</small></div><div><strong>4,096</strong><small>action slots</small></div><div><strong>{currentInspection.legalActionCount}</strong><small>legal actions</small></div><div><strong>{currentInspection.parameterCount?.toLocaleString() ?? '—'}</strong><small>model parameters</small></div></div>}
            <div className="hover-readout" role="status"><span>{hover ? `${hover.square.toUpperCase()} · ${currentInspection.layers.find(item => item.id === hover.layerId)?.label}` : 'SELECT A LAYER, THEN AN ACTIVATION'}</span><strong>{hover ? format(hover.value) : 'Inspect a cell'}</strong><small>{hover?.channel.replaceAll('_', ' ') ?? 'Hover, tap, or use the keyboard grid.'}</small></div>
          </div>
        </div>
        {currentInspection.value !== null && <div className="prediction-summary"><div><span className="eyebrow">VALUE · {currentInspection.sideToMove.toUpperCase()} TO MOVE</span><strong>{currentInspection.value.toFixed(3)}</strong><p>A learned outcome estimate in [−1, +1], not a win probability. Its quality depends on training.</p></div><div><span className="eyebrow">HIGHEST LEGAL PRIORS</span>{topPolicy.map(action => <div className="prior-row" key={action.actionId}><span>{action.from} → {action.to}</span><meter min={0} max={1} value={action.prior} aria-label={`${action.from} to ${action.to} prior`} /><strong>{(action.prior * 100).toFixed(1)}%</strong></div>)}{topPolicy.length === 0 && <p>No legal actions in this terminal position.</p>}<p>MCTS search results remain available in the game sidebar.</p></div></div>}
      </> : !loading && !props.thinking && !error && <p className="experience-notice">Open the inspector to read the current position.</p>}
    </section>}
    {gallery && <Suspense fallback={<div className="experience-notice" role="status">Preparing the royal collection…</div>}><PieceGallery onClose={() => setGallery(false)} /></Suspense>}
  </div>
}
