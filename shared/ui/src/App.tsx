import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { Activity, ArrowDownToLine, ArrowRight, ArrowUpRight, BookOpen, Box, BrainCircuit, Check, Crown, Expand, Grid2X2, Hexagon, Layers3, Maximize2, Pause, Play, Plus, RotateCcw, RotateCw, Settings2, Sparkles, Swords, Volume2, VolumeX, X } from 'lucide-react'
import { api } from './services/httpGameAdapter'
import { Board2D } from './components/Board2D'
import { EnginePanel, NeuralInspector } from './components/Insights'
import { TrainingPage, RulesPage, army } from './components/Pages'
import type { Analysis, BoardState, LegalMove, ModelInfo, Side, SquareId, TrainingState } from './types/game'
const Board3D = lazy(() => import('./components/Board3D'))
type Page = 'arena' | 'lab' | 'training' | 'rules'
type Mode = 'ai' | 'local' | 'watch'

function preference<T extends string>(key: string, choices: readonly T[], fallback: T): T {
  const saved = sessionStorage.getItem(`cz-${key}`)
  return choices.includes(saved as T) ? saved as T : fallback
}

export function download(name: string, value: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }))
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = name; document.body.appendChild(anchor); anchor.click(); anchor.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export default function App() {
  const [page, setPage] = useState<Page>('arena')
  const [state, setState] = useState<BoardState | null>(null)
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [training, setTraining] = useState<TrainingState | null>(null)
  const [selected, setSelected] = useState<SquareId | null>(null)
  const [legal, setLegal] = useState<LegalMove[]>([])
  const [busy, setBusy] = useState('loading')
  const [error, setError] = useState('')
  const [view, setView] = useState<'3d' | '2d'>('3d')
  const [policy, setPolicy] = useState(false)
  const [layers, setLayers] = useState(false)
  const [paths, setPaths] = useState(false)
  const [flipped, setFlipped] = useState(() => preference('side', ['white', 'black'], 'white') === 'black')
  const [camera, setCamera] = useState(0)
  const [mode, setMode] = useState<Mode>(() => preference('mode', ['ai', 'local', 'watch'], 'ai'))
  const [side, setSide] = useState<Side>(() => preference('side', ['white', 'black'], 'white'))
  const [watching, setWatching] = useState(false)
  const [simulations, setSimulations] = useState(96)
  const [sound, setSound] = useState(false)
  const [settings, setSettings] = useState(false)
  const [draftMode, setDraftMode] = useState<Mode>('ai')
  const [draftSide, setDraftSide] = useState<Side>('white')
  const [draftSimulations, setDraftSimulations] = useState(96)
  const [draftScenario, setDraftScenario] = useState('opening')
  const [notice, setNotice] = useState('')
  const operation = useRef(false)
  const boardSection = useRef<HTMLElement>(null)
  const settingsDialog = useRef<HTMLDialogElement>(null)
  const audioContext = useRef<AudioContext | null>(null)
  const mounted = useRef(false)

  const clear = () => { setSelected(null); setLegal([]) }
  const openSettings = () => {
    setDraftMode(mode); setDraftSide(side); setDraftSimulations(simulations); setSettings(true)
    setDraftScenario(state?.scenario || 'opening')
  }
  const moveSound = useCallback(() => {
    if (!sound) return
    try {
      const ctx = audioContext.current || new AudioContext()
      audioContext.current = ctx
      void ctx.resume()
      const oscillator = ctx.createOscillator(), gain = ctx.createGain()
      oscillator.type = 'sine'; oscillator.frequency.setValueAtTime(440, ctx.currentTime)
      oscillator.frequency.exponentialRampToValueAtTime(180, ctx.currentTime + .1)
      gain.gain.setValueAtTime(.065, ctx.currentTime); gain.gain.exponentialRampToValueAtTime(.001, ctx.currentTime + .12)
      oscillator.connect(gain); gain.connect(ctx.destination); oscillator.start(); oscillator.stop(ctx.currentTime + .13)
    } catch { /* Optional sound never blocks a move. */ }
  }, [sound])

  const load = useCallback(async () => {
    if (operation.current) return
    operation.current = true; setBusy('loading'); setError('')
    try {
      const [board, info, metrics] = await Promise.all([api.state(), api.model(), api.training()])
      setState(board); setModel(info); setTraining(metrics); setAnalysis(null); clear()
      setBusy('thinking')
      const inspected = await api.analyze(96, board.revision)
      setAnalysis(inspected.analysis)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not load the game.') }
    finally { operation.current = false; setBusy('') }
  }, [])
  useEffect(() => {
    if (mounted.current) return
    mounted.current = true
    void load()
  }, [load])
  useEffect(() => {
    if (!notice) return
    const timer = setTimeout(() => setNotice(''), 5500)
    return () => clearTimeout(timer)
  }, [notice])
  useEffect(() => {
    if (settings) settingsDialog.current?.showModal()
    else settingsDialog.current?.close()
  }, [settings])
  useEffect(() => {
    sessionStorage.setItem('cz-mode', mode); sessionStorage.setItem('cz-side', side)
  }, [mode, side])
  useEffect(() => {
    if (page !== 'training' && !training?.job.running) return
    let cancelled = false
    const timer = setInterval(() => {
      void api.training().then(value => {
        if (cancelled) return
        setTraining(value)
        if (!value.job.running) void api.model().then(setModel).catch(() => {})
      }).catch(e => { if (!cancelled) setError(String(e.message)) })
    }, 3000)
    return () => { cancelled = true; clearInterval(timer) }
  }, [page, training?.job.running])

  const analyze = useCallback(async (play = false) => {
    if (!state || operation.current || (play && state.status.code === 'terminal')) return
    operation.current = true; setBusy('thinking'); setError(''); clear()
    try {
      const response = await api.analyze(simulations, state.revision, play)
      setState(response.state); setAnalysis(response.analysis); setModel(response.analysis.model)
      if (play) moveSound()
    } catch (e) { setError(e instanceof Error ? e.message : 'Analysis failed.'); setWatching(false) }
    finally { operation.current = false; setBusy('') }
  }, [state, simulations, moveSound])
  useEffect(() => {
    if (!state || busy || error || state.status.code === 'terminal' || settings) return
    if ((mode === 'ai' && state.currentTurn !== side) || (mode === 'watch' && watching)) {
      const timer = setTimeout(() => void analyze(true), mode === 'watch' ? 900 : 300)
      return () => clearTimeout(timer)
    }
  }, [state, mode, side, busy, error, watching, analyze, settings])

  const onSquare = async (square: SquareId) => {
    if (!state || operation.current || state.status.code === 'terminal' || mode === 'watch' || (mode === 'ai' && state.currentTurn !== side)) return
    if (square === selected) { clear(); return }
    operation.current = true; setBusy('moving'); setError('')
    try {
      if (selected && legal.some(move => move.to === square)) {
        setState(await api.move(selected, square, state.revision)); clear(); setAnalysis(null); moveSound()
      } else if (state.selectableSquares.includes(square)) {
        const moves = await api.legal(square); setSelected(square); setLegal(moves)
      } else clear()
    } catch (e) { setError(e instanceof Error ? e.message : 'Move failed.') }
    finally { operation.current = false; setBusy('') }
  }
  const reset = async () => {
    if (operation.current) return
    operation.current = true; setBusy('loading'); setError(''); setWatching(false); clear()
    try {
      setState(await api.reset(draftScenario)); setAnalysis(null); setMode(draftMode); setSide(draftSide)
      setSimulations(draftSimulations); setSettings(false); setFlipped(draftSide === 'black')
    }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not reset the board.') }
    finally { operation.current = false; setBusy('') }
  }
  const undo = async () => {
    if (!state || operation.current) return
    operation.current = true; setBusy('moving'); setError(''); setWatching(false); clear()
    try {
      const count = mode === 'ai' && state.currentTurn === side ? 2 : 1
      setState(await api.undo(count, state.revision)); setAnalysis(null)
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not undo the move.') }
    finally { operation.current = false; setBusy('') }
  }
  const toggle = (value: boolean, setter: (v: boolean) => void) => {
    setter(!value)
    if (!value && (!analysis || analysis.positionPly !== state?.moveHistory.length)) void analyze()
    if (!value && window.matchMedia('(max-width: 760px)').matches) {
      requestAnimationFrame(() => boardSection.current?.scrollIntoView({
        block: 'start', behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
      }))
    }
  }
  const startTraining = async () => {
    try { setTraining(await api.train()); setError('') }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not start training.') }
  }
  const navigate = (next: Page) => {
    setPage(next)
    window.scrollTo({ top: 0, behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' })
    if (next === 'lab' && state && (!analysis || analysis.positionPly !== state.moveHistory.length)) void analyze()
  }
  const activePiece = state?.pieces.find(piece => piece.square === selected)
  const boardProps = state ? { state, selected, legal, disabled: !!busy || state.status.code === 'terminal' || mode === 'watch' || (mode === 'ai' && state.currentTurn !== side), analysis, policy, layers: page === 'lab' || layers, paths, flipped, resetCamera: camera, onSquare: (square: SquareId) => void onSquare(square), onFallback: () => { setView('2d'); setNotice('3D is unavailable in this browser. The 2D board is ready to play.') } } : null

  return <div className="app-shell">
    <header className="site-header"><div className="header-inner">
      <a className="brand" href="#arena" onClick={e => { e.preventDefault(); navigate('arena') }}><span className="brand-emblem"><Crown size={25} strokeWidth={1.3} /></span><span>chaturang<span className="brand-zero">zero</span><small>ANCIENT GAME. EVOLVING MIND.</small></span></a>
      <nav aria-label="Main navigation">{([{ id: 'arena', icon: Swords, label: 'Arena' }, { id: 'lab', icon: BrainCircuit, label: 'Neural lab' }, { id: 'training', icon: Activity, label: 'Training' }, { id: 'rules', icon: BookOpen, label: 'The game' }] as const).map(({ id, icon: Icon, label }) => <button key={id} className={page === id ? 'active' : ''} onClick={() => navigate(id)} aria-current={page === id ? 'page' : undefined}><Icon size={16} /><span>{label}</span></button>)}</nav>
      <div className="connection"><span className={`live-dot ${error ? 'offline' : ''}`} />{model ? 'Engine online' : 'Connecting'}<span className="version">v1.0</span></div>
    </div></header>
    <main className="main-content">
      {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => void load()}>Reconnect</button><button aria-label="Dismiss error" onClick={() => setError('')}><X size={16} /></button></div>}
      {notice && <div className="notice" role="status"><Check size={15} />{notice}</div>}
      <div className="page-heading"><div><div className="eyebrow"><span className="eyebrow-line" />{page === 'arena' ? 'THE CHATURANGA EXPERIMENT' : page === 'lab' ? 'OPEN THE BLACK BOX' : page === 'training' ? 'LEARNING THROUGH EXPERIENCE' : 'SIX PIECES. ENDLESS POSSIBILITY.'}</div>
        <h1>{page === 'arena' ? <>Ancient strategy. <em>Artificial intuition.</em></> : page === 'lab' ? <>See the thinking. <em>Layer by layer.</em></> : page === 'training' ? <>Every game is <em>a lesson.</em></> : <>Before chess, <em>there was Chaturanga.</em></>}</h1>
        <p>{page === 'arena' ? 'An age-old game, a self-learning mind. Make your move and explore what happens beneath the board.' : page === 'lab' ? 'Follow a real position from its encoded representation to the search that chooses a move.' : page === 'training' ? 'Self-play creates experience. A residual network learns from it. The next candidate earns its place.' : 'Discover the four divisions of an ancient army, brought together on an eight-by-eight board.'}</p></div>
        {page === 'arena' ? <button className="button secondary" disabled={!!busy} onClick={openSettings}><Plus size={16} />New game</button> : page === 'lab' ? <button className="button secondary" disabled={!!busy} onClick={() => void analyze()}><Sparkles size={16} />{busy === 'thinking' ? 'Thinking…' : 'Refresh analysis'}</button> : page === 'training' ? <button className="button secondary" disabled={!training?.metrics} onClick={() => download('chaturangzero-training-report.json', training?.metrics)}><ArrowDownToLine size={16} />Export report</button> : <button className="button secondary" onClick={() => navigate('arena')}>Enter the arena<ArrowUpRight size={16} /></button>}
      </div>
      {(page === 'arena' || page === 'lab') && <>
        <div className={`arena-layout ${page === 'lab' ? 'lab-layout' : ''}`}>
          <section className="board-stage" ref={boardSection} aria-label="Playable Chaturanga board">
            <div className="board-topbar"><div className="stage-title"><span className="live-dot" /><span>{page === 'lab' ? 'NEURAL ARCHITECTURE' : 'THE ARENA'}</span><span className="stage-slash">/</span><span className="stage-mode">{page === 'lab' ? 'Exploded view' : mode === 'ai' ? 'Human vs. AI' : mode === 'local' ? 'Local two-player' : 'AI self-play'}</span></div><div className="segmented view-switch" aria-label="Board view"><button className={view === '3d' ? 'active' : ''} aria-pressed={view === '3d'} onClick={() => setView('3d')}><Box size={13} />3D</button><button className={view === '2d' ? 'active' : ''} aria-pressed={view === '2d'} onClick={() => setView('2d')}><Grid2X2 size={13} />2D</button></div></div>
            <div className="opponent-strip"><span className="avatar jade"><BrainCircuit size={20} /></span><div><strong>{mode === 'ai' && side === 'white' || mode === 'watch' ? 'ChaturangZero' : 'Midnight army'}</strong><small>{mode === 'ai' && side === 'white' || mode === 'watch' ? 'Neural network + MCTS' : 'Midnight · Black pieces'}</small></div><span className={`turn-tag ${state?.currentTurn === 'black' ? 'is-turn' : ''}`}>{state?.currentTurn === 'black' ? busy === 'thinking' ? 'THINKING' : 'TO MOVE' : 'MIDNIGHT'}</span></div>
            <div className={`board-viewport ${page === 'lab' || layers ? 'exploded' : ''}`}><div className="orbit-ring ring-1" /><div className="orbit-ring ring-2" /><span className="viewport-label">CHATURANGA<br /><small>8 × 8 · SIX ANCIENT PIECES</small></span>
              {boardProps ? view === '3d' ? <Suspense fallback={<div className="board-loading"><span className="spinner" />Sculpting the arena…</div>}><Board3D {...boardProps} /></Suspense> : <Board2D {...boardProps} /> : <div className="board-loading"><span className="spinner" />{error ? 'Waiting for the engine' : 'Preparing your board…'}</div>}

              {state?.status.code === 'terminal' && <div className="game-over-banner" role="status"><Crown size={22} /><div><span>GAME COMPLETE</span><strong>{state.status.result}</strong><small>{state.status.detail}</small></div><button className="button secondary" onClick={openSettings}>Play again</button></div>}
              {busy === 'thinking' && <div className="thinking-badge" role="status"><span className="spinner" />{mode === 'ai' && state?.currentTurn !== side || watching ? 'The engine is thinking' : 'Analyzing the position'}</div>}
            </div>
            <div className="player-strip"><span className="avatar ivory"><Crown size={20} /></span><div><strong>{mode === 'ai' && side === 'white' ? 'You' : mode === 'watch' || mode === 'ai' && side === 'black' ? 'ChaturangZero' : 'Ivory army'}</strong><small>Ivory · White pieces</small></div><span className={`turn-tag ${state?.currentTurn === 'white' ? 'is-turn' : ''}`}>{state?.currentTurn === 'white' ? 'TO MOVE' : 'IVORY'}</span></div>
            <div className="board-toolbar"><div className="toolbar-left"><button title="Undo move" aria-label="Undo move" disabled={!!busy || !state?.moveHistory.length} onClick={() => void undo()}><RotateCcw size={17} /></button><button title="Flip board" aria-label="Flip board" onClick={() => setFlipped(!flipped)}><RotateCw size={17} /></button><button title="Reset camera" aria-label="Reset camera" onClick={() => setCamera(camera + 1)}><Maximize2 size={17} /></button><span className="toolbar-divider" /><button title={sound ? 'Mute move sounds' : 'Enable move sounds'} aria-label={sound ? 'Mute move sounds' : 'Enable move sounds'} aria-pressed={sound} onClick={() => setSound(!sound)}>{sound ? <Volume2 size={17} /> : <VolumeX size={17} />}</button><button title="Game settings" aria-label="Game settings" disabled={!!busy} onClick={openSettings}><Settings2 size={17} /></button></div>
              <div className="toolbar-right">{mode === 'watch' ? <button className="watch-button" disabled={state?.status.code === 'terminal'} onClick={() => setWatching(!watching)}>{watching ? <Pause size={14} /> : <Play size={14} />}{state?.status.code === 'terminal' ? 'Game complete' : watching ? 'Pause self-play' : 'Watch self-play'}</button> : <span>{selected ? `SELECTED ${selected.toUpperCase()}` : state?.status.code === 'check' ? 'CHECK' : 'CLICK A PIECE TO BEGIN'}</span>}<button title="Fullscreen board" aria-label="Fullscreen board" onClick={() => { const action = document.fullscreenElement ? document.exitFullscreen() : boardSection.current?.requestFullscreen(); void action?.catch(() => setNotice('Fullscreen is unavailable in this browser.')) }}><Expand size={16} /></button></div>
            </div>
          </section>
          {page === 'arena' ? <EnginePanel analysis={analysis} model={model} busy={busy} onAnalyze={() => void analyze()} onLab={() => navigate('lab')} policy={policy} layers={layers} paths={paths} togglePolicy={() => toggle(policy, setPolicy)} toggleLayers={() => { setView('3d'); toggle(layers, setLayers) }} togglePaths={() => toggle(paths, setPaths)} currentPly={state?.moveHistory.length ?? 0} /> : <div className="lab-side surface"><div className="section-number">LIVE MODEL</div><BrainCircuit size={36} className="gold-text" /><h2>{model?.name || 'Connecting'}</h2><p>A compact residual network learns a policy over legal moves and a value for the player to move.</p><dl><div><dt>Trainable parameters</dt><dd>{model?.parameters.toLocaleString() || '—'}</dd></div><div><dt>Input planes</dt><dd>14 × 8 × 8</dd></div><div><dt>Residual blocks</dt><dd>2 × 32 channels</dd></div><div><dt>Action space</dt><dd>4,096</dd></div><div><dt>Inference</dt><dd>Local CPU</dd></div></dl><div className="lab-note"><Layers3 size={17} /><p>Explore six stages with the layer selector. Hover cells for measured values from {analysis ? `ply ${analysis.positionPly}` : 'the analyzed position'}. Zoom into a stage to inspect input planes, feature channels, policy priors, or the value prediction.</p></div><button className="button secondary" disabled={!!busy} onClick={() => void analyze()}>Inspect current position<ArrowRight size={15} /></button></div>}
        </div>
        {page === 'arena' && <div className="below-board"><section className="surface move-journal"><div className="journal-heading"><h2>Move journal <span>{state?.moveHistory.length || 0} plies</span></h2><button className="text-button" disabled={!state?.moveHistory.length} onClick={() => download('chaturangzero-game.json', { ruleset: 'CZ v1', state, analysis })}><ArrowDownToLine size={13} />Export game</button></div><div className="journal-moves">{state?.moveHistory.length ? Array.from({ length: Math.ceil(state.moveHistory.length / 2) }, (_, i) => <div className="move-pair" key={i}><span>{i + 1}.</span><strong>{state.moveHistory[i * 2].notation}</strong><strong>{state.moveHistory[i * 2 + 1]?.notation || '…'}</strong></div>) : <div className="journal-empty"><span className="empty-line" />Every great game begins with a single move.<span className="empty-line" /></div>}</div><div className={`game-status ${state?.status.code === 'terminal' ? 'finished' : ''}`} role="status"><span className="live-dot" /><span>{state?.status.code === 'terminal' ? `${state.status.result} · ${state.status.detail}` : activePiece ? `${army.find(p => p.kind === activePiece.kind)?.name} · ${legal.length} legal destination${legal.length === 1 ? '' : 's'}` : state?.status.code === 'check' ? state.status.label : state ? `${state.currentTurn === 'white' ? 'Ivory' : 'Midnight'} to move. Select a piece to reveal its legal moves.` : 'Connecting to the engine…'}</span></div></section><section className="about-card"><span className="eyebrow">LEARN BY LOOKING</span><h2>There’s more<br />beneath the board.</h2><p>Reveal the layers. Follow a search.<br />See how a machine finds its next move.</p><button className="text-button" onClick={() => navigate('lab')}>Explore the neural lab<ArrowUpRight size={15} /></button><Hexagon className="about-watermark" strokeWidth={.6} /></section></div>}
        {page === 'lab' && <NeuralInspector analysis={analysis} onAnalyze={() => void analyze()} busy={!!busy} />}
      </>}
      {page === 'training' && <TrainingPage training={training} model={model} onStart={() => void startTraining()} />}
      {page === 'rules' && <RulesPage />}
      <footer className="site-footer"><span><Crown size={13} />CHATURANGZERO <i />An AlphaZero-inspired research project</span><span>Built to play. Designed to explain.<span className="footer-dot">✦</span></span></footer>
    </main>
    <dialog ref={settingsDialog} className="settings-modal" onCancel={() => setSettings(false)} onClick={e => { if (e.target === e.currentTarget) setSettings(false) }}><div className="modal-heading"><span className="eyebrow">A NEW BEGINNING</span><button className="icon-button" aria-label="Close settings" onClick={() => setSettings(false)}><X size={19} /></button></div><h2>Set your stage.</h2><p>Choose who plays, and how deeply the engine thinks.</p><label className="field-label" htmlFor="game-mode">Game mode</label><select id="game-mode" value={draftMode} onChange={e => setDraftMode(e.target.value as Mode)}><option value="ai">Play against ChaturangZero</option><option value="local">Two players, one board</option><option value="watch">Watch AI self-play</option></select>{draftMode === 'ai' && <><label className="field-label" htmlFor="player-side">Your army</label><select id="player-side" value={draftSide} onChange={e => setDraftSide(e.target.value as Side)}><option value="white">Ivory · Move first</option><option value="black">Midnight · Move second</option></select></>}<label className="field-label" htmlFor="starting-position">Starting position</label><select id="starting-position" value={draftScenario} onChange={e => setDraftScenario(e.target.value)}><option value="opening">Traditional opening · Full army</option><option value="mate">Checkmate challenge · Find mate in one</option><option value="endgame">Endgame study · Eight pieces</option></select><label className="field-label" htmlFor="search-budget">Search budget</label><select id="search-budget" value={draftSimulations} onChange={e => setDraftSimulations(Number(e.target.value))}><option value="32">Quick · 32 simulations</option><option value="96">Balanced · 96 simulations</option><option value="192">Deep · 192 simulations</option></select><div className="modal-note"><BookOpen size={16} /><span>Chaturanga rules: pawns advance one square, elephants leap two diagonally, and there is no castling.</span></div><button className="button primary" disabled={!!busy} onClick={() => void reset()}>Start a new game<ArrowRight size={16} /></button><button className="text-button" onClick={() => setSettings(false)}>Return to current game</button></dialog>
  </div>
}
