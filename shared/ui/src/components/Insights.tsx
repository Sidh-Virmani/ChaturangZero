import { Activity, ArrowUpRight, BrainCircuit, ChevronRight, GitBranch, Layers3, Play, ScanLine, Sparkles } from 'lucide-react'
import { useState } from 'react'
import type { Analysis, ModelInfo, TrainingState } from '../types/game'

export function Toggle({ checked, onChange, label, detail, disabled = false }: { checked: boolean; onChange: () => void; label: string; detail?: string; disabled?: boolean }) {
  return <button className="toggle-row" role="switch" aria-checked={checked} aria-label={label} onClick={onChange} disabled={disabled}>
    <span><strong>{label}</strong>{detail && <small>{detail}</small>}</span><span className={`switch ${checked ? 'on' : ''}`}><i /></span>
  </button>
}

export function Heatmap({ values, label, input = false }: { values: number[]; label: string; input?: boolean }) {
  return <div className="heatmap" role="img" aria-label={label}>
    {Array.from({ length: 64 }, (_, i) => {
      const index = (7 - Math.floor(i / 8)) * 8 + i % 8
      const value = values[index] || 0
      return <span key={i} title={`${'abcdefgh'[i % 8]}${8 - Math.floor(i / 8)}: ${value.toFixed(3)}`}
        style={{ background: input ? `rgba(216,184,116,${.06 + value * .88})` : `hsl(${190 - value * 5} ${25 + value * 21}% ${10 + value * 60}%)` }} />
    })}
  </div>
}

export function SearchTree({ analysis }: { analysis: Analysis }) {
  const children = analysis.tree.children?.slice(0, 5) || []
  const gap = 680 / Math.max(children.length, 1)
  return <svg className="search-tree" viewBox="0 0 720 280" role="img" aria-label={`Search tree with ${analysis.simulations} simulations. Branches show actual visit counts.`}>
    <defs><linearGradient id="tree-line" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#d0b16f" /><stop offset="1" stopColor="#637eaa" /></linearGradient></defs>
    {children.map((child, i) => {
      const x = 20 + gap * (i + .5)
      return <g key={child.move}>
        <path d={`M360 53 C360 92 ${x} 77 ${x} 114`} stroke="url(#tree-line)" strokeWidth={i === 0 ? 2 : 1} fill="none" opacity={i === 0 ? 1 : .45} />
        {(child.children || []).slice(0, 2).map((leaf, j) => {
          const lx = x + (j - ((child.children?.length || 1) - 1) / 2) * 58
          return <g key={leaf.move}><path d={`M${x} 154 C${x} 185 ${lx} 176 ${lx} 206`} stroke="#394c6c" fill="none" /><circle cx={lx} cy="216" r="10" fill="#263b55" stroke="#78a9bb" /><text x={lx} y="245" textAnchor="middle" className="tree-tiny">{leaf.move}</text><text x={lx} y="262" textAnchor="middle" className="tree-tiny dim">{leaf.visits} visits</text></g>
        })}
        <rect x={x - 49} y="114" width="98" height="44" rx="7" fill={i === 0 ? '#303344' : '#1c2535'} stroke={i === 0 ? '#ad965f' : '#37495e'} />
        <text x={x} y="131" textAnchor="middle" className="tree-label">{child.move}</text>
        <text x={x} y="148" textAnchor="middle" className="tree-tiny">{child.visits} visits · Q {child.value.toFixed(2)}</text>
      </g>
    })}
    <rect x="297" y="12" width="126" height="42" rx="7" fill="#d2b575" />
    <text x="360" y="29" textAnchor="middle" className="tree-root">Current root</text><text x="360" y="44" textAnchor="middle" className="tree-root small">{analysis.simulations} simulations</text>
  </svg>
}

export function EnginePanel({ analysis, model, busy, onAnalyze, onLab, policy, layers, paths, togglePolicy, toggleLayers, togglePaths, currentPly }: {
  analysis: Analysis | null; model: ModelInfo | null; busy: string; onAnalyze: () => void; onLab: () => void
  policy: boolean; layers: boolean; paths: boolean; togglePolicy: () => void; toggleLayers: () => void; togglePaths: () => void; currentPly: number
}) {
  const [tab, setTab] = useState<'policy' | 'search'>('policy')
  return <aside className="engine-panel">
    <div className="panel-heading"><span className="small-icon"><BrainCircuit size={19} /></span><div><h2>Inside the mind</h2><p>Neural-guided tree search</p></div><span className={`live-dot ${busy ? 'pulsing' : ''}`} /></div>
    <div className="model-tag"><span>{model?.name || 'Connecting'}</span><span>{model?.trained ? 'TRAINED PILOT' : 'UNTRAINED'}</span></div>
    <div className="engine-summary"><div><span className="eyebrow">SEARCH VALUE</span><strong>{analysis ? `${analysis.searchValue >= 0 ? '+' : ''}${analysis.searchValue.toFixed(2)}` : '—'}</strong></div><div className="value-caption">{analysis ? `For ${analysis.positionTurn === 'white' ? 'Ivory' : 'Midnight'} at ply ${analysis.positionPly}` : 'Analyze a position to begin'}<span>−1 loss · 0 draw · +1 win</span></div></div>
    <div className="value-track"><i style={{ width: `${analysis ? (analysis.searchValue + 1) * 50 : 50}%` }} /></div>
    <div className="engine-stats"><div><strong>{analysis?.simulations ?? '—'}</strong><span>simulations</span></div><div><strong>{analysis?.depth ?? '—'}</strong><span>search depth</span></div><div><strong>{analysis ? `${(analysis.elapsedMs / 1000).toFixed(2)}s` : '—'}</strong><span>search time</span></div></div>
    <div className="insight-tabs" role="tablist" aria-label="Engine data"><button role="tab" aria-selected={tab === 'policy'} onClick={() => setTab('policy')}><Activity size={14} /> Move policy</button><button role="tab" aria-selected={tab === 'search'} onClick={() => setTab('search')}><GitBranch size={14} /> Search</button></div>
    <div className="policy-content">
      {busy === 'thinking' && <div className="thinking-line"><span className="spinner" /> Exploring possible futures…</div>}
      {analysis ? tab === 'policy' ? <><div className="candidate-heading"><span>CANDIDATE</span><span>VISIT SHARE</span></div>{analysis.candidates.slice(0, 4).map((candidate, i) => <div className="candidate" key={candidate.notation}><span className="candidate-rank">0{i + 1}</span><div><div className="candidate-text"><strong>{candidate.notation}</strong><span>{(candidate.probability * 100).toFixed(1)}%</span></div><div className="candidate-track"><i style={{ width: `${candidate.probability * 100}%` }} /></div></div></div>)}</> : <><SearchTree analysis={analysis} /><p className="micro-copy">PUCT balances a move’s value, prior probability, and exploration. Branches are actual visited nodes.</p></> : <div className="empty-insight"><ScanLine size={28} /><p>The next move starts<br />with a little exploration.</p></div>}
      <p className="data-caption">{analysis ? `${analysis.positionPly === currentPly ? 'Current position' : 'Last decision'} · ${analysis.candidates.length} legal actions` : 'Real model outputs appear here'}</p>
      {analysis && <details className="reading-guide"><summary>Read this decision</summary><p>{analysis.candidates[0] ? `${analysis.candidates[0].notation} received ${analysis.candidates[0].visits} of ${analysis.simulations} search visits. ` : 'This position has no legal action. '}A larger visit share means search explored a move more often, not that it has that chance of winning.</p><p>Search value is an estimate for {analysis.positionTurn === 'white' ? 'Ivory' : 'Midnight'}: −1 loss, 0 draw, +1 win. The neural lab shows the raw value before search. Neither is a calibrated probability.</p><p>In the tree, Q is the average backed-up value from the parent player’s perspective. Priors are the network’s initial suggestions; visits show how search refined them.</p></details>}

    </div>
    <div className="layer-controls"><div className="section-label"><Layers3 size={14} /> MAKE THINKING VISIBLE</div><Toggle checked={policy} onChange={togglePolicy} label="Policy heatmap" detail="Where the search wants to move" /><Toggle checked={paths} onChange={togglePaths} label="Search paths" detail="The three most explored moves" /><Toggle checked={layers} onChange={toggleLayers} label="Reveal neural layers" detail="Explode the board into activations" /></div>
    <button className="button primary analyze-button" onClick={onAnalyze} disabled={!!busy}><Sparkles size={15} /> {busy === 'thinking' ? 'Thinking…' : 'Analyze position'}<ChevronRight size={15} /></button>
    <button className="text-button lab-link" onClick={onLab}>Open the neural lab <ArrowUpRight size={14} /></button>
  </aside>
}

export function NeuralInspector({ analysis, onAnalyze, busy }: { analysis: Analysis | null; onAnalyze: () => void; busy: boolean }) {
  const [plane, setPlane] = useState(0)
  const names = ['Own raja', 'Own mantri', 'Own ratha', 'Own ashva', 'Own gaja', 'Own padati', 'Enemy raja', 'Enemy mantri', 'Enemy ratha', 'Enemy ashva', 'Enemy gaja', 'Enemy padati', 'No-progress clock', 'Repetition count']
  if (!analysis) return <div className="empty-lab"><BrainCircuit size={42} /><h2>Meet the network.</h2><p>Analyze a position to inspect its real tensors and search tree.</p><button className="button primary" disabled={busy} onClick={onAnalyze}><Play size={16} />Run inference</button></div>
  return <div className="lab-inspector">
    <div className="pipeline-strip"><span>14 × 8 × 8 input</span><ChevronRight /><span>32 feature channels</span><ChevronRight /><span>2 residual blocks</span><ChevronRight /><span>Policy + value</span></div>
    <div className="tensor-layout"><section className="surface tensor-input"><div className="section-number">01 / REPRESENTATION</div><h2>A position, translated.</h2><p>The current player’s army is encoded first. Black’s ranks are flipped so both sides learn the same forward direction.</p><label className="field-label" htmlFor="plane">Input plane</label><select id="plane" value={plane} onChange={e => setPlane(Number(e.target.value))}>{names.map((name, i) => <option key={name} value={i}>{i + 1}. {name}</option>)}</select><Heatmap values={analysis.input[plane].flat()} label={`${names[plane]} encoded input plane, canonical perspective`} input /><span className="data-caption">Canonical coordinates · Values from 0 to 1</span></section>
      <section className="surface tensor-features"><div className="section-number">02 / FEATURE EXTRACTION</div><h2>Patterns beneath the pieces.</h2><p>Each map shows mean absolute activation across 32 channels, normalized by that layer’s maximum. These are computed features, not an explanation of the model’s intent.</p><div className="feature-maps">{analysis.layers.map((layer, i) => <div key={layer.name}><Heatmap values={layer.heatmap} label={`${layer.name} mean activation`} /><strong>{i === 0 ? 'Convolution' : `Residual ${i}`}</strong><span>32 × 8 × 8 · max {layer.maxActivation.toFixed(2)}</span></div>)}</div><div className="heat-scale"><span>Low activation</span><i /><span>High</span></div></section></div>
    <section className="surface tree-section"><div className="section-number">03 / MONTE CARLO TREE SEARCH</div><div className="section-title-row"><div><h2>One move. Many possible futures.</h2><p>Top 5 root branches and their 2 most visited replies. Q is from the parent player’s perspective.</p></div><span className="outline-pill">{analysis.simulations} simulations</span></div><SearchTree analysis={analysis} /><div className="principal-variation"><span>PRINCIPAL VARIATION</span>{analysis.principalVariation.map((move, i) => <span key={i}>{move}{i < analysis.principalVariation.length - 1 && <ChevronRight size={12} />}</span>)}</div></section>
  </div>
}

export function LossChart({ training }: { training: TrainingState | null }) {
  const epochs = training?.metrics?.epochs || []
  if (!epochs.length) return <div className="chart-empty">Run a training cycle to see measured loss curves.</div>
  const max = Math.max(...epochs.map(e => e.loss), 1) * 1.12
  const x = (i: number) => 52 + i / Math.max(epochs.length - 1, 1) * 670
  const y = (v: number) => 228 - v / max * 192
  return <svg viewBox="0 0 760 270" className="loss-chart" role="img" aria-label={`Measured training loss over ${epochs.length} epochs, from ${epochs[0].loss.toFixed(3)} to ${epochs.at(-1)!.loss.toFixed(3)}`}>
    {[0, 1, 2, 3, 4].map(i => <g key={i}><line x1="52" y1={y(max * i / 4)} x2="722" y2={y(max * i / 4)} stroke="#ffffff0b" /><text x="38" y={y(max * i / 4) + 4} textAnchor="end" className="chart-label">{(max * i / 4).toFixed(1)}</text></g>)}
    {(['loss', 'policyLoss', 'valueLoss'] as const).map((key, k) => <path key={key} d={epochs.map((e, i) => `${i ? 'L' : 'M'}${x(i)} ${y(e[key])}`).join(' ')} fill="none" stroke={['#d5b578', '#72bca6', '#849bc9'][k]} strokeWidth={k === 0 ? 2.5 : 1.5} strokeLinejoin="round" />)}
    {epochs.filter((_, i) => i === 0 || i === epochs.length - 1 || i % Math.max(1, Math.floor(epochs.length / 6)) === 0).map(e => <text key={e.step} x={x(epochs.indexOf(e))} y="253" textAnchor="middle" className="chart-label">{e.step}</text>)}
    <text x="718" y="270" textAnchor="end" className="chart-label">Epoch</text>
  </svg>
}
