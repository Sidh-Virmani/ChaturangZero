import type { BoardState, MachineAnalysis, MachineProgress } from '../types/game'

interface MachineAnalysisPanelProps {
  state: BoardState
  thinking: boolean
  progress: MachineProgress | null
  result: MachineAnalysis | null
}

export function MachineAnalysisPanel({ state, thinking, progress, result }: MachineAnalysisPanelProps) {
  const analysis = result ?? state.analysis
  const whiteValue = analysis ? (analysis.sideToMove === 'white' ? analysis.value : -analysis.value) : 0
  const evaluationPosition = 50 + 50 * Math.max(-1, Math.min(1, whiteValue))
  const engine = state.engine
  const training = engine?.training
  const phase = thinking ? progress?.phase : analysis ? 'complete' : 'idle'
  const stage = phase === 'complete' || phase === 'selected' ? 3 :
    phase === 'mcts' || phase === 'search_complete' ? 2 :
    phase === 'network' || phase === 'network_ready' ? 1 : 0
  const count = thinking ? progress?.simulationsCompleted ?? 0 : analysis?.simulations ?? 0
  const budget = engine?.simulations ?? progress?.totalSimulations ?? analysis?.simulations ?? 0

  return <section className="ai-panel" aria-labelledby="ai-heading" aria-live="polite">
    <div className="ai-panel__heading">
      <span className="ai-panel__symbol" aria-hidden="true">✧</span>
      <div><span className="eyebrow">LIVE SEARCH</span><h2 id="ai-heading">Machine analysis</h2></div>
      <span className={`ai-panel__light ${thinking ? 'ai-panel__light--thinking' : ''}`} aria-hidden="true" />
    </div>
    <div className="ai-console">
      <div className="ai-console__model">MODEL <span>{state.checkpointLabel ?? 'No checkpoint loaded'}</span></div>
      {engine && <div className="ai-engine">
        <div className="ai-engine__identity"><strong>{training ? `Training iteration ${training.iteration}` : 'Checkpoint loaded'}</strong><code title={engine.sha256}>SHA {engine.sha256.slice(0, 10)}</code></div>
        <p>{engine.simulations} MCTS simulations · PUCT {engine.cPuct} · {engine.inferenceDevice.toUpperCase()} inference</p>
        {training ? <details><summary>Training provenance · {training.device?.toUpperCase() ?? 'device unrecorded'}</summary>
          <dl>
            {training.selfPlaySimulations != null && <><dt>Self-play search</dt><dd>{training.selfPlaySimulations} simulations / move</dd></>}
            {training.workers != null && <><dt>Self-play workers</dt><dd>{training.workers}</dd></>}
            {training.replayCapacity != null && <><dt>Replay capacity</dt><dd>{training.replayCapacity.toLocaleString()} positions</dd></>}
            {training.symmetryProbability != null && <><dt>Symmetry augmentation</dt><dd>{100 * training.symmetryProbability}% of samples</dd></>}
            <dt>Network</dt><dd>{engine.model.channels} channels · {engine.model.residual_blocks} residual blocks</dd>
          </dl><small>{training.source}</small>
        </details> : <small>Training provenance unavailable for this checkpoint.</small>}
      </div>}
      {state.machineReady && <>
        <ol className="ai-pipeline" aria-label="Real AI pipeline">
          {['Board / Legal Actions', 'Policy + Value Networks', 'MCTS / PUCT', 'Selected Action'].map((label, index) =>
            <li key={label} className={phase !== 'idle' && index === stage ? 'is-current' : phase !== 'idle' && index < stage ? 'is-done' : ''}>
              <span>{phase !== 'idle' && index < stage ? '✓' : index + 1}</span>{label}
            </li>)}
        </ol>
        <div className="ai-search-stats" aria-label="Search measurements">
          <span>Legal Actions <strong>{thinking ? progress?.legalActionCount ?? '—' : analysis?.legalActionCount ?? '—'}</strong></span>
          <span>Simulations <strong>{count} / {budget}</strong></span>
          {!thinking && analysis && <><span>Search <strong>{analysis.searchElapsedSeconds.toFixed(3)} s</strong></span><span>Throughput <strong>{analysis.simulationsPerSecond.toFixed(0)} sim/s</strong></span></>}
          {thinking && progress?.searchElapsedSeconds != null && <span>Search elapsed <strong>{progress.searchElapsedSeconds.toFixed(2)} s</strong></span>}
          {thinking && budget > 0 && <div className="ai-console__meter ai-search-stats__meter" role="progressbar" aria-label="Completed MCTS simulations" aria-valuenow={count} aria-valuemin={0} aria-valuemax={budget}><i style={{ width: `${100 * count / budget}%` }} /></div>}
        </div>
      </>}
      {!state.machineReady ? <p className="ai-console__empty">No checkpoint is loaded. Train a model and set CHATURANGZERO_CHECKPOINT to its latest_model.pt.</p> : thinking ? <>
        <div className="ai-console__progress-title"><span className="spinner" />Machine is thinking</div>
        <ol className="ai-console__events">
          {(progress?.events.length ? progress.events : ['Preparing the machine search...']).map((message, index) => <li key={`${index}-${message}`}>{message}</li>)}
        </ol>
      </> : analysis ? <>
        <div className="ai-console__summary">Last search · position before {analysis.sideToMove === 'black' ? "Black’s" : "White’s"} move. <strong>Policy prior</strong> is the network’s first preference; <strong>visits</strong> show where MCTS searched.</div>
        <div className="ai-console__section-title">Top Candidate Actions <small>network → search</small></div>
        <div className="ai-console__candidates">
          {analysis.candidates.slice(0, 3).map((action, index) => {
            const share = analysis.rootVisits ? 100 * (action.visits ?? 0) / analysis.rootVisits : 0
            return <div className="ai-console__candidate" key={action.actionId}>
              <div className="ai-console__candidate-line"><span><em>{index + 1}</em> {action.from} → {action.to}</span><strong>{action.visits ?? 0} visits</strong></div>
              <div className="ai-console__meter"><i style={{ width: `${share}%` }} /></div>
              <div className="ai-console__prior">Policy prior {(action.prior * 100).toFixed(1)}% <span>→</span> MCTS visit share {share.toFixed(1)}%</div>
            </div>
          })}
        </div>
        <div className="ai-console__selected"><span>SELECTED ACTION</span><strong>{analysis.selectedAction.from} → {analysis.selectedAction.to}</strong><small>Chosen by highest root Visit Count</small></div>
        <div className="ai-console__evaluation">
          <div className="ai-console__section-title">Position Evaluation / Value <strong>{whiteValue >= 0 ? '+' : ''}{whiteValue.toFixed(2)}</strong></div>
          <div className="ai-console__value-track" role="img" aria-label={`Position Value ${whiteValue.toFixed(2)} from White's perspective`}><i style={{ left: `${evaluationPosition}%` }} /></div>
          <div className="ai-console__value-labels"><span>BLACK advantage</span><span>0</span><span>WHITE advantage</span></div>
          <p>Pre-move network estimate from White's view; not a win probability.</p>
        </div>
      </> : <p className="ai-console__empty">Make a White move to see the Policy Network, Value Network and MCTS search the reply.</p>}
      {state.repetition && <div className="ai-repetition">Repetition draw active <strong>{state.repetition.occurrences} / {state.repetition.drawAt}</strong><small>Current board + turn occurrence · third occurrence draws</small></div>}
    </div>
  </section>
}
