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

  return <section className="ai-panel" aria-labelledby="ai-heading" aria-live="polite">
    <div className="ai-panel__heading">
      <span className="ai-panel__symbol" aria-hidden="true">✧</span>
      <div><span className="eyebrow">LIVE SEARCH</span><h2 id="ai-heading">Machine analysis</h2></div>
      <span className={`ai-panel__light ${thinking ? 'ai-panel__light--thinking' : ''}`} aria-hidden="true" />
    </div>
    <div className="ai-console">
      <div className="ai-console__model">MODEL <span>{state.checkpointLabel ?? 'No checkpoint loaded'}</span></div>
      {!state.machineReady ? <p className="ai-console__empty">No checkpoint is loaded. Train a model and set CHATURANGZERO_CHECKPOINT to its latest_model.pt.</p> : thinking ? <>
        <div className="ai-console__progress-title"><span className="spinner" />Machine is thinking</div>
        <ol className="ai-console__events">
          {(progress?.events.length ? progress.events : ['Preparing the machine search...']).map((message, index) => <li key={`${index}-${message}`}>{message}</li>)}
        </ol>
        {progress && progress.totalSimulations > 0 && progress.phase === 'mcts' && <div className="ai-console__simulation">
          <span>MCTS simulations</span><strong>{progress.simulationsCompleted} / {progress.totalSimulations}</strong>
          <div className="ai-console__meter"><i style={{ width: `${100 * progress.simulationsCompleted / progress.totalSimulations}%` }} /></div>
        </div>}
      </> : analysis ? <>
        <div className="ai-console__summary">The board State had <strong>{analysis.legalActionCount} Legal Actions.</strong> The Policy Network suggested moves; MCTS explored them using PUCT over <strong>{analysis.simulations} simulations.</strong></div>
        <div className="ai-console__section-title">Top Candidate Actions <small>share of root visits</small></div>
        <div className="ai-console__candidates">
          {analysis.candidates.slice(0, 3).map((action, index) => {
            const share = analysis.rootVisits ? 100 * (action.visits ?? 0) / analysis.rootVisits : 0
            return <div className="ai-console__candidate" key={action.actionId}>
              <div className="ai-console__candidate-line"><span><em>{index + 1}</em> {action.from} → {action.to}</span><strong>{action.visits ?? 0} visits · {share.toFixed(0)}%</strong></div>
              <div className="ai-console__meter"><i style={{ width: `${share}%` }} /></div>
            </div>
          })}
        </div>
        <div className="ai-console__selected"><span>SELECTED ACTION</span><strong>{analysis.selectedAction.from} → {analysis.selectedAction.to}</strong><small>Chosen by highest root Visit Count</small></div>
        <div className="ai-console__evaluation">
          <div className="ai-console__section-title">Position Evaluation / Value <strong>{whiteValue >= 0 ? '+' : ''}{whiteValue.toFixed(2)}</strong></div>
          <div className="ai-console__value-track" role="img" aria-label={`Position Value ${whiteValue.toFixed(2)} from White's perspective`}><i style={{ left: `${evaluationPosition}%` }} /></div>
          <div className="ai-console__value-labels"><span>BLACK advantage</span><span>0</span><span>WHITE advantage</span></div>
          <p>Network estimate, displayed from White's view.</p>
        </div>
      </> : <p className="ai-console__empty">Make a White move to see the Policy Network, Value Network and MCTS search the reply.</p>}
    </div>
  </section>
}
