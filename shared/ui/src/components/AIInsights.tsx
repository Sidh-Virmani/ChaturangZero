/**
 * AIInsights.tsx — ChaturangZero-X
 * ==================================
 * Shows the AI's move analysis: top MCTS candidates, visit %, Q-values,
 * PLUS the new explainability panel (Research Contribution #9):
 *   - Why did the AI choose this move?
 *   - Confidence and uncertainty label
 *   - Feature-based reasons (plain English)
 *   - Alternative moves considered
 */

interface TopMove {
  from: string
  to: string
  visitFraction: number
  qValue: number
}

interface Alternative {
  move: string
  visitPct: number
  qValue: number
}

interface Explanation {
  move: string
  confidence: number         // 0–100 %
  uncertainty: number        // raw std value
  uncertaintyLabel: string   // "low" | "medium" | "high"
  reasons: string[]
  alternatives: Alternative[]
}

interface AiInsightsProps {
  topMoves: TopMove[]
  numSims: number
  chosenMove?: { from: string; to: string }
  explanation?: Explanation | null
}

function UncertaintyBadge({ label }: { label: string }) {
  const colors: Record<string, string> = {
    low:    '#22c55e',
    medium: '#f59e0b',
    high:   '#ef4444',
  }
  return (
    <span style={{
      background: colors[label] ?? '#94a3b8',
      color: '#fff',
      fontSize: '0.65rem',
      padding: '1px 6px',
      borderRadius: '9999px',
      marginLeft: '6px',
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.04em',
    }}>
      {label} uncertainty
    </span>
  )
}

export function AIInsights({ topMoves, numSims, chosenMove, explanation }: AiInsightsProps) {
  if (topMoves.length === 0 && !explanation) return null

  return (
    <div className="ai-insights">
      <h3 className="ai-insights__title">
        <span className="ai-insights__brain" aria-hidden="true">🧠</span>
        AI Analysis
        <small>{numSims} simulations</small>
      </h3>

      {/* ── Explainability Panel ── */}
      {explanation && (
        <div className="ai-insights__explanation">
          <div className="ai-insights__exp-header">
            <span className="ai-insights__exp-move">▶ {explanation.move}</span>
            <span className="ai-insights__exp-conf">
              {explanation.confidence.toFixed(0)}% confident
            </span>
            <UncertaintyBadge label={explanation.uncertaintyLabel} />
          </div>

          {explanation.reasons.length > 0 && (
            <ul className="ai-insights__reasons">
              {explanation.reasons.map((r, i) => (
                <li key={i} className="ai-insights__reason">
                  <span className="ai-insights__reason-icon">
                    {r.startsWith('improves') ? '✓' : r.startsWith('weakens') ? '⚠' : '→'}
                  </span>
                  {r}
                </li>
              ))}
            </ul>
          )}

          {explanation.alternatives.length > 0 && (
            <div className="ai-insights__alternatives">
              <span className="ai-insights__alt-label">Alternatives considered:</span>
              {explanation.alternatives.map((alt, i) => (
                <span key={i} className="ai-insights__alt-item">
                  {alt.move}
                  <small>({alt.visitPct.toFixed(0)}% · Q{alt.qValue >= 0 ? '+' : ''}{alt.qValue.toFixed(2)})</small>
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── MCTS Top Moves Bar Chart ── */}
      {topMoves.length > 0 && (
        <>
          <div className="ai-insights__section-label">MCTS search tree</div>
          <ol className="ai-insights__list">
            {topMoves.map((m, i) => {
              const isChosen = chosenMove && m.from === chosenMove.from && m.to === chosenMove.to
              const barWidth = `${Math.round(m.visitFraction * 100)}%`
              const qColor   = m.qValue > 0.1 ? '#22c55e' : m.qValue < -0.1 ? '#ef4444' : '#94a3b8'
              return (
                <li key={i} className={`ai-insights__move${isChosen ? ' ai-insights__move--chosen' : ''}`}>
                  <span className="ai-insights__rank">#{i + 1}</span>
                  <span className="ai-insights__notation">
                    {m.from}→{m.to}
                    {isChosen && <span className="ai-insights__chosen-badge">Played</span>}
                  </span>
                  <div className="ai-insights__bar-wrap">
                    <div
                      className="ai-insights__bar"
                      style={{ width: barWidth, background: isChosen ? '#6366f1' : '#475569' }}
                    />
                    <span className="ai-insights__visits">{Math.round(m.visitFraction * 100)}%</span>
                  </div>
                  <span className="ai-insights__q" style={{ color: qColor }}>
                    {m.qValue >= 0 ? '+' : ''}{m.qValue.toFixed(2)}
                  </span>
                </li>
              )
            })}
          </ol>
          <p className="ai-insights__legend">
            <em>Bar = visit share · Number = value estimate (−1 losing, +1 winning)</em>
          </p>
        </>
      )}
    </div>
  )
}
