/**
 * AIInsights.tsx
 * ==============
 * Shows the top moves the AI considered, with visit percentages and Q-values.
 * This makes ChaturangZero unique: the AI is transparent and educational.
 *
 * Data is injected via the `insights` prop, which comes from the /ai_move
 * API response's `aiInsights` field.
 */

interface TopMove {
  from: string
  to: string
  visitFraction: number
  qValue: number
}

interface AiInsightsProps {
  topMoves: TopMove[]
  numSims: number
  chosenMove?: { from: string; to: string }
}

export function AIInsights({ topMoves, numSims, chosenMove }: AiInsightsProps) {
  if (topMoves.length === 0) return null

  return (
    <div className="ai-insights">
      <h3 className="ai-insights__title">
        <span className="ai-insights__brain" aria-hidden="true">🧠</span>
        AI Thinking
        <small>{numSims} simulations</small>
      </h3>
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
    </div>
  )
}
