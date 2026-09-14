/**
 * ModeSelector.tsx
 * ================
 * Toggle between Human vs Human and Human vs AI modes.
 * Also shows the difficulty slider (MCTS simulation budget).
 */

export type GameMode = 'hvh' | 'hvai'

interface ModeSelectorProps {
  mode:            GameMode
  difficulty:      number   // MCTS simulations: 25 | 100 | 400
  onModeChange:    (mode: GameMode) => void
  onDifficultyChange: (sims: number) => void
  disabled:        boolean
}

const DIFFICULTIES = [
  { label: 'Easy',   sims: 25,  desc: '25 sims' },
  { label: 'Medium', sims: 100, desc: '100 sims' },
  { label: 'Hard',   sims: 400, desc: '400 sims' },
]

export function ModeSelector({
  mode,
  difficulty,
  onModeChange,
  onDifficultyChange,
  disabled,
}: ModeSelectorProps) {
  return (
    <div className="mode-selector">
      <div className="mode-selector__tabs" role="group" aria-label="Game mode">
        <button
          type="button"
          className={`mode-tab${mode === 'hvh' ? ' mode-tab--active' : ''}`}
          onClick={() => onModeChange('hvh')}
          disabled={disabled}
          aria-pressed={mode === 'hvh'}
        >
          👥 Human vs Human
        </button>
        <button
          type="button"
          className={`mode-tab${mode === 'hvai' ? ' mode-tab--active' : ''}`}
          onClick={() => onModeChange('hvai')}
          disabled={disabled}
          aria-pressed={mode === 'hvai'}
        >
          🤖 Human vs AI
        </button>
      </div>

      {mode === 'hvai' && (
        <div className="difficulty-selector">
          <label className="difficulty-selector__label">AI Difficulty</label>
          <div className="difficulty-selector__options" role="group">
            {DIFFICULTIES.map(({ label, sims, desc }) => (
              <button
                key={sims}
                type="button"
                className={`difficulty-btn${difficulty === sims ? ' difficulty-btn--active' : ''}`}
                onClick={() => onDifficultyChange(sims)}
                disabled={disabled}
                title={desc}
              >
                {label}
                <small>{desc}</small>
              </button>
            ))}
          </div>
          <p className="difficulty-selector__hint">
            More simulations = stronger AI but slower response time.
          </p>
        </div>
      )}
    </div>
  )
}
