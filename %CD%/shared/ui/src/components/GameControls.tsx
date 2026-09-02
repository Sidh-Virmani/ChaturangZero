import type { Difficulty, PlayMode } from '../lib/prefs'

interface GameControlsProps {
  playMode: PlayMode
  difficulty: Difficulty
  onPlayMode: (mode: PlayMode) => void
  onDifficulty: (difficulty: Difficulty) => void
}

export function GameControls({
  playMode,
  difficulty,
  onPlayMode,
  onDifficulty,
}: GameControlsProps) {
  return (
    <section className="panel controls-panel" aria-labelledby="controls-heading">
      <h2 id="controls-heading">Players</h2>
      <div className="mode-toggle" role="group" aria-label="Match type">
        <button
          type="button"
          className={`mode-chip ${playMode === 'human' ? 'mode-chip--on' : ''}`}
          aria-pressed={playMode === 'human'}
          onClick={() => onPlayMode('human')}
        >
          Human vs Human
        </button>
        <button
          type="button"
          className={`mode-chip ${playMode === 'ai' ? 'mode-chip--on' : ''}`}
          aria-pressed={playMode === 'ai'}
          onClick={() => onPlayMode('ai')}
        >
          Human vs AI
        </button>
      </div>
      <label className="difficulty">
        <span>AI difficulty</span>
        <select
          value={difficulty}
          disabled={playMode !== 'ai'}
          onChange={(event) => onDifficulty(event.target.value as Difficulty)}
        >
          <option value="easy">Easy</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
      </label>
      {playMode === 'ai' && (
        <p className="hint">
          Placeholder only. The opponent will use {difficulty} search once game_core and MCTS exist.
          Both sides still play through the mock adapter.
        </p>
      )}
    </section>
  )
}
