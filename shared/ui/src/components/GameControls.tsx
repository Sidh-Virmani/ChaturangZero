import type { Side } from '../types/game'

interface GameControlsProps {
  disabled: boolean
  orientation: Side
  onReset: () => void
  onFlip: () => void
}

export function GameControls({ disabled, orientation, onReset, onFlip }: GameControlsProps) {
  return <section className="controls" aria-labelledby="controls-heading">
    <h2 id="controls-heading">Game controls</h2>
    <div className="controls__grid">
      <button className="button button--primary" type="button" onClick={onReset} disabled={disabled}><span className="button__icon" aria-hidden="true">＋</span> New game</button>
      <button className="button button--secondary" type="button" onClick={onFlip} disabled={disabled} aria-label={`Flip board to ${orientation === 'white' ? 'black' : 'white'} perspective`}><span className="button__icon" aria-hidden="true">⟳</span> Flip board</button>
    </div>
  </section>
}
