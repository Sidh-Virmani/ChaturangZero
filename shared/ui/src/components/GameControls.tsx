interface GameControlsProps {
  disabled: boolean
  onReset: () => void
}

export function GameControls({ disabled, onReset }: GameControlsProps) {
  return (
    <section className="panel controls" aria-labelledby="controls-heading">
      <h2 id="controls-heading">Game controls</h2>
      <button className="button button--primary" type="button" onClick={onReset} disabled={disabled}>
        <span aria-hidden="true">↻</span> New game
      </button>
    </section>
  )
}
