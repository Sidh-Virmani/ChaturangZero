import type { GameMode } from '../App'

interface HomeScreenProps { onStart: (mode: GameMode) => void }

export function HomeScreen({ onStart }: HomeScreenProps) {
  return (
    <section className="home-screen" aria-labelledby="home-heading">
      <div className="home-intro">
        <span className="eyebrow"><span className="eyebrow-line" /> THE BOARD IS READY</span>
        <h1 id="home-heading">A timeless game.<br /><em>Your next move.</em></h1>
        <p>Enter the world of Chaturanga. Settle in, take your side, and let the board tell the story.</p>
      </div>
      <div className="home-content">
        <div className="mode-list" aria-label="Choose a game mode">
          <button className="mode-card" type="button" onClick={() => onStart('human')}>
            <span className="mode-card__number">01 / LOCAL MATCH</span>
            <span className="mode-card__main"><span className="mode-card__symbol mode-card__symbol--people" aria-hidden="true"><i /><i /></span><span><strong>Human vs Human</strong><small>Two players. One board. Take turns on this device.</small></span></span>
            <span className="mode-card__footer"><span>PLAY NOW</span><span className="mode-card__arrow" aria-hidden="true">↗</span></span>
          </button>
          <button className="mode-card mode-card--machine" type="button" onClick={() => onStart('machine')}>
            <span className="mode-card__number">02 / MACHINE MATCH</span>
            <span className="mode-card__main"><span className="mode-card__symbol mode-card__symbol--machine" aria-hidden="true"><i /><i /><i /><i /></span><span><strong>Human vs Machine</strong><small>AI play will arrive when a trained model is connected.</small></span></span>
            <span className="mode-card__footer"><span>PREVIEW MODE</span><span className="mode-card__arrow" aria-hidden="true">↗</span></span>
          </button>
        </div>
        <div className="home-board" aria-hidden="true"><div className="home-board__inner"><span>अष्टापद</span><small>THE EIGHT BY EIGHT BOARD</small></div></div>
      </div>
      <div className="home-note"><span>ROOTED IN TRADITION</span><span>BUILT FOR THE NEXT MOVE</span></div>
    </section>
  )
}
