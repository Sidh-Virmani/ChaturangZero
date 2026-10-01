import type { GameMode, Side } from '../types/game'

interface PlayerStripProps { side: Side; currentTurn: Side; mode: GameMode }

export function PlayerStrip({ side, currentTurn, mode }: PlayerStripProps) {
  const name = mode === 'machine' && side === 'black' ? 'Machine' : side === 'white' ? 'Ivory' : 'Onyx'
  return <div className="player-strip"><span className={`player-token player-token--${side}`} aria-hidden="true" /><div className="player-strip__identity"><strong>{name}</strong><small>{side} side</small></div>{currentTurn === side && <span className="active-turn"><i /> To move</span>}</div>
}
