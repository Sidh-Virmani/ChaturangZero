import type { BoardState, GameMode } from '../types/game'

interface GameStatusProps { state: BoardState; mode: GameMode; thinking: boolean }

export function GameStatus({ state, mode, thinking }: GameStatusProps) {
  const turnName = state.currentTurn === 'white' ? 'Ivory' : 'Onyx'
  const isTerminal = state.status.code === 'terminal'
  return <section className="status-card" aria-labelledby="game-status-heading">
    <div className="status-card__label"><span>GAME STATUS</span><span className="status-card__live"><i /> {isTerminal ? 'FINISHED' : thinking ? 'SEARCHING' : 'LIVE'}</span></div>
    <div className="turn-display"><span className={`turn-piece turn-piece--${state.currentTurn}`} aria-hidden="true" /><div><h2 id="game-status-heading">{isTerminal ? 'Game complete' : !state.machineReady && mode === 'machine' ? 'Checkpoint needed' : thinking ? 'Machine is thinking' : `${turnName} to move`}</h2><p>{mode === 'machine' && !state.machineReady ? 'Train a model or set CHATURANGZERO_CHECKPOINT, then restart the app.' : thinking ? 'The policy/value model is guiding MCTS.' : state.status.detail}</p></div></div>
    {mode === 'human' && !isTerminal && <div className="status-card__hint">Select a piece to see its legal moves.</div>}
    {mode === 'machine' && state.smokeCheckpoint && <div className="status-card__hint">Using the smoke checkpoint for integration. Playing strength is unproven.</div>}
  </section>
}
