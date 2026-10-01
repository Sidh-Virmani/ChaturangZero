import type { GameStatus } from '../types/game'

interface ResultModalProps { status: GameStatus; onRematch: () => void; onHome: () => void }

export function ResultModal({ status, onRematch, onHome }: ResultModalProps) {
  return <div className="result-backdrop"><div className="result-modal" role="dialog" aria-modal="true" aria-labelledby="result-title" aria-describedby="result-detail"><div className="result-modal__crest" aria-hidden="true">✦</div><span className="eyebrow">MATCH COMPLETE</span><h2 id="result-title">{status.result ?? 'Game over'}</h2><p id="result-detail">{status.detail}</p><div className="result-modal__actions"><button className="button button--primary" type="button" onClick={onRematch}>Rematch</button><button className="button button--secondary" type="button" onClick={onHome}>Back to home</button></div></div></div>
}
