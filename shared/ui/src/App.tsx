import { useCallback, useState } from 'react'
import { Board } from './components/Board'
import { GameSidebar } from './components/GameSidebar'
import { HomeScreen } from './components/HomeScreen'
import { PlayerStrip } from './components/PlayerStrip'
import { ResultModal } from './components/ResultModal'
import { gameAdapter } from './services/httpGameAdapter'
import type { BoardState, LegalMove, Side, SquareId } from './types/game'

export type GameMode = 'human' | 'machine'

function App() {
  const [screen, setScreen] = useState<'home' | 'game'>('home')
  const [mode, setMode] = useState<GameMode>('human')
  const [orientation, setOrientation] = useState<Side>('white')
  const [state, setState] = useState<BoardState | null>(null)
  const [selectedSquare, setSelectedSquare] = useState<SquareId | null>(null)
  const [legalMoves, setLegalMoves] = useState<LegalMove[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const clearSelection = useCallback(() => {
    setSelectedSquare(null)
    setLegalMoves([])
  }, [])

  const startGame = async (nextMode: GameMode) => {
    setScreen('game')
    setMode(nextMode)
    setOrientation('white')
    setIsLoading(true)
    setError(null)
    setState(null)
    clearSelection()
    try {
      setState(await gameAdapter.resetGame())
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to start the game.')
    } finally {
      setIsLoading(false)
    }
  }

  const goHome = () => {
    clearSelection()
    setError(null)
    setScreen('home')
  }

  const resetGame = async () => {
    setIsLoading(true)
    setError(null)
    clearSelection()
    try {
      setState(await gameAdapter.resetGame())
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to reset the game.')
    } finally {
      setIsLoading(false)
    }
  }

  const selectSquare = async (square: SquareId) => {
    setIsLoading(true)
    setError(null)
    try {
      const suppliedMoves = await gameAdapter.getLegalMoves(square)
      if (suppliedMoves.length === 0) {
        clearSelection()
        return
      }
      setSelectedSquare(square)
      setLegalMoves(suppliedMoves)
    } catch (cause) {
      clearSelection()
      setError(cause instanceof Error ? cause.message : 'Unable to select that piece.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSquareClick = async (square: SquareId) => {
    if (!state || isLoading || mode === 'machine' || state.status.code === 'terminal' || state.isAiThinking) return
    if (square === selectedSquare) {
      clearSelection()
      return
    }
    const suppliedMove = legalMoves.find((move) => move.to === square)
    if (selectedSquare && suppliedMove) {
      setIsLoading(true)
      setError(null)
      try {
        setState(await gameAdapter.requestMove({ from: selectedSquare, to: square }))
        clearSelection()
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : 'The engine could not apply that move.')
      } finally {
        setIsLoading(false)
      }
      return
    }
    if (state.selectableSquares.includes(square)) {
      await selectSquare(square)
    } else {
      clearSelection()
    }
  }

  const flipBoard = () => {
    clearSelection()
    setOrientation((side) => side === 'white' ? 'black' : 'white')
  }

  const topSide: Side = orientation === 'white' ? 'black' : 'white'

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-inner">
          <button className="brand" type="button" onClick={goHome} aria-label="ChaturangZero home">
            <span className="brand__mark" aria-hidden="true"><i /><i /><i /><i /></span>
            <span><strong>ChaturangZero</strong><small>THE ROYAL GAME</small></span>
          </button>
          <div className="header-right"><span className="header-edition">A CLASSIC GAME OF STRATEGY</span><span className="header-indicator" aria-hidden="true" /></div>
        </div>
      </header>
      <main className="main-content">
        {screen === 'home' ? <HomeScreen onStart={(nextMode) => void startGame(nextMode)} /> : (
          <>
            <div className="game-topline">
              <div><span className="eyebrow">NOW PLAYING</span><h1>{mode === 'human' ? 'Local match' : 'Play the machine'}</h1></div>
              <span className="match-tag">{mode === 'human' ? 'HUMAN VS HUMAN' : 'HUMAN VS MACHINE'}</span>
            </div>
            {error && <div className="error-banner" role="alert"><div><strong>Game connection issue</strong><span>{error}</span></div><button type="button" onClick={() => void resetGame()}>Try again</button></div>}
            {state ? (
              <div className="game-layout">
                <section className="board-section" aria-label="Game board">
                  <PlayerStrip side={topSide} currentTurn={state.currentTurn} mode={mode} />
                  <Board state={state} orientation={orientation} selectedSquare={selectedSquare} legalMoves={legalMoves} disabled={isLoading || mode === 'machine' || state.isAiThinking || state.status.code === 'terminal'} busy={isLoading || state.isAiThinking} onSquareClick={(square) => void handleSquareClick(square)} />
                  <PlayerStrip side={orientation} currentTurn={state.currentTurn} mode={mode} />
                </section>
                <GameSidebar state={state} mode={mode} orientation={orientation} disabled={isLoading} onReset={() => void resetGame()} onFlip={flipBoard} onHome={goHome} />
                {state.status.code === 'terminal' && <ResultModal status={state.status} onRematch={() => void resetGame()} onHome={goHome} />}
              </div>
            ) : <div className="initial-loading" role="status"><span className="spinner" /><strong>Preparing the board</strong><p>Connecting to the Chaturanga game core...</p></div>}
          </>
        )}
      </main>
    </div>
  )
}

export default App
