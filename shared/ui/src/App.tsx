import { useCallback, useEffect, useRef, useState } from 'react'
import { Board } from './components/Board'
import { GameSidebar } from './components/GameSidebar'
import { gameAdapter } from './services/mockGameAdapter'
import type { BoardState, LegalMove, SquareId } from './types/game'

function App() {
  const hasLoaded = useRef(false)
  const [state, setState] = useState<BoardState | null>(null)
  const [selectedSquare, setSelectedSquare] = useState<SquareId | null>(null)
  const [legalMoves, setLegalMoves] = useState<LegalMove[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const clearSelection = useCallback(() => {
    setSelectedSquare(null)
    setLegalMoves([])
  }, [])

  const loadGame = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    clearSelection()
    try {
      setState(await gameAdapter.getState())
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to load the game.')
    } finally {
      setIsLoading(false)
    }
  }, [clearSelection])

  useEffect(() => {
    if (hasLoaded.current) return
    hasLoaded.current = true
    void loadGame()
  }, [loadGame])

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
    if (!state || isLoading || state.status.code === 'terminal' || state.isAiThinking) return

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
      return
    }

    clearSelection()
  }

  const handleReset = async () => {
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

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-inner">
          <a className="brand" href="/" aria-label="ChaturangZero home">
            <span className="brand__mark" aria-hidden="true">
              <i /><i /><i /><i />
            </span>
            <span>
              <strong>ChaturangZero</strong>
              <small>Play board</small>
            </span>
          </a>
          <div className="adapter-badge">
            <span /> Mock game connected
          </div>
        </div>
      </header>

      <main className="main-content">
        {error && (
          <div className="error-banner" role="alert">
            <div>
              <strong>Engine connection issue</strong>
              <span>{error}</span>
            </div>
            <button type="button" onClick={() => void loadGame()}>Try again</button>
          </div>
        )}

        {state ? (
          <div className="game-layout">
            <section className="board-section" aria-label="Game board">
              <div className="player-strip player-strip--black">
                <span className="player-token player-token--black" aria-hidden="true" />
                <div><strong>Onyx</strong><small>Black</small></div>
                {state.currentTurn === 'black' && <span className="active-turn">To move</span>}
              </div>
              <Board
                state={state}
                selectedSquare={selectedSquare}
                legalMoves={legalMoves}
                disabled={isLoading || state.isAiThinking || state.status.code === 'terminal'}
                onSquareClick={(square) => void handleSquareClick(square)}
              />
              <div className="player-strip">
                <span className="player-token player-token--white" aria-hidden="true" />
                <div><strong>Ivory</strong><small>White</small></div>
                {state.currentTurn === 'white' && <span className="active-turn">To move</span>}
              </div>
            </section>
            <GameSidebar state={state} disabled={isLoading} onReset={() => void handleReset()} />
          </div>
        ) : (
          <div className="initial-loading" role="status">
            <span className="spinner" />
            <strong>Preparing the board</strong>
            <p>Waiting for the game adapter...</p>
          </div>
        )}
      </main>

    </div>
  )
}

export default App
