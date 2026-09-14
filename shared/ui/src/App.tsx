import { useCallback, useEffect, useRef, useState } from 'react'
import { Board } from './components/Board'
import { GameSidebar } from './components/GameSidebar'
import { AIInsights } from './components/AIInsights'
import { ModeSelector, type GameMode } from './components/ModeSelector'
import { gameAdapter } from './services/mockGameAdapter'
import type { BoardState, LegalMove, SquareId } from './types/game'

// ── AI Insights type (mirrors what the backend returns) ───────────────────────
interface AiTopMove {
  from: SquareId
  to: SquareId
  visitFraction: number
  qValue: number
}
interface AiInsightsData {
  chosenMove: { from: SquareId; to: SquareId }
  topMoves: AiTopMove[]
  numSims: number
}

function App() {
  const hasLoaded = useRef(false)

  const [state, setState]               = useState<BoardState | null>(null)
  const [selectedSquare, setSelectedSquare] = useState<SquareId | null>(null)
  const [legalMoves, setLegalMoves]     = useState<LegalMove[]>([])
  const [isLoading, setIsLoading]       = useState(true)
  const [error, setError]               = useState<string | null>(null)

  // ── New: mode + difficulty + AI insights ─────────────────────────────────
  const [gameMode, setGameMode]         = useState<GameMode>('hvh')
  const [difficulty, setDifficulty]     = useState(50)
  const [aiInsights, setAiInsights]     = useState<AiInsightsData | null>(null)

  const clearSelection = useCallback(() => {
    setSelectedSquare(null)
    setLegalMoves([])
  }, [])

  const loadGame = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    clearSelection()
    setAiInsights(null)
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

  // ── Handle mode change — reset game when switching ────────────────────────
  const handleModeChange = (mode: GameMode) => {
    setGameMode(mode)
    void handleReset()
  }

  // ── After human move: let AI respond if in HvAI mode ─────────────────────
  const triggerAiMoveIfNeeded = async (newState: BoardState) => {
    if (gameMode !== 'hvai' || newState.status.code === 'terminal') return
    // AI always plays Black in HvAI mode
    if (newState.currentTurn !== 'black') return

    setState({ ...newState, isAiThinking: true })
    try {
      const aiState = await gameAdapter.requestAiMove(difficulty)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const insights = (aiState as any).aiInsights as AiInsightsData | undefined
      if (insights) setAiInsights(insights)
      setState({ ...aiState, isAiThinking: false })
    } catch {
      setState({ ...newState, isAiThinking: false })
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
    if (!state || isLoading || state.status.code === 'terminal' || state.isAiThinking) return

    // In HvAI mode, don't allow clicking when it's Black's turn
    if (gameMode === 'hvai' && state.currentTurn === 'black') return

    if (square === selectedSquare) {
      clearSelection()
      return
    }

    const suppliedMove = legalMoves.find((move) => move.to === square)
    if (selectedSquare && suppliedMove) {
      setIsLoading(true)
      setError(null)
      setAiInsights(null)
      try {
        const newState = await gameAdapter.requestMove({ from: selectedSquare, to: square })
        clearSelection()
        setState(newState)
        await triggerAiMoveIfNeeded(newState)
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
    setAiInsights(null)
    try {
      setState(await gameAdapter.resetGame())
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to reset the game.')
    } finally {
      setIsLoading(false)
    }
  }

  const isBackendMode = false  // flip to true when uvicorn is running

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
            <span />
            {isBackendMode ? '🟢 Engine connected' : '🟡 Mock game'}
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
                <div>
                  <strong>Onyx</strong>
                  <small>{gameMode === 'hvai' ? '🤖 AI' : 'Black'}</small>
                </div>
                {state.currentTurn === 'black' && (
                  <span className="active-turn">
                    {state.isAiThinking ? '⏳ Thinking…' : 'To move'}
                  </span>
                )}
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

            <aside className="right-panel">
              <ModeSelector
                mode={gameMode}
                difficulty={difficulty}
                onModeChange={handleModeChange}
                onDifficultyChange={setDifficulty}
                disabled={isLoading || state.isAiThinking}
              />

              <GameSidebar state={state} disabled={isLoading} onReset={() => void handleReset()} />

              {gameMode === 'hvai' && aiInsights && (
                <AIInsights
                  topMoves={aiInsights.topMoves}
                  numSims={aiInsights.numSims}
                  chosenMove={aiInsights.chosenMove}
                />
              )}
            </aside>
          </div>
        ) : (
          <div className="initial-loading" role="status">
            <span className="spinner" />
            <strong>Preparing the board</strong>
            <p>Waiting for the game adapter…</p>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
