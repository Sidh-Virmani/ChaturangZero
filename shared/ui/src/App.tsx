import { useCallback, useRef, useState } from 'react'
import { BoardExperience } from './components/BoardExperience'
import { GameSidebar } from './components/GameSidebar'
import { HomeScreen } from './components/HomeScreen'
import { PlayerStrip } from './components/PlayerStrip'
import { ResultModal } from './components/ResultModal'
import { gameAdapter } from './services/httpGameAdapter'
import type { BoardState, GameMode, LegalMove, MachineAnalysis, MachineProgress, Side, SquareId } from './types/game'

function App() {
  const [screen, setScreen] = useState<'home' | 'game'>('home')
  const [mode, setMode] = useState<GameMode>('human')
  const [orientation, setOrientation] = useState<Side>('white')
  const [state, setState] = useState<BoardState | null>(null)
  const [selectedSquare, setSelectedSquare] = useState<SquareId | null>(null)
  const [legalMoves, setLegalMoves] = useState<LegalMove[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isAiThinking, setIsAiThinking] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)
  const [machineProgress, setMachineProgress] = useState<MachineProgress | null>(null)
  const [machineResult, setMachineResult] = useState<MachineAnalysis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const requestEpoch = useRef(0)

  const clearSelection = useCallback(() => {
    setSelectedSquare(null)
    setLegalMoves([])
  }, [])

  const startGame = async (nextMode: GameMode) => {
    const epoch = ++requestEpoch.current
    setScreen('game')
    setMode(nextMode)
    setOrientation('white')
    setIsLoading(true)
    setIsAiThinking(false)
    setIsAnimating(false)
    setMachineProgress(null)
    setMachineResult(null)
    setError(null)
    setState(null)
    clearSelection()
    try {
      const next = await gameAdapter.resetGame(nextMode)
      if (epoch === requestEpoch.current) { setIsAnimating(true); setState(next) }
    } catch (cause) {
      if (epoch === requestEpoch.current) setError(cause instanceof Error ? cause.message : 'Unable to start the game.')
    } finally {
      if (epoch === requestEpoch.current) setIsLoading(false)
    }
  }

  const goHome = () => {
    ++requestEpoch.current
    clearSelection()
    setError(null)
    setIsLoading(false)
    setIsAiThinking(false)
    setIsAnimating(false)
    setMachineProgress(null)
    setMachineResult(null)
    setScreen('home')
  }

  const resetGame = async () => {
    const epoch = ++requestEpoch.current
    setIsLoading(true)
    setIsAiThinking(false)
    setIsAnimating(false)
    setMachineProgress(null)
    setMachineResult(null)
    setError(null)
    clearSelection()
    try {
      const next = await gameAdapter.resetGame(mode)
      if (epoch === requestEpoch.current) { setIsAnimating(true); setState(next) }
    } catch (cause) {
      if (epoch === requestEpoch.current) setError(cause instanceof Error ? cause.message : 'Unable to reset the game.')
    } finally {
      if (epoch === requestEpoch.current) setIsLoading(false)
    }
  }

  const playMachineTurn = async (epoch: number) => {
    setIsAiThinking(true)
    setMachineProgress(null)
    setMachineResult(null)
    setError(null)
    let polling = true
    const poll = async () => {
      while (polling && epoch === requestEpoch.current) {
        try {
          const progress = await gameAdapter.getMachineProgress()
          if (polling && epoch === requestEpoch.current && progress.phase !== 'idle') setMachineProgress(progress)
        } catch {
          // The move request reports connection errors; a missed progress poll is harmless.
        }
        await new Promise((resolve) => window.setTimeout(resolve, 160))
      }
    }
    try {
      const moveRequest = gameAdapter.requestMachineMove()
      void poll()
      const next = await moveRequest
      if (epoch === requestEpoch.current) {
        try { setMachineProgress(await gameAdapter.getMachineProgress()) } catch { /* final analysis still comes with the move */ }
        setMachineResult(next.analysis)
        // Let the completed search become visible before the piece arrives.
        await new Promise((resolve) => window.setTimeout(resolve, 320))
        if (epoch === requestEpoch.current) {
          setState(next)
          setIsAnimating(true)
          setMachineResult(null)
        }
      }
    } catch (cause) {
      if (epoch === requestEpoch.current) setError(cause instanceof Error ? cause.message : 'The machine search failed. Retry its move.')
    } finally {
      polling = false
      if (epoch === requestEpoch.current) setIsAiThinking(false)
    }
  }

  const selectSquare = async (square: SquareId) => {
    const epoch = requestEpoch.current
    setIsLoading(true)
    setError(null)
    try {
      const suppliedMoves = await gameAdapter.getLegalMoves(square)
      if (epoch !== requestEpoch.current) return
      if (suppliedMoves.length === 0) {
        clearSelection()
        return
      }
      setSelectedSquare(square)
      setLegalMoves(suppliedMoves)
    } catch (cause) {
      if (epoch === requestEpoch.current) {
        clearSelection()
        setError(cause instanceof Error ? cause.message : 'Unable to select that piece.')
      }
    } finally {
      if (epoch === requestEpoch.current) setIsLoading(false)
    }
  }

  const handleSquareClick = async (square: SquareId) => {
    if (!state || isLoading || isAiThinking || isAnimating || state.status.code === 'terminal' ||
      (mode === 'machine' && (!state.machineReady || state.currentTurn !== 'white'))) return
    if (square === selectedSquare) {
      clearSelection()
      return
    }
    const suppliedMove = legalMoves.find((move) => move.to === square)
    if (selectedSquare && suppliedMove) {
      const epoch = requestEpoch.current
      setIsLoading(true)
      setError(null)
      try {
        const next = await gameAdapter.requestMove({ from: selectedSquare, to: square })
        if (epoch !== requestEpoch.current) return
        setState(next)
        setIsAnimating(true)
        clearSelection()
        setIsLoading(false)
        if (mode === 'machine' && next.status.code !== 'terminal') {
          await playMachineTurn(epoch)
        }
      } catch (cause) {
        if (epoch === requestEpoch.current) setError(cause instanceof Error ? cause.message : 'The engine could not apply that move.')
      } finally {
        if (epoch === requestEpoch.current) setIsLoading(false)
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
      <main className={`main-content main-content--${screen}`}>
        {screen === 'home' ? <HomeScreen onStart={(nextMode) => void startGame(nextMode)} /> : (
          <>
            <div className="game-topline">
              <div><button className="game-back" type="button" onClick={goHome}><span aria-hidden="true">‹</span> Back to home</button><span className="eyebrow">NOW PLAYING</span><h1>{mode === 'human' ? 'Local match' : 'Play the machine'}</h1></div>
              <span className="match-tag">{mode === 'human' ? 'HUMAN VS HUMAN' : 'HUMAN VS MACHINE'}</span>
            </div>
            {error && <div className="error-banner" role="alert"><div><strong>Game connection issue</strong><span>{error}</span></div><button type="button" onClick={() => void (mode === 'machine' && state?.currentTurn === 'black' ? playMachineTurn(requestEpoch.current) : resetGame())}>Try again</button></div>}
            {state ? (
              <div className={`game-layout game-layout--${mode}`}>
                <section className="board-section" aria-label="Game board">
                  <PlayerStrip side={topSide} currentTurn={state.currentTurn} mode={mode} />
                  <BoardExperience state={state} orientation={orientation} selectedSquare={selectedSquare} legalMoves={legalMoves} disabled={isLoading || isAiThinking || isAnimating || state.status.code === 'terminal' || (mode === 'machine' && (!state.machineReady || state.currentTurn !== 'white'))} busy={isLoading} thinking={isAiThinking} onMotionChange={setIsAnimating} onSquareClick={(square) => void handleSquareClick(square)} />
                  <PlayerStrip side={orientation} currentTurn={state.currentTurn} mode={mode} />
                </section>
                <GameSidebar state={state} mode={mode} orientation={orientation} thinking={isAiThinking} progress={machineProgress} machineResult={machineResult} disabled={isLoading || isAiThinking} onReset={() => void resetGame()} onFlip={flipBoard} />
                {state.status.code === 'terminal' && !isLoading && !isAnimating && <ResultModal status={state.status} onRematch={() => void resetGame()} onHome={goHome} />}
              </div>
            ) : <div className="initial-loading" role="status"><span className="spinner" /><strong>Preparing the board</strong><p>Connecting to the Chaturanga game core...</p></div>}
          </>
        )}
      </main>
    </div>
  )
}

export default App
