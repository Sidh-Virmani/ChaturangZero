import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { Board } from './components/Board'
import { CapturedTray } from './components/CapturedTray'
import { GameSidebar } from './components/GameSidebar'
import { capturedBy } from './lib/capturedPieces'
import { applyTheme, loadPrefs, savePrefs, type Difficulty, type PlayMode, type ThemeName, type UiPrefs } from './lib/prefs'
import { neighborInView } from './lib/squares'
import { playMoveSound } from './lib/sound'
import { gameAdapter } from './services/mockGameAdapter'
import type { BoardState, LegalMove, Side, SquareId } from './types/game'

function App() {
  const hasLoaded = useRef(false)
  const [prefs, setPrefs] = useState<UiPrefs>(() => loadPrefs())
  const [state, setState] = useState<BoardState | null>(null)
  const [selectedSquare, setSelectedSquare] = useState<SquareId | null>(null)
  const [focusedSquare, setFocusedSquare] = useState<SquareId>('e2')
  const [keyboardNav, setKeyboardNav] = useState(false)
  const [legalMoves, setLegalMoves] = useState<LegalMove[]>([])
  const [error, setError] = useState<string | null>(null)
  const [bootstrapping, setBootstrapping] = useState(true)
  const [resetting, setResetting] = useState(false)
  const [movePending, setMovePending] = useState(false)

  const updatePrefs = useCallback((patch: Partial<UiPrefs>) => {
    setPrefs((current) => {
      const next = { ...current, ...patch }
      savePrefs(next)
      return next
    })
  }, [])

  useEffect(() => {
    applyTheme(prefs.theme)
  }, [prefs.theme])

  const clearSelection = useCallback(() => {
    setSelectedSquare(null)
    setLegalMoves([])
  }, [])

  const loadGame = useCallback(async () => {
    setBootstrapping(true)
    setError(null)
    clearSelection()
    try {
      setState(await gameAdapter.getState())
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to load the game.')
    } finally {
      setBootstrapping(false)
    }
  }, [clearSelection])

  useEffect(() => {
    if (hasLoaded.current) return
    hasLoaded.current = true
    void loadGame()
  }, [loadGame])

  const selectSquare = async (square: SquareId) => {
    if (!state) return
    setError(null)
    try {
      const suppliedMoves = await gameAdapter.getLegalMoves(square)
      if (suppliedMoves.length === 0) {
        clearSelection()
        return
      }
      setSelectedSquare(square)
      setLegalMoves(suppliedMoves)
      setFocusedSquare(square)
    } catch (cause) {
      clearSelection()
      setError(cause instanceof Error ? cause.message : 'Unable to select that piece.')
    }
  }

  const ignoreClickRef = useRef(false)
  const dragFromRef = useRef<SquareId | null>(null)
  const draggingRef = useRef(false)

  const applyMove = async (from: SquareId, to: SquareId, knownMoves = legalMoves) => {
    const suppliedMove = knownMoves.find((move) => move.from === from && move.to === to)
    if (!suppliedMove) return false
    setMovePending(true)
    setError(null)
    try {
      const next = await gameAdapter.requestMove({ from, to })
      setState(next)
      clearSelection()
      setFocusedSquare(to)
      if (prefs.sound) playMoveSound(suppliedMove.isCapture)
      return true
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The engine could not apply that move.')
      return false
    } finally {
      setMovePending(false)
    }
  }

  const handleSquareClick = async (square: SquareId) => {
    if (ignoreClickRef.current) {
      ignoreClickRef.current = false
      return
    }
    setKeyboardNav(false)
    if (!state || movePending || resetting) return
    if (state.status.code === 'terminal' || state.isAiThinking) return

    if (square === selectedSquare) {
      clearSelection()
      return
    }

    if (selectedSquare) {
      const moved = await applyMove(selectedSquare, square)
      if (moved) return
    }

    if (state.selectableSquares.includes(square)) {
      await selectSquare(square)
      return
    }

    clearSelection()
    setFocusedSquare(square)
  }

  const handleDragStart = (square: SquareId) => {
    if (!state || state.status.code === 'terminal' || state.isAiThinking) return
    draggingRef.current = true
    dragFromRef.current = square
    setFocusedSquare(square)
    void selectSquare(square)
  }

  const handleDropSquare = async (square: SquareId) => {
    if (!draggingRef.current) return
    draggingRef.current = false
    const from = dragFromRef.current ?? selectedSquare
    dragFromRef.current = null
    if (!from || square === from) return
    ignoreClickRef.current = true
    let moves = legalMoves
    if (!moves.some((move) => move.from === from && move.to === square)) {
      try {
        moves = await gameAdapter.getLegalMoves(from)
      } catch {
        return
      }
    }
    await applyMove(from, square, moves)
  }

  const handleReset = async () => {
    setResetting(true)
    setError(null)
    clearSelection()
    try {
      setState(await gameAdapter.resetGame())
      setFocusedSquare(prefs.orientation === 'white' ? 'e2' : 'e7')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to reset the game.')
    } finally {
      setResetting(false)
    }
  }

  const handleBoardKey = (event: KeyboardEvent<HTMLDivElement>) => {
    setKeyboardNav(true)
    if (event.key === 'Escape') {
      event.preventDefault()
      clearSelection()
      return
    }

    if (
      event.key === 'ArrowUp' ||
      event.key === 'ArrowDown' ||
      event.key === 'ArrowLeft' ||
      event.key === 'ArrowRight'
    ) {
      event.preventDefault()
      const next = neighborInView(focusedSquare, event.key, prefs.orientation)
      if (next) setFocusedSquare(next)
      return
    }

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      void handleSquareClick(focusedSquare)
    }
  }

  const locked =
    Boolean(state && (state.status.code === 'terminal' || state.isAiThinking)) || resetting
  const busyLabel = state?.isAiThinking
    ? 'Opponent thinking…'
    : resetting
      ? 'Starting a new game…'
      : null

  const topSide: Side = prefs.orientation === 'white' ? 'black' : 'white'
  const bottomSide: Side = prefs.orientation

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="header-inner">
          <a className="brand" href="/" aria-label="ChaturangZero home">
            <span className="brand__mark" aria-hidden="true">
              <i />
              <i />
              <i />
              <i />
            </span>
            <span>
              <strong>ChaturangZero</strong>
              <small>Chaturanga play board</small>
            </span>
          </a>
          <div className="header-actions">
            <span className="adapter-badge" title="MockGameAdapter is serving demo replies">
              <span /> Mock
            </span>
            <button
              type="button"
              className="icon-button"
              onClick={() =>
                updatePrefs({ orientation: prefs.orientation === 'white' ? 'black' : 'white' })
              }
              aria-pressed={prefs.orientation === 'black'}
            >
              Flip board
            </button>
            <button
              type="button"
              className="icon-button"
              onClick={() => updatePrefs({ theme: invertTheme(prefs.theme) })}
            >
              {prefs.theme === 'dark' ? 'Light board' : 'Dark board'}
            </button>
            <button
              type="button"
              className="icon-button"
              aria-pressed={prefs.sound}
              onClick={() => updatePrefs({ sound: !prefs.sound })}
            >
              {prefs.sound ? 'Sound on' : 'Sound off'}
            </button>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="mock-banner" role="status">
          <strong>Demo adapter only.</strong>
          <span>
            Highlighted pieces can make a handful of scripted replies. This is not Chaturanga
            legality. Real rules arrive when <code>shared/game_core</code> is wired through
            GameAdapter.
          </span>
        </div>

        {error && (
          <div className="error-banner" role="alert">
            <div>
              <strong>Engine connection issue</strong>
              <span>{error}</span>
            </div>
            <button type="button" onClick={() => void loadGame()}>
              Try again
            </button>
          </div>
        )}

        {state ? (
          <div className="game-layout">
            <section className="board-section" aria-label="Game board">
              <PlayerStrip
                side={topSide}
                isToMove={state.currentTurn === topSide}
                captured={capturedBy(topSide, state.pieces)}
              />
              <Board
                state={state}
                selectedSquare={selectedSquare}
                focusedSquare={focusedSquare}
                showFocus={keyboardNav}
                legalMoves={legalMoves}
                orientation={prefs.orientation}
                locked={locked}
                busyLabel={busyLabel}
                onSquareClick={(square) => void handleSquareClick(square)}
                onDragStart={handleDragStart}
                onDropSquare={(square) => void handleDropSquare(square)}
                onKeyNavigate={handleBoardKey}
              />
              <PlayerStrip
                side={bottomSide}
                isToMove={state.currentTurn === bottomSide}
                captured={capturedBy(bottomSide, state.pieces)}
              />
            </section>
            <GameSidebar
              state={state}
              disabled={resetting || bootstrapping}
              playMode={prefs.playMode}
              difficulty={prefs.difficulty}
              onReset={() => void handleReset()}
              onPlayMode={(mode: PlayMode) => updatePrefs({ playMode: mode })}
              onDifficulty={(difficulty: Difficulty) => updatePrefs({ difficulty })}
            />
          </div>
        ) : bootstrapping ? (
          <div className="initial-loading" role="status">
            <span className="spinner" />
            <strong>Preparing the board</strong>
            <p>Waiting for the game adapter…</p>
          </div>
        ) : (
          <div className="initial-loading" role="status">
            <strong>No position loaded</strong>
            <p>The adapter did not return a board. Retry or check the mock connection.</p>
            <button className="button button--primary" type="button" onClick={() => void loadGame()}>
              Load board
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

function invertTheme(theme: ThemeName): ThemeName {
  return theme === 'dark' ? 'light' : 'dark'
}

function PlayerStrip({
  side,
  isToMove,
  captured,
}: {
  side: Side
  isToMove: boolean
  captured: ReturnType<typeof capturedBy>
}) {
  const name = side === 'white' ? 'White' : 'Black'
  const flavor = side === 'white' ? 'Ivory' : 'Onyx'
  return (
    <div className={`player-strip ${side === 'black' ? 'player-strip--black' : ''}`}>
      <span className={`player-token player-token--${side}`} aria-hidden="true" />
      <div>
        <strong>{name}</strong>
        <small>{flavor}</small>
      </div>
      <CapturedTray
        side={side === 'white' ? 'black' : 'white'}
        pieces={captured}
        label={`Pieces captured by ${name}`}
      />
      {isToMove && <span className="active-turn">To move</span>}
    </div>
  )
}

export default App
