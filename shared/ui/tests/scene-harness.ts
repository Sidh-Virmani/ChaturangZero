import { createBoardScene, type SceneSettings } from '../src/three/scene'
import type { BoardState, Move, Piece, PieceKind, Side, SquareId } from '../src/types/game'

const stage = document.querySelector<HTMLDivElement>('#stage')!
const runButton = document.querySelector<HTMLButtonElement>('#run')!
const results = document.querySelector<HTMLOListElement>('#results')!
const summary = document.querySelector<HTMLParagraphElement>('#summary')!
const modeLabel = document.querySelector<HTMLSpanElement>('#mode')!
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)')
const updateMode = () => { modeLabel.textContent = motionPreference.matches ? 'Motion preference: reduced' : 'Motion preference: normal' }
motionPreference.addEventListener('change', updateMode)
updateMode()

interface MotionEvent { busy: boolean; move: Move | null; time: number }
interface Report { name: string; passed: boolean; detail: string }
let scene: ReturnType<typeof createBoardScene> | null = null
let settings: SceneSettings | null = null
let events: MotionEvent[] = []
let reservedBusy = false
let failures = 0
let revision = 0
let running = false
const runtimeErrors: string[] = []
const reports: Report[] = []

// Useful to browser automation without exposing anything in the product scene.
const diagnostics = {
  get mode() { return motionPreference.matches ? 'reduced' : 'normal' },
  get busy() { return reservedBusy },
  get running() { return running },
  get events() { return events },
  get reports() { return reports },
  get errors() { return runtimeErrors },
}
;(window as Window & { sceneHarness?: typeof diagnostics }).sceneHarness = diagnostics
window.addEventListener('error', event => runtimeErrors.push(event.message))
window.addEventListener('unhandledrejection', event => runtimeErrors.push(String(event.reason)))

const piece = (kind: PieceKind, side: Side, square: SquareId): Piece => ({ id: `${side}-${kind}-${square}`, kind, side, square })
const state = (pieces: Piece[], moveHistory: Move[] = [], currentTurn: Side = 'white'): BoardState => ({
  positionKey: `fixture-${++revision}`, pieces, moveHistory, currentTurn,
  selectableSquares: [], lastMove: moveHistory.at(-1) ?? null,
  status: { code: 'playing', label: 'Local scene fixture', detail: 'No engine request is involved.' },
  isAiThinking: false, mode: 'human', machineReady: false, checkpointLabel: null, smokeCheckpoint: false, analysis: null,
})
const baseline = () => state([
  piece('raja', 'white', 'a1'), piece('raja', 'black', 'h8'),
  piece('padati', 'white', 'e4'), piece('padati', 'black', 'f5'), piece('padati', 'black', 'd7'),
  piece('ashva', 'white', 'b1'), piece('gaja', 'black', 'c8'), piece('ratha', 'white', 'h1'), piece('mantri', 'white', 'd1'),
])

/** This constructs explicit UI snapshots; it does not pretend to implement rules. */
function fixtureMove(before: BoardState, from: SquareId, to: SquareId, promote?: PieceKind): BoardState {
  const attacker = before.pieces.find(candidate => candidate.square === from)
  if (!attacker) throw new Error(`Fixture has no piece at ${from}`)
  const victim = before.pieces.find(candidate => candidate.square === to)
  const ply = before.moveHistory.length + 1
  const move: Move = { id: `move-${ply}`, ply, side: attacker.side, piece: attacker.kind, from, to,
    notation: `${from}${victim ? 'x' : '–'}${to}`, ...(victim ? { captured: victim.kind } : {}) }
  const pieces = before.pieces.filter(candidate => candidate.square !== from && candidate.square !== to)
  pieces.push(piece(promote ?? attacker.kind, attacker.side, to))
  return state(pieces, [...before.moveHistory, move], attacker.side === 'white' ? 'black' : 'white')
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}
const frame = () => new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
const delay = (ms: number) => new Promise<void>(resolve => window.setTimeout(resolve, ms))
async function until(predicate: () => boolean, timeout: number, message: string) {
  const start = performance.now()
  while (!predicate()) {
    if (performance.now() - start > timeout) throw new Error(message)
    await frame()
  }
  await frame()
}
const canvas = () => stage.querySelector<HTMLCanvasElement>('canvas')
const visualPly = () => Number(canvas()?.dataset.visualPly ?? -1)
const moveEvents = () => events.filter(event => event.busy && event.move)

function send(next: BoardState, exploded = settings?.exploded ?? false, reserve = true) {
  assert(scene, 'Scene must exist')
  if (reserve) reservedBusy = true // mirrors App reserving the visual turn before setting state
  settings = { state: next, orientation: 'white', selectedSquare: null, legalMoves: [], disabled: false,
    inspection: null, exploded, focus: 'overview', channel: -1, replay: false, resetCamera: 0 }
  scene.update(settings)
}
async function prepare(next: BoardState) {
  // Inspection mode hydrates a fixture immediately, avoiding deployment between cases.
  send(next, true)
  send(next, false, false)
  await frame()
  await frame()
  assert(!reservedBusy, 'Fixture hydration must release the reserved turn')
  events = []
}

async function check(name: string, action: () => Promise<string>) {
  const row = document.createElement('li')
  row.dataset.result = 'running'
  const title = document.createElement('strong'); title.textContent = `… ${name}`
  const detail = document.createElement('small'); detail.textContent = 'Running against the real scene…'
  row.append(title, detail); results.append(row)
  try {
    const explanation = await action()
    row.dataset.result = 'pass'; title.textContent = `✓ ${name}`; detail.textContent = explanation
    reports.push({ name, passed: true, detail: explanation })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    row.dataset.result = 'fail'; title.textContent = `✕ ${name}`; detail.textContent = message
    reports.push({ name, passed: false, detail: message })
  }
}

async function run() {
  if (running) return
  running = true; runButton.disabled = true; runButton.dataset.status = 'running'
  reports.length = 0; runtimeErrors.length = 0; failures = 0; results.replaceChildren()
  summary.textContent = 'Checking local scene lifecycle fixtures…'
  scene?.dispose(); scene = null; settings = null; events = []; reservedBusy = false
  const started = performance.now(), reduced = motionPreference.matches
  try {
    scene = createBoardScene(stage, () => {}, () => {}, () => {}, () => { failures++ }, (busy, move) => {
      reservedBusy = busy; events.push({ busy, move: move ?? null, time: performance.now() })
    })
    await check('Initial scene releases the reserved turn', async () => {
      send(baseline())
      if (reduced) assert(!reservedBusy && events.at(-1)?.busy === false, 'Reduced-motion initial hydration failed to emit false')
      else assert(reservedBusy && events.at(-1)?.busy === true, 'Normal deployment must announce busy')
      await until(() => !reservedBusy, 1400, 'Initial deployment never released busy')
      assert(visualPly() === 0, 'Initial visual ply should be zero')
      return reduced ? 'Initial hydration emits false immediately; App-style reservation is released.' : 'Deployment starts busy and completes with false.'
    })

    await check('Rapid capture + reply stay in visual order', async () => {
      const initial = baseline(); await prepare(initial)
      const capture = fixtureMove(initial, 'e4', 'f5'), reply = fixtureMove(capture, 'd7', 'd6')
      send(capture); await delay(70); send(reply); await frame()
      if (!reduced) {
        assert(reservedBusy, 'Capture/reply queue must remain busy')
        assert(visualPly() === 1, 'Rapid reply should wait behind the capture')
        assert(moveEvents().map(event => event.move!.id).join(',') === 'move-1', 'Reply callback occurred before capture finished')
      } else assert(!reservedBusy, 'Reduced-motion reply failed to release its reservation')
      await until(() => !reservedBusy, 2600, 'Queued reply never completed')
      assert(visualPly() === 2, 'Final visual ply must include the reply')
      if (!reduced) {
        assert(moveEvents().map(event => event.move!.id).join(',') === 'move-1,move-2', 'Capture and reply must each announce in sequence')
        assert(events.slice(0, -1).every(event => event.busy), 'Queue released the turn between consecutive animations')
      } else assert(events.every(event => !event.busy), 'Reduced motion must not queue choreography')
      return reduced ? 'Both snapshots settle immediately and release the lock.' : 'Capture finishes before reply; one continuous busy interval reaches ply 2.'
    })

    await check('Reset cancels capture and its queued reply', async () => {
      const initial = baseline(); await prepare(initial)
      const capture = fixtureMove(initial, 'e4', 'f5')
      send(capture); send(fixtureMove(capture, 'd7', 'd6')); await delay(90)
      const resetAt = performance.now(); send(baseline())
      await until(() => !reservedBusy, 1400, 'Reset deployment never settled')
      assert(visualPly() === 0, 'Reset must return to visual ply zero')
      assert(events.filter(event => event.time >= resetAt).every(event => event.move === null), 'Old capture/reply metadata survived reset')
      return 'New fixture replaces actors and clears pending moves; no old move resumes.'
    })

    await check('Inspector entry settles an active capture', async () => {
      const initial = baseline(); await prepare(initial)
      const capture = fixtureMove(initial, 'e4', 'f5'), reply = fixtureMove(capture, 'd7', 'd6')
      send(capture); send(reply); await delay(80)
      send(reply, true, false)
      assert(!reservedBusy && events.at(-1)?.busy === false, 'Inspector entry must emit false immediately')
      await frame(); assert(visualPly() === 2, 'Inspector must hydrate the newest snapshot')
      send(reply, false, false); await frame()
      assert(!reservedBusy, 'Returning from inspection must not restart the capture')
      return 'Inspection flushes the queue to the newest position and releases the visual turn.'
    })

    await check('Promotion capture retains Padati choreography', async () => {
      const initial = state([piece('raja', 'white', 'a1'), piece('raja', 'black', 'h8'),
        piece('padati', 'white', 'e7'), piece('ratha', 'black', 'f8'), piece('padati', 'black', 'd7')])
      await prepare(initial)
      const promotion = fixtureMove(initial, 'e7', 'f8', 'mantri')
      const reply = fixtureMove(promotion, 'd7', 'd6'), advisorMove = fixtureMove(reply, 'f8', 'e7')
      send(promotion); send(reply); send(advisorMove)
      await until(() => !reservedBusy, 3300, 'Promotion + following moves never settled')
      assert(visualPly() === 3, 'Promoted actor must survive the following snapshots')
      if (!reduced) {
        const sequence = moveEvents().map(event => `${event.move!.id}:${event.move!.piece}`).join(',')
        assert(sequence === 'move-1:padati,move-2:padati,move-3:mantri', `Unexpected promotion choreography: ${sequence}`)
        assert(moveEvents()[0].move?.captured === 'ratha', 'Promotion capture must retain victim metadata')
      } else assert(events.every(event => !event.busy), 'Reduced-motion promotion left choreography active')
      return reduced ? 'Promoted snapshots hydrate immediately and all reservations release.' : 'Padati captures Ratha, becomes Mantri, and its later Mantri move animates successfully.'
    })

    await check('Disposal releases busy and removes the canvas', async () => {
      const initial = baseline(); await prepare(initial)
      send(fixtureMove(initial, 'e4', 'f5')); await delay(70)
      const ownedScene = scene!
      ownedScene.dispose(); scene = null
      assert(!reservedBusy, 'Disposal must release the active visual turn')
      assert(!canvas(), 'Disposed renderer canvas still exists')
      const eventCount = events.length; ownedScene.dispose(); await delay(80)
      assert(events.length === eventCount, 'Repeated disposal or retired timers emitted another callback')
      assert(failures === 0 && runtimeErrors.length === 0, `Runtime failure: ${runtimeErrors.join('; ') || `${failures} WebGL failures`}`)
      return 'Cleanup is idempotent; canvas, animation loop, and busy state are released.'
    })
  } catch (error) {
    runtimeErrors.push(error instanceof Error ? error.message : String(error))
  } finally {
    scene?.dispose(); scene = null
    const failed = reports.filter(report => !report.passed).length + runtimeErrors.length + failures
    const elapsed = ((performance.now() - started) / 1000).toFixed(1)
    const success = failed === 0 && reports.length === 6
    runButton.dataset.status = success ? 'passed' : 'failed'
    summary.textContent = success ? `All 6 lifecycle checks passed in ${elapsed}s (${reduced ? 'reduced' : 'normal'} motion).` : `${reports.filter(report => report.passed).length}/6 checks passed in ${elapsed}s. ${runtimeErrors.join('; ')}`
    running = false; runButton.disabled = false
  }
}
runButton.addEventListener('click', () => { void run() })
window.addEventListener('pagehide', () => scene?.dispose())
