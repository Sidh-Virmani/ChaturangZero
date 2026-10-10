import type { PieceKind } from '../types/game'

// One clock coordinates approach, contact, defeat and the final settle.
export const MOVE_MS = 680
export const IMPACT_MS = 620
export const CAPTURE_MS = 1450
export const DEPLOY_MS = 760
const clamp = (value: number) => Math.max(0, Math.min(1, value))
const smooth = (t: number) => t * t * (3 - 2 * t)

export interface MovePose {
  progress: number
  lift: number
  pitch: number
  yaw: number
  roll: number
  scale: number
}

/** Pure, time-based poses: frame rate never changes the move or its endpoint. */
export function movePose(kind: PieceKind, elapsed: number, capture: boolean, reduced = false): MovePose {
  const duration = capture ? CAPTURE_MS : MOVE_MS
  if (reduced || elapsed >= duration) return { progress: 1, lift: 0, pitch: 0, yaw: 0, roll: 0, scale: 1 }
  const travel = clamp(elapsed / (capture ? IMPACT_MS : MOVE_MS))
  const wave = Math.sin(Math.PI * travel)
  const settle = capture ? Math.sin(Math.PI * clamp((elapsed - IMPACT_MS) / 360)) : 0
  const power = capture ? 1 : .65
  const pose: MovePose = { progress: smooth(travel), lift: .11 * wave, pitch: 0, yaw: 0, roll: 0, scale: 1 }
  switch (kind) {
    case 'ashva':
      pose.lift = (capture ? 1.12 : .56) * wave
      pose.pitch = -.16 * wave * power
      pose.roll = .09 * wave
      break
    case 'gaja':
      pose.lift = .3 * wave * power + .075 * settle
      pose.pitch = -.07 * wave + .065 * settle
      break
    case 'ratha':
      pose.progress = 1 - (1 - travel) ** 3
      pose.lift = .045 * wave
      pose.pitch = -.11 * wave * power
      break
    case 'mantri':
      pose.lift = .42 * wave * power + .055 * settle
      pose.yaw = (capture ? Math.PI * 2 : .18) * smooth(travel)
      // A full turn lands in exactly the same orientation.
      if (travel === 1) pose.yaw = 0
      break
    case 'raja':
      pose.lift = .24 * wave * power
      pose.yaw = .17 * wave
      break
    case 'padati':
      pose.lift = .23 * wave * power
      pose.pitch = -.2 * wave * power + .045 * settle
      break
  }
  return pose
}

export function defeatPose(kind: PieceKind, elapsed: number, reduced = false) {
  const t = reduced ? 1 : clamp((elapsed - IMPACT_MS) / 570)
  const amount = smooth(t)
  const force = kind === 'gaja' ? .48 : kind === 'ratha' ? .35 : kind === 'ashva' ? .27 : .16
  return { progress: t, scale: Math.max(0, 1 - amount), lift: Math.sin(Math.PI * t) * .2 - amount * .15,
    displacement: amount * force, tilt: amount * (kind === 'mantri' || kind === 'raja' ? .25 : 1.15),
    spin: amount * (kind === 'mantri' ? Math.PI * 1.3 : .38) }
}

/** Resets / jumps must never be mistaken for a capture, even with reused IDs. */
export function isSequentialMove(previous: { moveHistory: { id: string; from: string; to: string; piece: PieceKind; side: string }[] },
  next: { moveHistory: { id: string; from: string; to: string; piece: PieceKind; side: string }[]; lastMove: { id: string } | null }) {
  if (!next.lastMove || next.moveHistory.length !== previous.moveHistory.length + 1 || next.moveHistory.at(-1)?.id !== next.lastMove.id) return false
  return previous.moveHistory.every((move, index) => {
    const other = next.moveHistory[index]
    return move.id === other.id && move.from === other.from && move.to === other.to && move.side === other.side && move.piece === other.piece
  })
}
