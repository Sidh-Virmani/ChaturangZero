import { FILES, RANKS, type File, type Rank, type Side, type SquareId } from '../types/game'

export function squareId(file: File, rank: Rank): SquareId {
  return `${file}${rank}`
}

export function parseSquare(id: SquareId): { file: File; rank: Rank } {
  return { file: id[0] as File, rank: Number(id.slice(1)) as Rank }
}

export function displayFiles(orientation: Side): readonly File[] {
  return orientation === 'white' ? FILES : [...FILES].reverse()
}

export function displayRanks(orientation: Side): readonly Rank[] {
  return orientation === 'white' ? RANKS : [...RANKS].reverse()
}

export function squareTone(file: File, rank: Rank): 'light' | 'dark' {
  return (FILES.indexOf(file) + rank) % 2 === 0 ? 'light' : 'dark'
}

export function shiftSquare(
  id: SquareId,
  fileDelta: number,
  rankDelta: number,
): SquareId | null {
  const { file, rank } = parseSquare(id)
  const nextFile = FILES[FILES.indexOf(file) + fileDelta]
  const nextRank = rank + rankDelta
  if (!nextFile || nextRank < 1 || nextRank > 8) return null
  return squareId(nextFile, nextRank as Rank)
}

/** Arrow keys move in screen space; orientation decides which way rank/file increase. */
export function neighborInView(
  id: SquareId,
  key: 'ArrowUp' | 'ArrowDown' | 'ArrowLeft' | 'ArrowRight',
  orientation: Side,
): SquareId | null {
  const dir = orientation === 'white' ? 1 : -1
  switch (key) {
    case 'ArrowRight':
      return shiftSquare(id, dir, 0)
    case 'ArrowLeft':
      return shiftSquare(id, -dir, 0)
    case 'ArrowUp':
      return shiftSquare(id, 0, dir)
    case 'ArrowDown':
      return shiftSquare(id, 0, -dir)
  }
}
