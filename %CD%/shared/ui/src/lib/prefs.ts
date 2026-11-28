import type { Side } from '../types/game'

export type ThemeName = 'dark' | 'light'
export type PlayMode = 'human' | 'ai'
export type Difficulty = 'easy' | 'medium' | 'hard'

export interface UiPrefs {
  theme: ThemeName
  sound: boolean
  orientation: Side
  playMode: PlayMode
  difficulty: Difficulty
}

const KEY = 'chaturangzero-ui-prefs'

const defaults: UiPrefs = {
  theme: 'dark',
  sound: true,
  orientation: 'white',
  playMode: 'human',
  difficulty: 'medium',
}

export function loadPrefs(): UiPrefs {
  try {
    const raw = window.localStorage.getItem(KEY)
    if (!raw) return { ...defaults }
    const parsed = JSON.parse(raw) as Partial<UiPrefs>
    return { ...defaults, ...parsed }
  } catch {
    return { ...defaults }
  }
}

export function savePrefs(next: UiPrefs): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(next))
  } catch {
    // Ignore quota / private-mode failures.
  }
}

export function applyTheme(theme: ThemeName): void {
  document.documentElement.dataset.theme = theme
  const meta = document.querySelector('meta[name="theme-color"]')
  if (meta) {
    meta.setAttribute('content', theme === 'light' ? '#efe8d4' : '#151815')
  }
}
