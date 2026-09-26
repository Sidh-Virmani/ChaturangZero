import type { PieceKind, Side } from '../types/game'

export function PieceIcon({ kind, side = 'white', className = '' }: { kind: PieceKind; side?: Side; className?: string }) {
  return <svg viewBox="0 0 64 72" className={`piece-icon ${className} ${side}`} role="img" aria-label={kind}>
    <g fill="currentColor" stroke="var(--piece-line, #a58a59)" strokeWidth="1.3" strokeLinejoin="round">
      <path d="M17 59h30l5 7H12zM20 54h24l3 5H17z" />
      {kind === 'padati' && <><path d="M24 32h16l-3 9 7 13H20l7-13z" /><circle cx="32" cy="24" r="11" /><path d="M20 23c0-17 24-17 24 0z" /><path d="M30 12l2-7 2 7" /></>}
      {kind === 'raja' && <><path d="M24 35h16l-4 8 8 11H20l8-11z" /><path d="M18 19l8 7 6-15 6 15 8-7-5 16H23z" /><circle cx="32" cy="9" r="3" /><path d="M22 36h20v5H22z" /></>}
      {kind === 'mantri' && <><path d="M23 33h18l-5 11 8 10H20l8-10z" /><path d="M32 9c3 9 13 12 13 22 0 9-26 9-26 0 0-10 10-13 13-22z" /><path d="M21 29h22" /><circle cx="32" cy="9" r="3" /></>}
      {kind === 'ratha' && <><path d="M19 17v17l6 5-3 15h20l-3-15 6-5V17h-6v9h-5v-9h-5v9h-5v-9z" /><circle cx="23" cy="45" r="5" /><circle cx="41" cy="45" r="5" /></>}
      {kind === 'ashva' && <><path d="M19 54c0-10 2-17 9-25l-8 4-6-6 13-15 4-7 4 8c16 8 13 26 11 41z" /><circle cx="30" cy="21" r="1.6" fill="#182c29" /><path d="M39 18c7 9 6 21 3 28" fill="none" /></>}
      {kind === 'gaja' && <><path d="M23 41l-3 13h24l-3-13" /><ellipse cx="19" cy="29" rx="9" ry="13" /><ellipse cx="45" cy="29" rx="9" ry="13" /><path d="M20 25c0-20 24-20 24 0v10l-7 4v7c0 10-13 10-13 2l6-2V32" /><path d="M22 33l-4 12 8-7m15-5 5 12-8-7" fill="#f4e6c6" /><path d="M22 16h20" /><circle cx="25" cy="27" r="1.5" fill="#182c29" /><circle cx="39" cy="27" r="1.5" fill="#182c29" /></>}
    </g>
  </svg>
}
