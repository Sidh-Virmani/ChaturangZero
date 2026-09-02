import type { ReactNode } from 'react'
import type { PieceKind, Side } from '../types/game'

interface PieceGlyphProps {
  kind: PieceKind
  side: Side
  className?: string
}

function GlyphFrame({
  children,
  label,
  className,
}: {
  children: ReactNode
  label: string
  className?: string
}) {
  return (
    <svg
      className={className ?? 'piece__glyph'}
      viewBox="0 0 64 64"
      role="img"
      aria-hidden="true"
      focusable="false"
    >
      <title>{label}</title>
      {children}
    </svg>
  )
}

function Padati() {
  return (
    <>
      <ellipse cx="32" cy="54" rx="16" ry="5.5" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <path
        d="M18 50c3-9 7-16 14-16s11 7 14 16"
        fill="var(--piece-fill)"
        stroke="var(--piece-stroke)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <circle cx="32" cy="28" r="9.5" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <circle cx="32" cy="16" r="5" fill="var(--piece-accent)" stroke="var(--piece-stroke)" strokeWidth="1.6" />
    </>
  )
}

function Ashva() {
  return (
    <>
      <ellipse cx="32" cy="54" rx="17" ry="5.5" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <path
        d="M20 50c1-12 6-18 14-18 3 0 4 3 7 3 6 0 11-7 14-11 1 6-1 12-6 15 3 2 5 6 5 11H20z"
        fill="var(--piece-fill)"
        stroke="var(--piece-stroke)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <path
        d="M29 22c2-8 9-12 16-10 2 4-1 8-5 9 5 1 8 5 8 9-6-1-11-4-14-9-3 2-5 5-7 5-1-2 0-4 2-4z"
        fill="var(--piece-fill)"
        stroke="var(--piece-stroke)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <circle cx="42" cy="18" r="1.6" fill="var(--piece-stroke)" />
      <path d="M46 14c3-1 6 1 7 4" fill="none" stroke="var(--piece-accent)" strokeWidth="1.8" strokeLinecap="round" />
    </>
  )
}

function Gaja() {
  return (
    <>
      <ellipse cx="32" cy="54" rx="18" ry="5.5" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <path
        d="M16 50c2-14 8-22 16-22s14 8 16 22"
        fill="var(--piece-fill)"
        stroke="var(--piece-stroke)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <ellipse cx="32" cy="26" rx="13" ry="11" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <path
        d="M19 26c-6-2-9 2-8 8 4 1 7-2 8-6zM45 26c6-2 9 2 8 8-4 1-7-2-8-6z"
        fill="var(--piece-fill)"
        stroke="var(--piece-stroke)"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path
        d="M40 30c8 4 10 14 6 20"
        fill="none"
        stroke="var(--piece-stroke)"
        strokeWidth="3.2"
        strokeLinecap="round"
      />
      <circle cx="28" cy="24" r="1.7" fill="var(--piece-stroke)" />
      <rect x="27" y="12" width="10" height="6" rx="1.5" fill="var(--piece-accent)" stroke="var(--piece-stroke)" strokeWidth="1.4" />
    </>
  )
}

function Ratha() {
  return (
    <>
      <ellipse cx="32" cy="54" rx="18" ry="5.5" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <rect x="14" y="40" width="36" height="12" rx="2" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <rect x="18" y="22" width="28" height="18" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <path d="M18 22h28l-4-8H22z" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" strokeLinejoin="round" />
      <path d="M22 14v-4h4v4M30 14v-6h4v6M38 14v-4h4v4" fill="var(--piece-accent)" stroke="var(--piece-stroke)" strokeWidth="1.6" />
      <circle cx="20" cy="52" r="4.2" fill="var(--piece-accent)" stroke="var(--piece-stroke)" strokeWidth="1.6" />
      <circle cx="44" cy="52" r="4.2" fill="var(--piece-accent)" stroke="var(--piece-stroke)" strokeWidth="1.6" />
    </>
  )
}

function Mantri() {
  return (
    <>
      <ellipse cx="32" cy="54" rx="16" ry="5.5" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <path
        d="M20 50c2-16 6-24 12-24s10 8 12 24"
        fill="var(--piece-fill)"
        stroke="var(--piece-stroke)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <path
        d="M32 8l10 16H22z"
        fill="var(--piece-fill)"
        stroke="var(--piece-stroke)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <rect x="24" y="24" width="16" height="8" rx="1" fill="var(--piece-accent)" stroke="var(--piece-stroke)" strokeWidth="1.6" />
      <circle cx="32" cy="7" r="3.2" fill="var(--piece-accent)" stroke="var(--piece-stroke)" strokeWidth="1.5" />
    </>
  )
}

function Raja() {
  return (
    <>
      <ellipse cx="32" cy="54" rx="17" ry="5.5" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <path
        d="M18 50c3-16 7-24 14-24s11 8 14 24"
        fill="var(--piece-fill)"
        stroke="var(--piece-stroke)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <rect x="22" y="22" width="20" height="8" rx="1.5" fill="var(--piece-fill)" stroke="var(--piece-stroke)" strokeWidth="2" />
      <path
        d="M20 22l4-12 8 8 8-8 4 12H20z"
        fill="var(--piece-fill)"
        stroke="var(--piece-stroke)"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <path d="M32 4v8M28 8h8" stroke="var(--piece-accent)" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="24" cy="10" r="2.2" fill="var(--piece-accent)" />
      <circle cx="40" cy="10" r="2.2" fill="var(--piece-accent)" />
    </>
  )
}

const glyphs: Record<PieceKind, () => ReactNode> = {
  padati: Padati,
  ashva: Ashva,
  gaja: Gaja,
  ratha: Ratha,
  mantri: Mantri,
  raja: Raja,
}

export function PieceGlyph({ kind, side, className }: PieceGlyphProps) {
  const Glyph = glyphs[kind]
  return (
    <GlyphFrame className={className} label={`${side} ${kind}`}>
      <Glyph />
    </GlyphFrame>
  )
}
