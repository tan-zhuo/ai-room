const base = { width: 15, height: 15, viewBox: '0 0 24 24', fill: 'currentColor' as const }

export const IconPlay = () => (
  <svg {...base}>
    <path d="M8 5v14l11-7z" />
  </svg>
)
export const IconPause = () => (
  <svg {...base}>
    <path d="M6 5h4v14H6zM14 5h4v14h-4z" />
  </svg>
)
export const IconPrev = () => (
  <svg {...base}>
    <path d="M6 6h2v12H6zM18 6l-8.5 6L18 18z" />
  </svg>
)
export const IconNext = () => (
  <svg {...base}>
    <path d="M16 6h2v12h-2zM6 6l8.5 6L6 18z" />
  </svg>
)
export const IconReset = () => (
  <svg {...base}>
    <path d="M12 5V2L7 6l5 4V7a5 5 0 1 1-5 5H5a7 7 0 1 0 7-7z" />
  </svg>
)
export const IconShuffle = () => (
  <svg {...base}>
    <path d="M17.65 6.35A8 8 0 1 0 20 12h-2a6 6 0 1 1-1.76-4.24L13 11h7V4z" />
  </svg>
)
