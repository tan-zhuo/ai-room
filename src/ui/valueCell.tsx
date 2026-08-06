import { CSSProperties } from 'react'

export function fmt(v: number, digits = 3): string {
  return v.toFixed(digits)
}

/** Background tint for a numeric cell: cyan positive, orange negative. */
export function cellStyle(v: number, scale: number): CSSProperties {
  const t = Math.min(1, Math.abs(v) / (scale || 1))
  const alpha = 0.08 + 0.5 * t
  return {
    background: v >= 0 ? `rgba(56, 214, 255, ${alpha * 0.55})` : `rgba(255, 138, 60, ${alpha * 0.55})`,
  }
}

export function NumGrid({
  values,
  scale,
  highlight,
  digits = 2,
}: {
  values: number[][]
  scale: number
  highlight?: { row: number; col: number }
  digits?: number
}) {
  return (
    <div className="num-grid" style={{ gridTemplateColumns: `repeat(${values[0].length}, 1fr)` }}>
      {values.map((row, r) =>
        row.map((v, c) => (
          <span
            key={`${r}-${c}`}
            className={`num-cell${highlight && highlight.row === r && highlight.col === c ? ' hl' : ''}`}
            style={cellStyle(v, scale)}
          >
            {fmt(v, digits)}
          </span>
        )),
      )}
    </div>
  )
}
