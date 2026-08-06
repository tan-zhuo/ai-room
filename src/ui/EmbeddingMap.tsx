import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { MODELS } from '../nn/models'
import { useT } from '../store'

/** Top-2 principal components via power iteration on X^T X. */
function pca2(rows: number[][]): { x: number; y: number }[] {
  const n = rows.length
  const d = rows[0].length
  const mean = Array(d).fill(0)
  rows.forEach((r) => r.forEach((v, i) => (mean[i] += v / n)))
  const X = rows.map((r) => r.map((v, i) => v - mean[i]))
  const mul = (v: number[]) => {
    const t = X.map((row) => row.reduce((s, rv, i) => s + rv * v[i], 0))
    const out = Array(d).fill(0)
    X.forEach((row, k) => row.forEach((rv, i) => (out[i] += rv * t[k])))
    return out
  }
  const norm = (v: number[]) => {
    const m = Math.hypot(...v) || 1
    return v.map((x) => x / m)
  }
  let v1 = norm(Array.from({ length: d }, (_, i) => Math.sin(i + 1)))
  for (let i = 0; i < 50; i++) v1 = norm(mul(v1))
  let v2 = norm(Array.from({ length: d }, (_, i) => Math.cos(i * 1.7 + 2)))
  for (let i = 0; i < 50; i++) {
    let w = mul(v2)
    const dp = w.reduce((s, x, k) => s + x * v1[k], 0)
    w = w.map((x, k) => x - dp * v1[k])
    v2 = norm(w)
  }
  return X.map((r) => ({
    x: r.reduce((s, v, i) => s + v * v1[i], 0),
    y: r.reduce((s, v, i) => s + v * v2[i], 0),
  }))
}

const VOWELS = 'aeiou'

/** PCA scatter of the transformer's learned character embeddings.
 *  Rendered in a portal (true viewport centering) and draggable by its header. */
export function EmbeddingMap({ onClose }: { onClose: () => void }) {
  const t = useT()
  const model = MODELS.llm.model
  const [offset, setOffset] = useState({ x: 0, y: 0 })
  const drag = useRef<{ px: number; py: number; ox: number; oy: number } | null>(null)

  useEffect(() => {
    const move = (e: PointerEvent) => {
      const d = drag.current
      if (!d) return
      setOffset({ x: d.ox + e.clientX - d.px, y: d.oy + e.clientY - d.py })
    }
    const up = () => (drag.current = null)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [])

  const points = useMemo(() => {
    const raw = pca2(model.E)
    const xs = raw.map((p) => p.x)
    const ys = raw.map((p) => p.y)
    const sx = Math.max(...xs.map(Math.abs)) || 1
    const sy = Math.max(...ys.map(Math.abs)) || 1
    return raw.map((p, i) => ({
      x: 160 + (p.x / sx) * 135,
      y: 160 - (p.y / sy) * 135,
      ch: model.vocab[i],
    }))
  }, [model])

  const colorOf = (ch: string) => {
    if (ch === ' ' || ch === '.') return '#ff8a3c'
    if (VOWELS.includes(ch)) return '#38d6ff'
    return '#d9e4f2'
  }

  return createPortal(
    <div className="modal-backdrop floating" onClick={onClose}>
      <div
        className="modal embed-modal"
        style={{ transform: `translate(${offset.x}px, ${offset.y}px)` }}
        onClick={(e) => e.stopPropagation()}
      >
        <header
          className="drag-handle"
          onPointerDown={(e) => {
            drag.current = { px: e.clientX, py: e.clientY, ox: offset.x, oy: offset.y }
          }}
        >
          <h3>{t('llm.embedMap')}</h3>
          <button className="icon-btn" onClick={onClose}>
            ×
          </button>
        </header>
        <svg viewBox="0 0 320 320" className="embed-svg">
          <line x1="160" y1="8" x2="160" y2="312" stroke="rgba(97,168,255,0.12)" />
          <line x1="8" y1="160" x2="312" y2="160" stroke="rgba(97,168,255,0.12)" />
          {points.map((p) => (
            <text
              key={p.ch}
              x={p.x}
              y={p.y}
              fill={colorOf(p.ch)}
              fontSize="15"
              fontFamily="ui-monospace, monospace"
              fontWeight="700"
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {p.ch === ' ' ? '␣' : p.ch}
            </text>
          ))}
        </svg>
        <p className="muted">{t('llm.embedMapNote')}</p>
      </div>
    </div>,
    document.body,
  )
}
