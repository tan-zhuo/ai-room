import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { totalSteps, useStore, useT } from '../store'
import { GridSlot, diffSlot, gridPos, slotLabelAnchor } from './layout'
import { Segment } from './common'
import { GridNodes } from './GridNodes'
import { FlowParticles } from './FlowParticles'
import { LayerLabel } from './Labels'
import { SelectionMarker } from './SelectionMarker'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

function toGrid(flat: number[], n: number): Tensor3 {
  return [Array.from({ length: n }, (_, r) => flat.slice(r * n, (r + 1) * n))]
}

/** DDPM: pure noise is denoised step by step — every playback step is one
 *  reverse-diffusion iteration. x_t → denoiser hidden → x̂₀ guess → x_{t-1}. */
export function DiffusionScene() {
  const trace = useStore((s) => s.diffTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const task = MODELS.diff
  const model = task.model

  const xSlot = diffSlot(-1) as GridSlot
  const hSlot = diffSlot(0) as GridSlot
  const predSlot = diffSlot(1) as GridSlot
  const nextSlot = diffSlot(2) as GridSlot

  const total = totalSteps('diff')
  const si = Math.min(step, total - 1)
  const cur = trace.steps[si]
  const done = step >= total

  const xVals = useMemo(() => toGrid(trace.xs[Math.min(step, total)], task.n), [trace, step, total, task.n])
  const hVals = useMemo(() => {
    const padded = [...cur.h2.a]
    while (padded.length < hSlot.rows * hSlot.cols) padded.push(0)
    return [
      Array.from({ length: hSlot.rows }, (_, r) => padded.slice(r * hSlot.cols, (r + 1) * hSlot.cols)),
    ]
  }, [cur, hSlot])
  const predVals = useMemo(() => toGrid(cur.pred.a, task.n), [cur, task.n])
  const nextVals = useMemo(
    () => toGrid(trace.xs[Math.min(step + 1, total)], task.n),
    [trace, step, total, task.n],
  )

  const flows = useMemo(() => {
    const mk = (src: GridSlot, dst: GridSlot, stride: number): Segment[] => {
      const segs: Segment[] = []
      for (let r = 0; r < dst.rows; r++)
        for (let c = 0; c < dst.cols; c += stride)
          segs.push({
            a: v(gridPos(src, 0, Math.min(r, src.rows - 1), Math.min(c, src.cols - 1))),
            b: v(gridPos(dst, 0, r, c)),
            w: 0.5,
            target: 0,
          })
      return segs
    }
    return [mk(xSlot, hSlot, 1), mk(hSlot, predSlot, 1), mk(predSlot, nextSlot, 1)]
  }, [xSlot, hSlot, predSlot, nextSlot])

  const tNow = done ? 0 : cur.t

  return (
    <group>
      <GridNodes slot={xSlot} values={xVals} scale={2} alwaysOn />
      <GridNodes slot={hSlot} values={hVals} scale={Math.max(...cur.h2.a.map(Math.abs), 1)} alwaysOn />
      <GridNodes slot={predSlot} values={predVals} scale={1} alwaysOn />
      <GridNodes slot={nextSlot} values={nextVals} scale={2} alwaysOn />

      {flows.map((segs, k) => (
        <FlowParticles key={`p${k}`} segments={segs} layerIndex={k} size={0.055} continuous />
      ))}

      {/* live timestep badge */}
      <Html
        position={[xSlot.x, ((xSlot.rows - 1) / 2) * xSlot.cell + 2.5, 0]}
        center
        zIndexRange={[20, 0]}
        style={{ pointerEvents: 'none' }}
      >
        <div className="next-char-banner">
          t = <b>{tNow}</b> / {model.T}
        </div>
      </Html>
      {done && (
        <Html
          position={[nextSlot.x, ((nextSlot.rows - 1) / 2) * nextSlot.cell + 2.5, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="next-char-banner">✨ {t('diff.doneBadge')}</div>
        </Html>
      )}

      <LayerLabel position={slotLabelAnchor(xSlot)} title={t('layer.xt')} sub="x_t" layer={-1} />
      <LayerLabel position={slotLabelAnchor(hSlot)} title={t('layer.denoiser')} sub={`MLP + t-embed`} layer={0} />
      <LayerLabel position={slotLabelAnchor(predSlot)} title={t('layer.x0hat')} sub="x̂₀ = f(x_t, t)" layer={1} />
      <LayerLabel
        position={slotLabelAnchor(nextSlot)}
        title={t('layer.ddpmStep')}
        sub="c₀·x̂₀ + cₜ·x_t + σ·z"
        layer={2}
      />

      <SelectionMarker />
    </group>
  )
}
