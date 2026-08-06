import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS, getVAETask } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { maxAbs } from '../nn/mlp'
import { totalSteps, useStore, useT } from '../store'
import { GridSlot, VecSlot, aeSlot, gridPos, slotLabelAnchor, vecPos } from './layout'
import { Segment } from './common'
import { GridNodes } from './GridNodes'
import { VectorNodes } from './VectorNodes'
import { Connections } from './Connections'
import { FlowParticles } from './FlowParticles'
import { LayerLabel } from './Labels'
import { SelectionMarker } from './SelectionMarker'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

function topSegments(segs: Segment[], cap: number): Segment[] {
  if (segs.length <= cap) return segs
  return [...segs].sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, cap)
}

/** VAE flavour: encoder → (μ, log σ²) → sampled z → decoder → reconstruction. */
function VAESceneInner() {
  const input = useStore((s) => s.aeInput)
  const trace = useStore((s) => s.vaeTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const task = getVAETask()
  const model = task.model

  const inSlot = aeSlot(-1) as GridSlot
  const encSlot = aeSlot(0) as VecSlot
  const msSlot = aeSlot(1) as GridSlot
  const zSlot = aeSlot(2) as VecSlot
  const decSlot = aeSlot(3) as VecSlot
  const outSlot = aeSlot(4) as GridSlot

  const encPositions = useMemo(
    () => Array.from({ length: encSlot.size }, (_, i) => vecPos(encSlot, i)),
    [encSlot],
  )
  const zPositions = useMemo(
    () => Array.from({ length: zSlot.size }, (_, i) => vecPos(zSlot, i)),
    [zSlot],
  )
  const decPositions = useMemo(
    () => Array.from({ length: decSlot.size }, (_, i) => vecPos(decSlot, i)),
    [decSlot],
  )

  const outVals = useMemo<Tensor3>(() => {
    const a = trace?.o.a ?? Array.from({ length: task.n * task.n }, () => 0)
    return [Array.from({ length: task.n }, (_, r) => a.slice(r * task.n, (r + 1) * task.n))]
  }, [trace, task.n])
  const msVals = useMemo<Tensor3>(
    () => [
      Array.from({ length: task.latent }, (_, r) => [trace?.mu[r] ?? 0, trace?.logvar[r] ?? 0]),
    ],
    [trace, task.latent],
  )

  const segs = useMemo(() => {
    const enc: Segment[] = []
    model.enc.weights.forEach((row, j) =>
      row.forEach((w, i) =>
        enc.push({
          a: v(gridPos(inSlot, 0, Math.floor(i / task.n), i % task.n)),
          b: v(encPositions[j]),
          w,
          target: j,
        }),
      ),
    )
    const ms: Segment[] = []
    for (let r = 0; r < task.latent; r++) {
      model.muHead.weights[r].forEach((w, i) =>
        ms.push({ a: v(encPositions[i]), b: v(gridPos(msSlot, 0, r, 0)), w, target: 0 }),
      )
      model.lvHead.weights[r].forEach((w, i) =>
        ms.push({ a: v(encPositions[i]), b: v(gridPos(msSlot, 0, r, 1)), w, target: 0 }),
      )
    }
    const z: Segment[] = []
    for (let r = 0; r < task.latent; r++) {
      z.push({ a: v(gridPos(msSlot, 0, r, 0)), b: v(zPositions[r]), w: 0.8, target: r })
      z.push({ a: v(gridPos(msSlot, 0, r, 1)), b: v(zPositions[r]), w: -0.8, target: r })
    }
    const dec: Segment[] = []
    model.dec.weights.forEach((row, j) =>
      row.forEach((w, i) => dec.push({ a: v(zPositions[i]), b: v(decPositions[j]), w, target: j })),
    )
    const out: Segment[] = []
    model.out.weights.forEach((row, j) =>
      row.forEach((w, i) =>
        out.push({
          a: v(decPositions[i]),
          b: v(gridPos(outSlot, 0, Math.floor(j / task.n), j % task.n)),
          w,
          target: j,
        }),
      ),
    )
    return { enc, ms, z, dec, out }
  }, [model, inSlot, msSlot, outSlot, encPositions, zPositions, decPositions, task])

  const done = step >= totalSteps('ae')
  const mse = useMemo(() => {
    if (!trace || trace.generated) return null
    const flat = input[0].flat()
    return trace.o.a.reduce((s, val, i) => s + (val - flat[i]) ** 2, 0) / trace.o.a.length
  }, [input, trace])

  if (!trace) return null

  return (
    <group>
      <GridNodes slot={inSlot} values={input} scale={1} paintable />
      <VectorNodes
        positions={encPositions}
        values={trace.h.a.length ? trace.h.a : Array.from({ length: encSlot.size }, () => 0)}
        scale={maxAbs(trace.h.a.length ? trace.h.a : [1])}
        layerIndex={0}
        radius={Math.min(0.24, encSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 0, index: i })}
      />
      <GridNodes slot={msSlot} values={msVals} scale={Math.max(maxAbs(trace.mu), maxAbs(trace.logvar))} />
      <VectorNodes
        positions={zPositions}
        values={trace.z}
        scale={maxAbs(trace.z)}
        layerIndex={2}
        radius={0.38}
        refFor={(i) => ({ space: 'vector', layer: 2, index: i })}
      />
      <VectorNodes
        positions={decPositions}
        values={trace.d.a}
        scale={maxAbs(trace.d.a)}
        layerIndex={3}
        radius={Math.min(0.24, decSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 3, index: i })}
      />
      <GridNodes slot={outSlot} values={outVals} scale={1} />

      <Connections segments={topSegments(segs.enc, 500)} layerIndex={0} maxRadius={0.01} />
      <Connections segments={topSegments(segs.ms, 200)} layerIndex={1} maxRadius={0.02} />
      <Connections segments={segs.z} layerIndex={2} maxRadius={0.035} />
      <Connections segments={segs.dec} layerIndex={3} maxRadius={0.03} />
      <Connections segments={topSegments(segs.out, 500)} layerIndex={4} maxRadius={0.01} />
      <FlowParticles segments={topSegments(segs.enc, 300)} layerIndex={0} size={0.07} />
      <FlowParticles segments={topSegments(segs.ms, 200)} layerIndex={1} size={0.07} />
      <FlowParticles segments={segs.z} layerIndex={2} size={0.1} />
      <FlowParticles segments={segs.dec} layerIndex={3} size={0.09} />
      <FlowParticles segments={topSegments(segs.out, 300)} layerIndex={4} size={0.07} />

      {done && (
        <Html
          position={[outSlot.x, ((outSlot.rows - 1) / 2) * outSlot.cell + 2.3, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="next-char-banner">
            {trace.generated ? (
              t('vae.generatedBadge')
            ) : (
              <>
                MSE: <b>{mse?.toFixed(4)}</b>
              </>
            )}
          </div>
        </Html>
      )}

      <LayerLabel position={slotLabelAnchor(inSlot)} title={t('layer.input')} sub={`${task.n}×${task.n}`} layer={-1} />
      <LayerLabel position={slotLabelAnchor(encSlot)} title={t('layer.encoder')} sub={`${encSlot.size} · ReLU`} layer={0} />
      <LayerLabel position={slotLabelAnchor(msSlot)} title="μ · log σ²" sub={`${task.latent}×2`} layer={1} />
      <LayerLabel position={slotLabelAnchor(zSlot)} title={t('layer.sampleZ')} sub="z = μ + ε·σ" layer={2} />
      <LayerLabel position={slotLabelAnchor(decSlot)} title={t('layer.decoder')} sub={`${decSlot.size} · ReLU`} layer={3} />
      <LayerLabel position={slotLabelAnchor(outSlot)} title={t('layer.recon')} sub="sigmoid" layer={4} />

      <SelectionMarker />
    </group>
  )
}

/** Autoencoder: input image → encoder → latent bottleneck → decoder → reconstruction. */
export function AEScene() {
  const variant = useStore((s) => s.aeVariant)
  if (variant === 'vae') return <VAESceneInner />
  return <AESceneInner />
}

function AESceneInner() {
  const input = useStore((s) => s.aeInput)
  const trace = useStore((s) => s.aeTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const task = MODELS.ae
  const model = task.model

  const inSlot = aeSlot(-1) as GridSlot
  const encSlot = aeSlot(0) as VecSlot
  const latSlot = aeSlot(1) as VecSlot
  const decSlot = aeSlot(2) as VecSlot
  const outSlot = aeSlot(3) as GridSlot

  const recon = trace[3].a
  const outVals = useMemo<Tensor3>(
    () => [Array.from({ length: task.n }, (_, r) => recon.slice(r * task.n, (r + 1) * task.n))],
    [recon, task.n],
  )

  const encPositions = useMemo(
    () => Array.from({ length: encSlot.size }, (_, i) => vecPos(encSlot, i)),
    [encSlot],
  )
  const latPositions = useMemo(
    () => Array.from({ length: latSlot.size }, (_, i) => vecPos(latSlot, i)),
    [latSlot],
  )
  const decPositions = useMemo(
    () => Array.from({ length: decSlot.size }, (_, i) => vecPos(decSlot, i)),
    [decSlot],
  )

  const segs = useMemo(() => {
    const enc: Segment[] = []
    model.layers[0].weights.forEach((row, j) =>
      row.forEach((w, i) =>
        enc.push({
          a: v(gridPos(inSlot, 0, Math.floor(i / task.n), i % task.n)),
          b: v(encPositions[j]),
          w,
          target: j,
        }),
      ),
    )
    const lat: Segment[] = []
    model.layers[1].weights.forEach((row, j) =>
      row.forEach((w, i) => lat.push({ a: v(encPositions[i]), b: v(latPositions[j]), w, target: j })),
    )
    const dec: Segment[] = []
    model.layers[2].weights.forEach((row, j) =>
      row.forEach((w, i) => dec.push({ a: v(latPositions[i]), b: v(decPositions[j]), w, target: j })),
    )
    const out: Segment[] = []
    model.layers[3].weights.forEach((row, j) =>
      row.forEach((w, i) =>
        out.push({
          a: v(decPositions[i]),
          b: v(gridPos(outSlot, 0, Math.floor(j / task.n), j % task.n)),
          w,
          target: j,
        }),
      ),
    )
    return { enc, lat, dec, out }
  }, [model, inSlot, outSlot, encPositions, latPositions, decPositions, task.n])

  const done = step >= totalSteps('ae')
  const mse = useMemo(() => {
    const flat = input[0].flat()
    return recon.reduce((s, val, i) => s + (val - flat[i]) ** 2, 0) / recon.length
  }, [input, recon])

  return (
    <group>
      <GridNodes slot={inSlot} values={input} scale={1} paintable />
      <VectorNodes
        positions={encPositions}
        values={trace[0].a}
        scale={maxAbs(trace[0].a)}
        layerIndex={0}
        radius={Math.min(0.24, encSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 0, index: i })}
      />
      <VectorNodes
        positions={latPositions}
        values={trace[1].a}
        scale={1}
        layerIndex={1}
        radius={0.4}
        refFor={(i) => ({ space: 'vector', layer: 1, index: i })}
      />
      <VectorNodes
        positions={decPositions}
        values={trace[2].a}
        scale={maxAbs(trace[2].a)}
        layerIndex={2}
        radius={Math.min(0.24, decSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 2, index: i })}
      />
      <GridNodes slot={outSlot} values={outVals} scale={1} />

      <Connections segments={topSegments(segs.enc, 500)} layerIndex={0} maxRadius={0.01} />
      <Connections segments={segs.lat} layerIndex={1} maxRadius={0.03} />
      <Connections segments={segs.dec} layerIndex={2} maxRadius={0.03} />
      <Connections segments={topSegments(segs.out, 500)} layerIndex={3} maxRadius={0.01} />
      <FlowParticles segments={topSegments(segs.enc, 300)} layerIndex={0} size={0.07} />
      <FlowParticles segments={segs.lat} layerIndex={1} size={0.09} />
      <FlowParticles segments={segs.dec} layerIndex={2} size={0.09} />
      <FlowParticles segments={topSegments(segs.out, 300)} layerIndex={3} size={0.07} />

      {done && (
        <Html
          position={[outSlot.x, ((outSlot.rows - 1) / 2) * outSlot.cell + 2.3, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="next-char-banner">
            MSE: <b>{mse.toFixed(4)}</b>
          </div>
        </Html>
      )}

      <LayerLabel position={slotLabelAnchor(inSlot)} title={t('layer.input')} sub={`${task.n}×${task.n}`} layer={-1} />
      <LayerLabel position={slotLabelAnchor(encSlot)} title={t('layer.encoder')} sub={`${encSlot.size} · ReLU`} layer={0} />
      <LayerLabel position={slotLabelAnchor(latSlot)} title={t('layer.latent')} sub={`${latSlot.size} · tanh`} layer={1} />
      <LayerLabel position={slotLabelAnchor(decSlot)} title={t('layer.decoder')} sub={`${decSlot.size} · ReLU`} layer={2} />
      <LayerLabel position={slotLabelAnchor(outSlot)} title={t('layer.recon')} sub="sigmoid" layer={3} />

      <SelectionMarker />
    </group>
  )
}
