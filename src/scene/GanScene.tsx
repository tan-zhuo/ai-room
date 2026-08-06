import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { maxAbs } from '../nn/mlp'
import { totalSteps, useStore, useT } from '../store'
import { GridSlot, VecSlot, ganSlot, gridPos, slotLabelAnchor, vecPos } from './layout'
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

function toGrid(flat: number[], n: number): Tensor3 {
  return [Array.from({ length: n }, (_, r) => flat.slice(r * n, (r + 1) * n))]
}

/** GAN: z → generator → fake image; fake AND a real sample both go through the
 *  same discriminator, which scores each one's probability of being real. */
export function GanScene() {
  const trace = useStore((s) => s.ganTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const task = MODELS.gan
  const model = task.model

  const zSlot = ganSlot(-1) as VecSlot
  const gSlot = ganSlot(0) as VecSlot
  const fakeSlot = ganSlot(1) as GridSlot
  const dSlot = ganSlot(2) as VecSlot
  const vSlot = ganSlot(3) as VecSlot
  const realSlot = ganSlot(4) as GridSlot

  const zPositions = useMemo(
    () => Array.from({ length: zSlot.size }, (_, i) => vecPos(zSlot, i)),
    [zSlot],
  )
  const gPositions = useMemo(
    () => Array.from({ length: gSlot.size }, (_, i) => vecPos(gSlot, i)),
    [gSlot],
  )
  const dPositions = useMemo(
    () => Array.from({ length: dSlot.size }, (_, i) => vecPos(dSlot, i)),
    [dSlot],
  )
  const vPositions = useMemo(
    () => Array.from({ length: vSlot.size }, (_, i) => vecPos(vSlot, i)),
    [vSlot],
  )

  const fakeVals = useMemo(() => toGrid(trace.img.a, task.n), [trace, task.n])
  const realVals = useMemo(() => toGrid(trace.real, task.n), [trace, task.n])

  const segs = useMemo(() => {
    const gz: Segment[] = []
    model.g1.weights.forEach((row, j) =>
      row.forEach((w, i) => gz.push({ a: v(zPositions[i]), b: v(gPositions[j]), w, target: j })),
    )
    const gimg: Segment[] = []
    model.g2.weights.forEach((row, j) =>
      row.forEach((w, i) =>
        gimg.push({
          a: v(gPositions[i]),
          b: v(gridPos(fakeSlot, 0, Math.floor(j / task.n), j % task.n)),
          w,
          target: j,
        }),
      ),
    )
    const dIn: Segment[] = []
    model.d1.weights.forEach((row, j) =>
      row.forEach((w, i) => {
        const r = Math.floor(i / task.n)
        const c = i % task.n
        dIn.push({ a: v(gridPos(fakeSlot, 0, r, c)), b: v(dPositions[j]), w, target: j })
        dIn.push({ a: v(gridPos(realSlot, 0, r, c)), b: v(dPositions[j]), w, target: j })
      }),
    )
    const verdict: Segment[] = []
    model.d2.weights[0].forEach((w, i) => {
      verdict.push({ a: v(dPositions[i]), b: v(vPositions[0]), w, target: 0 })
      verdict.push({ a: v(dPositions[i]), b: v(vPositions[1]), w, target: 1 })
    })
    return { gz, gimg, dIn, verdict }
  }, [model, zPositions, gPositions, dPositions, vPositions, fakeSlot, realSlot, task.n])

  const done = step >= totalSteps('gan')
  const dFake = trace.out.a[0]
  const dReal = trace.realOut.a[0]

  return (
    <group>
      <VectorNodes
        positions={zPositions}
        values={trace.z}
        scale={maxAbs(trace.z)}
        layerIndex={-1}
        radius={0.4}
        refFor={(i) => ({ space: 'vector', layer: -1, index: i })}
      />
      <VectorNodes
        positions={gPositions}
        values={trace.g1.a}
        scale={maxAbs(trace.g1.a)}
        layerIndex={0}
        radius={Math.min(0.24, gSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 0, index: i })}
      />
      <GridNodes slot={fakeSlot} values={fakeVals} scale={1} />
      <GridNodes slot={realSlot} values={realVals} scale={1} alwaysOn />
      <VectorNodes
        positions={dPositions}
        values={trace.d1.a}
        scale={maxAbs(trace.d1.a)}
        layerIndex={2}
        radius={Math.min(0.24, dSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 2, index: i })}
      />
      <VectorNodes
        positions={vPositions}
        values={[dFake, dReal]}
        scale={1}
        layerIndex={3}
        radius={0.45}
        refFor={(i) => ({ space: 'vector', layer: 3, index: i })}
      />

      <Connections segments={topSegments(segs.gz, 150)} layerIndex={0} maxRadius={0.02} />
      <Connections segments={topSegments(segs.gimg, 350)} layerIndex={1} maxRadius={0.01} />
      <Connections segments={topSegments(segs.dIn, 320)} layerIndex={2} maxRadius={0.01} />
      <Connections segments={segs.verdict} layerIndex={3} maxRadius={0.03} />
      <FlowParticles segments={topSegments(segs.gz, 200)} layerIndex={0} size={0.08} />
      <FlowParticles segments={topSegments(segs.gimg, 300)} layerIndex={1} size={0.07} />
      <FlowParticles segments={topSegments(segs.dIn, 300)} layerIndex={2} size={0.07} />
      <FlowParticles segments={segs.verdict} layerIndex={3} size={0.09} />

      {done && (
        <Html
          position={[vSlot.x, 1.2, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="next-char-banner">
            {dFake > 0.5 ? '🎭 ' + t('gan.fooled') : '🔍 ' + t('gan.caught')}
          </div>
        </Html>
      )}
      {done && (
        <>
          <Html position={[vSlot.x + 0.9, vPositions[0][1], 0]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className="value-tag">D = {(dFake * 100).toFixed(0)}%</div>
          </Html>
          <Html position={[vSlot.x + 0.9, vPositions[1][1], 0]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className="value-tag">D = {(dReal * 100).toFixed(0)}%</div>
          </Html>
        </>
      )}

      <LayerLabel position={slotLabelAnchor(zSlot)} title={t('layer.zLatent')} sub="z ~ N(0,1)" layer={-1} />
      <LayerLabel
        position={slotLabelAnchor(gSlot)}
        title={t('layer.generator')}
        sub={`${gSlot.size} · ReLU`}
        layer={0}
      />
      <LayerLabel position={slotLabelAnchor(fakeSlot)} title={t('layer.fakeImg')} sub="G(z) · tanh" layer={1} />
      <LayerLabel position={slotLabelAnchor(realSlot)} title={t('layer.realImg')} sub={`${task.n}×${task.n}`} layer={4} />
      <LayerLabel
        position={slotLabelAnchor(dSlot)}
        title={t('layer.discriminator')}
        sub={`${dSlot.size} · LeakyReLU`}
        layer={2}
      />
      <LayerLabel position={slotLabelAnchor(vSlot)} title={t('layer.verdict')} sub="sigmoid = P(real)" layer={3} />

      <SelectionMarker />
    </group>
  )
}
