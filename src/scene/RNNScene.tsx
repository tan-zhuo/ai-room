import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { totalSteps, useStore, useT } from '../store'
import { GridSlot, VecSlot, gridPos, rnnSlot, slotLabelAnchor, vecPos } from './layout'
import { Segment } from './common'
import { GridNodes } from './GridNodes'
import { VectorNodes } from './VectorNodes'
import { Connections } from './Connections'
import { FlowParticles } from './FlowParticles'
import { LayerLabel } from './Labels'
import { SelectionMarker } from './SelectionMarker'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

function tensorMax(t: Tensor3): number {
  let m = 0
  for (const ch of t) for (const row of ch) for (const val of row) m = Math.max(m, Math.abs(val))
  return m || 1
}

/** Elman RNN: tokens → embeddings → recurrent hidden states (row = timestep) → next char. */
export function RNNScene() {
  const trace = useStore((s) => s.rnnTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const model = MODELS.rnn.model

  const tokSlot = rnnSlot(-1) as GridSlot
  const embSlot = rnnSlot(0) as GridSlot
  const hidSlot = rnnSlot(1) as GridSlot
  const outSlot = rnnSlot(2) as VecSlot

  const T = model.T
  const vocabN = model.vocab.length

  const tokVals = useMemo<Tensor3>(
    () => [[trace.ids.map((id) => 0.25 + (0.7 * id) / vocabN)]],
    [trace, vocabN],
  )
  const embVals = useMemo<Tensor3>(() => [trace.X], [trace])
  const hidVals = useMemo<Tensor3>(() => [trace.H], [trace])

  const outPositions = useMemo(
    () => Array.from({ length: vocabN }, (_, i) => vecPos(outSlot, i)),
    [outSlot, vocabN],
  )

  // recurrence arcs between consecutive hidden rows — the defining feature
  const recurrent = useMemo<Segment[]>(() => {
    const segs: Segment[] = []
    for (let ts = 1; ts < T; ts++)
      for (let j = 0; j < model.h; j++)
        segs.push({ a: v(gridPos(hidSlot, 0, ts - 1, j)), b: v(gridPos(hidSlot, 0, ts, j)), w: 0.45, target: 0 })
    return segs
  }, [T, model.h, hidSlot])

  const flows = useMemo(() => {
    const embSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let k = 0; k < model.d; k++)
        embSegs.push({ a: v(gridPos(tokSlot, 0, 0, i)), b: v(gridPos(embSlot, 0, i, k)), w: 0.5, target: 0 })
    const hidSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let j = 0; j < model.h; j++)
        hidSegs.push({ a: v(gridPos(embSlot, 0, i, j % model.d)), b: v(gridPos(hidSlot, 0, i, j)), w: 0.5, target: 0 })
    const outSegs: Segment[] = []
    for (let j = 0; j < model.h; j++)
      for (let vi = 0; vi < vocabN; vi += 2)
        outSegs.push({ a: v(gridPos(hidSlot, 0, T - 1, j)), b: v(outPositions[vi]), w: 0.5, target: vi })
    return [embSegs, [...hidSegs, ...recurrent], outSegs]
  }, [T, model, tokSlot, embSlot, hidSlot, outPositions, vocabN, recurrent])

  const done = step >= totalSteps('rnn')
  const pred = trace.probs.indexOf(Math.max(...trace.probs))
  const topIdx = [...trace.probs.keys()].sort((a, b) => trace.probs[b] - trace.probs[a]).slice(0, 5)
  const showChar = (c: string) => (c === ' ' ? '␣' : c)

  return (
    <group>
      <GridNodes slot={tokSlot} values={tokVals} scale={1} />
      <GridNodes slot={embSlot} values={embVals} scale={tensorMax(embVals)} />
      <GridNodes slot={hidSlot} values={hidVals} scale={1} />
      <VectorNodes
        positions={outPositions}
        values={trace.probs}
        scale={Math.max(...trace.probs)}
        layerIndex={2}
        radius={Math.min(0.22, outSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 2, index: i })}
      />

      <Connections segments={recurrent} layerIndex={1} maxRadius={0.014} />
      {flows.map((segs, k) => (
        <FlowParticles key={`p${k}`} segments={segs} layerIndex={k} size={0.06} />
      ))}

      {trace.chars.map((ch, i) => {
        const p = gridPos(tokSlot, 0, 0, i)
        return (
          <Html key={`tk${i}`} position={[p[0], p[1], p[2]]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className="token-char">{showChar(ch)}</div>
          </Html>
        )
      })}
      {topIdx.map((i) => {
        const p = outPositions[i]
        return (
          <Html key={`ov${i}`} position={[p[0] + 0.4, p[1], p[2]]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className={`out-label${done ? ' done' : ''}${done && i === pred ? ' pred' : ''}`}>
              <span className="out-name mono-char">{showChar(model.vocab[i])}</span>
              {done && <span className="out-prob">{(trace.probs[i] * 100).toFixed(1)}%</span>}
            </div>
          </Html>
        )
      })}
      {done && (
        <Html
          position={[outSlot.x, ((outSlot.size - 1) / 2) * outSlot.gapY + 2.2, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="next-char-banner">
            {t('llm.next')}: <b>{showChar(model.vocab[pred])}</b> ({(trace.probs[pred] * 100).toFixed(1)}%)
          </div>
        </Html>
      )}

      <LayerLabel position={slotLabelAnchor(tokSlot)} title={t('layer.tokens')} sub={`${T}`} layer={-1} />
      <LayerLabel position={slotLabelAnchor(embSlot)} title={t('layer.embed')} sub={`${T}×${model.d}`} layer={0} />
      <LayerLabel
        position={slotLabelAnchor(hidSlot)}
        title={t('layer.rnnHidden')}
        sub={`h_t = tanh(W·x + U·h_{t-1})`}
        layer={1}
      />
      <LayerLabel position={slotLabelAnchor(outSlot)} title={t('layer.output')} sub={`${vocabN} · softmax`} layer={2} />

      <SelectionMarker />
    </group>
  )
}
