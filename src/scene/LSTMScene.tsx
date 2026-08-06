import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { totalSteps, useStore, useT } from '../store'
import { GridSlot, VecSlot, gridPos, lstmSlot, slotLabelAnchor, vecPos } from './layout'
import { Segment } from './common'
import { GridNodes } from './GridNodes'
import { VectorNodes } from './VectorNodes'
import { FlowParticles } from './FlowParticles'
import { LayerLabel } from './Labels'
import { SelectionMarker } from './SelectionMarker'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

function tensorMax(t: Tensor3): number {
  let m = 0
  for (const ch of t) for (const row of ch) for (const val of row) m = Math.max(m, Math.abs(val))
  return m || 1
}

const GATE_NAMES = ['f', 'i', 'g', 'o']

/** LSTM: tokens → embeddings → gates f/i/g/o → cell state → hidden state → next char. */
export function LSTMScene() {
  const trace = useStore((s) => s.lstmTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const model = MODELS.lstm.model

  const tokSlot = lstmSlot(-1) as GridSlot
  const embSlot = lstmSlot(0) as GridSlot
  const gateSlot = lstmSlot(1) as GridSlot
  const cellSlot = lstmSlot(2) as GridSlot
  const hidSlot = lstmSlot(3) as GridSlot
  const outSlot = lstmSlot(4) as VecSlot

  const T = model.T
  const vocabN = model.vocab.length

  const tokVals = useMemo<Tensor3>(
    () => [[trace.ids.map((id) => 0.25 + (0.7 * id) / vocabN)]],
    [trace, vocabN],
  )
  const embVals = useMemo<Tensor3>(() => [trace.X], [trace])
  const gateVals = useMemo<Tensor3>(() => [trace.F, trace.I, trace.G, trace.O], [trace])
  const cellVals = useMemo<Tensor3>(() => [trace.C], [trace])
  const hidVals = useMemo<Tensor3>(() => [trace.H], [trace])

  const outPositions = useMemo(
    () => Array.from({ length: vocabN }, (_, i) => vecPos(outSlot, i)),
    [outSlot, vocabN],
  )

  const flows = useMemo(() => {
    const embSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let k = 0; k < model.d; k++)
        embSegs.push({ a: v(gridPos(tokSlot, 0, 0, i)), b: v(gridPos(embSlot, 0, i, k)), w: 0.5, target: 0 })
    const gateSegs: Segment[] = []
    for (let ch = 0; ch < 4; ch++)
      for (let i = 0; i < T; i++)
        for (let j = 0; j < model.h; j += 2)
          gateSegs.push({ a: v(gridPos(embSlot, 0, i, j % model.d)), b: v(gridPos(gateSlot, ch, i, j)), w: 0.5, target: 0 })
    const cellSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let j = 0; j < model.h; j++)
        cellSegs.push({ a: v(gridPos(gateSlot, (i + j) % 4, i, j)), b: v(gridPos(cellSlot, 0, i, j)), w: 0.5, target: 0 })
    const hidSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let j = 0; j < model.h; j++)
        hidSegs.push({ a: v(gridPos(cellSlot, 0, i, j)), b: v(gridPos(hidSlot, 0, i, j)), w: 0.5, target: 0 })
    const outSegs: Segment[] = []
    for (let j = 0; j < model.h; j++)
      for (let vi = 0; vi < vocabN; vi += 2)
        outSegs.push({ a: v(gridPos(hidSlot, 0, T - 1, j)), b: v(outPositions[vi]), w: 0.5, target: vi })
    return [embSegs, gateSegs, cellSegs, hidSegs, outSegs]
  }, [T, model, tokSlot, embSlot, gateSlot, cellSlot, hidSlot, outPositions, vocabN])

  const done = step >= totalSteps('lstm')
  const pred = trace.probs.indexOf(Math.max(...trace.probs))
  const topIdx = [...trace.probs.keys()].sort((a, b) => trace.probs[b] - trace.probs[a]).slice(0, 5)
  const showChar = (c: string) => (c === ' ' ? '␣' : c)

  return (
    <group>
      <GridNodes slot={tokSlot} values={tokVals} scale={1} />
      <GridNodes slot={embSlot} values={embVals} scale={tensorMax(embVals)} />
      <GridNodes slot={gateSlot} values={gateVals} scale={1} />
      <GridNodes slot={cellSlot} values={cellVals} scale={tensorMax(cellVals)} />
      <GridNodes slot={hidSlot} values={hidVals} scale={1} />
      <VectorNodes
        positions={outPositions}
        values={trace.probs}
        scale={Math.max(...trace.probs)}
        layerIndex={4}
        radius={Math.min(0.22, outSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 4, index: i })}
      />

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
      {/* gate letters above each gate sheet */}
      {GATE_NAMES.map((g, ch) => {
        const p = gridPos(gateSlot, ch, 0, Math.floor(model.h / 2))
        return (
          <Html key={`g${ch}`} position={[p[0], p[1] + 0.7, p[2]]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className="gate-letter">{g}</div>
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
      <LayerLabel position={slotLabelAnchor(gateSlot)} title={t('layer.gates')} sub="σ · σ · tanh · σ" layer={1} />
      <LayerLabel position={slotLabelAnchor(cellSlot)} title={t('layer.cell')} sub="c = f⊙c′ + i⊙g" layer={2} />
      <LayerLabel position={slotLabelAnchor(hidSlot)} title={t('layer.rnnHidden')} sub="h = o⊙tanh(c)" layer={3} />
      <LayerLabel position={slotLabelAnchor(outSlot)} title={t('layer.output')} sub={`${vocabN} · softmax`} layer={4} />

      <SelectionMarker />
    </group>
  )
}
