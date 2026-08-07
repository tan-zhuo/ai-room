import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { totalSteps, useStore, useT } from '../store'
import { GridSlot, VecSlot, gridPos, mambaSlot, slotLabelAnchor, vecPos } from './layout'
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

/** Mamba (selective SSM): tokens → embeddings → input-dependent Δ selectivity →
 *  linear-time selective scan (fixed-size state, recurrence arcs) → SiLU gate →
 *  next char. Same task as RNN / LSTM / Transformer, so compare the losses. */
export function MambaScene() {
  const trace = useStore((s) => s.mambaTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const model = MODELS.mamba.model

  const tokSlot = mambaSlot(-1) as GridSlot
  const embSlot = mambaSlot(0) as GridSlot
  const selSlot = mambaSlot(1) as GridSlot
  const ssmSlot = mambaSlot(2) as GridSlot
  const gateSlot = mambaSlot(3) as GridSlot
  const outSlot = mambaSlot(4) as VecSlot

  const T = model.T
  const vocabN = model.vocab.length

  const tokVals = useMemo<Tensor3>(
    () => [[trace.ids.map((id) => 0.25 + (0.7 * id) / vocabN)]],
    [trace, vocabN],
  )
  const embVals = useMemo<Tensor3>(() => [trace.X], [trace])
  const selVals = useMemo<Tensor3>(() => [trace.Delta], [trace])
  const ssmVals = useMemo<Tensor3>(() => [trace.Y], [trace])
  const gateVals = useMemo<Tensor3>(() => [trace.O], [trace])

  const outPositions = useMemo(
    () => Array.from({ length: vocabN }, (_, i) => vecPos(outSlot, i)),
    [outSlot, vocabN],
  )

  // the state flows down the sequence in O(T): one arc per channel per step
  const recurrent = useMemo<Segment[]>(() => {
    const segs: Segment[] = []
    for (let ts = 1; ts < T; ts++)
      for (let j = 0; j < model.dI; j++)
        segs.push({ a: v(gridPos(ssmSlot, 0, ts - 1, j)), b: v(gridPos(ssmSlot, 0, ts, j)), w: 0.45, target: 0 })
    return segs
  }, [T, model.dI, ssmSlot])

  const flows = useMemo(() => {
    const embSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let k = 0; k < model.d; k++)
        embSegs.push({ a: v(gridPos(tokSlot, 0, 0, i)), b: v(gridPos(embSlot, 0, i, k)), w: 0.5, target: 0 })
    const selSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let j = 0; j < model.dI; j++)
        selSegs.push({
          a: v(gridPos(embSlot, 0, i, j % model.d)),
          b: v(gridPos(selSlot, 0, i, j)),
          w: 0.5,
          target: 0,
        })
    const ssmSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let j = 0; j < model.dI; j++)
        ssmSegs.push({ a: v(gridPos(selSlot, 0, i, j)), b: v(gridPos(ssmSlot, 0, i, j)), w: 0.5, target: 0 })
    const gateSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let j = 0; j < model.dI; j++)
        gateSegs.push({ a: v(gridPos(ssmSlot, 0, i, j)), b: v(gridPos(gateSlot, 0, i, j)), w: 0.5, target: 0 })
    const outSegs: Segment[] = []
    for (let j = 0; j < model.dI; j++)
      for (let vi = 0; vi < vocabN; vi += 2)
        outSegs.push({ a: v(gridPos(gateSlot, 0, T - 1, j)), b: v(outPositions[vi]), w: 0.5, target: vi })
    return [embSegs, selSegs, [...ssmSegs, ...recurrent], gateSegs, outSegs]
  }, [T, model, tokSlot, embSlot, selSlot, ssmSlot, gateSlot, outPositions, vocabN, recurrent])

  const done = step >= totalSteps('mamba')
  const pred = trace.probs.indexOf(Math.max(...trace.probs))
  const topIdx = [...trace.probs.keys()].sort((a, b) => trace.probs[b] - trace.probs[a]).slice(0, 5)
  const showChar = (c: string) => (c === ' ' ? '␣' : c)

  return (
    <group>
      <GridNodes slot={tokSlot} values={tokVals} scale={1} />
      <GridNodes slot={embSlot} values={embVals} scale={tensorMax(embVals)} />
      <GridNodes slot={selSlot} values={selVals} scale={tensorMax(selVals)} />
      <GridNodes slot={ssmSlot} values={ssmVals} scale={tensorMax(ssmVals)} />
      <GridNodes slot={gateSlot} values={gateVals} scale={tensorMax(gateVals)} />
      <VectorNodes
        positions={outPositions}
        values={trace.probs}
        scale={Math.max(...trace.probs)}
        layerIndex={4}
        radius={Math.min(0.22, outSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 4, index: i })}
      />

      <Connections segments={recurrent} layerIndex={2} maxRadius={0.014} />
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
      {outPositions.map((p, i) => {
        const isTop = topIdx.includes(i)
        return (
          <Html key={`ov${i}`} position={[p[0] + 0.4, p[1], p[2]]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className={`out-label${done ? ' done' : ''}${done && i === pred ? ' pred' : ''}`}>
              <span className={`out-name mono-char${isTop ? '' : ' dim'}`}>{showChar(model.vocab[i])}</span>
              {done && isTop && <span className="out-prob">{(trace.probs[i] * 100).toFixed(1)}%</span>}
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

      {/* the headline: linear time, constant state */}
      <Html
        position={[ssmSlot.x, slotLabelAnchor(ssmSlot)[1] + 1.6, 0]}
        center
        zIndexRange={[20, 0]}
        style={{ pointerEvents: 'none' }}
      >
        <div className="next-char-banner">
          {t('mamba.badge', { s: `${model.dI}×${model.N}` })}
        </div>
      </Html>

      <LayerLabel position={slotLabelAnchor(tokSlot)} title={t('layer.tokens')} sub={`${T}`} layer={-1} />
      <LayerLabel position={slotLabelAnchor(embSlot)} title={t('layer.embed')} sub={`${T}×${model.d}`} layer={0} />
      <LayerLabel
        position={slotLabelAnchor(selSlot)}
        title={t('layer.selectivity')}
        sub="Δ = softplus(x·WΔ)"
        layer={1}
      />
      <LayerLabel
        position={slotLabelAnchor(ssmSlot)}
        title={t('layer.ssmState')}
        sub="h = ā⊙h′ + Δ·B·u"
        layer={2}
      />
      <LayerLabel position={slotLabelAnchor(gateSlot)} title={t('layer.gatedOut')} sub="y ⊙ silu(z)" layer={3} />
      <LayerLabel position={slotLabelAnchor(outSlot)} title={t('layer.output')} sub={`${vocabN} · softmax`} layer={4} />

      <SelectionMarker />
    </group>
  )
}
