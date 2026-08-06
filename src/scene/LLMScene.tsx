import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { useStore, useT } from '../store'
import { GridSlot, VecSlot, gridPos, llmLabelAnchor, llmSlot, vecPos } from './layout'
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

/** Tiny transformer scene: tokens → embeddings → Q/K/V → attention → A·V → FFN → next char. */
export function LLMScene() {
  const trace = useStore((s) => s.llmTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const model = MODELS.llm.model

  const tokSlot = llmSlot(-1) as GridSlot
  const embSlot = llmSlot(0) as GridSlot
  const qkvSlot = llmSlot(1) as GridSlot
  const attSlot = llmSlot(2) as GridSlot
  const zSlot = llmSlot(3) as GridSlot
  const ffnSlot = llmSlot(4) as GridSlot
  const outSlot = llmSlot(5) as VecSlot

  const T = model.T
  const vocabN = model.vocab.length

  // tensors for each sheet
  const tokVals = useMemo<Tensor3>(
    () => [[trace.ids.map((id) => 0.25 + (0.7 * id) / vocabN)]],
    [trace, vocabN],
  )
  const embVals = useMemo<Tensor3>(() => [trace.X], [trace])
  const qkvVals = useMemo<Tensor3>(() => [trace.Q, trace.K, trace.V], [trace])
  const attVals = useMemo<Tensor3>(() => [trace.A], [trace])
  const zVals = useMemo<Tensor3>(() => [trace.Z], [trace])
  const ffnVals = useMemo<Tensor3>(() => [trace.H], [trace])

  const outPositions = useMemo(
    () => Array.from({ length: vocabN }, (_, i) => vecPos(outSlot, i)),
    [outSlot, vocabN],
  )

  // decorative particle routes between consecutive sheets
  const flows = useMemo(() => {
    const mk = (segs: Segment[]) => segs
    const embSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let k = 0; k < model.d; k++)
        embSegs.push({ a: v(gridPos(tokSlot, 0, 0, i)), b: v(gridPos(embSlot, 0, i, k)), w: 0.5, target: 0 })
    const qkvSegs: Segment[] = []
    for (let ch = 0; ch < 3; ch++)
      for (let i = 0; i < T; i++)
        for (let k = 0; k < model.d; k += 2)
          qkvSegs.push({ a: v(gridPos(embSlot, 0, i, k)), b: v(gridPos(qkvSlot, ch, i, k)), w: 0.5, target: 0 })
    const attSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let j = 0; j <= i; j++) {
        attSegs.push({ a: v(gridPos(qkvSlot, 0, i, model.d - 1)), b: v(gridPos(attSlot, 0, i, j)), w: 0.5, target: 0 })
        attSegs.push({ a: v(gridPos(qkvSlot, 1, j, model.d - 1)), b: v(gridPos(attSlot, 0, i, j)), w: -0.5, target: 0 })
      }
    const zSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let k = 0; k < model.d; k++)
        zSegs.push({ a: v(gridPos(attSlot, 0, i, Math.min(i, T - 1))), b: v(gridPos(zSlot, 0, i, k)), w: 0.5, target: 0 })
    const ffnSegs: Segment[] = []
    for (let i = 0; i < T; i++)
      for (let k = 0; k < model.h; k++)
        ffnSegs.push({ a: v(gridPos(zSlot, 0, i, k % model.d)), b: v(gridPos(ffnSlot, 0, i, k)), w: 0.5, target: 0 })
    const outSegs: Segment[] = []
    for (let k = 0; k < model.h; k++)
      for (let j = 0; j < vocabN; j += 2)
        outSegs.push({ a: v(gridPos(ffnSlot, 0, T - 1, k)), b: v(outPositions[j]), w: 0.5, target: j })
    return [mk(embSegs), mk(qkvSegs), mk(attSegs), mk(zSegs), mk(ffnSegs), mk(outSegs)]
  }, [T, model, tokSlot, embSlot, qkvSlot, attSlot, zSlot, ffnSlot, outPositions, vocabN])

  const done = step >= 6
  const pred = trace.probs.indexOf(Math.max(...trace.probs))
  const topIdx = [...trace.probs.keys()].sort((a, b) => trace.probs[b] - trace.probs[a]).slice(0, 5)
  const showChar = (c: string) => (c === ' ' ? '␣' : c)

  return (
    <group>
      <GridNodes slot={tokSlot} values={tokVals} scale={1} />
      <GridNodes slot={embSlot} values={embVals} scale={tensorMax(embVals)} />
      <GridNodes slot={qkvSlot} values={qkvVals} scale={tensorMax(qkvVals)} />
      <GridNodes slot={attSlot} values={attVals} scale={1} />
      <GridNodes slot={zSlot} values={zVals} scale={tensorMax(zVals)} />
      <GridNodes slot={ffnSlot} values={ffnVals} scale={tensorMax(ffnVals)} />
      <VectorNodes
        positions={outPositions}
        values={trace.probs}
        scale={Math.max(...trace.probs)}
        layerIndex={5}
        radius={Math.min(0.22, outSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: 5, index: i })}
      />

      {flows.map((segs, k) => (
        <FlowParticles key={`p${k}`} segments={segs} layerIndex={k} size={0.06} />
      ))}

      {/* token characters on the input tiles */}
      {trace.chars.map((ch, i) => {
        const p = gridPos(tokSlot, 0, 0, i)
        return (
          <Html key={`tk${i}`} position={[p[0], p[1], p[2]]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className="token-char">{showChar(ch)}</div>
          </Html>
        )
      })}

      {/* vocab characters + probabilities beside the output column (top candidates only) */}
      {topIdx.map((i) => {
        const p = outPositions[i]
        return (
          <Html
            key={`ov${i}`}
            position={[p[0] + 0.4, p[1], p[2]]}
            zIndexRange={[20, 0]}
            style={{ pointerEvents: 'none' }}
          >
            <div className={`out-label${done ? ' done' : ''}${done && i === pred ? ' pred' : ''}`}>
              <span className="out-name mono-char">{showChar(model.vocab[i])}</span>
              {done && <span className="out-prob">{(trace.probs[i] * 100).toFixed(1)}%</span>}
            </div>
          </Html>
        )
      })}
      {done && (
        <Html
          position={[outSlot.x, ((outSlot.size - 1) / 2) * outSlot.gapY + 2.4, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="next-char-banner">
            {t('llm.next')}: <b>{showChar(model.vocab[pred])}</b> ({(trace.probs[pred] * 100).toFixed(1)}%)
          </div>
        </Html>
      )}

      <LayerLabel position={llmLabelAnchor(-1)} title={t('layer.tokens')} sub={`${T}`} layer={-1} />
      <LayerLabel position={llmLabelAnchor(0)} title={t('layer.embed')} sub={`${T}×${model.d}`} layer={0} />
      <LayerLabel position={llmLabelAnchor(1)} title="Q · K · V" sub={`3 × ${T}×${model.d}`} layer={1} />
      <LayerLabel position={llmLabelAnchor(2)} title={t('layer.attn')} sub={`${T}×${T} · softmax`} layer={2} />
      <LayerLabel position={llmLabelAnchor(3)} title={t('layer.attnout')} sub={`${T}×${model.d}`} layer={3} />
      <LayerLabel position={llmLabelAnchor(4)} title={t('layer.ffn')} sub={`${model.h} · ReLU`} layer={4} />
      <LayerLabel position={llmLabelAnchor(5)} title={t('layer.output')} sub={`${vocabN} · softmax`} layer={5} />

      <SelectionMarker />
    </group>
  )
}
