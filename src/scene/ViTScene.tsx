import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { maxAbs } from '../nn/mlp'
import { totalSteps, useStore, useT } from '../store'
import { GridSlot, VecSlot, gridPos, slotLabelAnchor, vecPos, vitSlot } from './layout'
import { Segment } from './common'
import { GridNodes } from './GridNodes'
import { VectorNodes } from './VectorNodes'
import { FlowParticles } from './FlowParticles'
import { LayerLabel } from './Labels'
import { SelectionMarker } from './SelectionMarker'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

/** ViT: image → patch tokens (+CLS, +pos) → bidirectional attention →
 *  residual → FFN residual → classification from the [CLS] token. */
export function ViTScene() {
  const input = useStore((s) => s.vitInput)
  const trace = useStore((s) => s.vitTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const task = MODELS.vit
  const model = task.model

  const imgSlot = vitSlot(-1) as GridSlot
  const embSlot = vitSlot(0) as GridSlot
  const attSlot = vitSlot(1) as GridSlot
  const x1Slot = vitSlot(2) as GridSlot
  const x2Slot = vitSlot(3) as GridSlot
  const outSlot = vitSlot(4) as VecSlot

  const embVals = useMemo<Tensor3>(() => [trace.X], [trace])
  const attVals = useMemo<Tensor3>(() => trace.att, [trace])
  const x1Vals = useMemo<Tensor3>(() => [trace.X1], [trace])
  const x2Vals = useMemo<Tensor3>(() => [trace.X2], [trace])

  const outPositions = useMemo(
    () => Array.from({ length: outSlot.size }, (_, i) => vecPos(outSlot, i)),
    [outSlot],
  )

  const flows = useMemo(() => {
    // image patches → their token rows
    const embed: Segment[] = []
    for (let tk = 1; tk < model.T; tk++) {
      const pi = tk - 1
      const gr = Math.floor(pi / model.grid)
      const gc = pi % model.grid
      for (let j = 0; j < model.d; j += 2)
        embed.push({
          a: v(gridPos(imgSlot, 0, gr * model.p + model.p / 2, gc * model.p + model.p / 2)),
          b: v(gridPos(embSlot, 0, tk, j)),
          w: 0.5,
          target: 0,
        })
    }
    // tokens → attention rows
    const att: Segment[] = []
    for (let h = 0; h < model.heads; h++)
      for (let i = 0; i < model.T; i++)
        att.push({
          a: v(gridPos(embSlot, 0, i, Math.floor(model.d / 2))),
          b: v(gridPos(attSlot, h, i, i)),
          w: 0.5,
          target: 0,
        })
    // attention → X1 rows, X1 → X2 rows
    const mk = (src: GridSlot, dst: GridSlot): Segment[] => {
      const segs: Segment[] = []
      for (let i = 0; i < model.T; i++)
        for (let j = 0; j < model.d; j += 2)
          segs.push({
            a: v(gridPos(src, 0, i, Math.min(j, src.cols - 1))),
            b: v(gridPos(dst, 0, i, j)),
            w: 0.5,
            target: 0,
          })
      return segs
    }
    // CLS row → output classes
    const out: Segment[] = []
    for (let c = 0; c < model.classes; c++)
      for (let j = 0; j < model.d; j += 2)
        out.push({ a: v(gridPos(x2Slot, 0, 0, j)), b: v(outPositions[c]), w: 0.5, target: c })
    return [embed, att, mk(attSlot, x1Slot), mk(x1Slot, x2Slot), out]
  }, [model, imgSlot, embSlot, attSlot, x1Slot, x2Slot, outPositions])

  const done = step >= totalSteps('vit')
  const classNames = [0, 1, 2, 3].map((c) => t(`class.cnn.${c}`))

  return (
    <group>
      <GridNodes slot={imgSlot} values={input} scale={1} />
      <GridNodes slot={embSlot} values={embVals} scale={maxAbs(trace.X.flat())} />
      <GridNodes slot={attSlot} values={attVals} scale={1} />
      <GridNodes slot={x1Slot} values={x1Vals} scale={maxAbs(trace.X1.flat())} />
      <GridNodes slot={x2Slot} values={x2Vals} scale={maxAbs(trace.X2.flat())} />
      <VectorNodes
        positions={outPositions}
        values={trace.probs}
        scale={1}
        layerIndex={4}
        radius={0.42}
        refFor={(i) => ({ space: 'vector', layer: 4, index: i })}
      />

      {flows.map((segs, k) => (
        <FlowParticles key={`f${k}`} segments={segs} layerIndex={k} size={0.07} />
      ))}

      {/* patch grid lines over the input image */}
      {Array.from({ length: model.grid + 1 }, (_, k) => {
        const half = (imgSlot.rows * imgSlot.cell) / 2
        const o = -half + k * model.p * imgSlot.cell
        return (
          <group key={`pg${k}`}>
            <mesh position={[imgSlot.x, o, 0]}>
              <boxGeometry args={[0.02, 0.02, half * 2]} />
              <meshBasicMaterial color="#3d5a80" toneMapped={false} />
            </mesh>
            <mesh position={[imgSlot.x, 0, o]}>
              <boxGeometry args={[0.02, half * 2, 0.02]} />
              <meshBasicMaterial color="#3d5a80" toneMapped={false} />
            </mesh>
          </group>
        )
      })}

      {done && (
        <Html
          position={[outSlot.x, ((outSlot.size - 1) / 2) * outSlot.gapY + 2.1, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="next-char-banner">
            {classNames[trace.pred]} · <b>{(trace.probs[trace.pred] * 100).toFixed(1)}%</b>
          </div>
        </Html>
      )}

      <LayerLabel
        position={slotLabelAnchor(imgSlot)}
        title={t('layer.input')}
        sub={`${model.n}×${model.n} → ${model.grid}×${model.grid} patches`}
        layer={-1}
      />
      <LayerLabel
        position={slotLabelAnchor(embSlot)}
        title={t('layer.patchEmbed')}
        sub={`[CLS] + ${model.T - 1} × ${model.p}×${model.p} · +pos`}
        layer={0}
      />
      <LayerLabel
        position={slotLabelAnchor(attSlot)}
        title={t('layer.attn')}
        sub={`${model.heads} × ${model.T}×${model.T} · softmax`}
        layer={1}
      />
      <LayerLabel position={slotLabelAnchor(x1Slot)} title={t('layer.vitResid')} sub="X + MHA(LN(X))" layer={2} />
      <LayerLabel position={slotLabelAnchor(x2Slot)} title={t('layer.vitFfn')} sub="X₁ + FFN(LN(X₁))" layer={3} />
      <LayerLabel position={slotLabelAnchor(outSlot)} title={t('layer.clsHead')} sub="LN(CLS)·W → softmax" layer={4} />

      <SelectionMarker />
    </group>
  )
}
