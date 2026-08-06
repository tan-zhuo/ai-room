import { useMemo } from 'react'
import * as THREE from 'three'
import { MODELS } from '../nn/models'
import { maxAbs } from '../nn/mlp'
import { tensorMaxAbs } from '../nn/cnn'
import { useStore, useT } from '../store'
import {
  FlattenSlot,
  GridSlot,
  VecSlot,
  cnnLabelAnchor,
  cnnSlot,
  flattenPos,
  vecPos,
} from './layout'
import { Segment } from './common'
import { VectorNodes } from './VectorNodes'
import { GridNodes } from './GridNodes'
import { Connections } from './Connections'
import { FlowParticles } from './FlowParticles'
import { LayerLabel, OutputLabels } from './Labels'
import { SelectionMarker } from './SelectionMarker'
import { SlideAnim } from './SlideAnim'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

export function CNNScene() {
  const model = MODELS.cnn.model
  const input = useStore((s) => s.cnnInput)
  const trace = useStore((s) => s.cnnTrace)
  const t = useT()

  const inputSlot = cnnSlot(-1) as GridSlot
  const convSlot = cnnSlot(0) as GridSlot
  const poolSlot = cnnSlot(1) as GridSlot
  const flatSlot = cnnSlot(2) as FlattenSlot
  const denseSlot = cnnSlot(3) as VecSlot
  const outSlot = cnnSlot(4) as VecSlot

  const convOut = trace[0].kind === 'tensor' ? trace[0].out : []
  const poolOut = trace[1].kind === 'tensor' ? trace[1].out : []
  const flatOut = trace[2].kind === 'vector' ? trace[2].a : []
  const denseOut = trace[3].kind === 'vector' ? trace[3].a : []
  const finalOut = trace[4].kind === 'vector' ? trace[4].a : []

  const flatPositions = useMemo(
    () => Array.from({ length: flatSlot.size }, (_, i) => flattenPos(flatSlot, i)),
    [flatSlot],
  )
  const densePositions = useMemo(
    () => Array.from({ length: denseSlot.size }, (_, i) => vecPos(denseSlot, i)),
    [denseSlot],
  )
  const outPositions = useMemo(
    () => Array.from({ length: outSlot.size }, (_, i) => vecPos(outSlot, i)),
    [outSlot],
  )

  // pool -> flatten: identity mapping, uniform weight
  const flatSegments = useMemo<Segment[]>(() => {
    const segs: Segment[] = []
    for (let i = 0; i < flatSlot.size; i++) {
      const perCh = flatSlot.srcShape[1] * flatSlot.srcShape[2]
      const ch = Math.floor(i / perCh)
      const rem = i % perCh
      const row = Math.floor(rem / flatSlot.srcShape[2])
      const col = rem % flatSlot.srcShape[2]
      segs.push({
        a: v([
          poolSlot.x + (ch - (poolSlot.channels - 1) / 2) * poolSlot.chGap,
          ((poolSlot.rows - 1) / 2 - row) * poolSlot.cell,
          (col - (poolSlot.cols - 1) / 2) * poolSlot.cell,
        ]),
        b: v(flatPositions[i]),
        w: 0.5,
        target: i,
      })
    }
    return segs
  }, [flatSlot, poolSlot, flatPositions])

  const denseSegments = useMemo<Segment[]>(() => {
    const def = model.layers[3]
    if (def.type !== 'dense') return []
    const segs: Segment[] = []
    def.layer.weights.forEach((row, j) =>
      row.forEach((w, i) => segs.push({ a: v(flatPositions[i]), b: v(densePositions[j]), w, target: j })),
    )
    return segs
  }, [model, flatPositions, densePositions])

  const outSegments = useMemo<Segment[]>(() => {
    const def = model.layers[4]
    if (def.type !== 'dense') return []
    const segs: Segment[] = []
    def.layer.weights.forEach((row, j) =>
      row.forEach((w, i) => segs.push({ a: v(densePositions[i]), b: v(outPositions[j]), w, target: j })),
    )
    return segs
  }, [model, densePositions, outPositions])

  return (
    <group>
      <GridNodes slot={inputSlot} values={input} scale={1} />
      <GridNodes slot={convSlot} values={convOut} scale={tensorMaxAbs(convOut)} />
      <GridNodes slot={poolSlot} values={poolOut} scale={tensorMaxAbs(poolOut)} />
      <VectorNodes
        positions={flatPositions}
        values={flatOut}
        scale={maxAbs(flatOut)}
        layerIndex={2}
        radius={0.18}
        refFor={(i) => ({ space: 'vector', layer: 2, index: i })}
      />
      <VectorNodes
        positions={densePositions}
        values={denseOut}
        scale={maxAbs(denseOut)}
        layerIndex={3}
        radius={0.28}
        refFor={(i) => ({ space: 'vector', layer: 3, index: i })}
      />
      <VectorNodes
        positions={outPositions}
        values={finalOut}
        scale={1}
        layerIndex={4}
        radius={0.34}
        refFor={(i) => ({ space: 'vector', layer: 4, index: i })}
      />

      <Connections segments={flatSegments} layerIndex={2} maxRadius={0.015} />
      <Connections segments={denseSegments} layerIndex={3} maxRadius={0.022} />
      <Connections segments={outSegments} layerIndex={4} maxRadius={0.03} />
      <FlowParticles segments={flatSegments} layerIndex={2} size={0.08} />
      <FlowParticles segments={denseSegments} layerIndex={3} size={0.07} />
      <FlowParticles segments={outSegments} layerIndex={4} size={0.1} />

      <SlideAnim mode="conv" />
      <SlideAnim mode="pool" />

      <LayerLabel position={cnnLabelAnchor(-1)} title={t('layer.input')} sub="1×8×8" />
      <LayerLabel position={cnnLabelAnchor(0)} title={t('layer.conv')} sub="3×3 · 3 ch · ReLU" />
      <LayerLabel position={cnnLabelAnchor(1)} title={t('layer.pool')} sub="2×2" />
      <LayerLabel position={cnnLabelAnchor(2)} title={t('layer.flatten')} sub="27" />
      <LayerLabel position={cnnLabelAnchor(3)} title={t('layer.dense')} sub="10 · ReLU" />
      <LayerLabel position={cnnLabelAnchor(4)} title={t('layer.output')} sub="softmax" />

      <OutputLabels arch="cnn" />
      <SelectionMarker />
    </group>
  )
}
