import { useMemo } from 'react'
import * as THREE from 'three'
import { MODELS } from '../nn/models'
import { maxAbs } from '../nn/mlp'
import { useStore, useT } from '../store'
import { MLP_SIZES, mlpLabelAnchor, mlpPos } from './layout'
import { Segment } from './common'
import { VectorNodes } from './VectorNodes'
import { Connections } from './Connections'
import { FlowParticles } from './FlowParticles'
import { InputLabels, LayerLabel, OutputLabels } from './Labels'
import { SelectionMarker } from './SelectionMarker'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

export function MLPScene() {
  const model = MODELS.mlp.model
  const input = useStore((s) => s.mlpInput)
  const trace = useStore((s) => s.mlpTrace)
  const t = useT()

  const layerValues = useMemo(() => [input, ...trace.map((l) => l.a)], [input, trace])
  const scales = useMemo(() => layerValues.map((vs) => maxAbs(vs)), [layerValues])
  const positions = useMemo(
    () => MLP_SIZES.map((n, li) => Array.from({ length: n }, (_, i) => mlpPos(li - 1, i))),
    [],
  )

  const segmentsByLayer = useMemo(
    () =>
      model.layers.map((layer, k) => {
        const segs: Segment[] = []
        layer.weights.forEach((row, j) =>
          row.forEach((w, i) => segs.push({ a: v(mlpPos(k - 1, i)), b: v(mlpPos(k, j)), w, target: j })),
        )
        return segs
      }),
    [model],
  )

  const lastLayer = model.layers.length - 1

  return (
    <group>
      {layerValues.map((vals, li) => (
        <VectorNodes
          key={li}
          positions={positions[li]}
          values={vals}
          scale={scales[li]}
          layerIndex={li - 1}
          radius={0.3}
          refFor={(i) => ({ space: 'vector', layer: li - 1, index: i })}
        />
      ))}
      {segmentsByLayer.map((segs, k) => (
        <Connections key={`c${k}`} segments={segs} layerIndex={k} />
      ))}
      {segmentsByLayer.map((segs, k) => (
        <FlowParticles key={`p${k}`} segments={segs} layerIndex={k} />
      ))}

      <LayerLabel position={mlpLabelAnchor(-1)} title={t('layer.input')} sub={`${MLP_SIZES[0]}`} />
      {model.layers.map((layer, k) => (
        <LayerLabel
          key={`lb${k}`}
          position={mlpLabelAnchor(k)}
          title={k === lastLayer ? t('layer.output') : t('layer.hidden', { n: k + 1 })}
          sub={k === lastLayer ? 'softmax' : `${layer.biases.length} · ReLU`}
        />
      ))}

      <InputLabels />
      <OutputLabels arch="mlp" />
      <SelectionMarker />
    </group>
  )
}
