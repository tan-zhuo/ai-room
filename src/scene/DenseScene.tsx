import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { maxAbs } from '../nn/mlp'
import { useStore, useT } from '../store'
import { DenseArch, denseLabelAnchor, densePos, denseSizes } from './layout'
import { Segment } from './common'
import { VectorNodes } from './VectorNodes'
import { Connections } from './Connections'
import { FlowParticles } from './FlowParticles'
import { InputLabels, LayerLabel, OutputLabels } from './Labels'
import { SelectionMarker } from './SelectionMarker'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

/** Fully-connected network scene, shared by the MLP and TEXT architectures. */
export function DenseScene({ arch }: { arch: DenseArch }) {
  const model = MODELS[arch].model
  const input = useStore((s) => (arch === 'mlp' ? s.mlpInput : s.textFeatures))
  const trace = useStore((s) => (arch === 'mlp' ? s.mlpTrace : s.textTrace))
  const textRaw = useStore((s) => s.textRaw)
  const t = useT()

  const sizes = denseSizes(arch)
  const layerValues = useMemo(() => [input, ...trace.map((l) => l.a)], [input, trace])
  const scales = useMemo(() => layerValues.map((vs) => maxAbs(vs)), [layerValues])
  const positions = useMemo(
    () => sizes.map((n, li) => Array.from({ length: n }, (_, i) => densePos(arch, li - 1, i))),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [arch],
  )

  const segmentsByLayer = useMemo(
    () =>
      model.layers.map((layer, k) => {
        const segs: Segment[] = []
        layer.weights.forEach((row, j) =>
          row.forEach((w, i) =>
            segs.push({ a: v(densePos(arch, k - 1, i)), b: v(densePos(arch, k, j)), w, target: j }),
          ),
        )
        return segs
      }),
    [model, arch],
  )

  const lastLayer = model.layers.length - 1
  const inputAnchor = denseLabelAnchor(arch, -1)

  return (
    <group>
      {layerValues.map((vals, li) => (
        <VectorNodes
          key={li}
          positions={positions[li]}
          values={vals}
          scale={scales[li]}
          layerIndex={li - 1}
          radius={Math.min(0.3, 2.4 / sizes[li] + 0.14)}
          refFor={(i) => ({ space: 'vector', layer: li - 1, index: i })}
        />
      ))}
      {segmentsByLayer.map((segs, k) => (
        <Connections key={`c${k}`} segments={segs} layerIndex={k} />
      ))}
      {segmentsByLayer.map((segs, k) => (
        <FlowParticles key={`p${k}`} segments={segs} layerIndex={k} />
      ))}

      <LayerLabel position={inputAnchor} title={t('layer.input')} sub={`${sizes[0]}`} layer={-1} />
      {model.layers.map((layer, k) => (
        <LayerLabel
          key={`lb${k}`}
          position={denseLabelAnchor(arch, k)}
          title={k === lastLayer ? t('layer.output') : t('layer.hidden', { n: k + 1 })}
          sub={k === lastLayer ? 'softmax' : `${layer.biases.length} · ReLU`}
          layer={k}
        />
      ))}

      {arch === 'text' && (
        <Html
          position={[inputAnchor[0], -inputAnchor[1] - 0.6, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="raw-text-label">
            “{textRaw.length > 26 ? textRaw.slice(0, 26) + '…' : textRaw}”
          </div>
        </Html>
      )}

      <InputLabels arch={arch} />
      <OutputLabels arch={arch} />
      <SelectionMarker />
    </group>
  )
}
