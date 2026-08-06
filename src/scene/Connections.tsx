import { useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { useStore } from '../store'
import { Segment, orientSegment, segMaxW, weightColor } from './common'

interface Props {
  segments: Segment[]
  /** target layer index — connections brighten once step > layerIndex */
  layerIndex: number
  maxRadius?: number
  /** always-bright mode used for selection highlights */
  highlight?: boolean
}

export function Connections({ segments, layerIndex, maxRadius = 0.03, highlight = false }: Props) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = useMemo(() => new THREE.Color(), [])
  const maxW = useMemo(() => segMaxW(segments), [segments])

  useLayoutEffect(() => {
    const m = mesh.current
    if (!m) return
    segments.forEach((s, i) => {
      const radius = (0.012 + maxRadius * (Math.abs(s.w) / maxW)) * (highlight ? 1.7 : 1)
      orientSegment(dummy, s.a, s.b, radius)
      dummy.updateMatrix()
      m.setMatrixAt(i, dummy.matrix)
    })
    m.instanceMatrix.needsUpdate = true
  }, [segments, maxW, maxRadius, highlight, dummy])

  // dense scales produce thousands of lines — attenuate so additive blending doesn't blow out
  const crowd = Math.min(1, 700 / Math.max(1, segments.length))

  useFrame(() => {
    const m = mesh.current
    if (!m) return
    const { step, selected, explain } = useStore.getState()
    const active = highlight || step > layerIndex
    const selHere =
      !highlight && selected && selected.space === 'vector' && selected.layer === layerIndex
    for (let i = 0; i < segments.length; i++) {
      let b = active ? 1 : 0.3
      if (selHere && selected.space === 'vector')
        b = selected.index === segments[i].target ? 2.6 : 0.1
      if (highlight) b = 1.8
      else if (explain !== null) b *= explain === layerIndex ? 1.3 : 0.1
      if (!highlight) b *= crowd
      weightColor(segments[i].w, maxW, b, color)
      m.setColorAt(i, color)
    }
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  })

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, segments.length]}
      frustumCulled={false}
      raycast={() => null}
    >
      <cylinderGeometry args={[1, 1, 1, 5, 1, true]} />
      <meshBasicMaterial
        transparent
        opacity={0.9}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        toneMapped={false}
      />
    </instancedMesh>
  )
}
