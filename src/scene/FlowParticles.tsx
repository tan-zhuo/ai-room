import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { flow, useStore } from '../store'
import { COLOR_NEG, COLOR_PARTICLE, COLOR_POS, Segment, ease } from './common'

interface Props {
  segments: Segment[]
  layerIndex: number
  size?: number
  /** fire whenever playback is running, regardless of the current step */
  continuous?: boolean
}

/** Glowing particles that travel along connections while a layer is being computed. */
export function FlowParticles({ segments, layerIndex, size = 0.1, continuous = false }: Props) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = useMemo(() => new THREE.Color(), [])

  useFrame(() => {
    const m = mesh.current
    if (!m) return
    const { step, playing, transitioning } = useStore.getState()
    const active = (continuous || step === layerIndex) && (playing || transitioning)
    m.visible = active
    if (!active) return
    for (let i = 0; i < segments.length; i++) {
      const s = segments[i]
      const stagger = (((i * 7919) % 23) / 23) * 0.3
      const t = ease(flow.phase * 1.35 - stagger)
      const vis = t > 0.001 && t < 0.999
      dummy.position.lerpVectors(s.a, s.b, t)
      dummy.scale.setScalar(vis ? size * (0.6 + 0.7 * Math.sin(t * Math.PI)) : 0.0001)
      dummy.updateMatrix()
      m.setMatrixAt(i, dummy.matrix)
      color.copy(COLOR_PARTICLE).lerp(s.w >= 0 ? COLOR_POS : COLOR_NEG, 0.45).multiplyScalar(1.6)
      m.setColorAt(i, color)
    }
    m.instanceMatrix.needsUpdate = true
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  })

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, segments.length]}
      frustumCulled={false}
      raycast={() => null}
    >
      <sphereGeometry args={[1, 10, 10]} />
      <meshBasicMaterial
        transparent
        opacity={0.95}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        toneMapped={false}
      />
    </instancedMesh>
  )
}
