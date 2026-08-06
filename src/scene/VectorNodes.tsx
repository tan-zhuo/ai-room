import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { ThreeEvent, useFrame } from '@react-three/fiber'
import { NodeRef, sameRef, useStore } from '../store'
import { Vec3 } from './layout'
import { COLOR_IDLE, activationColor } from './common'

interface Props {
  positions: Vec3[]
  values: number[]
  scale: number
  /** model layer index; -1 = input (always visible) */
  layerIndex: number
  radius: number
  refFor: (i: number) => NodeRef
}

export function VectorNodes({ positions, values, scale, layerIndex, radius, refFor }: Props) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = useMemo(() => new THREE.Color(), [])
  const select = useStore((s) => s.select)
  const setHover = useStore((s) => s.setHover)

  useFrame((state) => {
    const m = mesh.current
    if (!m) return
    const { step, playing, transitioning, explain } = useStore.getState()
    const t = state.clock.elapsedTime
    const computed = step > layerIndex
    const inFlight = step === layerIndex && (playing || transitioning)
    const dim = explain !== null && explain !== layerIndex
    for (let i = 0; i < positions.length; i++) {
      const p = positions[i]
      const norm = computed ? Math.min(1, Math.abs(values[i]) / (scale || 1)) : 0
      const pulse = computed ? 1 + 0.08 * norm * Math.sin(t * 2.5 + i * 0.9) : 1
      dummy.position.set(p[0], p[1], p[2])
      dummy.scale.setScalar(radius * pulse)
      dummy.updateMatrix()
      m.setMatrixAt(i, dummy.matrix)
      if (computed) activationColor(values[i], scale, color)
      else color.copy(COLOR_IDLE).multiplyScalar(inFlight ? 1.7 + 0.6 * Math.sin(t * 6) : 1)
      if (dim) color.multiplyScalar(0.22)
      m.setColorAt(i, color)
    }
    m.instanceMatrix.needsUpdate = true
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  })

  const onClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.delta > 5 || e.instanceId === undefined) return
    e.stopPropagation()
    select(refFor(e.instanceId))
  }
  const onMove = (e: ThreeEvent<PointerEvent>) => {
    if (e.instanceId === undefined) return
    e.stopPropagation()
    const ref = refFor(e.instanceId)
    const cur = useStore.getState().hoverInfo
    if (cur && sameRef(cur.ref, ref)) return
    setHover({ ref, value: values[e.instanceId] })
  }

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, positions.length]}
      frustumCulled={false}
      onClick={onClick}
      onPointerMove={onMove}
      onPointerOver={() => (document.body.style.cursor = 'pointer')}
      onPointerOut={() => {
        document.body.style.cursor = ''
        setHover(null)
      }}
    >
      <sphereGeometry args={[1, 24, 24]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  )
}
