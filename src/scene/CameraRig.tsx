import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useStore } from '../store'
import { DEFAULT_VIEW } from './layout'
import { ease } from './common'

interface CamAnim {
  t: number
  fromPos: THREE.Vector3
  toPos: THREE.Vector3
  fromTarget: THREE.Vector3
  toTarget: THREE.Vector3
}

export function CameraRig() {
  const controls = useRef<OrbitControlsImpl>(null)
  const anim = useRef<CamAnim | null>(null)
  const focusNonce = useStore((s) => s.focusNonce)
  const drawMode = useStore((s) => s.drawMode)
  const { camera, size } = useThree()

  useEffect(() => {
    const c = controls.current
    if (!c) return
    const { focusTarget, focusDistance, arch } = useStore.getState()
    let toTarget: THREE.Vector3
    let toPos: THREE.Vector3
    if (focusTarget) {
      toTarget = new THREE.Vector3(...focusTarget)
      const dir = camera.position.clone().sub(c.target).normalize()
      toPos = toTarget.clone().add(dir.multiplyScalar(focusDistance))
    } else {
      if (focusNonce === 0) return
      const view = DEFAULT_VIEW[arch]
      toTarget = new THREE.Vector3(...view.target)
      // narrow (portrait) screens need more distance to fit the network
      const aspect = size.width / size.height
      const f = Math.min(2.8, Math.max(1, 1.15 / aspect))
      toPos = new THREE.Vector3(...view.position).multiplyScalar(f)
    }
    anim.current = {
      t: 0,
      fromPos: camera.position.clone(),
      toPos,
      fromTarget: c.target.clone(),
      toTarget,
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusNonce])

  useEffect(() => {
    const c = controls.current
    if (!c) return
    const cancel = () => (anim.current = null)
    c.addEventListener('start', cancel)
    return () => c.removeEventListener('start', cancel)
  }, [])

  useFrame((_, dt) => {
    const a = anim.current
    const c = controls.current
    if (!a || !c) return
    a.t = Math.min(1, a.t + dt / 0.9)
    const k = ease(a.t)
    camera.position.lerpVectors(a.fromPos, a.toPos, k)
    c.target.lerpVectors(a.fromTarget, a.toTarget, k)
    c.update()
    if (a.t >= 1) anim.current = null
  })

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enabled={!drawMode}
      enableDamping
      dampingFactor={0.08}
      minDistance={2}
      maxDistance={70}
      rotateSpeed={0.7}
    />
  )
}
