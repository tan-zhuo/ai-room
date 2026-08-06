import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { flow, useStore } from '../store'
import { GridSlot, cnnSlot, gridPos } from './layout'

interface Props {
  mode: 'conv' | 'pool'
}

/**
 * The sliding-window animation: a highlighted receptive field moves across the
 * source layer while the corresponding output pixel lights up, one pixel at a
 * time — exactly mirroring the real computation order.
 */
export function SlideAnim({ mode }: Props) {
  const layerIndex = mode === 'conv' ? 0 : 1
  const srcSlot = cnnSlot(layerIndex - 1) as GridSlot
  const outSlot = cnnSlot(layerIndex) as GridSlot
  const win = mode === 'conv' ? 3 : 2
  const stride = mode === 'conv' ? 1 : 2

  const group = useRef<THREE.Group>(null)
  const windowRefs = useRef<(THREE.LineSegments | null)[]>([])
  const outRefs = useRef<(THREE.Mesh | null)[]>([])

  const boxEdges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), [])

  // conv: one source window feeding every output channel; pool: one window per channel.
  const pairs = useMemo(
    () =>
      Array.from({ length: outSlot.channels }, (_, outCh) => ({
        srcCh: mode === 'conv' ? 0 : outCh,
        outCh,
      })),
    [mode, outSlot.channels],
  )
  const uniqueWindows = mode === 'conv' ? 1 : outSlot.channels

  const lines = useMemo(
    () =>
      pairs.map(() => {
        const g = new THREE.BufferGeometry()
        g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3))
        const mat = new THREE.LineBasicMaterial({
          color: '#5bd7ff',
          transparent: true,
          opacity: 0.8,
          blending: THREE.AdditiveBlending,
        })
        mat.toneMapped = false
        return new THREE.Line(g, mat)
      }),
    [pairs],
  )

  useFrame(() => {
    const g = group.current
    if (!g) return
    const { step, playing, transitioning } = useStore.getState()
    const active = step === layerIndex && (playing || transitioning)
    g.visible = active
    if (!active) return
    const total = outSlot.rows * outSlot.cols
    const oi = Math.min(total - 1, Math.floor(flow.phase * total))
    const oy = Math.floor(oi / outSlot.cols)
    const ox = oi % outSlot.cols
    const centerR = oy * stride + (win - 1) / 2
    const centerC = ox * stride + (win - 1) / 2

    for (let w = 0; w < uniqueWindows; w++) {
      const box = windowRefs.current[w]
      if (!box) continue
      const srcCh = mode === 'conv' ? 0 : w
      const p = gridPos(srcSlot, srcCh, centerR, centerC)
      box.position.set(p[0], p[1], p[2])
      box.scale.set(srcSlot.cell * 0.5, win * srcSlot.cell, win * srcSlot.cell)
    }
    pairs.forEach((pair, i) => {
      const out = outRefs.current[i]
      const src = gridPos(srcSlot, pair.srcCh, centerR, centerC)
      const dst = gridPos(outSlot, pair.outCh, oy, ox)
      if (out) {
        out.position.set(dst[0], dst[1], dst[2])
        out.scale.setScalar(outSlot.cell * 1.05)
      }
      const attr = lines[i].geometry.getAttribute('position') as THREE.BufferAttribute
      attr.setXYZ(0, src[0], src[1], src[2])
      attr.setXYZ(1, dst[0], dst[1], dst[2])
      attr.needsUpdate = true
    })
  })

  return (
    <group ref={group} visible={false}>
      {Array.from({ length: uniqueWindows }, (_, w) => (
        <lineSegments key={`w${w}`} ref={(el) => (windowRefs.current[w] = el)} geometry={boxEdges}>
          <lineBasicMaterial color="#9be8ff" transparent opacity={0.95} toneMapped={false} />
        </lineSegments>
      ))}
      {pairs.map((_, i) => (
        <mesh key={`o${i}`} ref={(el) => (outRefs.current[i] = el)}>
          <boxGeometry args={[1, 1, 1]} />
          <meshBasicMaterial
            color="#c9f2ff"
            transparent
            opacity={0.35}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            toneMapped={false}
          />
        </mesh>
      ))}
      {lines.map((l, i) => (
        <primitive key={`l${i}`} object={l} />
      ))}
    </group>
  )
}
