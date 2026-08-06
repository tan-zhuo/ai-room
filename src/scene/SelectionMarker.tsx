import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { MODELS } from '../nn/models'
import { unflattenIndex } from '../nn/cnn'
import { NodeRef, useStore } from '../store'
import {
  FlattenSlot,
  GridSlot,
  cnnPos,
  cnnSlot,
  gridPos,
  mlpPos,
  positionOf,
} from './layout'
import { Segment } from './common'
import { Connections } from './Connections'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

/** Ring + highlighted incoming connections for the selected node. */
export function SelectionMarker() {
  const arch = useStore((s) => s.arch)
  const selected = useStore((s) => s.selected)
  if (!selected) return null
  const key = `${arch}:${JSON.stringify(selected)}`
  return <Marker key={key} arch={arch} sel={selected} />
}

function Marker({ arch, sel }: { arch: 'mlp' | 'cnn'; sel: NodeRef }) {
  const pos = positionOf(arch, sel)
  const ring = useRef<THREE.Mesh>(null)

  const radius = useMemo(() => {
    if (arch === 'mlp') return 0.48
    if (sel.space === 'grid') {
      const slot = cnnSlot(sel.layer) as GridSlot
      return slot.cell * 0.75
    }
    return sel.layer === 2 ? 0.4 : 0.55
  }, [arch, sel])

  const segments = useMemo<Segment[]>(() => {
    const target = v(pos)
    const segs: Segment[] = []
    if (arch === 'mlp') {
      if (sel.space !== 'vector' || sel.layer < 0) return segs
      const weights = MODELS.mlp.model.layers[sel.layer].weights[sel.index]
      weights.forEach((w, i) => segs.push({ a: v(mlpPos(sel.layer - 1, i)), b: target, w, target: sel.index }))
      return segs
    }
    // CNN
    if (sel.space === 'grid' && sel.layer === 0) {
      // conv pixel <- 3x3 input patch, weighted by the kernel
      const inSlot = cnnSlot(-1) as GridSlot
      const def = MODELS.cnn.model.layers[0]
      if (def.type !== 'conv') return segs
      const kernel = def.kernels[sel.channel][0]
      for (let ky = 0; ky < 3; ky++)
        for (let kx = 0; kx < 3; kx++)
          segs.push({
            a: v(gridPos(inSlot, 0, sel.row + ky, sel.col + kx)),
            b: target,
            w: kernel[ky][kx],
            target: 0,
          })
      return segs
    }
    if (sel.space === 'grid' && sel.layer === 1) {
      // pool pixel <- 2x2 window on its own channel
      const convSlot = cnnSlot(0) as GridSlot
      for (let dy = 0; dy < 2; dy++)
        for (let dx = 0; dx < 2; dx++)
          segs.push({
            a: v(gridPos(convSlot, sel.channel, sel.row * 2 + dy, sel.col * 2 + dx)),
            b: target,
            w: 0.6,
            target: 0,
          })
      return segs
    }
    if (sel.space === 'vector' && sel.layer === 2) {
      // flatten node <- its pool pixel
      const slot = cnnSlot(2) as FlattenSlot
      const { channel, row, col } = unflattenIndex(slot.srcShape, sel.index)
      const poolSlot = cnnSlot(1) as GridSlot
      segs.push({ a: v(gridPos(poolSlot, channel, row, col)), b: target, w: 0.6, target: 0 })
      return segs
    }
    if (sel.space === 'vector' && (sel.layer === 3 || sel.layer === 4)) {
      const def = MODELS.cnn.model.layers[sel.layer]
      if (def.type !== 'dense') return segs
      def.layer.weights[sel.index].forEach((w, i) =>
        segs.push({ a: v(cnnPos(sel.layer - 1, { index: i })), b: target, w, target: sel.index }),
      )
      return segs
    }
    return segs
  }, [arch, sel, pos])

  useFrame((state) => {
    const r = ring.current
    if (!r) return
    const t = state.clock.elapsedTime
    r.scale.setScalar(1 + 0.12 * Math.sin(t * 4))
    r.rotation.y = t * 0.8
  })

  return (
    <group>
      <mesh ref={ring} position={pos}>
        <sphereGeometry args={[radius, 16, 12]} />
        <meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.4} toneMapped={false} />
      </mesh>
      {segments.length > 0 && (
        <Connections segments={segments} layerIndex={-999} highlight maxRadius={0.04} />
      )}
    </group>
  )
}
