import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { Arch, useStore } from '../store'
import { layerBounds } from './layout'

/** Pulsing bounding box around the layer whose module explanation is open. */
export function LayerHighlight() {
  const arch = useStore((s) => s.arch)
  const explain = useStore((s) => s.explain)
  if (explain === null) return null
  return <HighlightBox key={`${arch}:${explain}`} arch={arch} layer={explain} />
}

function HighlightBox({ arch, layer }: { arch: Arch; layer: number }) {
  const edgeMat = useRef<THREE.LineBasicMaterial>(null)
  const fillMat = useRef<THREE.MeshBasicMaterial>(null)

  const { center, size } = useMemo(() => {
    const { min, max } = layerBounds(arch, layer)
    const pad = 0.9
    return {
      center: [
        (min[0] + max[0]) / 2,
        (min[1] + max[1]) / 2,
        (min[2] + max[2]) / 2,
      ] as [number, number, number],
      size: [
        Math.max(1.4, max[0] - min[0] + pad * 2),
        Math.max(1.4, max[1] - min[1] + pad * 2),
        Math.max(1.4, max[2] - min[2] + pad * 2),
      ] as [number, number, number],
    }
  }, [arch, layer])

  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)), [])

  useFrame((state) => {
    const pulse = 0.55 + 0.3 * Math.sin(state.clock.elapsedTime * 2.6)
    if (edgeMat.current) edgeMat.current.opacity = pulse
    if (fillMat.current) fillMat.current.opacity = 0.035 + 0.025 * pulse
  })

  return (
    <group position={center}>
      <lineSegments geometry={edges} scale={size}>
        <lineBasicMaterial ref={edgeMat} color="#38d6ff" transparent toneMapped={false} />
      </lineSegments>
      <mesh scale={size}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial
          ref={fillMat}
          color="#38d6ff"
          transparent
          depthWrite={false}
          side={THREE.DoubleSide}
          toneMapped={false}
        />
      </mesh>
    </group>
  )
}
