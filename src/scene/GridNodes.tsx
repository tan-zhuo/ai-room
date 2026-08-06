import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { ThreeEvent, useFrame } from '@react-three/fiber'
import { Tensor3 } from '../nn/cnn'
import { NodeRef, flow, sameRef, useStore } from '../store'
import { GridSlot, gridPos } from './layout'
import { COLOR_IDLE, activationColor } from './common'

interface Props {
  slot: GridSlot
  values: Tensor3
  scale: number
  /** allow painting cells with the mouse while draw mode is on */
  paintable?: boolean
}

/** A stack of feature-map sheets (or the input image) rendered as instanced cells. */
export function GridNodes({ slot, values, scale, paintable = false }: Props) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = useMemo(() => new THREE.Color(), [])
  const select = useStore((s) => s.select)
  const setHover = useStore((s) => s.setHover)
  const paintPixel = useStore((s) => s.paintPixel)
  const drawMode = useStore((s) => s.drawMode)
  const painting = paintable && drawMode

  const { channels, rows, cols, cell, layer } = slot
  const perCh = rows * cols
  const count = channels * perCh

  const toRef = (id: number): NodeRef => {
    const channel = Math.floor(id / perCh)
    const rem = id % perCh
    return { space: 'grid', layer, channel, row: Math.floor(rem / cols), col: rem % cols }
  }

  useFrame((state) => {
    const m = mesh.current
    if (!m) return
    const { step, playing, transitioning, explain } = useStore.getState()
    const t = state.clock.elapsedTime
    const computed = step > layer
    const inFlight = step === layer && (playing || transitioning)
    const dim = explain !== null && explain !== layer
    const revealCount = computed ? perCh : inFlight ? Math.floor(flow.phase * perCh) : 0
    for (let ch = 0; ch < channels; ch++) {
      for (let k = 0; k < perCh; k++) {
        const id = ch * perCh + k
        const r = Math.floor(k / cols)
        const c = k % cols
        const p = gridPos(slot, ch, r, c)
        const shown = k < revealCount
        const isCurrent = inFlight && k === revealCount
        dummy.position.set(p[0], p[1], p[2])
        const s = cell * (isCurrent ? 0.95 : 0.8)
        dummy.scale.set(s * 0.4, s, s)
        dummy.updateMatrix()
        m.setMatrixAt(id, dummy.matrix)
        if (shown) activationColor(values[ch][r][c], scale, color)
        else if (isCurrent) color.set('#ffffff').multiplyScalar(1.4 + 0.5 * Math.sin(t * 20))
        else color.copy(COLOR_IDLE)
        if (dim) color.multiplyScalar(0.22)
        m.setColorAt(id, color)
      }
    }
    m.instanceMatrix.needsUpdate = true
    if (m.instanceColor) m.instanceColor.needsUpdate = true
  })

  const onClick = (e: ThreeEvent<MouseEvent>) => {
    if (e.instanceId === undefined) return
    if (painting) {
      e.stopPropagation()
      const ref = toRef(e.instanceId)
      if (ref.space === 'grid') paintPixel(ref.row, ref.col)
      return
    }
    if (e.delta > 5) return
    e.stopPropagation()
    select(toRef(e.instanceId))
  }
  const onMove = (e: ThreeEvent<PointerEvent>) => {
    if (e.instanceId === undefined) return
    e.stopPropagation()
    const ref = toRef(e.instanceId)
    if (ref.space !== 'grid') return
    if (painting && e.buttons === 1) {
      paintPixel(ref.row, ref.col)
      return
    }
    const cur = useStore.getState().hoverInfo
    if (cur && sameRef(cur.ref, ref)) return
    setHover({ ref, value: values[ref.channel][ref.row][ref.col] })
  }
  const onDown = (e: ThreeEvent<PointerEvent>) => {
    if (!painting || e.instanceId === undefined) return
    e.stopPropagation()
    const ref = toRef(e.instanceId)
    if (ref.space === 'grid') paintPixel(ref.row, ref.col)
  }

  return (
    <instancedMesh
      ref={mesh}
      args={[undefined, undefined, count]}
      frustumCulled={false}
      onClick={onClick}
      onPointerMove={onMove}
      onPointerDown={onDown}
      onPointerOver={() => (document.body.style.cursor = painting ? 'crosshair' : 'pointer')}
      onPointerOut={() => {
        document.body.style.cursor = ''
        setHover(null)
      }}
    >
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  )
}
