import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { MODELS, getVAETask } from '../nn/models'
import { unflattenIndex } from '../nn/cnn'
import { Arch, NodeRef, useStore } from '../store'
import {
  FlattenSlot,
  GridSlot,
  aeSlot,
  cnnPos,
  cnnSlot,
  densePos,
  diffSlot,
  ganSlot,
  gridPos,
  llmLayerOf,
  llmSlot,
  llmStageKind,
  lstmSlot,
  positionOf,
  rnnSlot,
  slotFor,
  vecPos,
  VecSlot,
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

function denseSegments(arch: 'mlp' | 'text', sel: NodeRef, target: THREE.Vector3): Segment[] {
  const segs: Segment[] = []
  if (sel.space !== 'vector' || sel.layer < 0) return segs
  const weights = MODELS[arch].model.layers[sel.layer].weights[sel.index]
  weights.forEach((w, i) =>
    segs.push({ a: v(densePos(arch, sel.layer - 1, i)), b: target, w, target: sel.index }),
  )
  return segs
}

function cnnSegments(sel: NodeRef, target: THREE.Vector3): Segment[] {
  const segs: Segment[] = []
  const model = MODELS.cnn.model
  if (sel.space === 'grid' && sel.layer === 0) {
    const inSlot = cnnSlot(-1) as GridSlot
    const def = model.layers[0]
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
    const slot = cnnSlot(2) as FlattenSlot
    const { channel, row, col } = unflattenIndex(slot.srcShape, sel.index)
    const poolSlot = cnnSlot(1) as GridSlot
    segs.push({ a: v(gridPos(poolSlot, channel, row, col)), b: target, w: 0.6, target: 0 })
    return segs
  }
  if (sel.space === 'vector' && (sel.layer === 3 || sel.layer === 4)) {
    const def = model.layers[sel.layer]
    if (def.type !== 'dense') return segs
    def.layer.weights[sel.index].forEach((w, i) =>
      segs.push({ a: v(cnnPos(sel.layer - 1, { index: i })), b: target, w, target: sel.index }),
    )
    return segs
  }
  return segs
}

function llmSegments(sel: NodeRef, target: THREE.Vector3): Segment[] {
  const segs: Segment[] = []
  const model = MODELS.llm.model
  const trace = useStore.getState().llmTrace
  const { d, dff, heads } = model
  const dh = d / heads
  const slot = (layer: number) => llmSlot(layer) as GridSlot
  const at = (kind: Parameters<typeof llmLayerOf>[0]) => slot(llmLayerOf(kind))
  const kind = llmStageKind(sel.layer)

  if (sel.space === 'grid' && kind === 'embed') {
    segs.push({ a: v(gridPos(slot(-1), 0, 0, sel.row)), b: target, w: 0.6, target: 0 })
    return segs
  }
  if (sel.space === 'grid' && kind === 'posenc') {
    segs.push({ a: v(gridPos(at('embed'), 0, sel.row, sel.col)), b: target, w: 0.6, target: 0 })
    return segs
  }
  if (sel.space === 'grid' && kind === 'qkv') {
    const W = [model.Wq, model.Wk, model.Wv][sel.channel]
    for (let m = 0; m < d; m++)
      segs.push({ a: v(gridPos(at('posenc'), 0, sel.row, m)), b: target, w: W[m][sel.col], target: 0 })
    return segs
  }
  if (sel.space === 'grid' && kind === 'attn') {
    const off = sel.channel * dh
    for (let m = 0; m < dh; m++) {
      segs.push({ a: v(gridPos(at('qkv'), 0, sel.row, off + m)), b: target, w: trace.Q[sel.row][off + m], target: 0 })
      segs.push({ a: v(gridPos(at('qkv'), 1, sel.col, off + m)), b: target, w: trace.K[sel.col][off + m], target: 0 })
    }
    return segs
  }
  if (sel.space === 'grid' && kind === 'attnout') {
    for (let h = 0; h < heads; h++) {
      for (let j = 0; j <= sel.row; j++) {
        segs.push({ a: v(gridPos(at('attn'), h, sel.row, j)), b: target, w: trace.A[h][sel.row][j], target: 0 })
      }
    }
    for (let j = 0; j <= sel.row; j++) {
      segs.push({ a: v(gridPos(at('qkv'), 2, j, Math.min(sel.col, d - 1))), b: target, w: trace.V[j][sel.col], target: 0 })
    }
    return segs
  }
  if (sel.space === 'grid' && kind === 'addnorm1') {
    segs.push({ a: v(gridPos(at('posenc'), 0, sel.row, sel.col)), b: target, w: 0.7, target: 0 })
    segs.push({ a: v(gridPos(at('attnout'), 0, sel.row, sel.col)), b: target, w: -0.7, target: 0 })
    return segs
  }
  if (sel.space === 'grid' && kind === 'ffn') {
    for (let m = 0; m < d; m++)
      segs.push({ a: v(gridPos(at('addnorm1'), 0, sel.row, m)), b: target, w: model.W1[m][sel.col], target: 0 })
    return segs
  }
  if (sel.space === 'grid' && kind === 'router') {
    for (let m = 0; m < d; m++)
      segs.push({ a: v(gridPos(at('addnorm1'), 0, sel.row, m)), b: target, w: model.Wr[m][sel.col], target: 0 })
    return segs
  }
  if (sel.space === 'grid' && kind === 'experts') {
    // expert cell <- routed input row (via that expert's W1) + its router gate
    segs.push({ a: v(gridPos(at('router'), 0, sel.row, sel.channel)), b: target, w: 0.8, target: 0 })
    for (let m = 0; m < d; m++)
      segs.push({ a: v(gridPos(at('addnorm1'), 0, sel.row, m)), b: target, w: model.We1[sel.channel][m][sel.col], target: 0 })
    return segs
  }
  if (sel.space === 'grid' && kind === 'combine') {
    for (const e of trace.topIdx[sel.row] ?? []) {
      for (let j = 0; j < model.dffE; j++) {
        segs.push({ a: v(gridPos(at('experts'), e, sel.row, j)), b: target, w: model.We2[e][j][sel.col], target: 0 })
      }
    }
    return segs
  }
  if (sel.space === 'grid' && kind === 'addnorm2') {
    segs.push({ a: v(gridPos(at('addnorm1'), 0, sel.row, sel.col)), b: target, w: 0.7, target: 0 })
    if (model.moe) {
      segs.push({ a: v(gridPos(at('combine'), 0, sel.row, sel.col)), b: target, w: -0.7, target: 0 })
    } else {
      for (let m = 0; m < dff; m++)
        segs.push({ a: v(gridPos(at('ffn'), 0, sel.row, m)), b: target, w: model.W2[m][sel.col], target: 0 })
    }
    return segs
  }
  if (sel.space === 'vector' && kind === 'output') {
    const T = trace.ids.length
    for (let k = 0; k < d; k++)
      segs.push({
        a: v(gridPos(at('addnorm2'), 0, T - 1, k)),
        b: target,
        w: model.Wout[k][sel.index],
        target: sel.index,
      })
    return segs
  }
  return segs
}

function rnnSegments(sel: NodeRef, target: THREE.Vector3): Segment[] {
  const segs: Segment[] = []
  const model = MODELS.rnn.model
  const trace = useStore.getState().rnnTrace
  if (sel.space === 'grid' && sel.layer === 0) {
    segs.push({ a: v(gridPos(rnnSlot(-1) as GridSlot, 0, 0, sel.row)), b: target, w: 0.6, target: 0 })
    return segs
  }
  if (sel.space === 'grid' && sel.layer === 1) {
    // h_t[j] <- x_t (via Wx) and h_{t-1} (via Wh)
    const embS = rnnSlot(0) as GridSlot
    const hidS = rnnSlot(1) as GridSlot
    for (let i = 0; i < model.d; i++)
      segs.push({ a: v(gridPos(embS, 0, sel.row, i)), b: target, w: model.Wx[i][sel.col], target: 0 })
    if (sel.row > 0)
      for (let i = 0; i < model.h; i++)
        segs.push({ a: v(gridPos(hidS, 0, sel.row - 1, i)), b: target, w: model.Wh[i][sel.col], target: 0 })
    return segs
  }
  if (sel.space === 'vector' && sel.layer === 2) {
    const hidS = rnnSlot(1) as GridSlot
    const T = trace.ids.length
    for (let j = 0; j < model.h; j++)
      segs.push({ a: v(gridPos(hidS, 0, T - 1, j)), b: target, w: model.Wy[j][sel.index], target: sel.index })
    return segs
  }
  return segs
}

function lstmSegments(sel: NodeRef, target: THREE.Vector3): Segment[] {
  const segs: Segment[] = []
  const model = MODELS.lstm.model
  const trace = useStore.getState().lstmTrace
  const { d, h } = model
  if (sel.space === 'grid' && sel.layer === 0) {
    segs.push({ a: v(gridPos(lstmSlot(-1) as GridSlot, 0, 0, sel.row)), b: target, w: 0.6, target: 0 })
    return segs
  }
  if (sel.space === 'grid' && sel.layer === 1) {
    // gate cell <- x_t and h_{t-1}
    const W = [model.Wf, model.Wi, model.Wg, model.Wo][sel.channel]
    const embS = lstmSlot(0) as GridSlot
    const hidS = lstmSlot(3) as GridSlot
    for (let i = 0; i < d; i++)
      segs.push({ a: v(gridPos(embS, 0, sel.row, i)), b: target, w: W[i][sel.col], target: 0 })
    if (sel.row > 0)
      for (let i = 0; i < h; i++)
        segs.push({ a: v(gridPos(hidS, 0, sel.row - 1, i)), b: target, w: W[d + i][sel.col], target: 0 })
    return segs
  }
  if (sel.space === 'grid' && sel.layer === 2) {
    // c = f⊙c_prev + i⊙g
    const gateS = lstmSlot(1) as GridSlot
    const cellS = lstmSlot(2) as GridSlot
    segs.push({ a: v(gridPos(gateS, 0, sel.row, sel.col)), b: target, w: trace.F[sel.row][sel.col], target: 0 })
    segs.push({ a: v(gridPos(gateS, 1, sel.row, sel.col)), b: target, w: trace.I[sel.row][sel.col], target: 0 })
    segs.push({ a: v(gridPos(gateS, 2, sel.row, sel.col)), b: target, w: trace.G[sel.row][sel.col], target: 0 })
    if (sel.row > 0)
      segs.push({ a: v(gridPos(cellS, 0, sel.row - 1, sel.col)), b: target, w: 0.5, target: 0 })
    return segs
  }
  if (sel.space === 'grid' && sel.layer === 3) {
    // h = o⊙tanh(c)
    const gateS = lstmSlot(1) as GridSlot
    const cellS = lstmSlot(2) as GridSlot
    segs.push({ a: v(gridPos(gateS, 3, sel.row, sel.col)), b: target, w: trace.O[sel.row][sel.col], target: 0 })
    segs.push({ a: v(gridPos(cellS, 0, sel.row, sel.col)), b: target, w: trace.Ct[sel.row][sel.col], target: 0 })
    return segs
  }
  if (sel.space === 'vector' && sel.layer === 4) {
    const hidS = lstmSlot(3) as GridSlot
    const T = trace.ids.length
    for (let j = 0; j < h; j++)
      segs.push({ a: v(gridPos(hidS, 0, T - 1, j)), b: target, w: model.Wy[j][sel.index], target: sel.index })
    return segs
  }
  return segs
}

function vaeSegments(sel: NodeRef, target: THREE.Vector3): Segment[] {
  const segs: Segment[] = []
  const task = getVAETask()
  const model = task.model
  if (sel.space === 'vector' && sel.layer === 0) {
    const inS = aeSlot(-1) as GridSlot
    model.enc.weights[sel.index].forEach((w, i) =>
      segs.push({
        a: v(gridPos(inS, 0, Math.floor(i / task.n), i % task.n)),
        b: target,
        w,
        target: sel.index,
      }),
    )
    return segs
  }
  if (sel.space === 'grid' && sel.layer === 1) {
    const encS = aeSlot(0) as VecSlot
    const head = sel.col === 0 ? model.muHead : model.lvHead
    head.weights[sel.row].forEach((w, i) =>
      segs.push({ a: v(vecPos(encS, i)), b: target, w, target: 0 }),
    )
    return segs
  }
  if (sel.space === 'vector' && sel.layer === 2) {
    const msS = aeSlot(1) as GridSlot
    segs.push({ a: v(gridPos(msS, 0, sel.index, 0)), b: target, w: 0.8, target: sel.index })
    segs.push({ a: v(gridPos(msS, 0, sel.index, 1)), b: target, w: -0.8, target: sel.index })
    return segs
  }
  if (sel.space === 'vector' && sel.layer === 3) {
    const zS = aeSlot(2) as VecSlot
    model.dec.weights[sel.index].forEach((w, i) =>
      segs.push({ a: v(vecPos(zS, i)), b: target, w, target: sel.index }),
    )
    return segs
  }
  if (sel.space === 'grid' && sel.layer === 4) {
    const decS = aeSlot(3) as VecSlot
    const idx = sel.row * task.n + sel.col
    model.out.weights[idx].forEach((w, i) =>
      segs.push({ a: v(vecPos(decS, i)), b: target, w, target: 0 }),
    )
    return segs
  }
  return segs
}

function aeSegments(sel: NodeRef, target: THREE.Vector3): Segment[] {
  if (useStore.getState().aeVariant === 'vae') return vaeSegments(sel, target)
  const segs: Segment[] = []
  const task = MODELS.ae
  const model = task.model
  if (sel.space === 'vector' && sel.layer >= 0 && sel.layer <= 2) {
    const weights = model.layers[sel.layer].weights[sel.index]
    if (sel.layer === 0) {
      const inS = aeSlot(-1) as GridSlot
      weights.forEach((w, i) =>
        segs.push({
          a: v(gridPos(inS, 0, Math.floor(i / task.n), i % task.n)),
          b: target,
          w,
          target: sel.index,
        }),
      )
    } else {
      const prevS = aeSlot(sel.layer - 1) as VecSlot
      weights.forEach((w, i) => segs.push({ a: v(vecPos(prevS, i)), b: target, w, target: sel.index }))
    }
    return segs
  }
  if (sel.space === 'grid' && sel.layer === 3) {
    const decS = aeSlot(2) as VecSlot
    const idx = sel.row * task.n + sel.col
    model.layers[3].weights[idx].forEach((w, i) =>
      segs.push({ a: v(vecPos(decS, i)), b: target, w, target: 0 }),
    )
    return segs
  }
  return segs
}

function diffSegments(sel: NodeRef, target: THREE.Vector3): Segment[] {
  const segs: Segment[] = []
  if (sel.space !== 'grid') return segs
  const task = MODELS.diff
  const model = task.model
  if (sel.layer === 1) {
    // x̂₀ pixel <- denoiser hidden sheet, weighted by the output layer
    const hS = diffSlot(0) as GridSlot
    const idx = sel.row * task.n + sel.col
    model.out.weights[idx].forEach((w, m) =>
      segs.push({ a: v(gridPos(hS, 0, Math.floor(m / hS.cols), m % hS.cols)), b: target, w, target: 0 }),
    )
    return segs
  }
  if (sel.layer === 2) {
    // x_{t-1} pixel <- its x̂₀ and x_t counterparts
    segs.push({ a: v(gridPos(diffSlot(1) as GridSlot, 0, sel.row, sel.col)), b: target, w: 0.8, target: 0 })
    segs.push({ a: v(gridPos(diffSlot(-1) as GridSlot, 0, sel.row, sel.col)), b: target, w: -0.8, target: 0 })
    return segs
  }
  return segs
}

function ganSegments(sel: NodeRef, target: THREE.Vector3): Segment[] {
  const segs: Segment[] = []
  const task = MODELS.gan
  const model = task.model
  if (sel.space === 'vector') {
    if (sel.layer === 0) {
      const zS = ganSlot(-1) as VecSlot
      model.g1.weights[sel.index].forEach((w, i) =>
        segs.push({ a: v(vecPos(zS, i)), b: target, w, target: sel.index }),
      )
    } else if (sel.layer === 2) {
      // discriminator hidden <- both images through the same weights
      const fakeS = ganSlot(1) as GridSlot
      const realS = ganSlot(4) as GridSlot
      model.d1.weights[sel.index].forEach((w, i) => {
        const r = Math.floor(i / task.n)
        const c = i % task.n
        segs.push({ a: v(gridPos(fakeS, 0, r, c)), b: target, w, target: sel.index })
        segs.push({ a: v(gridPos(realS, 0, r, c)), b: target, w, target: sel.index })
      })
    } else if (sel.layer === 3) {
      const dS = ganSlot(2) as VecSlot
      model.d2.weights[0].forEach((w, i) => segs.push({ a: v(vecPos(dS, i)), b: target, w, target: 0 }))
    }
    return segs
  }
  if (sel.layer === 1) {
    const gS = ganSlot(0) as VecSlot
    const idx = sel.row * task.n + sel.col
    model.g2.weights[idx].forEach((w, i) => segs.push({ a: v(vecPos(gS, i)), b: target, w, target: 0 }))
  }
  return segs
}

function Marker({ arch, sel }: { arch: Arch; sel: NodeRef }) {
  const pos = positionOf(arch, sel)
  const ring = useRef<THREE.Mesh>(null)

  const radius = useMemo(() => {
    if (arch === 'mlp' || arch === 'text') return 0.48
    if (sel.space === 'grid') {
      const slot = slotFor(arch, sel.layer) as GridSlot
      return slot.cell * 0.75
    }
    return sel.layer === 2 && arch === 'cnn' ? 0.4 : 0.5
  }, [arch, sel])

  const segments = useMemo<Segment[]>(() => {
    const target = v(pos)
    if (arch === 'mlp' || arch === 'text') return denseSegments(arch, sel, target)
    if (arch === 'cnn') return cnnSegments(sel, target)
    if (arch === 'llm') return llmSegments(sel, target)
    if (arch === 'rnn') return rnnSegments(sel, target)
    if (arch === 'lstm') return lstmSegments(sel, target)
    if (arch === 'diff') return diffSegments(sel, target)
    if (arch === 'gan') return ganSegments(sel, target)
    return aeSegments(sel, target)
  }, [arch, sel, pos])

  useFrame((state) => {
    const r = ring.current
    if (!r) return
    // fixed size — only a slow rotation so it reads as "selected", not alive
    r.rotation.y = state.clock.elapsedTime * 0.8
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
