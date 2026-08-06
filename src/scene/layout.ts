// Pure geometry: where every node of every layer lives in 3D space.
// Layouts are derived from the CURRENT models (which can be rebuilt at a new
// scale), so cached slot tables are refreshed via refreshLayout().

import { MODELS } from '../nn/models'
import { CNNLayerDef, unflattenIndex } from '../nn/cnn'
import type { Arch, NodeRef } from '../store'

export type Vec3 = [number, number, number]
export type DenseArch = 'mlp' | 'text'

// ------------------------------------------------------------------ dense (MLP / TEXT)

export function denseSizes(arch: DenseArch): number[] {
  const task = MODELS[arch]
  return [task.model.inputSize, ...task.model.layers.map((l) => l.biases.length)]
}

const DENSE_GAP_X = 4.4

function denseGapY(n: number): number {
  return Math.min(1.4, 11 / n)
}

/** layer: -1 for input, otherwise model layer index. */
export function densePos(arch: DenseArch, layer: number, index: number): Vec3 {
  const sizes = denseSizes(arch)
  const li = layer + 1
  const n = sizes[li]
  return [(li - (sizes.length - 1) / 2) * DENSE_GAP_X, ((n - 1) / 2 - index) * denseGapY(n), 0]
}

export function denseLabelAnchor(arch: DenseArch, layer: number): Vec3 {
  const sizes = denseSizes(arch)
  const n = sizes[layer + 1]
  const [x] = densePos(arch, layer, 0)
  return [x, ((n - 1) / 2) * denseGapY(n) + 1.5, 0]
}

// ------------------------------------------------------------------ slot shapes

export interface GridSlot {
  kind: 'grid'
  layer: number
  channels: number
  rows: number
  cols: number
  cell: number
  x: number
  chGap: number
}

export interface VecSlot {
  kind: 'vector'
  layer: number
  size: number
  x: number
  gapY: number
}

export interface FlattenSlot {
  kind: 'flatten'
  layer: number
  size: number
  x: number
  gapY: number
  srcShape: [number, number, number]
}

export type CNNSlot = GridSlot | VecSlot | FlattenSlot

// ------------------------------------------------------------------ CNN slots

function computeCnnSlots(): CNNSlot[] {
  const model = MODELS.cnn.model
  const slots: CNNSlot[] = []
  const xs = [-9.5, -5, -0.8, 2.6, 5.6, 8.6]
  let shape: [number, number, number] = model.inputShape

  slots.push({
    kind: 'grid',
    layer: -1,
    channels: shape[0],
    rows: shape[1],
    cols: shape[2],
    cell: Math.min(0.55, 6 / shape[1]),
    x: xs[0],
    chGap: 0.7,
  })

  model.layers.forEach((def: CNNLayerDef, layer: number) => {
    const x = xs[layer + 1] ?? xs[xs.length - 1] + 3 * (layer + 2 - xs.length)
    if (def.type === 'conv') {
      const kh = def.kernels[0][0].length
      shape = [def.kernels.length, shape[1] - kh + 1, shape[2] - kh + 1]
      slots.push({
        kind: 'grid',
        layer,
        channels: shape[0],
        rows: shape[1],
        cols: shape[2],
        cell: Math.min(0.5, 5.6 / shape[1]),
        x,
        chGap: 0.75,
      })
    } else if (def.type === 'pool') {
      shape = [shape[0], Math.floor(shape[1] / def.size), Math.floor(shape[2] / def.size)]
      slots.push({
        kind: 'grid',
        layer,
        channels: shape[0],
        rows: shape[1],
        cols: shape[2],
        cell: Math.min(0.62, 4.6 / shape[1]),
        x,
        chGap: 0.75,
      })
    } else if (def.type === 'flatten') {
      const size = shape[0] * shape[1] * shape[2]
      const perCh = shape[1] * shape[2]
      slots.push({
        kind: 'flatten',
        layer,
        size,
        x,
        gapY: Math.min(0.52, 10.5 / perCh),
        srcShape: shape,
      })
    } else {
      const size = def.layer.biases.length
      slots.push({
        kind: 'vector',
        layer,
        size,
        x,
        gapY: size > 6 ? Math.min(0.72, 13 / size) : 1.15,
      })
    }
  })
  return slots
}

// ------------------------------------------------------------------ LLM slots
// Layer indices: -1 tokens, 0 embeddings, 1 Q/K/V, 2 attention, 3 A·V, 4 FFN, 5 output.

export const LLM_STEPS = 6

function computeLlmSlots(): CNNSlot[] {
  const m = MODELS.llm.model
  const { T, d, h } = m
  const V = m.vocab.length
  const xs = [-11, -7.4, -3.4, 0.4, 3.6, 6.6, 10]
  return [
    { kind: 'grid', layer: -1, channels: 1, rows: 1, cols: T, cell: 0.8, x: xs[0], chGap: 0 },
    { kind: 'grid', layer: 0, channels: 1, rows: T, cols: d, cell: 0.42, x: xs[1], chGap: 0 },
    { kind: 'grid', layer: 1, channels: 3, rows: T, cols: d, cell: 0.36, x: xs[2], chGap: 0.7 },
    { kind: 'grid', layer: 2, channels: 1, rows: T, cols: T, cell: 0.52, x: xs[3], chGap: 0 },
    { kind: 'grid', layer: 3, channels: 1, rows: T, cols: d, cell: 0.42, x: xs[4], chGap: 0 },
    { kind: 'grid', layer: 4, channels: 1, rows: T, cols: h, cell: 0.34, x: xs[5], chGap: 0 },
    { kind: 'vector', layer: 5, size: V, x: xs[6], gapY: Math.min(0.72, 12.5 / V) },
  ]
}

// ------------------------------------------------------------------ cached slot tables

let cnnSlots = computeCnnSlots()
let llmSlots = computeLlmSlots()

/** Re-derive slot tables after a model was rebuilt at a new scale. */
export function refreshLayout(): void {
  cnnSlots = computeCnnSlots()
  llmSlots = computeLlmSlots()
}

export function cnnSlot(layer: number): CNNSlot {
  return cnnSlots[layer + 1]
}

export function llmSlot(layer: number): CNNSlot {
  return llmSlots[layer + 1]
}

// ------------------------------------------------------------------ positions

export function gridPos(slot: GridSlot, ch: number, row: number, col: number): Vec3 {
  return [
    slot.x + (ch - (slot.channels - 1) / 2) * slot.chGap,
    ((slot.rows - 1) / 2 - row) * slot.cell,
    (col - (slot.cols - 1) / 2) * slot.cell,
  ]
}

export function flattenPos(slot: FlattenSlot, i: number): Vec3 {
  const { channel, row, col } = unflattenIndex(slot.srcShape, i)
  const perCh = slot.srcShape[1] * slot.srcShape[2]
  const k = row * slot.srcShape[2] + col
  return [slot.x, ((perCh - 1) / 2 - k) * slot.gapY, (channel - (slot.srcShape[0] - 1) / 2) * 1.2]
}

export function vecPos(slot: VecSlot, i: number): Vec3 {
  return [slot.x, ((slot.size - 1) / 2 - i) * slot.gapY, 0]
}

export function slotPos(slot: CNNSlot, opts: { index?: number; channel?: number; row?: number; col?: number }): Vec3 {
  if (slot.kind === 'grid') return gridPos(slot, opts.channel ?? 0, opts.row ?? 0, opts.col ?? 0)
  if (slot.kind === 'flatten') return flattenPos(slot, opts.index ?? 0)
  return vecPos(slot, opts.index ?? 0)
}

export function cnnPos(layer: number, opts: { index?: number; channel?: number; row?: number; col?: number }): Vec3 {
  return slotPos(cnnSlot(layer), opts)
}

export function llmPos(layer: number, opts: { index?: number; channel?: number; row?: number; col?: number }): Vec3 {
  return slotPos(llmSlot(layer), opts)
}

export function slotLabelAnchor(slot: CNNSlot): Vec3 {
  if (slot.kind === 'grid') return [slot.x, ((slot.rows - 1) / 2) * slot.cell + 1.6, 0]
  if (slot.kind === 'flatten') {
    const perCh = slot.srcShape[1] * slot.srcShape[2]
    return [slot.x, ((perCh - 1) / 2) * slot.gapY + 1.4, 0]
  }
  return [slot.x, ((slot.size - 1) / 2) * slot.gapY + 1.5, 0]
}

export function cnnLabelAnchor(layer: number): Vec3 {
  return slotLabelAnchor(cnnSlot(layer))
}

export function llmLabelAnchor(layer: number): Vec3 {
  return slotLabelAnchor(llmSlot(layer))
}

// ------------------------------------------------------------------ shared

export function positionOf(arch: Arch, ref: NodeRef): Vec3 {
  if (arch === 'mlp' || arch === 'text') {
    return densePos(arch, ref.layer, ref.space === 'vector' ? ref.index : 0)
  }
  const posFn = arch === 'cnn' ? cnnPos : llmPos
  if (ref.space === 'vector') return posFn(ref.layer, { index: ref.index })
  return posFn(ref.layer, { channel: ref.channel, row: ref.row, col: ref.col })
}

/** Axis-aligned bounds of all node centers in a layer (-1 = input). */
export function layerBounds(arch: Arch, layer: number): { min: Vec3; max: Vec3 } {
  const pts: Vec3[] = []
  if (arch === 'mlp' || arch === 'text') {
    const n = denseSizes(arch)[layer + 1]
    for (let i = 0; i < n; i++) pts.push(densePos(arch, layer, i))
  } else {
    const slot = arch === 'cnn' ? cnnSlot(layer) : llmSlot(layer)
    if (slot.kind === 'grid') {
      for (let ch = 0; ch < slot.channels; ch++) {
        pts.push(gridPos(slot, ch, 0, 0), gridPos(slot, ch, slot.rows - 1, slot.cols - 1))
      }
    } else if (slot.kind === 'flatten') {
      for (let i = 0; i < slot.size; i++) pts.push(flattenPos(slot, i))
    } else {
      pts.push(vecPos(slot, 0), vecPos(slot, slot.size - 1))
    }
  }
  const min: Vec3 = [Infinity, Infinity, Infinity]
  const max: Vec3 = [-Infinity, -Infinity, -Infinity]
  for (const p of pts) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], p[a])
      max[a] = Math.max(max[a], p[a])
    }
  }
  return { min, max }
}

export const DEFAULT_VIEW: Record<Arch, { position: Vec3; target: Vec3 }> = {
  mlp: { position: [8, 4.5, 13], target: [0, 0, 0] },
  text: { position: [8, 4.5, 13], target: [0, 0, 0] },
  cnn: { position: [11, 6, 18], target: [0, 0, 0] },
  llm: { position: [12.5, 6, 20], target: [0.5, 0, 0] },
}
