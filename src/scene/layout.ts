// Pure geometry: where every node of every layer lives in 3D space.
// Used by meshes, connections, particles, selection highlights and camera focus.

import { MODELS } from '../nn/models'
import { CNNLayerDef, tensorShape, Tensor3, unflattenIndex } from '../nn/cnn'
import { Arch, NodeRef } from '../store'

export type Vec3 = [number, number, number]

// ------------------------------------------------------------------ MLP

export const MLP_SIZES = [MODELS.mlp.model.inputSize, ...MODELS.mlp.model.layers.map((l) => l.biases.length)]
const MLP_GAP_X = 4.4
const MLP_GAP_Y = 1.4

/** layer: -1 for input, otherwise model layer index. */
export function mlpPos(layer: number, index: number): Vec3 {
  const li = layer + 1
  const n = MLP_SIZES[li]
  return [(li - (MLP_SIZES.length - 1) / 2) * MLP_GAP_X, ((n - 1) / 2 - index) * MLP_GAP_Y, 0]
}

export function mlpLabelAnchor(layer: number): Vec3 {
  const li = layer + 1
  const n = MLP_SIZES[li]
  const [x] = mlpPos(layer, 0)
  return [x, ((n - 1) / 2) * MLP_GAP_Y + 1.5, 0]
}

// ------------------------------------------------------------------ CNN

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
  srcShape: [number, number, number]
}

export type CNNSlot = GridSlot | VecSlot | FlattenSlot

function computeCnnSlots(): CNNSlot[] {
  const model = MODELS.cnn.model
  const slots: CNNSlot[] = []
  const xs = [-9.5, -5, -0.8, 2.4, 5.4, 8.6]
  let shape: [number, number, number] = model.inputShape

  slots.push({ kind: 'grid', layer: -1, channels: shape[0], rows: shape[1], cols: shape[2], cell: 0.55, x: xs[0], chGap: 0.7 })

  model.layers.forEach((def: CNNLayerDef, layer: number) => {
    const x = xs[layer + 1] ?? xs[xs.length - 1] + 3 * (layer + 2 - xs.length)
    if (def.type === 'conv') {
      const kh = def.kernels[0][0].length
      shape = [def.kernels.length, shape[1] - kh + 1, shape[2] - kh + 1]
      slots.push({ kind: 'grid', layer, channels: shape[0], rows: shape[1], cols: shape[2], cell: 0.5, x, chGap: 0.75 })
    } else if (def.type === 'pool') {
      shape = [shape[0], Math.floor(shape[1] / def.size), Math.floor(shape[2] / def.size)]
      slots.push({ kind: 'grid', layer, channels: shape[0], rows: shape[1], cols: shape[2], cell: 0.62, x, chGap: 0.75 })
    } else if (def.type === 'flatten') {
      slots.push({ kind: 'flatten', layer, size: shape[0] * shape[1] * shape[2], x, srcShape: shape })
    } else {
      const size = def.layer.biases.length
      slots.push({ kind: 'vector', layer, size, x, gapY: size > 6 ? 0.72 : 1.15 })
    }
  })
  return slots
}

export const CNN_SLOTS = computeCnnSlots()

export function cnnSlot(layer: number): CNNSlot {
  return CNN_SLOTS[layer + 1]
}

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
  return [slot.x, ((perCh - 1) / 2 - k) * 0.52, (channel - (slot.srcShape[0] - 1) / 2) * 1.2]
}

export function vecPos(slot: VecSlot, i: number): Vec3 {
  return [slot.x, ((slot.size - 1) / 2 - i) * slot.gapY, 0]
}

export function cnnPos(layer: number, opts: { index?: number; channel?: number; row?: number; col?: number }): Vec3 {
  const slot = cnnSlot(layer)
  if (slot.kind === 'grid') return gridPos(slot, opts.channel ?? 0, opts.row ?? 0, opts.col ?? 0)
  if (slot.kind === 'flatten') return flattenPos(slot, opts.index ?? 0)
  return vecPos(slot, opts.index ?? 0)
}

export function cnnLabelAnchor(layer: number): Vec3 {
  const slot = cnnSlot(layer)
  if (slot.kind === 'grid') return [slot.x, ((slot.rows - 1) / 2) * slot.cell + 1.6, 0]
  if (slot.kind === 'flatten') {
    const perCh = slot.srcShape[1] * slot.srcShape[2]
    return [slot.x, ((perCh - 1) / 2) * 0.52 + 1.4, 0]
  }
  return [slot.x, ((slot.size - 1) / 2) * slot.gapY + 1.5, 0]
}

// ------------------------------------------------------------------ shared

export function positionOf(arch: Arch, ref: NodeRef): Vec3 {
  if (arch === 'mlp') {
    return mlpPos(ref.layer, ref.space === 'vector' ? ref.index : 0)
  }
  if (ref.space === 'vector') return cnnPos(ref.layer, { index: ref.index })
  return cnnPos(ref.layer, { channel: ref.channel, row: ref.row, col: ref.col })
}

export const DEFAULT_VIEW: Record<Arch, { position: Vec3; target: Vec3 }> = {
  mlp: { position: [8, 4.5, 13], target: [0, 0, 0] },
  cnn: { position: [10, 5.5, 15.5], target: [0, 0, 0] },
}

export function cnnTensorAt(input: Tensor3): [number, number, number] {
  return tensorShape(input)
}
