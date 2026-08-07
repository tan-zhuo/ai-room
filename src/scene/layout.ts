// Pure geometry: where every node of every layer lives in 3D space.
// Layouts are derived from the CURRENT models (which can be rebuilt at a new
// scale), so cached slot tables are refreshed via refreshLayout().

import { AE_VARIANT, MODELS, getVAETask } from '../nn/models'
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
  /** vertical offset of the whole sheet (GAN stacks fake/real images) */
  y?: number
}

export interface VecSlot {
  kind: 'vector'
  layer: number
  size: number
  x: number
  gapY: number
  y?: number
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
// Dense: tokenizer → embed → posenc → QKV → attention → A·V·Wo → Add&Norm →
// FFN → Add&Norm → output. MoE swaps the FFN stage for router → experts →
// weighted combine (11 steps instead of 9).

export type LLMStageKind =
  | 'tokens'
  | 'embed'
  | 'posenc'
  | 'qkv'
  | 'attn'
  | 'attnout'
  | 'addnorm1'
  | 'ffn'
  | 'router'
  | 'experts'
  | 'combine'
  | 'addnorm2'
  | 'output'

let llmKinds: LLMStageKind[] = []

function computeLlmSlots(): CNNSlot[] {
  const m = MODELS.llm.model
  const { T, d, dff, heads } = m
  const V = m.vocab.length
  const grid = (
    layer: number,
    channels: number,
    rows: number,
    cols: number,
    cell: number,
    x: number,
    chGap = 0,
  ): CNNSlot => ({ kind: 'grid', layer, channels, rows, cols, cell, x, chGap })

  // cells shrink as widths grow so larger scales still fit the fixed x gaps
  const cd = Math.min(0.34, 4.2 / d)
  const cq = Math.min(0.3, 3.6 / d)
  const ca = Math.min(0.44, 4.6 / T)
  const cf = Math.min(0.27, 4.4 / dff)
  if (!m.moe) {
    llmKinds = ['tokens', 'embed', 'posenc', 'qkv', 'attn', 'attnout', 'addnorm1', 'ffn', 'addnorm2', 'output']
    const xs = [-14, -10.9, -7.9, -4.7, -1.2, 1.9, 4.6, 7.3, 9.9, 12.7]
    return [
      grid(-1, 1, 1, T, Math.min(0.7, 6.4 / T), xs[0]),
      grid(0, 1, T, d, cd, xs[1]),
      grid(1, 1, T, d, cd, xs[2]),
      grid(2, 3, T, d, cq, xs[3], 0.62),
      grid(3, heads, T, T, ca, xs[4], Math.min(0.85, 3 / heads)),
      grid(4, 1, T, d, cd, xs[5]),
      grid(5, 1, T, d, cd, xs[6]),
      grid(6, 1, T, dff, cf, xs[7]),
      grid(7, 1, T, d, cd, xs[8]),
      { kind: 'vector', layer: 8, size: V, x: xs[9], gapY: Math.min(0.72, 12.5 / V) },
    ]
  }

  llmKinds = ['tokens', 'embed', 'posenc', 'qkv', 'attn', 'attnout', 'addnorm1', 'router', 'experts', 'combine', 'addnorm2', 'output']
  const xs = [-16.4, -13.4, -10.6, -7.6, -4.2, -1.3, 1.4, 3.9, 6.7, 9.6, 12.2, 15]
  const ce = Math.min(0.26, 3.4 / m.dffE)
  return [
    grid(-1, 1, 1, T, Math.min(0.7, 6.4 / T), xs[0]),
    grid(0, 1, T, d, cd, xs[1]),
    grid(1, 1, T, d, cd, xs[2]),
    grid(2, 3, T, d, cq, xs[3], 0.6),
    grid(3, heads, T, T, Math.min(0.4, 4.2 / T), xs[4], Math.min(0.8, 2.8 / heads)),
    grid(4, 1, T, d, cd, xs[5]),
    grid(5, 1, T, d, cd, xs[6]),
    grid(6, 1, T, m.nExperts, Math.min(0.5, 3.2 / m.nExperts), xs[7]),
    grid(7, m.nExperts, T, m.dffE, ce, xs[8], Math.min(0.6, 3.4 / m.nExperts)),
    grid(8, 1, T, d, cd, xs[9]),
    grid(9, 1, T, d, cd, xs[10]),
    { kind: 'vector', layer: 10, size: V, x: xs[11], gapY: Math.min(0.72, 12.5 / V) },
  ]
}

/** Semantic stage of an LLM layer (-1 = tokenizer) under the current variant. */
export function llmStageKind(layer: number): LLMStageKind {
  return llmKinds[layer + 1] ?? 'output'
}

/** Layer index of a stage kind, or -2 if absent in the current variant. */
export function llmLayerOf(kind: LLMStageKind): number {
  const i = llmKinds.indexOf(kind)
  return i === -1 ? -2 : i - 1
}

export function llmSteps(): number {
  return llmKinds.length - 1
}

// ------------------------------------------------------------------ RNN slots
// -1 tokens, 0 embeddings, 1 recurrent hidden states (row = timestep), 2 output.

export const RNN_STEPS = 3

function computeRnnSlots(): CNNSlot[] {
  const m = MODELS.rnn.model
  const { T, d, h } = m
  const V = m.vocab.length
  const xs = [-8.5, -4.6, 0.2, 5.4]
  return [
    { kind: 'grid', layer: -1, channels: 1, rows: 1, cols: T, cell: Math.min(0.7, 6.4 / T), x: xs[0], chGap: 0 },
    { kind: 'grid', layer: 0, channels: 1, rows: T, cols: d, cell: Math.min(0.4, 4.2 / d), x: xs[1], chGap: 0 },
    { kind: 'grid', layer: 1, channels: 1, rows: T, cols: h, cell: Math.min(0.4, 4.6 / h), x: xs[2], chGap: 0 },
    { kind: 'vector', layer: 2, size: V, x: xs[3], gapY: Math.min(0.72, 12.5 / V) },
  ]
}

// ------------------------------------------------------------------ LSTM slots
// -1 tokens, 0 embeddings, 1 gates f/i/g/o (4 sheets), 2 cell state, 3 hidden, 4 output.

export const LSTM_STEPS = 5

function computeLstmSlots(): CNNSlot[] {
  const m = MODELS.lstm.model
  const { T, d, h } = m
  const V = m.vocab.length
  const xs = [-12, -8.4, -4, 0.6, 4.2, 8.6]
  const ch = Math.min(0.38, 4.4 / h)
  return [
    { kind: 'grid', layer: -1, channels: 1, rows: 1, cols: T, cell: Math.min(0.7, 6.4 / T), x: xs[0], chGap: 0 },
    { kind: 'grid', layer: 0, channels: 1, rows: T, cols: d, cell: Math.min(0.38, 4 / d), x: xs[1], chGap: 0 },
    { kind: 'grid', layer: 1, channels: 4, rows: T, cols: h, cell: Math.min(0.3, 3 / h), x: xs[2], chGap: 0.65 },
    { kind: 'grid', layer: 2, channels: 1, rows: T, cols: h, cell: ch, x: xs[3], chGap: 0 },
    { kind: 'grid', layer: 3, channels: 1, rows: T, cols: h, cell: ch, x: xs[4], chGap: 0 },
    { kind: 'vector', layer: 4, size: V, x: xs[5], gapY: Math.min(0.72, 12.5 / V) },
  ]
}

// ------------------------------------------------------------------ Autoencoder slots
// -1 input image, 0 encoder, 1 latent bottleneck, 2 decoder, 3 reconstruction.

export const AE_STEPS = 4
export const VAE_STEPS = 5

function computeAeSlots(): CNNSlot[] {
  if (AE_VARIANT === 'vae') {
    const t = getVAETask()
    const enc = t.model.enc.biases.length
    const dec = t.model.dec.biases.length
    const cell = Math.min(0.5, 4.2 / t.n)
    const xs = [-9.4, -5.4, -1.9, 1.2, 4.6, 8.6]
    return [
      { kind: 'grid', layer: -1, channels: 1, rows: t.n, cols: t.n, cell, x: xs[0], chGap: 0 },
      { kind: 'vector', layer: 0, size: enc, x: xs[1], gapY: Math.min(0.72, 13 / enc) },
      { kind: 'grid', layer: 1, channels: 1, rows: t.latent, cols: 2, cell: Math.min(0.7, 8.4 / t.latent), x: xs[2], chGap: 0 },
      { kind: 'vector', layer: 2, size: t.latent, x: xs[3], gapY: Math.min(1.1, 11 / t.latent) },
      { kind: 'vector', layer: 3, size: dec, x: xs[4], gapY: Math.min(0.72, 13 / dec) },
      { kind: 'grid', layer: 4, channels: 1, rows: t.n, cols: t.n, cell, x: xs[5], chGap: 0 },
    ]
  }
  const t = MODELS.ae
  const enc = t.model.layers[0].biases.length
  const dec = t.model.layers[2].biases.length
  const cell = Math.min(0.5, 4.2 / t.n)
  const xs = [-8, -3.6, 0, 3.6, 8]
  return [
    { kind: 'grid', layer: -1, channels: 1, rows: t.n, cols: t.n, cell, x: xs[0], chGap: 0 },
    { kind: 'vector', layer: 0, size: enc, x: xs[1], gapY: Math.min(0.72, 13 / enc) },
    { kind: 'vector', layer: 1, size: t.latent, x: xs[2], gapY: Math.min(1.15, 11 / t.latent) },
    { kind: 'vector', layer: 2, size: dec, x: xs[3], gapY: Math.min(0.72, 13 / dec) },
    { kind: 'grid', layer: 3, channels: 1, rows: t.n, cols: t.n, cell, x: xs[4], chGap: 0 },
  ]
}

// ------------------------------------------------------------------ Diffusion slots
// -1 noisy image x_t, 0 denoiser hidden (as an 8×8 sheet), 1 predicted x̂₀,
// 2 next image x_{t-1}. Playback steps are denoising iterations, so every
// slot stays visible and its contents evolve.

function computeDiffSlots(): CNNSlot[] {
  const n = MODELS.diff.n
  // denoiser hidden activations shown as a near-square sheet
  const h2 = MODELS.diff.model.l2.biases.length
  const hCols = Math.ceil(Math.sqrt(h2))
  const hRows = Math.ceil(h2 / hCols)
  const cell = Math.min(0.55, 4.4 / n)
  const xs = [-7.8, -2.6, 2.6, 7.8]
  return [
    { kind: 'grid', layer: -1, channels: 1, rows: n, cols: n, cell, x: xs[0], chGap: 0 },
    { kind: 'grid', layer: 0, channels: 1, rows: hRows, cols: hCols, cell: Math.min(0.42, 4.2 / hCols), x: xs[1], chGap: 0 },
    { kind: 'grid', layer: 1, channels: 1, rows: n, cols: n, cell: Math.min(0.5, 4.2 / n), x: xs[2], chGap: 0 },
    { kind: 'grid', layer: 2, channels: 1, rows: n, cols: n, cell, x: xs[3], chGap: 0 },
  ]
}

// ------------------------------------------------------------------ GAN slots
// -1 latent z, 0 generator hidden, 1 generated (fake) image, 2 discriminator
// hidden, 3 verdict D(fake)/D(real), 4 real sample (decorative, always on).
// Fake image sits on top, the real sample below — both feed the same D.

export const GAN_STEPS = 4

function computeGanSlots(): CNNSlot[] {
  const t = MODELS.gan
  const m = t.model
  const gh = m.g1.biases.length
  const dh = m.d1.biases.length
  const cell = Math.min(0.5, 4 / t.n)
  const imgY = (t.n * cell) / 2 + 1 // keep a gap between the stacked fake/real sheets
  const xs = [-11, -6.6, -1.8, 3.4, 7.6]
  return [
    { kind: 'vector', layer: -1, size: m.zdim, x: xs[0], gapY: Math.min(1.0, 9.5 / m.zdim) },
    { kind: 'vector', layer: 0, size: gh, x: xs[1], gapY: Math.min(0.72, 13 / gh) },
    { kind: 'grid', layer: 1, channels: 1, rows: t.n, cols: t.n, cell, x: xs[2], chGap: 0, y: imgY },
    { kind: 'vector', layer: 2, size: dh, x: xs[3], gapY: Math.min(0.62, 12 / dh) },
    { kind: 'vector', layer: 3, size: 2, x: xs[4], gapY: 2 * imgY },
    { kind: 'grid', layer: 4, channels: 1, rows: t.n, cols: t.n, cell, x: xs[2], chGap: 0, y: -imgY },
  ]
}

// ------------------------------------------------------------------ ViT slots
// -1 input image, 0 patch tokens + CLS [T][d], 1 attention [heads][T][T],
// 2 after attention residual [T][d], 3 after FFN residual [T][d], 4 classes.

export const VIT_STEPS = 5

function computeVitSlots(): CNNSlot[] {
  const m = MODELS.vit.model
  const { T, d, heads, n } = m
  const cd = Math.min(0.34, 4 / d)
  const xs = [-11.4, -6.4, -1.4, 3.2, 6.9, 10.4]
  return [
    { kind: 'grid', layer: -1, channels: 1, rows: n, cols: n, cell: Math.min(0.42, 5 / n), x: xs[0], chGap: 0 },
    { kind: 'grid', layer: 0, channels: 1, rows: T, cols: d, cell: cd, x: xs[1], chGap: 0 },
    { kind: 'grid', layer: 1, channels: heads, rows: T, cols: T, cell: Math.min(0.4, 4.4 / T), x: xs[2], chGap: Math.min(0.8, 2.8 / heads) },
    { kind: 'grid', layer: 2, channels: 1, rows: T, cols: d, cell: cd, x: xs[3], chGap: 0 },
    { kind: 'grid', layer: 3, channels: 1, rows: T, cols: d, cell: cd, x: xs[4], chGap: 0 },
    { kind: 'vector', layer: 4, size: m.classes, x: xs[5], gapY: 1.6 },
  ]
}

// ------------------------------------------------------------------ GNN layout
// Four x-shifted copies of the SAME graph: input features, hidden after
// message passing 1, logits after message passing 2, per-node prediction.
// Node positions are irregular (community clusters), so the GNN bypasses the
// slot system: gnnNodePos() gives world positions directly.

export const GNN_STEPS = 3
export const GNN_XS = [-8.2, -2.8, 2.8, 8.2]

let gnnLocal: [number, number][] = []

/** In-plane (y,z) position of each graph node — three community clusters. */
function computeGnnPositions(): [number, number][] {
  const m = MODELS.gnn.model
  const per = m.perClass
  const R = 2.5
  const r = Math.max(1.05, per * 0.28)
  const pts: [number, number][] = []
  for (let c = 0; c < m.classes; c++) {
    const base = (c / m.classes) * Math.PI * 2 + Math.PI / 2
    const cy = Math.sin(base) * R
    const cz = Math.cos(base) * R
    for (let k = 0; k < per; k++) {
      const a = (k / per) * Math.PI * 2 + c * 0.7
      pts.push([cy + Math.sin(a) * r, cz + Math.cos(a) * r])
    }
  }
  return pts
}

/** World position of graph node i in the layer-k copy (-1 = input). */
export function gnnNodePos(layer: number, i: number): Vec3 {
  const p = gnnLocal[i] ?? [0, 0]
  return [GNN_XS[layer + 1] ?? GNN_XS[3], p[0], p[1]]
}

export function gnnLabelAnchor(layer: number): Vec3 {
  const top = gnnLocal.reduce((mx, p) => Math.max(mx, p[0]), 0)
  return [GNN_XS[layer + 1] ?? GNN_XS[3], top + 1.9, 0]
}

// ------------------------------------------------------------------ cached slot tables

let cnnSlots = computeCnnSlots()
let llmSlots = computeLlmSlots()
let rnnSlots = computeRnnSlots()
let lstmSlots = computeLstmSlots()
let aeSlots = computeAeSlots()
let diffSlots = computeDiffSlots()
let ganSlots = computeGanSlots()
let vitSlots = computeVitSlots()
gnnLocal = computeGnnPositions()

/** Re-derive slot tables after a model was rebuilt at a new scale. */
export function refreshLayout(): void {
  cnnSlots = computeCnnSlots()
  llmSlots = computeLlmSlots()
  rnnSlots = computeRnnSlots()
  lstmSlots = computeLstmSlots()
  aeSlots = computeAeSlots()
  diffSlots = computeDiffSlots()
  ganSlots = computeGanSlots()
  vitSlots = computeVitSlots()
  gnnLocal = computeGnnPositions()
}

export function cnnSlot(layer: number): CNNSlot {
  return cnnSlots[layer + 1]
}

export function llmSlot(layer: number): CNNSlot {
  return llmSlots[layer + 1]
}

export function rnnSlot(layer: number): CNNSlot {
  return rnnSlots[layer + 1]
}

export function lstmSlot(layer: number): CNNSlot {
  return lstmSlots[layer + 1]
}

export function aeSlot(layer: number): CNNSlot {
  return aeSlots[layer + 1]
}

export function diffSlot(layer: number): CNNSlot {
  return diffSlots[layer + 1]
}

export function ganSlot(layer: number): CNNSlot {
  return ganSlots[layer + 1]
}

export function vitSlot(layer: number): CNNSlot {
  return vitSlots[layer + 1]
}

/** Slot lookup for every grid-based architecture. */
export function slotFor(arch: Arch, layer: number): CNNSlot {
  switch (arch) {
    case 'cnn':
      return cnnSlot(layer)
    case 'llm':
      return llmSlot(layer)
    case 'rnn':
      return rnnSlot(layer)
    case 'lstm':
      return lstmSlot(layer)
    case 'diff':
      return diffSlot(layer)
    case 'gan':
      return ganSlot(layer)
    case 'vit':
      return vitSlot(layer)
    default:
      return aeSlot(layer)
  }
}

// ------------------------------------------------------------------ positions

export function gridPos(slot: GridSlot, ch: number, row: number, col: number): Vec3 {
  return [
    slot.x + (ch - (slot.channels - 1) / 2) * slot.chGap,
    (slot.y ?? 0) + ((slot.rows - 1) / 2 - row) * slot.cell,
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
  return [slot.x, (slot.y ?? 0) + ((slot.size - 1) / 2 - i) * slot.gapY, 0]
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
  if (slot.kind === 'grid')
    return [slot.x, (slot.y ?? 0) + ((slot.rows - 1) / 2) * slot.cell + 1.6, 0]
  if (slot.kind === 'flatten') {
    const perCh = slot.srcShape[1] * slot.srcShape[2]
    return [slot.x, ((perCh - 1) / 2) * slot.gapY + 1.4, 0]
  }
  return [slot.x, (slot.y ?? 0) + ((slot.size - 1) / 2) * slot.gapY + 1.5, 0]
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
  if (arch === 'gnn') {
    return gnnNodePos(ref.layer, ref.space === 'vector' ? ref.index : 0)
  }
  const slot = slotFor(arch, ref.layer)
  if (ref.space === 'vector') return slotPos(slot, { index: ref.index })
  return slotPos(slot, { channel: ref.channel, row: ref.row, col: ref.col })
}

/** Axis-aligned bounds of all node centers in a layer (-1 = input). */
export function layerBounds(arch: Arch, layer: number): { min: Vec3; max: Vec3 } {
  const pts: Vec3[] = []
  if (arch === 'mlp' || arch === 'text') {
    const n = denseSizes(arch)[layer + 1]
    for (let i = 0; i < n; i++) pts.push(densePos(arch, layer, i))
  } else if (arch === 'gnn') {
    for (let i = 0; i < gnnLocal.length; i++) pts.push(gnnNodePos(layer, i))
  } else {
    const slot = slotFor(arch, layer)
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
  llm: { position: [15, 7.5, 26], target: [0, 0, 0] },
  rnn: { position: [8.5, 5, 14], target: [0, 0, 0] },
  lstm: { position: [12, 6.5, 20], target: [0, 0, 0] },
  ae: { position: [9, 5, 15], target: [0, 0, 0] },
  diff: { position: [12.5, 5.5, 19], target: [0, 0, 0] },
  gan: { position: [13.5, 2.2, 19], target: [0, 0.4, 0] },
  gnn: { position: [12, 5, 18], target: [0, 0, 0] },
  vit: { position: [13, 6, 20], target: [0, 0, 0] },
}
