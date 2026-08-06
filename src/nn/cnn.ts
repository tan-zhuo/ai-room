// Convolutional network primitives: conv / max-pool / flatten / dense,
// all computed for real with every intermediate value inspectable.
// Includes end-to-end training with a hand-written conv backward pass,
// so the kernels themselves can be learned rather than hand-crafted.

import { DenseLayer, DenseTrace, denseForward } from './mlp'
import { Rng, shuffleInPlace } from './rng'

/** [channel][row][col] */
export type Tensor3 = number[][][]

export interface ConvLayerDef {
  type: 'conv'
  /** kernels[outChannel][inChannel][ky][kx] — followed by ReLU */
  kernels: number[][][][]
  biases: number[]
}

export interface PoolLayerDef {
  type: 'pool'
  size: number
}

export interface FlattenLayerDef {
  type: 'flatten'
}

export interface DenseWrapDef {
  type: 'dense'
  layer: DenseLayer
}

export type CNNLayerDef = ConvLayerDef | PoolLayerDef | FlattenLayerDef | DenseWrapDef

export interface CNNModel {
  /** [channels, rows, cols] */
  inputShape: [number, number, number]
  layers: CNNLayerDef[]
}

export type CNNStep =
  | { kind: 'tensor'; pre: Tensor3 | null; out: Tensor3 }
  | { kind: 'vector'; z: number[] | null; a: number[] }

export function tensorShape(t: Tensor3): [number, number, number] {
  return [t.length, t[0].length, t[0][0].length]
}

export function tensorMaxAbs(t: Tensor3): number {
  let m = 0
  for (const ch of t) for (const row of ch) for (const v of row) m = Math.max(m, Math.abs(v))
  return m || 1
}

/** Full detail of a single conv output pixel — used by the inspector panel. */
export interface ConvDetail {
  patch: number[][][]
  kernel: number[][][]
  products: number[][][]
  sum: number
  bias: number
  z: number
  a: number
}

export function convAt(input: Tensor3, def: ConvLayerDef, c: number, y: number, x: number): ConvDetail {
  const kernel = def.kernels[c]
  const inC = kernel.length
  const kh = kernel[0].length
  const kw = kernel[0][0].length
  const patch: number[][][] = []
  const products: number[][][] = []
  let sum = 0
  for (let ic = 0; ic < inC; ic++) {
    const p: number[][] = []
    const pr: number[][] = []
    for (let ky = 0; ky < kh; ky++) {
      const pRow: number[] = []
      const prRow: number[] = []
      for (let kx = 0; kx < kw; kx++) {
        const v = input[ic][y + ky][x + kx]
        const prod = v * kernel[ic][ky][kx]
        pRow.push(v)
        prRow.push(prod)
        sum += prod
      }
      p.push(pRow)
      pr.push(prRow)
    }
    patch.push(p)
    products.push(pr)
  }
  const bias = def.biases[c]
  const z = sum + bias
  return { patch, kernel, products, sum, bias, z, a: Math.max(0, z) }
}

export function convForward(input: Tensor3, def: ConvLayerDef): { pre: Tensor3; out: Tensor3 } {
  const [, h, w] = tensorShape(input)
  const kh = def.kernels[0][0].length
  const kw = def.kernels[0][0][0].length
  const outH = h - kh + 1
  const outW = w - kw + 1
  const pre: Tensor3 = []
  const out: Tensor3 = []
  for (let c = 0; c < def.kernels.length; c++) {
    const preCh: number[][] = []
    const outCh: number[][] = []
    for (let y = 0; y < outH; y++) {
      const preRow: number[] = []
      const outRow: number[] = []
      for (let x = 0; x < outW; x++) {
        const d = convAt(input, def, c, y, x)
        preRow.push(d.z)
        outRow.push(d.a)
      }
      preCh.push(preRow)
      outCh.push(outRow)
    }
    pre.push(preCh)
    out.push(outCh)
  }
  return { pre, out }
}

export interface PoolDetail {
  values: number[][]
  max: number
  argRow: number
  argCol: number
}

export function poolAt(input: Tensor3, size: number, c: number, y: number, x: number): PoolDetail {
  const values: number[][] = []
  let max = -Infinity
  let argRow = 0
  let argCol = 0
  for (let dy = 0; dy < size; dy++) {
    const row: number[] = []
    for (let dx = 0; dx < size; dx++) {
      const v = input[c][y * size + dy][x * size + dx]
      row.push(v)
      if (v > max) {
        max = v
        argRow = dy
        argCol = dx
      }
    }
    values.push(row)
  }
  return { values, max, argRow, argCol }
}

export function poolForward(input: Tensor3, size: number): Tensor3 {
  const [ch, h, w] = tensorShape(input)
  const outH = Math.floor(h / size)
  const outW = Math.floor(w / size)
  const out: Tensor3 = []
  for (let c = 0; c < ch; c++) {
    const outCh: number[][] = []
    for (let y = 0; y < outH; y++) {
      const row: number[] = []
      for (let x = 0; x < outW; x++) row.push(poolAt(input, size, c, y, x).max)
      outCh.push(row)
    }
    out.push(outCh)
  }
  return out
}

export function flattenTensor(t: Tensor3): number[] {
  const out: number[] = []
  for (const ch of t) for (const row of ch) for (const v of row) out.push(v)
  return out
}

/** Inverse of flattenTensor ordering: index -> (channel, row, col). */
export function unflattenIndex(shape: [number, number, number], i: number): { channel: number; row: number; col: number } {
  const [, h, w] = shape
  const perCh = h * w
  const channel = Math.floor(i / perCh)
  const rem = i % perCh
  return { channel, row: Math.floor(rem / w), col: rem % w }
}

export interface CNNTrainSample {
  x: Tensor3
  y: number[]
}

/**
 * End-to-end SGD through dense head, flatten, max-pool, ReLU and the conv
 * layer itself. Assumes the layer stack [conv, pool, flatten, dense…, dense]
 * with softmax + cross-entropy on the final layer.
 */
export function trainCNNEndToEnd(
  model: CNNModel,
  data: CNNTrainSample[],
  opts: { lr: number; epochs: number; rng: Rng },
): void {
  const { lr, epochs, rng } = opts
  const conv = model.layers[0]
  const pool = model.layers[1]
  if (conv.type !== 'conv' || pool.type !== 'pool') return
  const denseLayers = model.layers.filter((l): l is DenseWrapDef => l.type === 'dense').map((l) => l.layer)
  const order = data.map((_, i) => i)

  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    for (const idx of order) {
      const { x, y } = data[idx]
      // ---- forward with caches
      const { pre, out } = convForward(x, conv)
      const pooled = poolForward(out, pool.size)
      const flat = flattenTensor(pooled)
      const traces: DenseTrace[] = []
      let cur = flat
      for (const layer of denseLayers) {
        const tr = denseForward(layer, cur)
        traces.push(tr)
        cur = tr.a
      }
      // ---- dense backward (softmax + CE)
      const L = denseLayers.length
      let delta = traces[L - 1].a.map((v, j) => v - y[j])
      for (let k = L - 1; k >= 0; k--) {
        const layer = denseLayers[k]
        const aPrev = k === 0 ? flat : traces[k - 1].a
        const prevDelta = aPrev.map((_, i) => layer.weights.reduce((s, row, j) => s + row[i] * delta[j], 0))
        for (let j = 0; j < layer.weights.length; j++) {
          const row = layer.weights[j]
          const d = lr * delta[j]
          for (let i = 0; i < aPrev.length; i++) row[i] -= d * aPrev[i]
          layer.biases[j] -= d
        }
        if (k > 0) {
          const zPrev = traces[k - 1].z
          delta = prevDelta.map((d, i) => (zPrev[i] > 0 ? d : 0)) // hidden dense layers are ReLU
        } else {
          delta = prevDelta // gradient w.r.t. the flat vector
        }
      }
      // ---- unflatten + max-pool backward (gradient routes to the argmax)
      const [ch, ph, pw] = tensorShape(pooled)
      const dOut: Tensor3 = out.map((c) => c.map((row) => row.map(() => 0)))
      for (let c = 0; c < ch; c++) {
        for (let py = 0; py < ph; py++) {
          for (let px = 0; px < pw; px++) {
            const g = delta[c * ph * pw + py * pw + px]
            if (g === 0) continue
            const d = poolAt(out, pool.size, c, py, px)
            dOut[c][py * pool.size + d.argRow][px * pool.size + d.argCol] += g
          }
        }
      }
      // ---- ReLU + conv backward
      const kh = conv.kernels[0][0].length
      const kw = conv.kernels[0][0][0].length
      const [, outH, outW] = tensorShape(out)
      for (let c = 0; c < conv.kernels.length; c++) {
        let dBias = 0
        for (let yy = 0; yy < outH; yy++) {
          for (let xx = 0; xx < outW; xx++) {
            const g = pre[c][yy][xx] > 0 ? dOut[c][yy][xx] : 0
            if (g === 0) continue
            dBias += g
            for (let ic = 0; ic < conv.kernels[c].length; ic++) {
              for (let ky = 0; ky < kh; ky++) {
                for (let kx = 0; kx < kw; kx++) {
                  conv.kernels[c][ic][ky][kx] -= lr * g * x[ic][yy + ky][xx + kx]
                }
              }
            }
          }
        }
        conv.biases[c] -= lr * dBias
      }
    }
  }
}

export function forwardCNN(model: CNNModel, input: Tensor3): CNNStep[] {
  const steps: CNNStep[] = []
  let tensor: Tensor3 | null = input
  let vector: number[] | null = null
  for (const def of model.layers) {
    switch (def.type) {
      case 'conv': {
        const { pre, out } = convForward(tensor!, def)
        steps.push({ kind: 'tensor', pre, out })
        tensor = out
        break
      }
      case 'pool': {
        const out = poolForward(tensor!, def.size)
        steps.push({ kind: 'tensor', pre: null, out })
        tensor = out
        break
      }
      case 'flatten': {
        vector = flattenTensor(tensor!)
        steps.push({ kind: 'vector', z: null, a: vector })
        break
      }
      case 'dense': {
        const t = denseForward(def.layer, vector!)
        steps.push({ kind: 'vector', z: t.z, a: t.a })
        vector = t.a
        break
      }
    }
  }
  return steps
}
