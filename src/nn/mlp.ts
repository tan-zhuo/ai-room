// Fully-connected network: real forward pass, real SGD training.

import { Rng, gaussian, shuffleInPlace } from './rng'

export type ActivationKind = 'relu' | 'tanh' | 'sigmoid' | 'softmax' | 'linear'

export interface DenseLayer {
  /** weights[out][in] */
  weights: number[][]
  biases: number[]
  activation: ActivationKind
}

export interface MLPModel {
  inputSize: number
  layers: DenseLayer[]
}

export interface DenseTrace {
  z: number[]
  a: number[]
}

export function applyActivation(kind: ActivationKind, z: number[]): number[] {
  switch (kind) {
    case 'relu':
      return z.map((v) => Math.max(0, v))
    case 'tanh':
      return z.map((v) => Math.tanh(v))
    case 'sigmoid':
      return z.map((v) => 1 / (1 + Math.exp(-v)))
    case 'linear':
      return [...z]
    case 'softmax': {
      const m = Math.max(...z)
      const e = z.map((v) => Math.exp(v - m))
      const s = e.reduce((acc, v) => acc + v, 0)
      return e.map((v) => v / s)
    }
  }
}

function activationDeriv(kind: ActivationKind, z: number, a: number): number {
  switch (kind) {
    case 'relu':
      return z > 0 ? 1 : 0
    case 'tanh':
      return 1 - a * a
    case 'sigmoid':
      return a * (1 - a)
    default:
      return 1
  }
}

export function denseForward(layer: DenseLayer, input: number[]): DenseTrace {
  const z = layer.biases.map((b, j) =>
    layer.weights[j].reduce((s, w, i) => s + w * input[i], b),
  )
  return { z, a: applyActivation(layer.activation, z) }
}

export function forwardMLP(model: MLPModel, input: number[]): DenseTrace[] {
  const traces: DenseTrace[] = []
  let cur = input
  for (const layer of model.layers) {
    const t = denseForward(layer, cur)
    traces.push(t)
    cur = t.a
  }
  return traces
}

export function createMLP(rng: Rng, sizes: number[], activations: ActivationKind[]): MLPModel {
  const layers: DenseLayer[] = []
  for (let k = 1; k < sizes.length; k++) {
    const fanIn = sizes[k - 1]
    const std = Math.sqrt(2 / fanIn)
    layers.push({
      weights: Array.from({ length: sizes[k] }, () =>
        Array.from({ length: fanIn }, () => gaussian(rng) * std),
      ),
      biases: Array.from({ length: sizes[k] }, () => 0),
      activation: activations[k - 1],
    })
  }
  return { inputSize: sizes[0], layers }
}

export interface TrainSample {
  x: number[]
  y: number[]
}

/** In-place SGD with softmax + cross-entropy loss on the final layer. */
export function trainMLP(
  model: MLPModel,
  data: TrainSample[],
  opts: { lr: number; epochs: number; rng: Rng },
): void {
  const { lr, epochs, rng } = opts
  const order = data.map((_, i) => i)
  const L = model.layers.length
  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    for (const idx of order) {
      const { x, y } = data[idx]
      const traces = forwardMLP(model, x)
      let delta = traces[L - 1].a.map((v, j) => v - y[j])
      for (let k = L - 1; k >= 0; k--) {
        const layer = model.layers[k]
        const aPrev = k === 0 ? x : traces[k - 1].a
        const prevDelta =
          k > 0
            ? aPrev.map((_, i) => layer.weights.reduce((s, row, j) => s + row[i] * delta[j], 0))
            : null
        for (let j = 0; j < layer.weights.length; j++) {
          const row = layer.weights[j]
          const d = lr * delta[j]
          for (let i = 0; i < aPrev.length; i++) row[i] -= d * aPrev[i]
          layer.biases[j] -= d
        }
        if (k > 0 && prevDelta) {
          const prevLayer = model.layers[k - 1]
          const zPrev = traces[k - 1].z
          const aPrevTrace = traces[k - 1].a
          delta = prevDelta.map((d, i) =>
            d * activationDeriv(prevLayer.activation, zPrev[i], aPrevTrace[i]),
          )
        }
      }
    }
  }
}

/** In-place SGD with mean-squared-error loss and a sigmoid output layer. */
export function trainMLPMSE(
  model: MLPModel,
  data: TrainSample[],
  opts: { lr: number; epochs: number; rng: Rng },
): number {
  const { lr, epochs, rng } = opts
  const order = data.map((_, i) => i)
  const L = model.layers.length
  let mse = 0
  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    mse = 0
    for (const idx of order) {
      const { x, y } = data[idx]
      const traces = forwardMLP(model, x)
      const out = traces[L - 1]
      mse += out.a.reduce((s, v, j) => s + (v - y[j]) ** 2, 0) / y.length
      // sigmoid output: delta = (a - y) ⊙ a(1 - a)
      let delta = out.a.map((v, j) => (v - y[j]) * v * (1 - v))
      for (let k = L - 1; k >= 0; k--) {
        const layer = model.layers[k]
        const aPrev = k === 0 ? x : traces[k - 1].a
        const prevDelta =
          k > 0
            ? aPrev.map((_, i) => layer.weights.reduce((s, row, j) => s + row[i] * delta[j], 0))
            : null
        for (let j = 0; j < layer.weights.length; j++) {
          const row = layer.weights[j]
          const d = lr * delta[j]
          for (let i = 0; i < aPrev.length; i++) row[i] -= d * aPrev[i]
          layer.biases[j] -= d
        }
        if (k > 0 && prevDelta) {
          const prevLayer = model.layers[k - 1]
          const zPrev = traces[k - 1].z
          const aPrevTrace = traces[k - 1].a
          delta = prevDelta.map((d, i) =>
            d * activationDeriv(prevLayer.activation, zPrev[i], aPrevTrace[i]),
          )
        }
      }
    }
    mse /= data.length
  }
  return mse
}

export function argmax(arr: number[]): number {
  let best = 0
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i
  return best
}

export function maxAbs(arr: number[]): number {
  let m = 0
  for (const v of arr) m = Math.max(m, Math.abs(v))
  return m || 1
}
