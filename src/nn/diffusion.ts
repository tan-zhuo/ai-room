// A tiny DDPM (denoising diffusion probabilistic model) on 8×8 patterns.
// Forward process q(x_t|x_0) adds Gaussian noise on a fixed schedule; a small
// MLP is trained to predict that noise; sampling runs the learned reverse
// process step by step — the full trajectory is kept for visualization.

import { DenseLayer, DenseTrace, createMLP, denseForward } from './mlp'
import { Rng, gaussian, mulberry32, shuffleInPlace } from './rng'
import { Tensor3 } from './cnn'

export interface DiffusionModel {
  n: number
  /** number of diffusion timesteps */
  T: number
  betas: number[]
  alphas: number[]
  alphaBars: number[]
  /** time-embedding width (sin/cos features of t) */
  tdim: number
  /** denoiser predicting x̂₀: [n² + tdim] → h1 → h2 → n² (tanh out) */
  l1: DenseLayer
  l2: DenseLayer
  out: DenseLayer
}

export function timeEmbed(model: DiffusionModel, t: number): number[] {
  const e: number[] = []
  for (let k = 0; k < model.tdim / 2; k++) {
    const f = (t / model.T) * Math.PI * 2 ** k
    e.push(Math.sin(f), Math.cos(f))
  }
  return e
}

export interface DenoiseTrace {
  temb: number[]
  h1: DenseTrace
  h2: DenseTrace
  /** predicted clean image x̂₀ */
  pred: DenseTrace
}

export function denoise(model: DiffusionModel, x: number[], t: number): DenoiseTrace {
  const temb = timeEmbed(model, t)
  const h1 = denseForward(model.l1, [...x, ...temb])
  const h2 = denseForward(model.l2, h1.a)
  const pred = denseForward(model.out, h2.a)
  return { temb, h1, h2, pred }
}

export interface DiffusionStepTrace extends DenoiseTrace {
  /** timestep being denoised */
  t: number
  x: number[]
  /** fresh noise injected by the sampler (zero on the final step) */
  z: number[]
  xNext: number[]
}

export interface DiffusionTrace {
  /** trajectory of images, xs[0] = pure noise … xs[T] = final sample */
  xs: number[][]
  steps: DiffusionStepTrace[]
}

/** Run the full reverse process from pure noise, recording every step. */
export function sampleDiffusion(model: DiffusionModel, rng: Rng): DiffusionTrace {
  const size = model.n * model.n
  let x = Array.from({ length: size }, () => gaussian(rng))
  const xs = [x]
  const steps: DiffusionStepTrace[] = []
  for (let t = model.T; t >= 1; t--) {
    const d = denoise(model, x, t)
    const beta = model.betas[t - 1]
    const alpha = model.alphas[t - 1]
    const aBar = model.alphaBars[t - 1]
    const aBarPrev = t > 1 ? model.alphaBars[t - 2] : 1
    // DDPM posterior mean under the x̂₀ parameterization
    const c0 = (Math.sqrt(aBarPrev) * beta) / (1 - aBar)
    const ct = (Math.sqrt(alpha) * (1 - aBarPrev)) / (1 - aBar)
    const sigma = t > 1 ? Math.sqrt(((1 - aBarPrev) / (1 - aBar)) * beta) : 0
    const z = Array.from({ length: size }, () => (t > 1 ? gaussian(rng) : 0))
    const xNext = x.map((v, i) => c0 * d.pred.a[i] + ct * v + sigma * z[i])
    steps.push({ ...d, t, x, z, xNext })
    x = xNext
    xs.push(x)
  }
  return { xs, steps }
}

function backDense(layer: DenseLayer, input: number[], delta: number[], lr: number): number[] {
  const prev = input.map((_, i) => layer.weights.reduce((s, row, j) => s + row[i] * delta[j], 0))
  for (let j = 0; j < layer.weights.length; j++) {
    const row = layer.weights[j]
    const d = lr * delta[j]
    for (let i = 0; i < input.length; i++) row[i] -= d * input[i]
    layer.biases[j] -= d
  }
  return prev
}

export interface DiffusionTask {
  model: DiffusionModel
  n: number
  finalLoss: number
}

const DIFF_SCALE_CFG = {
  s: { n: 8, T: 20, h1: 96, h2: 64, epochs: 220 },
  m: { n: 10, T: 25, h1: 128, h2: 96, epochs: 170 },
  l: { n: 12, T: 30, h1: 160, h2: 128, epochs: 130 },
} as const

export function buildDiffusionTask(
  patternSample: (n: number, cls: number, rng: Rng) => Tensor3,
  scale: 's' | 'm' | 'l' = 's',
): DiffusionTask {
  const rng = mulberry32(0xd1ff + scale.charCodeAt(0))
  const { n, T, h1, h2, epochs } = DIFF_SCALE_CFG[scale]
  const tdim = 6
  const betas = Array.from({ length: T }, (_, i) => 0.02 + (0.26 - 0.02) * (i / (T - 1)))
  const alphas = betas.map((b) => 1 - b)
  const alphaBars: number[] = []
  alphas.reduce((acc, a) => {
    const v = acc * a
    alphaBars.push(v)
    return v
  }, 1)

  const scaffold = createMLP(rng, [n * n + tdim, h1, h2, n * n], ['relu', 'relu', 'tanh'])
  const model: DiffusionModel = {
    n,
    T,
    betas,
    alphas,
    alphaBars,
    tdim,
    l1: scaffold.layers[0],
    l2: scaffold.layers[1],
    out: scaffold.layers[2],
  }

  // patterns scaled to [-1, 1]
  const data: number[][] = []
  for (let c = 0; c < 4; c++)
    for (let i = 0; i < 40; i++)
      data.push(patternSample(n, c, rng)[0].flat().map((v) => v * 2 - 1))

  const order = data.map((_, i) => i)
  const lr = 0.02
  let loss = 0
  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    loss = 0
    for (const idx of order) {
      const x0 = data[idx]
      const t = 1 + Math.floor(rng() * T)
      const aBar = model.alphaBars[t - 1]
      const eps = x0.map(() => gaussian(rng))
      const xt = x0.map((v, i) => Math.sqrt(aBar) * v + Math.sqrt(1 - aBar) * eps[i])
      const d = denoise(model, xt, t)
      loss += d.pred.a.reduce((s, v, i) => s + (v - x0[i]) ** 2, 0) / x0.length

      // backward: MSE on the predicted clean image, tanh output
      let delta = d.pred.a.map((v, i) => (((v - x0[i]) * 2) / x0.length) * (1 - v * v))
      delta = backDense(model.out, d.h2.a, delta, lr).map((g, i) => (d.h2.z[i] > 0 ? g : 0))
      delta = backDense(model.l2, d.h1.a, delta, lr).map((g, i) => (d.h1.z[i] > 0 ? g : 0))
      backDense(model.l1, [...xt, ...d.temb], delta, lr)
    }
    loss /= data.length
  }

  return { model, n, finalLoss: loss }
}
