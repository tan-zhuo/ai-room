// A tiny vanilla GAN (Goodfellow 2014) on 8×8 patterns. The generator maps a
// Gaussian latent z to an image; the discriminator scores real vs generated.
// Both are trained adversarially with the non-saturating G loss — real SGD,
// alternating D and G updates.

import { DenseLayer, DenseTrace, createMLP, denseForward } from './mlp'
import { Rng, gaussian, mulberry32, shuffleInPlace } from './rng'
import { Tensor3 } from './cnn'

export interface GANModel {
  n: number
  /** latent dimension */
  zdim: number
  /** generator: z → hidden (relu) → n² image (tanh, [-1,1]) */
  g1: DenseLayer
  g2: DenseLayer
  /** discriminator: n² → hidden (leaky relu) → 1 (sigmoid = P(real)) */
  d1: DenseLayer
  d2: DenseLayer
}

export interface GenTrace {
  g1: DenseTrace
  /** generated image, tanh in [-1,1] */
  img: DenseTrace
}

export interface DiscTrace {
  d1: DenseTrace
  /** single sigmoid unit: P(input is real) */
  out: DenseTrace
}

export function generate(model: GANModel, z: number[]): GenTrace {
  const g1 = denseForward(model.g1, z)
  const img = denseForward(model.g2, g1.a)
  return { g1, img }
}

export function discriminate(model: GANModel, x: number[]): DiscTrace {
  const d1 = denseForward(model.d1, x)
  const out = denseForward(model.d2, d1.a)
  return { d1, out }
}

export interface GANTrace extends GenTrace, DiscTrace {
  z: number[]
  /** a real training pattern in [-1,1], judged by the same discriminator */
  real: number[]
  realD1: DenseTrace
  realOut: DenseTrace
}

/** Sample z ~ N(0,1), generate, and run BOTH images through the discriminator. */
export function forwardGAN(model: GANModel, realBank: number[][], rng: Rng): GANTrace {
  const z = Array.from({ length: model.zdim }, () => gaussian(rng))
  const g = generate(model, z)
  const d = discriminate(model, g.img.a)
  const real = realBank[Math.floor(rng() * realBank.length)]
  const dr = discriminate(model, real)
  return { z, ...g, ...d, real, realD1: dr.d1, realOut: dr.out }
}

/** Backprop delta through a layer; updates weights iff lr is given. */
function backDense(layer: DenseLayer, input: number[], delta: number[], lr: number | null): number[] {
  const prev = input.map((_, i) => layer.weights.reduce((s, row, j) => s + row[i] * delta[j], 0))
  if (lr !== null) {
    for (let j = 0; j < layer.weights.length; j++) {
      const row = layer.weights[j]
      const d = lr * delta[j]
      for (let i = 0; i < input.length; i++) row[i] -= d * input[i]
      layer.biases[j] -= d
    }
  }
  return prev
}

const leakyD = (z: number) => (z > 0 ? 1 : 0.2)

export interface GANTask {
  model: GANModel
  n: number
  /** clean training patterns in [-1,1] shown as the "real" branch */
  realBank: number[][]
  /** mean D(real) / D(G(z)) over the last epoch — near equilibrium both ≈ 0.5 */
  dReal: number
  dFake: number
  /** mean min-MSE (in [0,1] space) of generated samples vs clean patterns */
  quality: number
  /** how many of the 4 pattern classes the generator covers */
  modes: number
}

const GAN_SCALE_CFG = {
  s: { n: 8, zdim: 8, gh: 32, dh: 24, epochs: 120 },
  m: { n: 8, zdim: 12, gh: 48, dh: 32, epochs: 130 },
  l: { n: 12, zdim: 16, gh: 72, dh: 48, epochs: 120 },
} as const

export function buildGANTask(
  patternSample: (n: number, cls: number, rng: Rng) => Tensor3,
  scale: 's' | 'm' | 'l' = 's',
): GANTask {
  const rng = mulberry32(0x9a4 + scale.charCodeAt(0))
  const { n, zdim, gh, dh, epochs } = GAN_SCALE_CFG[scale]

  const gScaffold = createMLP(rng, [zdim, gh, n * n], ['relu', 'tanh'])
  const dScaffold = createMLP(rng, [n * n, dh, 1], ['leaky', 'sigmoid'])
  const model: GANModel = {
    n,
    zdim,
    g1: gScaffold.layers[0],
    g2: gScaffold.layers[1],
    d1: dScaffold.layers[0],
    d2: dScaffold.layers[1],
  }

  // patterns scaled to [-1, 1]
  const data: number[][] = []
  for (let c = 0; c < 4; c++)
    for (let i = 0; i < 40; i++)
      data.push(patternSample(n, c, rng)[0].flat().map((v) => v * 2 - 1))

  const order = data.map((_, i) => i)
  const lrD = 0.008
  const lrG = 0.02
  let dReal = 0
  let dFake = 0
  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    dReal = 0
    dFake = 0
    for (const idx of order) {
      // --- D step on a real sample (one-sided label smoothing: target 0.9)
      const real = data[idx]
      const dr = discriminate(model, real)
      dReal += dr.out.a[0]
      let delta = [dr.out.a[0] - 0.9]
      delta = backDense(model.d2, dr.d1.a, delta, lrD).map((g, i) => g * leakyD(dr.d1.z[i]))
      backDense(model.d1, real, delta, lrD)

      // --- D step on a fake sample (target 0)
      const z = Array.from({ length: zdim }, () => gaussian(rng))
      const g = generate(model, z)
      const df = discriminate(model, g.img.a)
      dFake += df.out.a[0]
      delta = [df.out.a[0]]
      delta = backDense(model.d2, df.d1.a, delta, lrD).map((gv, i) => gv * leakyD(df.d1.z[i]))
      backDense(model.d1, g.img.a, delta, lrD)

      // --- G step (non-saturating: push D(G(z)) toward 1), D frozen
      const df2 = discriminate(model, g.img.a)
      delta = [df2.out.a[0] - 1]
      delta = backDense(model.d2, df2.d1.a, delta, null).map((gv, i) => gv * leakyD(df2.d1.z[i]))
      const gradImg = backDense(model.d1, g.img.a, delta, null)
      let gDelta = gradImg.map((gv, i) => gv * (1 - g.img.a[i] * g.img.a[i]))
      gDelta = backDense(model.g2, g.g1.a, gDelta, lrG).map((gv, i) => (g.g1.z[i] > 0 ? gv : 0))
      backDense(model.g1, z, gDelta, lrG)
    }
    dReal /= data.length
    dFake /= data.length
  }

  // quality: generated samples should land near SOME clean pattern
  const clean: { img: number[]; cls: number }[] = []
  for (let c = 0; c < 4; c++)
    for (let i = 0; i < 20; i++) clean.push({ img: patternSample(n, c, rng)[0].flat(), cls: c })
  let quality = 0
  const covered = new Set<number>()
  const probes = 48
  for (let s = 0; s < probes; s++) {
    const z = Array.from({ length: zdim }, () => gaussian(rng))
    const img = generate(model, z).img.a.map((v) => (v + 1) / 2)
    let best = Infinity
    let bestCls = 0
    for (const c of clean) {
      const mse = img.reduce((acc, v, i) => acc + (v - c.img[i]) ** 2, 0) / img.length
      if (mse < best) {
        best = mse
        bestCls = c.cls
      }
    }
    quality += best
    covered.add(bestCls)
  }
  quality /= probes

  const realBank = data.slice(0, 24)
  return { model, n, realBank, dReal, dFake, quality, modes: covered.size }
}
