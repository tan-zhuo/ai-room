// Variational autoencoder: the bottleneck becomes a probability distribution.
// Encoder → (μ, log σ²) → z = μ + ε·σ (reparameterization) → decoder.
// Trained with MSE reconstruction + KL divergence, backprop hand-written.

import { DenseLayer, DenseTrace, denseForward, createMLP } from './mlp'
import { Rng, gaussian, mulberry32, shuffleInPlace } from './rng'
import { Tensor3 } from './cnn'

export interface VAEModel {
  n: number
  latent: number
  /** 64 → 24 ReLU */
  enc: DenseLayer
  /** 24 → latent, linear heads */
  muHead: DenseLayer
  lvHead: DenseLayer
  /** latent → 24 ReLU */
  dec: DenseLayer
  /** 24 → 64 sigmoid */
  out: DenseLayer
}

export interface VAETrace {
  /** encoder hidden */
  h: DenseTrace
  mu: number[]
  logvar: number[]
  sigma: number[]
  eps: number[]
  z: number[]
  d: DenseTrace
  o: DenseTrace
  /** true when z was sampled from the prior, not encoded from an input */
  generated: boolean
}

export function forwardVAE(model: VAEModel, x: number[], eps?: number[], rng?: Rng): VAETrace {
  const h = denseForward(model.enc, x)
  const mu = denseForward(model.muHead, h.a).a
  const logvar = denseForward(model.lvHead, h.a).a.map((v) => Math.max(-6, Math.min(6, v)))
  const sigma = logvar.map((v) => Math.exp(0.5 * v))
  const e = eps ?? mu.map(() => (rng ? gaussian(rng) : 0))
  const z = mu.map((m, i) => m + e[i] * sigma[i])
  const d = denseForward(model.dec, z)
  const o = denseForward(model.out, d.a)
  return { h, mu, logvar, sigma, eps: e, z, d, o, generated: false }
}

/** Decode a z sampled straight from the prior N(0, I) — pure generation. */
export function generateVAE(model: VAEModel, rng: Rng): VAETrace {
  const z = Array.from({ length: model.latent }, () => gaussian(rng))
  const d = denseForward(model.dec, z)
  const o = denseForward(model.out, d.a)
  const zero = z.map(() => 0)
  return {
    h: { z: [], a: [] },
    mu: zero,
    logvar: zero,
    sigma: zero.map(() => 1),
    eps: z,
    z,
    d,
    o,
    generated: true,
  }
}

function backDense(
  layer: DenseLayer,
  input: number[],
  delta: number[],
  lr: number,
): number[] {
  const prev = input.map((_, i) => layer.weights.reduce((s, row, j) => s + row[i] * delta[j], 0))
  for (let j = 0; j < layer.weights.length; j++) {
    const row = layer.weights[j]
    const d = lr * delta[j]
    for (let i = 0; i < input.length; i++) row[i] -= d * input[i]
    layer.biases[j] -= d
  }
  return prev
}

export interface VAETask {
  model: VAEModel
  n: number
  latent: number
  classCount: number
  makeSample: (cls: number, rng: Rng) => Tensor3
  finalMSE: number
  finalKL: number
}

export function buildVAETask(patternSample: (n: number, cls: number, rng: Rng) => Tensor3): VAETask {
  const rng = mulberry32(0x7ae1)
  const n = 8
  const latent = 6
  const beta = 0.001 // KL weight — small so reconstructions stay sharp at this tiny scale

  const scaffold = createMLP(rng, [n * n, 24, latent, 24, n * n], ['relu', 'linear', 'relu', 'sigmoid'])
  const lvScaffold = createMLP(rng, [24, latent], ['linear'])
  const model: VAEModel = {
    n,
    latent,
    enc: scaffold.layers[0],
    muHead: scaffold.layers[1],
    lvHead: lvScaffold.layers[0],
    dec: scaffold.layers[2],
    out: scaffold.layers[3],
  }

  const makeSample = (cls: number, r: Rng) => patternSample(n, cls, r)
  const data: number[][] = []
  for (let c = 0; c < 4; c++) for (let i = 0; i < 30; i++) data.push(makeSample(c, rng)[0].flat())

  const order = data.map((_, i) => i)
  const epochs = 200
  const lr = 0.05
  let mse = 0
  let kl = 0
  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    mse = 0
    kl = 0
    for (const idx of order) {
      const x = data[idx]
      const eps = Array.from({ length: latent }, () => gaussian(rng))
      const tr = forwardVAE(model, x, eps)
      mse += tr.o.a.reduce((s, v, i) => s + (v - x[i]) ** 2, 0) / x.length
      kl += -0.5 * tr.logvar.reduce((s, lv, i) => s + 1 + lv - tr.mu[i] ** 2 - Math.exp(lv), 0)

      // ---- backward
      // output sigmoid + MSE
      let delta = tr.o.a.map((v, i) => ((v - x[i]) * 2 * v * (1 - v)) / x.length)
      let dDec = backDense(model.out, tr.d.a, delta, lr)
      dDec = dDec.map((g, i) => (tr.d.z[i] > 0 ? g : 0))
      const dz = backDense(model.dec, tr.z, dDec, lr)
      // reparameterization: z = μ + ε·σ, σ = exp(logvar/2)
      // plus KL gradients: dKL/dμ = μ, dKL/dlogvar = (exp(logvar) − 1)/2
      const dMu = dz.map((g, i) => g + beta * tr.mu[i])
      const dLv = dz.map(
        (g, i) => g * tr.eps[i] * tr.sigma[i] * 0.5 + beta * 0.5 * (Math.exp(tr.logvar[i]) - 1),
      )
      const dhMu = backDense(model.muHead, tr.h.a, dMu, lr)
      const dhLv = backDense(model.lvHead, tr.h.a, dLv, lr)
      const dh = dhMu.map((g, i) => (g + dhLv[i]) * (tr.h.z[i] > 0 ? 1 : 0))
      backDense(model.enc, x, dh, lr)
    }
    mse /= data.length
    kl /= data.length
  }

  return { model, n, latent, classCount: 4, makeSample, finalMSE: mse, finalKL: kl }
}
