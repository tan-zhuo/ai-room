// A minimal Mamba (selective state space model, Gu & Dao 2023) doing the same
// char-level next-token task as the RNN / LSTM / Transformer, so the four can
// be compared head-to-head. The heart is the SELECTIVE scan: unlike earlier
// SSMs, Δ, B and C are functions of the input, so the state decides per token
// what to remember and what to forget — in O(T) time with a fixed-size state.
//
//   Δ_t = softplus(x_t·WΔ + bΔ)          (input-dependent step size)
//   ā_t = exp(Δ_t ⊗ A)   A = -exp(logA)  (per-channel decay, always < 1)
//   h_t = ā_t ⊙ h_{t-1} + (Δ_t·u_t) ⊗ B_t
//   y_t = h_t · C_t + D ⊙ u_t            (B_t = x_t·WB, C_t = x_t·WC)
//   o_t = y_t ⊙ silu(z_t)                (gate branch z_t = x_t·Wz)
//
// Forward AND backward (BPTT through the scan) are hand-written.

import { Rng, gaussian, mulberry32, shuffleInPlace } from './rng'
import { buildVocab, corpusWindows } from './rnn'

export interface MambaModel {
  vocab: string[]
  d: number
  /** inner (expanded) width */
  dI: number
  /** state size per channel */
  N: number
  T: number
  E: number[][]
  /** [d][dI] input projection */
  Win: number[][]
  /** [d][dI] gate branch */
  Wz: number[][]
  /** Δ projection [d][dI] + bias */
  Wd: number[][]
  bd: number[]
  /** [d][N] input-dependent B and C */
  WB: number[][]
  WC: number[][]
  /** [dI][N]; A = -exp(logA) keeps the decay stable */
  logA: number[][]
  /** skip connection weight per channel */
  Dskip: number[]
  Wout: number[][]
  bout: number[]
}

export interface MambaTrace {
  ids: number[]
  chars: string[]
  /** embeddings [T][d] */
  X: number[][]
  /** inner input u [T][dI] */
  U: number[][]
  /** selectivity Δ (post-softplus) [T][dI] */
  Delta: number[][]
  /** input-dependent B_t, C_t [T][N] */
  Bt: number[][]
  Ct: number[][]
  /** state after each step [T][dI][N] */
  H: number[][][]
  /** SSM output before gating [T][dI] */
  Y: number[][]
  /** gate silu(z) [T][dI] */
  G: number[][]
  /** gated output [T][dI] */
  O: number[][]
  /** next-char distribution from the LAST position */
  probs: number[]
}

function randn(r: number, c: number, std: number, rng: Rng): number[][] {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => gaussian(rng) * std))
}

const sig = (v: number) => 1 / (1 + Math.exp(-v))
const softplus = (v: number) => (v > 20 ? v : Math.log(1 + Math.exp(v)))
const silu = (v: number) => v * sig(v)

function proj(x: number[], W: number[][]): number[] {
  return Array.from({ length: W[0].length }, (_, j) => x.reduce((s, v, k) => s + v * W[k][j], 0))
}

export function forwardMamba(model: MambaModel, ids: number[]): MambaTrace {
  const { dI, N } = model
  const T = ids.length
  const X: number[][] = []
  const U: number[][] = []
  const Delta: number[][] = []
  const Bt: number[][] = []
  const Ct: number[][] = []
  const H: number[][][] = []
  const Y: number[][] = []
  const G: number[][] = []
  const O: number[][] = []
  let h = Array.from({ length: dI }, () => Array.from({ length: N }, () => 0))

  for (let t = 0; t < T; t++) {
    const x = model.E[ids[t]]
    const u = proj(x, model.Win)
    const z = proj(x, model.Wz)
    const dPre = proj(x, model.Wd).map((v, i) => v + model.bd[i])
    const delta = dPre.map(softplus)
    const b = proj(x, model.WB)
    const c = proj(x, model.WC)
    const hNext = h.map((row, i) =>
      row.map((hv, n) => {
        const a = Math.exp(-Math.exp(model.logA[i][n]) * delta[i])
        return a * hv + delta[i] * b[n] * u[i]
      }),
    )
    const y = hNext.map((row, i) => row.reduce((s, hv, n) => s + hv * c[n], 0) + model.Dskip[i] * u[i])
    const g = z.map(silu)
    const o = y.map((v, i) => v * g[i])
    X.push(x)
    U.push(u)
    Delta.push(delta)
    Bt.push(b)
    Ct.push(c)
    H.push(hNext)
    Y.push(y)
    G.push(g)
    O.push(o)
    h = hNext
  }

  const logits = model.bout.map((bb, v) => O[T - 1].reduce((s, ov, i) => s + ov * model.Wout[i][v], bb))
  const m = Math.max(...logits)
  const e = logits.map((v) => Math.exp(v - m))
  const su = e.reduce((acc, v) => acc + v, 0)
  return {
    ids,
    chars: ids.map((i) => model.vocab[i]),
    X,
    U,
    Delta,
    Bt,
    Ct,
    H,
    Y,
    G,
    O,
    probs: e.map((v) => v / su),
  }
}

export interface MambaTask {
  model: MambaModel
  samples: string[]
  finalLoss: number
}

const MAMBA_SCALE_CFG = {
  s: { d: 12, dI: 24, N: 8, T: 8, epochs: 22 },
  m: { d: 16, dI: 32, N: 8, T: 10, epochs: 26 },
  l: { d: 20, dI: 44, N: 12, T: 12, epochs: 26 },
} as const

export function buildMambaTask(scale: 's' | 'm' | 'l' = 's'): MambaTask {
  const rng = mulberry32(0x3a3ba + scale.charCodeAt(0))
  const vocab = buildVocab()
  const { d, dI, N, T, epochs } = MAMBA_SCALE_CFG[scale]
  const V = vocab.length
  const model: MambaModel = {
    vocab,
    d,
    dI,
    N,
    T,
    E: randn(V, d, 0.3, rng),
    Win: randn(d, dI, 0.3, rng),
    Wz: randn(d, dI, 0.3, rng),
    Wd: randn(d, dI, 0.25, rng),
    bd: Array.from({ length: dI }, () => -0.5), // start with mild decay
    WB: randn(d, N, 0.3, rng),
    WC: randn(d, N, 0.3, rng),
    logA: Array.from({ length: dI }, () =>
      Array.from({ length: N }, (_, n) => Math.log(0.5 + (n / N) * 1.5)),
    ),
    Dskip: Array.from({ length: dI }, () => 1),
    Wout: randn(dI, V, 0.3, rng),
    bout: Array.from({ length: V }, () => 0),
  }

  const windows = corpusWindows(vocab, T)
  const order = windows.map((_, i) => i)
  let loss = 0
  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    const lr = 0.09 * (1 - (e / epochs) * 0.65)
    loss = 0
    for (const i of order) loss += mambaStep(model, windows[i].ids, windows[i].targets, lr)
    loss /= windows.length
  }
  return { model, samples: ['hello wo', 'the ai r', 'attentio', 'deep net'], finalLoss: loss }
}

/** One BPTT step over a window; loss on every position. */
function mambaStep(model: MambaModel, ids: number[], targets: number[], lr: number): number {
  const { d, dI, N } = model
  const T = ids.length
  const V = model.vocab.length

  // ---- forward, keeping everything (incl. per-position logits)
  const X: number[][] = []
  const U: number[][] = []
  const Z: number[][] = []
  const DPre: number[][] = []
  const Delta: number[][] = []
  const Bt: number[][] = []
  const Ct: number[][] = []
  const Abar: number[][][] = []
  const H: number[][][] = []
  const Y: number[][] = []
  const O: number[][] = []
  const P: number[][] = []
  let h = Array.from({ length: dI }, () => Array.from({ length: N }, () => 0))
  let loss = 0

  for (let t = 0; t < T; t++) {
    const x = model.E[ids[t]]
    const u = proj(x, model.Win)
    const z = proj(x, model.Wz)
    const dPre = proj(x, model.Wd).map((v, i) => v + model.bd[i])
    const delta = dPre.map(softplus)
    const b = proj(x, model.WB)
    const c = proj(x, model.WC)
    const abar = Array.from({ length: dI }, (_, i) =>
      Array.from({ length: N }, (_, n) => Math.exp(-Math.exp(model.logA[i][n]) * delta[i])),
    )
    const hN = h.map((row, i) => row.map((hv, n) => abar[i][n] * hv + delta[i] * b[n] * u[i]))
    const y = hN.map((row, i) => row.reduce((s, hv, n) => s + hv * c[n], 0) + model.Dskip[i] * u[i])
    const o = y.map((v, i) => v * silu(z[i]))
    const logits = model.bout.map((bb, v) => o.reduce((s, ov, i) => s + ov * model.Wout[i][v], bb))
    const mx = Math.max(...logits)
    const ex = logits.map((v) => Math.exp(v - mx))
    const su = ex.reduce((acc, v) => acc + v, 0)
    const p = ex.map((v) => v / su)
    loss -= Math.log(Math.max(p[targets[t]], 1e-9)) / T
    X.push(x)
    U.push(u)
    Z.push(z)
    DPre.push(dPre)
    Delta.push(delta)
    Bt.push(b)
    Ct.push(c)
    Abar.push(abar)
    H.push(hN)
    Y.push(y)
    O.push(o)
    P.push(p)
    h = hN
  }

  // ---- grads
  const gE = new Map<number, number[]>()
  const gWin = randnZero(d, dI)
  const gWz = randnZero(d, dI)
  const gWd = randnZero(d, dI)
  const gBd = Array.from({ length: dI }, () => 0)
  const gWB = randnZero(d, N)
  const gWC = randnZero(d, N)
  const gLogA = randnZero(dI, N)
  const gD = Array.from({ length: dI }, () => 0)
  const gWout = randnZero(dI, V)
  const gBout = Array.from({ length: V }, () => 0)

  let dhNext = Array.from({ length: dI }, () => Array.from({ length: N }, () => 0))

  for (let t = T - 1; t >= 0; t--) {
    // head
    const dLogits = P[t].map((p, v) => (p - (v === targets[t] ? 1 : 0)) / T)
    const dO = Array.from({ length: dI }, (_, i) =>
      dLogits.reduce((s, g, v) => s + g * model.Wout[i][v], 0),
    )
    for (let i = 0; i < dI; i++) for (let v = 0; v < V; v++) gWout[i][v] += O[t][i] * dLogits[v]
    for (let v = 0; v < V; v++) gBout[v] += dLogits[v]

    // gate: o = y·silu(z)
    const dY = Array.from({ length: dI }, () => 0)
    const dZ = Array.from({ length: dI }, () => 0)
    for (let i = 0; i < dI; i++) {
      const s = sig(Z[t][i])
      dY[i] = dO[i] * Z[t][i] * s
      dZ[i] = dO[i] * Y[t][i] * s * (1 + Z[t][i] * (1 - s))
    }

    // y = Σ h·C + D·u
    const dH = dhNext.map((row) => [...row])
    const dC = Array.from({ length: N }, () => 0)
    const dU = Array.from({ length: dI }, () => 0)
    for (let i = 0; i < dI; i++) {
      for (let n = 0; n < N; n++) {
        dH[i][n] += dY[i] * Ct[t][n]
        dC[n] += dY[i] * H[t][i][n]
      }
      gD[i] += dY[i] * U[t][i]
      dU[i] = dY[i] * model.Dskip[i]
    }

    // scan: h_t = ā ⊙ h_{t-1} + Δ·B·u
    const hPrev = t > 0 ? H[t - 1] : Array.from({ length: dI }, () => Array.from({ length: N }, () => 0))
    const dDelta = Array.from({ length: dI }, () => 0)
    const dB = Array.from({ length: N }, () => 0)
    dhNext = Array.from({ length: dI }, () => Array.from({ length: N }, () => 0))
    for (let i = 0; i < dI; i++) {
      for (let n = 0; n < N; n++) {
        const g = dH[i][n]
        const A = Math.exp(model.logA[i][n])
        // ā = exp(-A·Δ):  dā = g·h_prev;  dΔ += dā·(-A)·ā;  dlogA += dā·(-Δ·A)·ā
        const dabar = g * hPrev[i][n]
        dDelta[i] += dabar * -A * Abar[t][i][n] + g * Bt[t][n] * U[t][i]
        gLogA[i][n] += dabar * -Delta[t][i] * A * Abar[t][i][n]
        dB[n] += g * Delta[t][i] * U[t][i]
        dU[i] += g * Delta[t][i] * Bt[t][n]
        dhNext[i][n] = g * Abar[t][i][n]
      }
    }

    // Δ = softplus(dPre)
    const dDPre = dDelta.map((g, i) => g * sig(DPre[t][i]))
    for (let i = 0; i < dI; i++) gBd[i] += dDPre[i]

    // project everything back to x
    const dX = Array.from({ length: d }, () => 0)
    for (let k = 0; k < d; k++) {
      for (let i = 0; i < dI; i++) {
        gWin[k][i] += X[t][k] * dU[i]
        gWz[k][i] += X[t][k] * dZ[i]
        gWd[k][i] += X[t][k] * dDPre[i]
        dX[k] += model.Win[k][i] * dU[i] + model.Wz[k][i] * dZ[i] + model.Wd[k][i] * dDPre[i]
      }
      for (let n = 0; n < N; n++) {
        gWB[k][n] += X[t][k] * dB[n]
        gWC[k][n] += X[t][k] * dC[n]
        dX[k] += model.WB[k][n] * dB[n] + model.WC[k][n] * dC[n]
      }
    }
    const id = ids[t]
    if (!gE.has(id)) gE.set(id, Array.from({ length: d }, () => 0))
    const ge = gE.get(id)!
    for (let k = 0; k < d; k++) ge[k] += dX[k]
  }

  // ---- SGD
  const upd = (W: number[][], G: number[][]) => {
    for (let a = 0; a < W.length; a++) for (let b2 = 0; b2 < W[a].length; b2++) W[a][b2] -= lr * G[a][b2]
  }
  upd(model.Win, gWin)
  upd(model.Wz, gWz)
  upd(model.Wd, gWd)
  upd(model.WB, gWB)
  upd(model.WC, gWC)
  upd(model.logA, gLogA)
  upd(model.Wout, gWout)
  for (let i = 0; i < dI; i++) {
    model.bd[i] -= lr * gBd[i]
    model.Dskip[i] -= lr * gD[i]
  }
  for (let v = 0; v < V; v++) model.bout[v] -= lr * gBout[v]
  for (const [id, ge] of gE) for (let k = 0; k < d; k++) model.E[id][k] -= lr * ge[k]

  return loss
}

function randnZero(r: number, c: number): number[][] {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => 0))
}
