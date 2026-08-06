// A tiny but structurally faithful character-level transformer:
// tokenizer → embedding → sinusoidal positional encoding → multi-head causal
// attention → residual + LayerNorm → feed-forward → residual + LayerNorm →
// output head. Forward AND backward passes are hand-written and the model is
// trained for real in the browser; every intermediate is kept in the trace.

import { Rng, gaussian, mulberry32, shuffleInPlace } from './rng'

export const LN_EPS = 1e-5

export type LLMVariant = 'dense' | 'moe'

export interface LLMModel {
  vocab: string[]
  /** model width */
  d: number
  /** feed-forward hidden width */
  dff: number
  /** attention heads (d is split evenly across them) */
  heads: number
  /** max sequence length */
  T: number
  /** mixture-of-experts FFN instead of a dense one */
  moe: boolean
  nExperts: number
  dffE: number
  topK: number
  /** router [d][nExperts] */
  Wr: number[][]
  br: number[]
  /** expert FFNs: [expert][d][dffE] and [expert][dffE][d] */
  We1: number[][][]
  bE1: number[][]
  We2: number[][][]
  bE2: number[][]
  /** token embeddings [vocab][d] */
  E: number[][]
  /** fixed sinusoidal positional encodings [T][d] */
  P: number[][]
  /** projections [d][d], layout [in][out]; head h owns output columns h·dh .. h·dh+dh-1 */
  Wq: number[][]
  Wk: number[][]
  Wv: number[][]
  /** multi-head output projection applied to the concatenated heads */
  Wo: number[][]
  bo: number[]
  /** FFN */
  W1: number[][]
  b1: number[]
  W2: number[][]
  b2: number[]
  /** LayerNorm 1 (after attention residual) and 2 (after FFN residual) */
  g1: number[]
  be1: number[]
  g2: number[]
  be2: number[]
  /** output head [d][vocab] */
  Wout: number[][]
  bout: number[]
}

export interface LLMTrace {
  ids: number[]
  chars: string[]
  /** embedding lookup E[token], [T][d] */
  E: number[][]
  /** X = E + P */
  X: number[][]
  Q: number[][]
  K: number[][]
  V: number[][]
  /** per-head raw scaled scores [heads][T][T] (upper triangle masked) */
  S: number[][][]
  /** per-head attention weights [heads][T][T] */
  A: number[][][]
  /** concatenated per-head weighted sums, [T][d] */
  Zc: number[][]
  /** attention output after the W_O projection: Z = Zc·Wo + bo */
  Z: number[][]
  /** residual sums and LayerNorm stats for both Add&Norm stages */
  res1: number[][]
  mu1: number[]
  sig1: number[]
  R1: number[][]
  Hpre: number[][]
  H: number[][]
  /** FFN output projected back to d */
  F: number[][]
  /** MoE (empty when dense): raw router scores, softmax gates, top-k picks,
   *  per-expert hidden activations (zero rows = token not routed = skipped),
   *  and the gate-weighted combination */
  Sr: number[][]
  G: number[][]
  topIdx: number[][]
  expertH: number[][][]
  Y: number[][]
  res2: number[][]
  mu2: number[]
  sig2: number[]
  R2: number[][]
  /** logits per position [T][vocab] */
  U: number[][]
  /** softmax of the last position's logits */
  probs: number[]
}

function zeros(r: number, c: number): number[][] {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => 0))
}

function randn(r: number, c: number, std: number, rng: Rng): number[][] {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => gaussian(rng) * std))
}

function matmul(a: number[][], b: number[][]): number[][] {
  const out = zeros(a.length, b[0].length)
  for (let i = 0; i < a.length; i++) {
    for (let k = 0; k < b.length; k++) {
      const v = a[i][k]
      if (v === 0) continue
      for (let j = 0; j < b[0].length; j++) out[i][j] += v * b[k][j]
    }
  }
  return out
}

function softmaxRow(row: number[]): number[] {
  const m = Math.max(...row)
  const e = row.map((v) => Math.exp(v - m))
  const s = e.reduce((acc, v) => acc + v, 0)
  return e.map((v) => v / s)
}

function sinusoidalP(T: number, d: number): number[][] {
  const P = zeros(T, d)
  for (let i = 0; i < T; i++) {
    for (let k = 0; k < d; k++) {
      const freq = Math.pow(10000, -Math.floor(k / 2) * 2 / d)
      P[i][k] = k % 2 === 0 ? Math.sin(i * freq) : Math.cos(i * freq)
    }
  }
  return P
}

function layerNormRow(x: number[], g: number[], b: number[]): { y: number[]; mu: number; sig: number } {
  const d = x.length
  const mu = x.reduce((s, v) => s + v, 0) / d
  const variance = x.reduce((s, v) => s + (v - mu) ** 2, 0) / d
  const sig = Math.sqrt(variance + LN_EPS)
  return { y: x.map((v, k) => ((v - mu) / sig) * g[k] + b[k]), mu, sig }
}

export function encodeLLM(model: LLMModel, text: string): number[] {
  const space = model.vocab.indexOf(' ')
  const ids = [...text.toLowerCase()]
    .map((ch) => {
      const i = model.vocab.indexOf(ch)
      return i >= 0 ? i : space
    })
    .slice(-model.T)
  return ids.length > 0 ? ids : [space]
}

export function forwardLLM(model: LLMModel, ids: number[]): LLMTrace {
  const { d, heads } = model
  const dh = d / heads
  const T = ids.length
  const E = ids.map((id) => [...model.E[id]])
  const X = E.map((row, i) => row.map((v, k) => v + model.P[i][k]))
  const Q = matmul(X, model.Wq)
  const K = matmul(X, model.Wk)
  const V = matmul(X, model.Wv)
  const scale = 1 / Math.sqrt(dh)
  const S: number[][][] = []
  const A: number[][][] = []
  const Zc = zeros(T, d)
  for (let h = 0; h < heads; h++) {
    const off = h * dh
    const Sh = zeros(T, T)
    const Ah = zeros(T, T)
    for (let i = 0; i < T; i++) {
      const masked: number[] = []
      for (let j = 0; j < T; j++) {
        let s = 0
        for (let k = 0; k < dh; k++) s += Q[i][off + k] * K[j][off + k]
        Sh[i][j] = s * scale
        if (j <= i) masked.push(Sh[i][j])
      }
      const soft = softmaxRow(masked)
      for (let j = 0; j <= i; j++) Ah[i][j] = soft[j]
      for (let j = 0; j <= i; j++) {
        for (let k = 0; k < dh; k++) Zc[i][off + k] += Ah[i][j] * V[j][off + k]
      }
    }
    S.push(Sh)
    A.push(Ah)
  }
  const Z = matmul(Zc, model.Wo).map((row) => row.map((v, k) => v + model.bo[k]))
  const res1 = X.map((row, i) => row.map((v, k) => v + Z[i][k]))
  const mu1: number[] = []
  const sig1: number[] = []
  const R1 = res1.map((row) => {
    const { y, mu, sig } = layerNormRow(row, model.g1, model.be1)
    mu1.push(mu)
    sig1.push(sig)
    return y
  })

  let Hpre: number[][] = []
  let H: number[][] = []
  let F: number[][] = []
  let Sr: number[][] = []
  let G: number[][] = []
  let topIdx: number[][] = []
  let expertH: number[][][] = []
  let Y: number[][] = []

  if (!model.moe) {
    Hpre = matmul(R1, model.W1).map((row) => row.map((v, j) => v + model.b1[j]))
    H = Hpre.map((row) => row.map((v) => Math.max(0, v)))
    F = matmul(H, model.W2).map((row) => row.map((v, j) => v + model.b2[j]))
  } else {
    const E = model.nExperts
    Sr = R1.map((row) => model.br.map((b, e) => row.reduce((s, v, m) => s + v * model.Wr[m][e], b)))
    G = Sr.map((row) => softmaxRow(row))
    topIdx = G.map((g) =>
      [...g.keys()].sort((a, b) => g[b] - g[a]).slice(0, model.topK).sort((a, b) => a - b),
    )
    expertH = Array.from({ length: E }, () => zeros(T, model.dffE))
    Y = zeros(T, d)
    for (let t = 0; t < T; t++) {
      for (const e of topIdx[t]) {
        for (let j = 0; j < model.dffE; j++) {
          let s = model.bE1[e][j]
          for (let m = 0; m < d; m++) s += R1[t][m] * model.We1[e][m][j]
          expertH[e][t][j] = Math.max(0, s)
        }
        const g = G[t][e]
        for (let k = 0; k < d; k++) {
          let s = model.bE2[e][k]
          for (let j = 0; j < model.dffE; j++) s += expertH[e][t][j] * model.We2[e][j][k]
          Y[t][k] += g * s
        }
      }
    }
  }

  const sub = model.moe ? Y : F
  const res2 = R1.map((row, i) => row.map((v, k) => v + sub[i][k]))
  const mu2: number[] = []
  const sig2: number[] = []
  const R2 = res2.map((row) => {
    const { y, mu, sig } = layerNormRow(row, model.g2, model.be2)
    mu2.push(mu)
    sig2.push(sig)
    return y
  })
  const U = matmul(R2, model.Wout).map((row) => row.map((v, j) => v + model.bout[j]))
  return {
    ids,
    chars: ids.map((id) => model.vocab[id]),
    E,
    X,
    Q,
    K,
    V,
    S,
    A,
    Zc,
    Z,
    res1,
    mu1,
    sig1,
    R1,
    Hpre,
    H,
    F,
    Sr,
    G,
    topIdx,
    expertH,
    Y,
    res2,
    mu2,
    sig2,
    R2,
    U,
    probs: softmaxRow(U[T - 1]),
  }
}

// ---------------------------------------------------------------- training

function addScaled(target: number[][], grad: number[][], lr: number): void {
  for (let i = 0; i < target.length; i++)
    for (let j = 0; j < target[0].length; j++) target[i][j] -= lr * grad[i][j]
}

function lnBackwardRow(
  dy: number[],
  x: number[],
  mu: number,
  sig: number,
  g: number[],
  dg: number[],
  db: number[],
): number[] {
  const d = dy.length
  const xhat = x.map((v) => (v - mu) / sig)
  const dxhat = dy.map((v, k) => v * g[k])
  let m1 = 0
  let m2 = 0
  for (let k = 0; k < d; k++) {
    dg[k] += dy[k] * xhat[k]
    db[k] += dy[k]
    m1 += dxhat[k]
    m2 += dxhat[k] * xhat[k]
  }
  m1 /= d
  m2 /= d
  return xhat.map((xh, k) => (dxhat[k] - m1 - xh * m2) / sig)
}

/** One SGD step on a single window; returns the mean cross-entropy loss. */
function trainStep(model: LLMModel, ids: number[], targets: number[], lr: number): number {
  const { d, dff, heads } = model
  const dh = d / heads
  const T = ids.length
  const tr = forwardLLM(model, ids)
  const vocabN = model.vocab.length

  let loss = 0
  const dU = zeros(T, vocabN)
  for (let i = 0; i < T; i++) {
    const p = softmaxRow(tr.U[i])
    loss += -Math.log(Math.max(1e-9, p[targets[i]]))
    for (let j = 0; j < vocabN; j++) dU[i][j] = (p[j] - (j === targets[i] ? 1 : 0)) / T
  }
  loss /= T

  // output head
  const dWout = zeros(d, vocabN)
  const dbout = Array.from({ length: vocabN }, () => 0)
  const dR2 = zeros(T, d)
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < vocabN; j++) {
      const gr = dU[i][j]
      if (gr === 0) continue
      dbout[j] += gr
      for (let k = 0; k < d; k++) {
        dWout[k][j] += tr.R2[i][k] * gr
        dR2[i][k] += gr * model.Wout[k][j]
      }
    }
  }

  // LayerNorm 2 → residual split
  const dg2 = Array.from({ length: d }, () => 0)
  const dbe2 = Array.from({ length: d }, () => 0)
  const dRes2 = tr.res2.map((row, i) =>
    lnBackwardRow(dR2[i], row, tr.mu2[i], tr.sig2[i], model.g2, dg2, dbe2),
  )
  const dR1 = dRes2.map((row) => [...row]) // residual branch
  const dF = dRes2

  // FFN backward — dense path, or MoE (combine → selected experts → router softmax)
  const dW1 = zeros(d, dff)
  const db1 = Array.from({ length: dff }, () => 0)
  const dW2 = zeros(dff, d)
  const db2 = Array.from({ length: d }, () => 0)
  if (!model.moe) {
    const dH = zeros(T, dff)
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < d; j++) {
        const gr = dF[i][j]
        if (gr === 0) continue
        db2[j] += gr
        for (let k = 0; k < dff; k++) {
          dW2[k][j] += tr.H[i][k] * gr
          dH[i][k] += gr * model.W2[k][j]
        }
      }
    }
    const dHpre = dH.map((row, i) => row.map((gr, k) => (tr.Hpre[i][k] > 0 ? gr : 0)))
    for (let i = 0; i < T; i++) {
      for (let k = 0; k < dff; k++) {
        const gr = dHpre[i][k]
        if (gr === 0) continue
        db1[k] += gr
        for (let m = 0; m < d; m++) {
          dW1[m][k] += tr.R1[i][m] * gr
          dR1[i][m] += gr * model.W1[m][k]
        }
      }
    }
  } else {
    const E = model.nExperts
    const dffE = model.dffE
    const dWr = zeros(d, E)
    const dbr = Array.from({ length: E }, () => 0)
    const dWe1 = Array.from({ length: E }, () => zeros(d, dffE))
    const dbE1 = Array.from({ length: E }, () => Array.from({ length: dffE }, () => 0))
    const dWe2 = Array.from({ length: E }, () => zeros(dffE, d))
    const dbE2 = Array.from({ length: E }, () => Array.from({ length: d }, () => 0))
    for (let t = 0; t < T; t++) {
      const dg = Array.from({ length: E }, () => 0)
      for (const e of tr.topIdx[t]) {
        const h = tr.expertH[e][t]
        // recompute this expert's output — needed for the gate gradient
        const out = model.bE2[e].map((b, k) => b + h.reduce((s, hv, j) => s + hv * model.We2[e][j][k], 0))
        const g = tr.G[t][e]
        const dh = Array.from({ length: dffE }, () => 0)
        for (let k = 0; k < d; k++) {
          dg[e] += dF[t][k] * out[k]
          const gy = g * dF[t][k]
          if (gy === 0) continue
          dbE2[e][k] += gy
          for (let j = 0; j < dffE; j++) {
            dWe2[e][j][k] += h[j] * gy
            dh[j] += gy * model.We2[e][j][k]
          }
        }
        for (let j = 0; j < dffE; j++) {
          if (h[j] <= 0) continue
          const dpre = dh[j]
          dbE1[e][j] += dpre
          for (let m = 0; m < d; m++) {
            dWe1[e][m][j] += tr.R1[t][m] * dpre
            dR1[t][m] += dpre * model.We1[e][m][j]
          }
        }
      }
      // router softmax backward (unselected experts contribute dg = 0)
      let dot = 0
      for (let e = 0; e < E; e++) dot += tr.G[t][e] * dg[e]
      for (let e = 0; e < E; e++) {
        const ds = tr.G[t][e] * (dg[e] - dot)
        if (ds === 0) continue
        dbr[e] += ds
        for (let m = 0; m < d; m++) {
          dWr[m][e] += tr.R1[t][m] * ds
          dR1[t][m] += ds * model.Wr[m][e]
        }
      }
    }
    for (let e = 0; e < E; e++) {
      model.br[e] -= lr * dbr[e]
      for (let m = 0; m < d; m++) model.Wr[m][e] -= lr * dWr[m][e]
      for (let j = 0; j < dffE; j++) {
        model.bE1[e][j] -= lr * dbE1[e][j]
        for (let m = 0; m < d; m++) model.We1[e][m][j] -= lr * dWe1[e][m][j]
      }
      for (let k = 0; k < d; k++) {
        model.bE2[e][k] -= lr * dbE2[e][k]
        for (let j = 0; j < dffE; j++) model.We2[e][j][k] -= lr * dWe2[e][j][k]
      }
    }
  }

  // LayerNorm 1 → residual split
  const dg1 = Array.from({ length: d }, () => 0)
  const dbe1 = Array.from({ length: d }, () => 0)
  const dRes1 = tr.res1.map((row, i) =>
    lnBackwardRow(dR1[i], row, tr.mu1[i], tr.sig1[i], model.g1, dg1, dbe1),
  )
  const dX = dRes1.map((row) => [...row]) // residual branch
  const dZ = dRes1

  // W_O projection backward: Z = Zc·Wo + bo
  const dWo = zeros(d, d)
  const dbo = Array.from({ length: d }, () => 0)
  const dZc = zeros(T, d)
  for (let i = 0; i < T; i++) {
    for (let k = 0; k < d; k++) {
      const gr = dZ[i][k]
      if (gr === 0) continue
      dbo[k] += gr
      for (let m = 0; m < d; m++) {
        dWo[m][k] += tr.Zc[i][m] * gr
        dZc[i][m] += gr * model.Wo[m][k]
      }
    }
  }

  // multi-head attention backward
  const scale = 1 / Math.sqrt(dh)
  const dQ = zeros(T, d)
  const dK = zeros(T, d)
  const dV = zeros(T, d)
  for (let h = 0; h < heads; h++) {
    const off = h * dh
    const Ah = tr.A[h]
    const dA = zeros(T, T)
    for (let i = 0; i < T; i++) {
      for (let j = 0; j <= i; j++) {
        let s = 0
        for (let m = 0; m < dh; m++) {
          s += dZc[i][off + m] * tr.V[j][off + m]
          dV[j][off + m] += Ah[i][j] * dZc[i][off + m]
        }
        dA[i][j] = s
      }
    }
    for (let i = 0; i < T; i++) {
      let dot = 0
      for (let j = 0; j <= i; j++) dot += dA[i][j] * Ah[i][j]
      for (let j = 0; j <= i; j++) {
        const dS = Ah[i][j] * (dA[i][j] - dot) * scale
        if (dS === 0) continue
        for (let k = 0; k < dh; k++) {
          dQ[i][off + k] += dS * tr.K[j][off + k]
          dK[j][off + k] += dS * tr.Q[i][off + k]
        }
      }
    }
  }

  const dWq = zeros(d, d)
  const dWk = zeros(d, d)
  const dWv = zeros(d, d)
  for (let i = 0; i < T; i++) {
    for (let m = 0; m < d; m++) {
      for (let k = 0; k < d; k++) {
        dWq[m][k] += tr.X[i][m] * dQ[i][k]
        dWk[m][k] += tr.X[i][m] * dK[i][k]
        dWv[m][k] += tr.X[i][m] * dV[i][k]
        dX[i][m] += dQ[i][k] * model.Wq[m][k] + dK[i][k] * model.Wk[m][k] + dV[i][k] * model.Wv[m][k]
      }
    }
  }

  addScaled(model.Wout, dWout, lr)
  addScaled(model.W2, dW2, lr)
  addScaled(model.W1, dW1, lr)
  addScaled(model.Wq, dWq, lr)
  addScaled(model.Wk, dWk, lr)
  addScaled(model.Wv, dWv, lr)
  addScaled(model.Wo, dWo, lr)
  for (let j = 0; j < vocabN; j++) model.bout[j] -= lr * dbout[j]
  for (let j = 0; j < d; j++) {
    model.b2[j] -= lr * db2[j]
    model.bo[j] -= lr * dbo[j]
    model.g1[j] -= lr * dg1[j]
    model.be1[j] -= lr * dbe1[j]
    model.g2[j] -= lr * dg2[j]
    model.be2[j] -= lr * dbe2[j]
  }
  for (let k = 0; k < dff; k++) model.b1[k] -= lr * db1[k]
  for (let i = 0; i < T; i++) {
    for (let m = 0; m < d; m++) model.E[ids[i]][m] -= lr * dX[i][m]
  }
  return loss
}

// ---------------------------------------------------------------- task

export const LLM_CORPUS =
  'hello world. the ai room. attention is all you need. neural networks learn deep patterns. ' +
  'we walk inside a living neural network. data flows through every layer. the model learns to ' +
  'read and the room comes alive. deep nets see the world. ' +
  'the quick brown fox jumps over the lazy dog. ' // pangram: every letter a–z is in the vocabulary

export interface LLMTask {
  model: LLMModel
  samples: string[]
  finalLoss: number
}

export function buildLLMTask(variant: LLMVariant = 'dense'): LLMTask {
  const rng = mulberry32(0x11a1)
  const vocab = [...new Set([...LLM_CORPUS.toLowerCase()])].sort()
  const d = 12
  const dff = 24
  const heads = 2
  const T = 8
  const moe = variant === 'moe'
  const nExperts = 4
  const dffE = 12
  const model: LLMModel = {
    vocab,
    d,
    dff,
    heads,
    T,
    moe,
    nExperts,
    dffE,
    topK: 2,
    E: randn(vocab.length, d, 0.3, rng),
    P: sinusoidalP(T, d).map((row) => row.map((v) => v * 0.4)),
    Wq: randn(d, d, 0.28, rng),
    Wk: randn(d, d, 0.28, rng),
    Wv: randn(d, d, 0.28, rng),
    Wo: randn(d, d, 0.28, rng),
    bo: Array.from({ length: d }, () => 0),
    W1: randn(d, dff, 0.28, rng),
    b1: Array.from({ length: dff }, () => 0),
    W2: randn(dff, d, 0.28, rng),
    b2: Array.from({ length: d }, () => 0),
    Wr: moe ? randn(d, nExperts, 0.3, rng) : [],
    br: moe ? Array.from({ length: nExperts }, () => 0) : [],
    We1: moe ? Array.from({ length: nExperts }, () => randn(d, dffE, 0.28, rng)) : [],
    bE1: moe ? Array.from({ length: nExperts }, () => Array.from({ length: dffE }, () => 0)) : [],
    We2: moe ? Array.from({ length: nExperts }, () => randn(dffE, d, 0.28, rng)) : [],
    bE2: moe ? Array.from({ length: nExperts }, () => Array.from({ length: d }, () => 0)) : [],
    g1: Array.from({ length: d }, () => 1),
    be1: Array.from({ length: d }, () => 0),
    g2: Array.from({ length: d }, () => 1),
    be2: Array.from({ length: d }, () => 0),
    Wout: randn(d, vocab.length, 0.28, rng),
    bout: Array.from({ length: vocab.length }, () => 0),
  }

  const chars = [...LLM_CORPUS.toLowerCase()]
  const idOf = new Map(vocab.map((c, i) => [c, i]))
  const windows: { ids: number[]; targets: number[] }[] = []
  for (let s = 0; s + T + 1 <= chars.length; s += 2) {
    const ids: number[] = []
    const targets: number[] = []
    for (let i = 0; i < T; i++) {
      ids.push(idOf.get(chars[s + i])!)
      targets.push(idOf.get(chars[s + i + 1])!)
    }
    windows.push({ ids, targets })
  }

  let loss = 0
  const order = windows.map((_, i) => i)
  const epochs = 42
  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleInPlace(order, rng)
    const lr = 0.1 * (1 - (epoch / epochs) * 0.7)
    loss = 0
    for (const i of order) loss += trainStep(model, windows[i].ids, windows[i].targets, lr)
    loss /= windows.length
  }

  return {
    model,
    samples: ['hello wo', 'the ai r', 'attentio', 'deep net'],
    finalLoss: loss,
  }
}

/** Top-1 next-char accuracy over the training corpus (for sanity checks). */
export function evalLLMAccuracy(task: LLMTask): number {
  const chars = [...LLM_CORPUS.toLowerCase()]
  const idOf = new Map(task.model.vocab.map((c, i) => [c, i]))
  let ok = 0
  let total = 0
  for (let s = 0; s + task.model.T + 1 <= chars.length; s += task.model.T) {
    const ids = chars.slice(s, s + task.model.T).map((c) => idOf.get(c)!)
    const tr = forwardLLM(task.model, ids)
    const pred = tr.probs.indexOf(Math.max(...tr.probs))
    if (task.model.vocab[pred] === chars[s + task.model.T]) ok++
    total++
  }
  return ok / total
}
