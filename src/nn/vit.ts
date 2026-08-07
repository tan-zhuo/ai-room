// A tiny Vision Transformer (Dosovitskiy et al. 2020) classifying the same
// procedural patterns as the CNN. Faithful structure at toy scale: the image
// is cut into patches, each patch linearly embedded, a learned [CLS] token
// prepended, learned positional embeddings added, then one pre-LN encoder
// block (bidirectional MHA + MLP, both with residuals) and a classification
// head on the final [CLS] vector. Forward AND backward are hand-written.

import { Rng, gaussian, mulberry32, shuffleInPlace } from './rng'
import { Tensor3 } from './cnn'

export interface ViTModel {
  /** image side */
  n: number
  /** patch side */
  p: number
  /** patches per row/col and total tokens (incl. CLS) */
  grid: number
  T: number
  d: number
  heads: number
  dff: number
  classes: number
  /** patch projection [p²][d] */
  Wp: number[][]
  bp: number[]
  /** learned CLS token [d] and positional embeddings [T][d] */
  cls: number[]
  pos: number[][]
  /** pre-LN block */
  g1: number[]
  b1ln: number[]
  Wq: number[][]
  Wk: number[][]
  Wv: number[][]
  Wo: number[][]
  bo: number[]
  g2: number[]
  b2ln: number[]
  W1: number[][]
  b1: number[]
  W2: number[][]
  b2: number[]
  /** final LN + classification head [d][classes] */
  g3: number[]
  b3ln: number[]
  Wh: number[][]
  bh: number[]
}

export interface ViTTrace {
  /** flattened patches [grid²][p²] */
  patches: number[][]
  /** tokens after embed + pos, row 0 = CLS, [T][d] */
  X: number[][]
  ln1: number[][]
  Q: number[][]
  K: number[][]
  V: number[][]
  /** per-head attention [heads][T][T] */
  att: number[][][]
  /** attention output after Wo [T][d] */
  Z: number[][]
  /** after first residual [T][d] */
  X1: number[][]
  ln2: number[][]
  /** FFN hidden [T][dff] */
  Fh: number[][]
  /** FFN output [T][d] */
  F: number[][]
  /** after second residual [T][d] */
  X2: number[][]
  /** final LN of the CLS row [d] */
  clsLn: number[]
  logits: number[]
  probs: number[]
  pred: number
}

function randn(r: number, c: number, std: number, rng: Rng): number[][] {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => gaussian(rng) * std))
}

function zeros(n: number): number[] {
  return Array.from({ length: n }, () => 0)
}

function layerNormRow(x: number[], g: number[], b: number[]): { y: number[]; mu: number; sd: number } {
  const mu = x.reduce((s, v) => s + v, 0) / x.length
  const va = x.reduce((s, v) => s + (v - mu) ** 2, 0) / x.length
  const sd = Math.sqrt(va + 1e-5)
  return { y: x.map((v, i) => ((v - mu) / sd) * g[i] + b[i]), mu, sd }
}

/** dL/dx for y = LN(x)·g + b given dL/dy. */
function layerNormBack(
  x: number[],
  g: number[],
  mu: number,
  sd: number,
  dy: number[],
): { dx: number[]; dg: number[]; db: number[] } {
  const n = x.length
  const xhat = x.map((v) => (v - mu) / sd)
  const dxhat = dy.map((v, i) => v * g[i])
  const sumDxhat = dxhat.reduce((s, v) => s + v, 0)
  const sumDxhatXhat = dxhat.reduce((s, v, i) => s + v * xhat[i], 0)
  const dx = xhat.map((xh, i) => (dxhat[i] - sumDxhat / n - (xh * sumDxhatXhat) / n) / sd)
  return { dx, dg: dy.map((v, i) => v * xhat[i]), db: [...dy] }
}

export function extractPatches(model: ViTModel, img: number[][]): number[][] {
  const { p, grid } = model
  const patches: number[][] = []
  for (let gr = 0; gr < grid; gr++)
    for (let gc = 0; gc < grid; gc++) {
      const flat: number[] = []
      for (let r = 0; r < p; r++)
        for (let c = 0; c < p; c++) flat.push(img[gr * p + r][gc * p + c])
      patches.push(flat)
    }
  return patches
}

export function forwardViT(model: ViTModel, img: number[][]): ViTTrace {
  const { d, heads, T } = model
  const dh = d / heads
  const patches = extractPatches(model, img)
  const X: number[][] = [
    model.cls.map((v, j) => v + model.pos[0][j]),
    ...patches.map((pt, i) =>
      model.bp.map(
        (b, j) => pt.reduce((s, v, k) => s + v * model.Wp[k][j], b) + model.pos[i + 1][j],
      ),
    ),
  ]
  const l1 = X.map((row) => layerNormRow(row, model.g1, model.b1ln))
  const ln1 = l1.map((o) => o.y)
  const proj = (W: number[][]) =>
    ln1.map((row) => Array.from({ length: d }, (_, j) => row.reduce((s, v, k) => s + v * W[k][j], 0)))
  const Q = proj(model.Wq)
  const K = proj(model.Wk)
  const V = proj(model.Wv)

  const att: number[][][] = []
  const headOut: number[][][] = []
  for (let h = 0; h < heads; h++) {
    const o = h * dh
    const scores = Array.from({ length: T }, (_, i) =>
      Array.from({ length: T }, (_, j) => {
        let s = 0
        for (let k = 0; k < dh; k++) s += Q[i][o + k] * K[j][o + k]
        return s / Math.sqrt(dh)
      }),
    )
    const A = scores.map((row) => {
      const m = Math.max(...row)
      const e = row.map((v) => Math.exp(v - m))
      const su = e.reduce((acc, v) => acc + v, 0)
      return e.map((v) => v / su)
    })
    att.push(A)
    headOut.push(
      A.map((row) =>
        Array.from({ length: dh }, (_, k) => row.reduce((s, a, j) => s + a * V[j][o + k], 0)),
      ),
    )
  }
  const concat = Array.from({ length: T }, (_, i) =>
    Array.from({ length: d }, (_, j) => headOut[Math.floor(j / dh)][i][j % dh]),
  )
  const Z = concat.map((row) =>
    model.bo.map((b, j) => row.reduce((s, v, k) => s + v * model.Wo[k][j], b)),
  )
  const X1 = X.map((row, i) => row.map((v, j) => v + Z[i][j]))

  const l2 = X1.map((row) => layerNormRow(row, model.g2, model.b2ln))
  const ln2 = l2.map((o) => o.y)
  const Fh = ln2.map((row) =>
    model.b1.map((b, j) => Math.max(0, row.reduce((s, v, k) => s + v * model.W1[k][j], b))),
  )
  const F = Fh.map((row) =>
    model.b2.map((b, j) => row.reduce((s, v, k) => s + v * model.W2[k][j], b)),
  )
  const X2 = X1.map((row, i) => row.map((v, j) => v + F[i][j]))

  const l3 = layerNormRow(X2[0], model.g3, model.b3ln)
  const clsLn = l3.y
  const logits = model.bh.map((b, c) => clsLn.reduce((s, v, k) => s + v * model.Wh[k][c], b))
  const m = Math.max(...logits)
  const e = logits.map((v) => Math.exp(v - m))
  const su = e.reduce((acc, v) => acc + v, 0)
  const probs = e.map((v) => v / su)
  return {
    patches,
    X,
    ln1,
    Q,
    K,
    V,
    att,
    Z,
    X1,
    ln2,
    Fh,
    F,
    X2,
    clsLn,
    logits,
    probs,
    pred: probs.indexOf(Math.max(...probs)),
  }
}

export interface ViTTask {
  model: ViTModel
  classCount: number
  makeSample: (cls: number, rng: Rng) => Tensor3
  finalLoss: number
  accuracy: number
}

const VIT_SCALE_CFG = {
  s: { n: 12, p: 4, d: 12, heads: 2, dff: 24, epochs: 30, perClass: 50 },
  m: { n: 12, p: 4, d: 16, heads: 4, dff: 32, epochs: 30, perClass: 50 },
  l: { n: 16, p: 4, d: 24, heads: 4, dff: 48, epochs: 18, perClass: 45 },
} as const

export function buildViTTask(
  patternSample: (n: number, cls: number, rng: Rng) => Tensor3,
  scale: 's' | 'm' | 'l' = 's',
): ViTTask {
  const rng = mulberry32(0x517 + scale.charCodeAt(0))
  const { n, p, d, heads, dff, epochs, perClass } = VIT_SCALE_CFG[scale]
  const grid = n / p
  const T = grid * grid + 1
  const classes = 4
  const pd = p * p
  const model: ViTModel = {
    n,
    p,
    grid,
    T,
    d,
    heads,
    dff,
    classes,
    Wp: randn(pd, d, Math.sqrt(2 / pd), rng),
    bp: zeros(d),
    cls: Array.from({ length: d }, () => gaussian(rng) * 0.3),
    pos: randn(T, d, 0.25, rng),
    g1: Array.from({ length: d }, () => 1),
    b1ln: zeros(d),
    Wq: randn(d, d, 0.3, rng),
    Wk: randn(d, d, 0.3, rng),
    Wv: randn(d, d, 0.3, rng),
    Wo: randn(d, d, 0.3, rng),
    bo: zeros(d),
    g2: Array.from({ length: d }, () => 1),
    b2ln: zeros(d),
    W1: randn(d, dff, Math.sqrt(2 / d), rng),
    b1: zeros(dff),
    W2: randn(dff, d, Math.sqrt(2 / dff), rng),
    b2: zeros(d),
    g3: Array.from({ length: d }, () => 1),
    b3ln: zeros(d),
    Wh: randn(d, classes, 0.3, rng),
    bh: zeros(classes),
  }

  const makeSample = (cls: number, r: Rng) => patternSample(n, cls, r)
  const data: { img: number[][]; y: number }[] = []
  for (let c = 0; c < classes; c++)
    for (let i = 0; i < perClass; i++) data.push({ img: makeSample(c, rng)[0], y: c })

  const lr0 = 0.05
  const order = data.map((_, i) => i)
  let loss = 0
  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    const lr = lr0 * (1 - (e / epochs) * 0.7)
    loss = 0
    for (const idx of order) {
      const { img, y } = data[idx]
      loss += trainStep(model, img, y, lr)
    }
    loss /= data.length
  }

  let ok = 0
  const evals = 80
  for (let i = 0; i < evals; i++) {
    const c = i % classes
    if (forwardViT(model, makeSample(c, rng)[0]).pred === c) ok++
  }
  return { model, classCount: classes, makeSample, finalLoss: loss, accuracy: ok / evals }
}

function trainStep(model: ViTModel, img: number[][], y: number, lr: number): number {
  const { d, heads, T, dff } = model
  const dh = d / heads
  const tr = forwardViT(model, img)
  const loss = -Math.log(Math.max(tr.probs[y], 1e-9))

  // ---- head
  const dLogits = tr.probs.map((pv, c) => pv - (c === y ? 1 : 0))
  const dClsLn = Array.from({ length: d }, (_, k) =>
    dLogits.reduce((s, v, c) => s + v * model.Wh[k][c], 0),
  )
  const gWh = model.Wh.map((row, k) => row.map((_, c) => tr.clsLn[k] * dLogits[c]))
  // final LN backward (CLS row of X2)
  {
    const mu = tr.X2[0].reduce((s, v) => s + v, 0) / d
    const sd = Math.sqrt(tr.X2[0].reduce((s, v) => s + (v - mu) ** 2, 0) / d + 1e-5)
    var ln3 = layerNormBack(tr.X2[0], model.g3, mu, sd, dClsLn)
  }
  const dX2: number[][] = Array.from({ length: T }, (_, i) => (i === 0 ? ln3.dx : zeros(d)))

  // ---- FFN residual: X2 = X1 + F(LN2(X1))
  const dF = dX2.map((row) => [...row])
  const dFh = dF.map((row, i) =>
    Array.from({ length: dff }, (_, j) =>
      tr.Fh[i][j] > 0 ? row.reduce((s, v, k) => s + v * model.W2[j][k], 0) : 0,
    ),
  )
  const gW2 = Array.from({ length: dff }, (_, j) =>
    Array.from({ length: d }, (_, k) => tr.Fh.reduce((s, row, i) => s + row[j] * dF[i][k], 0)),
  )
  const gB2 = Array.from({ length: d }, (_, k) => dF.reduce((s, row) => s + row[k], 0))
  const dLn2 = dFh.map((row) =>
    Array.from({ length: d }, (_, k) => row.reduce((s, v, j) => s + v * model.W1[k][j], 0)),
  )
  const gW1 = Array.from({ length: d }, (_, k) =>
    Array.from({ length: dff }, (_, j) => tr.ln2.reduce((s, row, i) => s + row[k] * dFh[i][j], 0)),
  )
  const gB1 = Array.from({ length: dff }, (_, j) => dFh.reduce((s, row) => s + row[j], 0))
  const gG2 = zeros(d)
  const gBl2 = zeros(d)
  const dX1 = dX2.map((row) => [...row]) // residual path
  for (let i = 0; i < T; i++) {
    const x = tr.X1[i]
    const mu = x.reduce((s, v) => s + v, 0) / d
    const sd = Math.sqrt(x.reduce((s, v) => s + (v - mu) ** 2, 0) / d + 1e-5)
    const b = layerNormBack(x, model.g2, mu, sd, dLn2[i])
    for (let k = 0; k < d; k++) {
      dX1[i][k] += b.dx[k]
      gG2[k] += b.dg[k]
      gBl2[k] += b.db[k]
    }
  }

  // ---- attention residual: X1 = X + Z, Z = concat(heads)·Wo + bo
  const dZ = dX1.map((row) => [...row])
  const gWo = Array.from({ length: d }, () => zeros(d))
  const gBo = zeros(d)
  const dConcat = Array.from({ length: T }, () => zeros(d))
  {
    // concat[i][k], Z[i][j] = Σ_k concat·Wo[k][j]
    const concat = Array.from({ length: T }, (_, i) =>
      Array.from({ length: d }, (_, j) => {
        const h = Math.floor(j / dh)
        const o = h * dh
        return tr.att[h][i].reduce((s, a, jj) => s + a * tr.V[jj][o + (j % dh)], 0)
      }),
    )
    for (let i = 0; i < T; i++)
      for (let j = 0; j < d; j++) {
        const g = dZ[i][j]
        gBo[j] += g
        for (let k = 0; k < d; k++) {
          gWo[k][j] += concat[i][k] * g
          dConcat[i][k] += model.Wo[k][j] * g
        }
      }
  }
  const dQ = Array.from({ length: T }, () => zeros(d))
  const dK = Array.from({ length: T }, () => zeros(d))
  const dV = Array.from({ length: T }, () => zeros(d))
  for (let h = 0; h < heads; h++) {
    const o = h * dh
    const A = tr.att[h]
    // headOut[i][k] = Σ_j A[i][j]·V[j][o+k]; dHead = slice of dConcat
    const dA = Array.from({ length: T }, () => zeros(T))
    for (let i = 0; i < T; i++)
      for (let k = 0; k < dh; k++) {
        const g = dConcat[i][o + k]
        for (let j = 0; j < T; j++) {
          dA[i][j] += g * tr.V[j][o + k]
          dV[j][o + k] += A[i][j] * g
        }
      }
    // softmax backward per row → dScores
    for (let i = 0; i < T; i++) {
      const dot = A[i].reduce((s, a, j) => s + a * dA[i][j], 0)
      for (let j = 0; j < T; j++) {
        const dS = (A[i][j] * (dA[i][j] - dot)) / Math.sqrt(dh)
        for (let k = 0; k < dh; k++) {
          dQ[i][o + k] += dS * tr.K[j][o + k]
          dK[j][o + k] += dS * tr.Q[i][o + k]
        }
      }
    }
  }
  // project back through Wq/Wk/Wv
  const gWq = Array.from({ length: d }, () => zeros(d))
  const gWk = Array.from({ length: d }, () => zeros(d))
  const gWv = Array.from({ length: d }, () => zeros(d))
  const dLn1 = Array.from({ length: T }, () => zeros(d))
  for (let i = 0; i < T; i++)
    for (let j = 0; j < d; j++) {
      for (let k = 0; k < d; k++) {
        gWq[k][j] += tr.ln1[i][k] * dQ[i][j]
        gWk[k][j] += tr.ln1[i][k] * dK[i][j]
        gWv[k][j] += tr.ln1[i][k] * dV[i][j]
        dLn1[i][k] += model.Wq[k][j] * dQ[i][j] + model.Wk[k][j] * dK[i][j] + model.Wv[k][j] * dV[i][j]
      }
    }
  const gG1 = zeros(d)
  const gBl1 = zeros(d)
  const dX = dX1.map((row) => [...row]) // residual path
  for (let i = 0; i < T; i++) {
    const x = tr.X[i]
    const mu = x.reduce((s, v) => s + v, 0) / d
    const sd = Math.sqrt(x.reduce((s, v) => s + (v - mu) ** 2, 0) / d + 1e-5)
    const b = layerNormBack(x, model.g1, mu, sd, dLn1[i])
    for (let k = 0; k < d; k++) {
      dX[i][k] += b.dx[k]
      gG1[k] += b.dg[k]
      gBl1[k] += b.db[k]
    }
  }

  // ---- embeddings: X[0] = cls + pos[0]; X[i+1] = patch·Wp + bp + pos[i+1]
  const gWp = Array.from({ length: model.p * model.p }, () => zeros(d))
  const gBp = zeros(d)
  for (let t = 1; t < T; t++) {
    const patch = tr.patches[t - 1]
    for (let j = 0; j < d; j++) {
      const g = dX[t][j]
      gBp[j] += g
      for (let k = 0; k < patch.length; k++) gWp[k][j] += patch[k] * g
    }
  }

  // ---- SGD updates
  const upd2 = (W: number[][], G: number[][]) => {
    for (let a = 0; a < W.length; a++)
      for (let b2i = 0; b2i < W[a].length; b2i++) W[a][b2i] -= lr * G[a][b2i]
  }
  const upd1 = (w: number[], g: number[]) => {
    for (let a = 0; a < w.length; a++) w[a] -= lr * g[a]
  }
  upd2(model.Wh, gWh)
  upd1(model.bh, dLogits)
  upd1(model.g3, ln3.dg)
  upd1(model.b3ln, ln3.db)
  upd2(model.W1, gW1)
  upd1(model.b1, gB1)
  upd2(model.W2, gW2)
  upd1(model.b2, gB2)
  upd1(model.g2, gG2)
  upd1(model.b2ln, gBl2)
  upd2(model.Wo, gWo)
  upd1(model.bo, gBo)
  upd2(model.Wq, gWq)
  upd2(model.Wk, gWk)
  upd2(model.Wv, gWv)
  upd1(model.g1, gG1)
  upd1(model.b1ln, gBl1)
  upd2(model.Wp, gWp)
  upd1(model.bp, gBp)
  for (let j = 0; j < d; j++) model.cls[j] -= lr * dX[0][j]
  for (let t = 0; t < T; t++) for (let j = 0; j < d; j++) model.pos[t][j] -= lr * dX[t][j]

  return loss
}
