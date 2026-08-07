// Recurrent architectures: a vanilla Elman RNN and an LSTM, both char-level
// next-character predictors trained on the same corpus as the transformer —
// with hand-written backpropagation through time (BPTT).

import { Rng, gaussian, mulberry32, shuffleInPlace } from './rng'
import { LLM_CORPUS } from './transformer'

function zeros(r: number, c: number): number[][] {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => 0))
}

function randn(r: number, c: number, std: number, rng: Rng): number[][] {
  return Array.from({ length: r }, () => Array.from({ length: c }, () => gaussian(rng) * std))
}

function softmaxRow(row: number[]): number[] {
  const m = Math.max(...row)
  const e = row.map((v) => Math.exp(v - m))
  const s = e.reduce((acc, v) => acc + v, 0)
  return e.map((v) => v / s)
}

export function corpusWindows(vocab: string[], T: number): { ids: number[]; targets: number[] }[] {
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
  return windows
}

export function buildVocab(): string[] {
  return [...new Set([...LLM_CORPUS.toLowerCase()])].sort()
}

export function encodeSeq(vocab: string[], T: number, text: string): number[] {
  const space = vocab.indexOf(' ')
  const ids = [...text.toLowerCase()]
    .map((ch) => {
      const i = vocab.indexOf(ch)
      return i >= 0 ? i : space
    })
    .slice(-T)
  while (ids.length < T) ids.unshift(space)
  return ids
}

// ================================================================= RNN

export interface RNNModel {
  vocab: string[]
  d: number
  h: number
  T: number
  E: number[][]
  /** [d][h] */
  Wx: number[][]
  /** [h][h] */
  Wh: number[][]
  bh: number[]
  /** [h][vocab] */
  Wy: number[][]
  by: number[]
}

export interface RNNTrace {
  ids: number[]
  chars: string[]
  /** embeddings [T][d] */
  X: number[][]
  /** pre-tanh [T][h] */
  Hpre: number[][]
  /** hidden states [T][h] */
  H: number[][]
  /** last-step logits */
  U: number[]
  probs: number[]
}

export function forwardRNN(model: RNNModel, ids: number[]): RNNTrace {
  const { d, h } = model
  const T = ids.length
  const X = ids.map((id) => [...model.E[id]])
  const Hpre = zeros(T, h)
  const H = zeros(T, h)
  let prev = Array.from({ length: h }, () => 0)
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < h; j++) {
      let s = model.bh[j]
      for (let i = 0; i < d; i++) s += X[t][i] * model.Wx[i][j]
      for (let i = 0; i < h; i++) s += prev[i] * model.Wh[i][j]
      Hpre[t][j] = s
      H[t][j] = Math.tanh(s)
    }
    prev = H[t]
  }
  const U = model.by.map((b, v) => model.Wy.reduce((s, row, j) => s + row[v] * H[T - 1][j], b))
  return { ids, chars: ids.map((i) => model.vocab[i]), X, Hpre, H, U, probs: softmaxRow(U) }
}

function rnnStep(model: RNNModel, ids: number[], targets: number[], lr: number): number {
  const { d, h } = model
  const T = ids.length
  const vocabN = model.vocab.length
  const tr = forwardRNN(model, ids)

  // per-step logits for training signal at every position
  const logits = tr.H.map((ht) =>
    model.by.map((b, v) => model.Wy.reduce((s, row, j) => s + row[v] * ht[j], b)),
  )
  let loss = 0
  const dU = zeros(T, vocabN)
  for (let t = 0; t < T; t++) {
    const p = softmaxRow(logits[t])
    loss += -Math.log(Math.max(1e-9, p[targets[t]]))
    for (let v = 0; v < vocabN; v++) dU[t][v] = (p[v] - (v === targets[t] ? 1 : 0)) / T
  }
  loss /= T

  const dWy = zeros(h, vocabN)
  const dby = Array.from({ length: vocabN }, () => 0)
  const dWx = zeros(d, h)
  const dWh = zeros(h, h)
  const dbh = Array.from({ length: h }, () => 0)
  let dhNext = Array.from({ length: h }, () => 0)

  for (let t = T - 1; t >= 0; t--) {
    const dh = [...dhNext]
    for (let v = 0; v < vocabN; v++) {
      const g = dU[t][v]
      if (g === 0) continue
      dby[v] += g
      for (let j = 0; j < h; j++) {
        dWy[j][v] += tr.H[t][j] * g
        dh[j] += g * model.Wy[j][v]
      }
    }
    const dpre = dh.map((g, j) => g * (1 - tr.H[t][j] ** 2))
    dhNext = Array.from({ length: h }, () => 0)
    for (let j = 0; j < h; j++) {
      const g = dpre[j]
      if (g === 0) continue
      dbh[j] += g
      for (let i = 0; i < d; i++) {
        dWx[i][j] += tr.X[t][i] * g
        model.E[ids[t]][i] -= lr * g * model.Wx[i][j]
      }
      if (t > 0) {
        for (let i = 0; i < h; i++) {
          dWh[i][j] += tr.H[t - 1][i] * g
          dhNext[i] += g * model.Wh[i][j]
        }
      }
    }
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < h; j++) model.Wx[i][j] -= lr * dWx[i][j]
  for (let i = 0; i < h; i++) for (let j = 0; j < h; j++) model.Wh[i][j] -= lr * dWh[i][j]
  for (let j = 0; j < h; j++) model.bh[j] -= lr * dbh[j]
  for (let j = 0; j < h; j++) for (let v = 0; v < vocabN; v++) model.Wy[j][v] -= lr * dWy[j][v]
  for (let v = 0; v < vocabN; v++) model.by[v] -= lr * dby[v]
  return loss
}

export interface RNNTask {
  model: RNNModel
  samples: string[]
  finalLoss: number
}

const RNN_SCALE_CFG = {
  s: { d: 10, h: 14, T: 8, epochs: 20 },
  m: { d: 14, h: 22, T: 10, epochs: 24 },
  l: { d: 16, h: 32, T: 12, epochs: 26 },
} as const

export function buildRNNTask(scale: 's' | 'm' | 'l' = 's'): RNNTask {
  const rng = mulberry32(0x4e4e + scale.charCodeAt(0))
  const vocab = buildVocab()
  const { d, h, T, epochs } = RNN_SCALE_CFG[scale]
  const model: RNNModel = {
    vocab,
    d,
    h,
    T,
    E: randn(vocab.length, d, 0.3, rng),
    Wx: randn(d, h, 0.3, rng),
    Wh: randn(h, h, 0.25, rng),
    bh: Array.from({ length: h }, () => 0),
    Wy: randn(h, vocab.length, 0.3, rng),
    by: Array.from({ length: vocab.length }, () => 0),
  }
  const windows = corpusWindows(vocab, T)
  const order = windows.map((_, i) => i)
  let loss = 0
  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    const lr = 0.1 * (1 - (e / epochs) * 0.6)
    loss = 0
    for (const i of order) loss += rnnStep(model, windows[i].ids, windows[i].targets, lr)
    loss /= windows.length
  }
  return { model, samples: ['hello wo', 'the ai r', 'attentio', 'deep net'], finalLoss: loss }
}

// ================================================================= LSTM

export interface LSTMModel {
  vocab: string[]
  d: number
  h: number
  T: number
  E: number[][]
  /** gate weights [d+h][h] over z = [x_t, h_{t-1}] */
  Wf: number[][]
  Wi: number[][]
  Wg: number[][]
  Wo: number[][]
  bf: number[]
  bi: number[]
  bg: number[]
  bo: number[]
  Wy: number[][]
  by: number[]
}

export interface LSTMTrace {
  ids: number[]
  chars: string[]
  X: number[][]
  /** gate activations, each [T][h] */
  F: number[][]
  I: number[][]
  G: number[][]
  O: number[][]
  /** cell state and its tanh, hidden state */
  C: number[][]
  Ct: number[][]
  H: number[][]
  U: number[]
  probs: number[]
}

const sig = (v: number) => 1 / (1 + Math.exp(-v))

export function forwardLSTM(model: LSTMModel, ids: number[]): LSTMTrace {
  const { d, h } = model
  const T = ids.length
  const X = ids.map((id) => [...model.E[id]])
  const F = zeros(T, h)
  const I = zeros(T, h)
  const G = zeros(T, h)
  const O = zeros(T, h)
  const C = zeros(T, h)
  const Ct = zeros(T, h)
  const H = zeros(T, h)
  let hPrev = Array.from({ length: h }, () => 0)
  let cPrev = Array.from({ length: h }, () => 0)
  for (let t = 0; t < T; t++) {
    const z = [...X[t], ...hPrev]
    for (let j = 0; j < h; j++) {
      let f = model.bf[j]
      let i = model.bi[j]
      let g = model.bg[j]
      let o = model.bo[j]
      for (let k = 0; k < d + h; k++) {
        const zv = z[k]
        f += zv * model.Wf[k][j]
        i += zv * model.Wi[k][j]
        g += zv * model.Wg[k][j]
        o += zv * model.Wo[k][j]
      }
      F[t][j] = sig(f)
      I[t][j] = sig(i)
      G[t][j] = Math.tanh(g)
      O[t][j] = sig(o)
      C[t][j] = F[t][j] * cPrev[j] + I[t][j] * G[t][j]
      Ct[t][j] = Math.tanh(C[t][j])
      H[t][j] = O[t][j] * Ct[t][j]
    }
    hPrev = H[t]
    cPrev = C[t]
  }
  const U = model.by.map((b, v) => model.Wy.reduce((s, row, j) => s + row[v] * H[T - 1][j], b))
  return { ids, chars: ids.map((i) => model.vocab[i]), X, F, I, G, O, C, Ct, H, U, probs: softmaxRow(U) }
}

function lstmStep(model: LSTMModel, ids: number[], targets: number[], lr: number): number {
  const { d, h } = model
  const D = d + h
  const T = ids.length
  const vocabN = model.vocab.length
  const tr = forwardLSTM(model, ids)

  const logits = tr.H.map((ht) =>
    model.by.map((b, v) => model.Wy.reduce((s, row, j) => s + row[v] * ht[j], b)),
  )
  let loss = 0
  const dU = zeros(T, vocabN)
  for (let t = 0; t < T; t++) {
    const p = softmaxRow(logits[t])
    loss += -Math.log(Math.max(1e-9, p[targets[t]]))
    for (let v = 0; v < vocabN; v++) dU[t][v] = (p[v] - (v === targets[t] ? 1 : 0)) / T
  }
  loss /= T

  const dWy = zeros(h, vocabN)
  const dby = Array.from({ length: vocabN }, () => 0)
  const dW = { f: zeros(D, h), i: zeros(D, h), g: zeros(D, h), o: zeros(D, h) }
  const db = {
    f: Array.from({ length: h }, () => 0),
    i: Array.from({ length: h }, () => 0),
    g: Array.from({ length: h }, () => 0),
    o: Array.from({ length: h }, () => 0),
  }
  let dhNext = Array.from({ length: h }, () => 0)
  let dcNext = Array.from({ length: h }, () => 0)

  for (let t = T - 1; t >= 0; t--) {
    const dh = [...dhNext]
    for (let v = 0; v < vocabN; v++) {
      const g = dU[t][v]
      if (g === 0) continue
      dby[v] += g
      for (let j = 0; j < h; j++) {
        dWy[j][v] += tr.H[t][j] * g
        dh[j] += g * model.Wy[j][v]
      }
    }
    const cPrev = t > 0 ? tr.C[t - 1] : Array.from({ length: h }, () => 0)
    const hPrev = t > 0 ? tr.H[t - 1] : Array.from({ length: h }, () => 0)
    const z = [...tr.X[t], ...hPrev]
    dhNext = Array.from({ length: h }, () => 0)
    const dcCur = Array.from({ length: h }, () => 0)
    for (let j = 0; j < h; j++) {
      const doRaw = dh[j] * tr.Ct[t][j]
      const dc = dcNext[j] + dh[j] * tr.O[t][j] * (1 - tr.Ct[t][j] ** 2)
      const dfRaw = dc * cPrev[j]
      const diRaw = dc * tr.G[t][j]
      const dgRaw = dc * tr.I[t][j]
      dcCur[j] = dc * tr.F[t][j]
      const dfp = dfRaw * tr.F[t][j] * (1 - tr.F[t][j])
      const dip = diRaw * tr.I[t][j] * (1 - tr.I[t][j])
      const dgp = dgRaw * (1 - tr.G[t][j] ** 2)
      const dop = doRaw * tr.O[t][j] * (1 - tr.O[t][j])
      db.f[j] += dfp
      db.i[j] += dip
      db.g[j] += dgp
      db.o[j] += dop
      for (let k = 0; k < D; k++) {
        const zv = z[k]
        dW.f[k][j] += zv * dfp
        dW.i[k][j] += zv * dip
        dW.g[k][j] += zv * dgp
        dW.o[k][j] += zv * dop
        const dz = dfp * model.Wf[k][j] + dip * model.Wi[k][j] + dgp * model.Wg[k][j] + dop * model.Wo[k][j]
        if (k < d) model.E[ids[t]][k] -= lr * dz
        else dhNext[k - d] += dz
      }
    }
    dcNext = dcCur
  }

  for (let k = 0; k < D; k++) {
    for (let j = 0; j < h; j++) {
      model.Wf[k][j] -= lr * dW.f[k][j]
      model.Wi[k][j] -= lr * dW.i[k][j]
      model.Wg[k][j] -= lr * dW.g[k][j]
      model.Wo[k][j] -= lr * dW.o[k][j]
    }
  }
  for (let j = 0; j < h; j++) {
    model.bf[j] -= lr * db.f[j]
    model.bi[j] -= lr * db.i[j]
    model.bg[j] -= lr * db.g[j]
    model.bo[j] -= lr * db.o[j]
  }
  for (let j = 0; j < h; j++) for (let v = 0; v < vocabN; v++) model.Wy[j][v] -= lr * dWy[j][v]
  for (let v = 0; v < vocabN; v++) model.by[v] -= lr * dby[v]
  return loss
}

export interface LSTMTask {
  model: LSTMModel
  samples: string[]
  finalLoss: number
}

const LSTM_SCALE_CFG = {
  s: { d: 10, h: 14, T: 8, epochs: 30 },
  m: { d: 14, h: 20, T: 10, epochs: 32 },
  l: { d: 16, h: 26, T: 12, epochs: 34 },
} as const

export function buildLSTMTask(scale: 's' | 'm' | 'l' = 's'): LSTMTask {
  const rng = mulberry32(0x157a + scale.charCodeAt(0))
  const vocab = buildVocab()
  const { d, h, T, epochs } = LSTM_SCALE_CFG[scale]
  const D = d + h
  const model: LSTMModel = {
    vocab,
    d,
    h,
    T,
    E: randn(vocab.length, d, 0.3, rng),
    Wf: randn(D, h, 0.25, rng),
    Wi: randn(D, h, 0.25, rng),
    Wg: randn(D, h, 0.25, rng),
    Wo: randn(D, h, 0.25, rng),
    bf: Array.from({ length: h }, () => 1), // forget-gate bias init: remember by default
    bi: Array.from({ length: h }, () => 0),
    bg: Array.from({ length: h }, () => 0),
    bo: Array.from({ length: h }, () => 0),
    Wy: randn(h, vocab.length, 0.3, rng),
    by: Array.from({ length: vocab.length }, () => 0),
  }
  const windows = corpusWindows(vocab, T)
  const order = windows.map((_, i) => i)
  let loss = 0
  for (let e = 0; e < epochs; e++) {
    shuffleInPlace(order, rng)
    const lr = 0.16 * (1 - (e / epochs) * 0.65)
    loss = 0
    for (const i of order) loss += lstmStep(model, windows[i].ids, windows[i].targets, lr)
    loss /= windows.length
  }
  return { model, samples: ['hello wo', 'the ai r', 'attentio', 'deep net'], finalLoss: loss }
}
