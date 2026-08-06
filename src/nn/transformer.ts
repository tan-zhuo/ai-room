// A tiny character-level transformer (1 head, 1 block, no layernorm) with a
// hand-written forward pass AND backprop — trained for real in the browser.
// Every intermediate (embeddings, Q/K/V, attention scores & weights, weighted
// sums, FFN, logits) is kept in the trace so the UI can show the true math.

import { Rng, gaussian, mulberry32, shuffleInPlace } from './rng'

export interface LLMModel {
  vocab: string[]
  d: number
  h: number
  /** max sequence length */
  T: number
  /** token embeddings [vocab][d] */
  E: number[][]
  /** positional embeddings [T][d] */
  P: number[][]
  /** projections [d][d], layout [in][out] */
  Wq: number[][]
  Wk: number[][]
  Wv: number[][]
  /** FFN [d][h], [h] */
  W1: number[][]
  b1: number[]
  /** output head [h][vocab], [vocab] */
  W2: number[][]
  b2: number[]
}

export interface LLMTrace {
  ids: number[]
  chars: string[]
  /** X = E[token] + P[pos], [T][d] */
  X: number[][]
  Q: number[][]
  K: number[][]
  V: number[][]
  /** raw scaled scores q·k/√d, [T][T] (upper triangle is masked out) */
  S: number[][]
  /** row-softmax of masked S */
  A: number[][]
  /** Z = A·V */
  Z: number[][]
  Hpre: number[][]
  H: number[][]
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

/** out[i][j] = Σ_k a[i][k] b[k][j] */
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
  const { d } = model
  const T = ids.length
  const X = ids.map((id, i) => model.E[id].map((v, k) => v + model.P[i][k]))
  const Q = matmul(X, model.Wq)
  const K = matmul(X, model.Wk)
  const V = matmul(X, model.Wv)
  const scale = 1 / Math.sqrt(d)
  const S = zeros(T, T)
  const A = zeros(T, T)
  for (let i = 0; i < T; i++) {
    const masked: number[] = []
    for (let j = 0; j < T; j++) {
      let s = 0
      for (let k = 0; k < d; k++) s += Q[i][k] * K[j][k]
      S[i][j] = s * scale
      if (j <= i) masked.push(S[i][j])
    }
    const soft = softmaxRow(masked)
    for (let j = 0; j <= i; j++) A[i][j] = soft[j]
  }
  const Z = matmul(A, V)
  const Hpre = matmul(Z, model.W1).map((row) => row.map((v, j) => v + model.b1[j]))
  const H = Hpre.map((row) => row.map((v) => Math.max(0, v)))
  const U = matmul(H, model.W2).map((row) => row.map((v, j) => v + model.b2[j]))
  return {
    ids,
    chars: ids.map((id) => model.vocab[id]),
    X,
    Q,
    K,
    V,
    S,
    A,
    Z,
    Hpre,
    H,
    U,
    probs: softmaxRow(U[T - 1]),
  }
}

// ---------------------------------------------------------------- training

function addScaled(target: number[][], grad: number[][], lr: number): void {
  for (let i = 0; i < target.length; i++)
    for (let j = 0; j < target[0].length; j++) target[i][j] -= lr * grad[i][j]
}

/** One SGD step on a single window; returns the mean cross-entropy loss. */
function trainStep(model: LLMModel, ids: number[], targets: number[], lr: number): number {
  const { d, h } = model
  const T = ids.length
  const tr = forwardLLM(model, ids)
  const vocabN = model.vocab.length

  // dU = (softmax(U) - onehot(target)) / T
  let loss = 0
  const dU = zeros(T, vocabN)
  for (let i = 0; i < T; i++) {
    const p = softmaxRow(tr.U[i])
    loss += -Math.log(Math.max(1e-9, p[targets[i]]))
    for (let j = 0; j < vocabN; j++) dU[i][j] = (p[j] - (j === targets[i] ? 1 : 0)) / T
  }
  loss /= T

  // output head
  const dW2 = zeros(h, vocabN)
  const db2 = Array.from({ length: vocabN }, () => 0)
  const dH = zeros(T, h)
  for (let i = 0; i < T; i++) {
    for (let j = 0; j < vocabN; j++) {
      const g = dU[i][j]
      if (g === 0) continue
      db2[j] += g
      for (let k = 0; k < h; k++) {
        dW2[k][j] += tr.H[i][k] * g
        dH[i][k] += g * model.W2[k][j]
      }
    }
  }

  // FFN
  const dPre = dH.map((row, i) => row.map((g, k) => (tr.Hpre[i][k] > 0 ? g : 0)))
  const dW1 = zeros(d, h)
  const db1 = Array.from({ length: h }, () => 0)
  const dZ = zeros(T, d)
  for (let i = 0; i < T; i++) {
    for (let k = 0; k < h; k++) {
      const g = dPre[i][k]
      if (g === 0) continue
      db1[k] += g
      for (let m = 0; m < d; m++) {
        dW1[m][k] += tr.Z[i][m] * g
        dZ[i][m] += g * model.W1[m][k]
      }
    }
  }

  // attention: Z = A·V
  const dA = zeros(T, T)
  const dV = zeros(T, d)
  for (let i = 0; i < T; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0
      for (let m = 0; m < d; m++) {
        s += dZ[i][m] * tr.V[j][m]
        dV[j][m] += tr.A[i][j] * dZ[i][m]
      }
      dA[i][j] = s
    }
  }
  // softmax backward per row: dS = A ⊙ (dA - Σ_j dA·A)
  const scale = 1 / Math.sqrt(d)
  const dQ = zeros(T, d)
  const dK = zeros(T, d)
  for (let i = 0; i < T; i++) {
    let dot = 0
    for (let j = 0; j <= i; j++) dot += dA[i][j] * tr.A[i][j]
    for (let j = 0; j <= i; j++) {
      const dS = tr.A[i][j] * (dA[i][j] - dot) * scale
      if (dS === 0) continue
      for (let k = 0; k < d; k++) {
        dQ[i][k] += dS * tr.K[j][k]
        dK[j][k] += dS * tr.Q[i][k]
      }
    }
  }

  // projections + embeddings
  const dX = zeros(T, d)
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

  addScaled(model.W2, dW2, lr)
  addScaled(model.W1, dW1, lr)
  addScaled(model.Wq, dWq, lr)
  addScaled(model.Wk, dWk, lr)
  addScaled(model.Wv, dWv, lr)
  for (let j = 0; j < vocabN; j++) model.b2[j] -= lr * db2[j]
  for (let k = 0; k < h; k++) model.b1[k] -= lr * db1[k]
  for (let i = 0; i < T; i++) {
    for (let m = 0; m < d; m++) {
      model.E[ids[i]][m] -= lr * dX[i][m]
      model.P[i][m] -= lr * dX[i][m]
    }
  }
  return loss
}

// ---------------------------------------------------------------- task

export const LLM_CORPUS =
  'hello world. the ai room. attention is all you need. neural networks learn deep patterns. ' +
  'we walk inside a living neural network. data flows through every layer. the model learns to ' +
  'read and the room comes alive. deep nets see the world. '

export interface LLMTask {
  model: LLMModel
  samples: string[]
  finalLoss: number
}

export function buildLLMTask(): LLMTask {
  const rng = mulberry32(0x11a1)
  const vocab = [...new Set([...LLM_CORPUS.toLowerCase()])].sort()
  const d = 10
  const h = 16
  const T = 8
  const model: LLMModel = {
    vocab,
    d,
    h,
    T,
    E: randn(vocab.length, d, 0.3, rng),
    P: randn(T, d, 0.15, rng),
    Wq: randn(d, d, 0.3, rng),
    Wk: randn(d, d, 0.3, rng),
    Wv: randn(d, d, 0.3, rng),
    W1: randn(d, h, 0.3, rng),
    b1: Array.from({ length: h }, () => 0),
    W2: randn(h, vocab.length, 0.3, rng),
    b2: Array.from({ length: vocab.length }, () => 0),
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
  const epochs = 40
  for (let epoch = 0; epoch < epochs; epoch++) {
    shuffleInPlace(order, rng)
    const lr = 0.12 * (1 - (epoch / epochs) * 0.7)
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
