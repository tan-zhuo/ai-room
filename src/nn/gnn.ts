// A graph convolutional network (GCN, Kipf & Welling 2017) doing node
// classification on small community graphs. Each layer is one round of
// message passing: H' = σ(Â H W) with Â = D^-1/2 (A+I) D^-1/2 — every node
// averages its neighbours' features (renormalized) before the linear map.

import { Rng, gaussian, mulberry32 } from './rng'

export interface Graph {
  n: number
  /** community id per node */
  labels: number[]
  /** adjacency list (undirected, no self loops) */
  edges: [number, number][]
  /** normalized adjacency Â incl. self loops, [n][n] */
  ahat: number[][]
  /** node features [n][d] */
  X: number[][]
}

export interface GNNModel {
  /** feature width */
  d: number
  /** hidden width */
  h: number
  classes: number
  /** class prototypes used to sample node features (fixed across graphs) */
  prototypes: number[][]
  /** layer 1: d → h */
  W1: number[][]
  b1: number[]
  /** layer 2: h → classes */
  W2: number[][]
  b2: number[]
  /** nodes per community at the current scale */
  perClass: number
}

export interface GNNTrace {
  graph: Graph
  /** neighbour-aggregated inputs Â·X [n][d] */
  agg1: number[][]
  /** hidden pre-activation [n][h] */
  z1: number[][]
  /** hidden after ReLU [n][h] */
  H1: number[][]
  /** second aggregation Â·H1 [n][h] */
  agg2: number[][]
  logits: number[][]
  /** per-node class probabilities [n][classes] */
  probs: number[][]
  /** argmax per node */
  pred: number[]
  /** fraction of nodes classified into their true community */
  acc: number
}

function matmul(A: number[][], B: number[][]): number[][] {
  return A.map((row) =>
    B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)),
  )
}

function softmaxRow(row: number[]): number[] {
  const m = Math.max(...row)
  const e = row.map((v) => Math.exp(v - m))
  const s = e.reduce((acc, v) => acc + v, 0)
  return e.map((v) => v / s)
}

/** Random community graph: dense edges inside communities, sparse across. */
export function sampleGraph(model: GNNModel, rng: Rng): Graph {
  const k = model.classes
  const per = model.perClass
  const n = k * per
  const labels = Array.from({ length: n }, (_, i) => Math.floor(i / per))
  const edges: [number, number][] = []
  const A = Array.from({ length: n }, () => Array.from({ length: n }, () => 0))
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      const p = labels[i] === labels[j] ? 0.7 : 0.06
      if (rng() < p) {
        edges.push([i, j])
        A[i][j] = A[j][i] = 1
      }
    }
  // guarantee no isolated node: connect to a same-community peer
  for (let i = 0; i < n; i++) {
    if (A[i].every((v) => v === 0)) {
      const c = labels[i]
      let j = c * per + Math.floor(rng() * per)
      if (j === i) j = c * per + ((i - c * per + 1) % per)
      edges.push([Math.min(i, j), Math.max(i, j)])
      A[i][j] = A[j][i] = 1
    }
  }
  // Â = D^-1/2 (A + I) D^-1/2
  const deg = A.map((row) => row.reduce((s, v) => s + v, 0) + 1)
  const ahat = A.map((row, i) =>
    row.map((v, j) => {
      const a = v + (i === j ? 1 : 0)
      return a / Math.sqrt(deg[i] * deg[j])
    }),
  )
  // features: community prototype + noise
  const X = labels.map((c) =>
    model.prototypes[c].map((v) => v + gaussian(rng) * 0.55),
  )
  return { n, labels, edges, ahat, X }
}

export function forwardGNN(model: GNNModel, graph: Graph): GNNTrace {
  const agg1 = matmul(graph.ahat, graph.X)
  const z1 = matmul(agg1, model.W1).map((row) => row.map((v, j) => v + model.b1[j]))
  const H1 = z1.map((row) => row.map((v) => Math.max(0, v)))
  const agg2 = matmul(graph.ahat, H1)
  const logits = matmul(agg2, model.W2).map((row) => row.map((v, j) => v + model.b2[j]))
  const probs = logits.map(softmaxRow)
  const pred = probs.map((row) => row.indexOf(Math.max(...row)))
  const acc = pred.filter((p, i) => p === graph.labels[i]).length / graph.n
  return { graph, agg1, z1, H1, agg2, logits, probs, pred, acc }
}

export interface GNNTask {
  model: GNNModel
  /** mean node accuracy over held-out sampled graphs */
  accuracy: number
  finalLoss: number
}

const GNN_SCALE_CFG = {
  s: { perClass: 4, d: 6, h: 10, epochs: 60 },
  m: { perClass: 5, d: 8, h: 14, epochs: 55 },
  l: { perClass: 6, d: 10, h: 20, epochs: 50 },
} as const

export function buildGNNTask(scale: 's' | 'm' | 'l' = 's'): GNNTask {
  const rng = mulberry32(0x64e0 + scale.charCodeAt(0))
  const { perClass, d, h, epochs } = GNN_SCALE_CFG[scale]
  const classes = 3
  const std = (fanIn: number) => Math.sqrt(2 / fanIn)
  const model: GNNModel = {
    d,
    h,
    classes,
    perClass,
    prototypes: Array.from({ length: classes }, () =>
      Array.from({ length: d }, () => gaussian(rng) * 0.8),
    ),
    W1: Array.from({ length: d }, () => Array.from({ length: h }, () => gaussian(rng) * std(d))),
    b1: Array.from({ length: h }, () => 0),
    W2: Array.from({ length: h }, () => Array.from({ length: classes }, () => gaussian(rng) * std(h))),
    b2: Array.from({ length: classes }, () => 0),
  }

  const lr = 0.08
  let loss = 0
  for (let e = 0; e < epochs; e++) {
    // a fresh random graph every epoch — the GCN must learn the mechanism,
    // not one particular graph
    const g = sampleGraph(model, rng)
    const tr = forwardGNN(model, g)
    loss = 0
    // dLogits = probs - onehot (mean over nodes)
    const dLogits = tr.probs.map((row, i) =>
      row.map((p, c) => (p - (c === g.labels[i] ? 1 : 0)) / g.n),
    )
    for (let i = 0; i < g.n; i++) loss -= Math.log(Math.max(tr.probs[i][g.labels[i]], 1e-9)) / g.n

    // backprop through logits = Â·H1·W2 + b2: dAgg2 = dLogits · W2^T
    const dAgg2 = dLogits.map((row) =>
      Array.from({ length: model.h }, (_, j) =>
        row.reduce((s, v, c) => s + v * model.W2[j][c], 0),
      ),
    )
    // grads for W2: agg2^T · dLogits
    const gW2 = Array.from({ length: model.h }, (_, j) =>
      Array.from({ length: classes }, (_, c) =>
        tr.agg2.reduce((s, row, i) => s + row[j] * dLogits[i][c], 0),
      ),
    )
    const gB2 = Array.from({ length: classes }, (_, c) =>
      dLogits.reduce((s, row) => s + row[c], 0),
    )
    // dH1 = Â^T · dAgg2 (Â symmetric) then ReLU mask
    const dH1 = matmul(g.ahat, dAgg2).map((row, i) => row.map((v, j) => (tr.z1[i][j] > 0 ? v : 0)))
    // grads for W1: agg1^T · dH1
    const gW1 = Array.from({ length: model.d }, (_, j) =>
      Array.from({ length: model.h }, (_, k) =>
        tr.agg1.reduce((s, row, i) => s + row[j] * dH1[i][k], 0),
      ),
    )
    const gB1 = Array.from({ length: model.h }, (_, k) => dH1.reduce((s, row) => s + row[k], 0))

    for (let j = 0; j < model.d; j++)
      for (let k = 0; k < model.h; k++) model.W1[j][k] -= lr * gW1[j][k]
    for (let k = 0; k < model.h; k++) model.b1[k] -= lr * gB1[k]
    for (let j = 0; j < model.h; j++)
      for (let c = 0; c < classes; c++) model.W2[j][c] -= lr * gW2[j][c]
    for (let c = 0; c < classes; c++) model.b2[c] -= lr * gB2[c]
  }

  // held-out evaluation on fresh graphs
  let acc = 0
  const evals = 20
  for (let i = 0; i < evals; i++) acc += forwardGNN(model, sampleGraph(model, rng)).acc
  return { model, accuracy: acc / evals, finalLoss: loss }
}
