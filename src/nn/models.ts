// Builds the two demo networks. Both are genuinely trained at load time
// (a few tens of milliseconds) so the predictions you watch are real.

import { Rng, mulberry32, gaussian, clamp01 } from './rng'
import { MLPModel, createMLP, trainMLP, TrainSample } from './mlp'
import { CNNModel, ConvLayerDef, Tensor3, convForward, poolForward, flattenTensor } from './cnn'

export const MLP_CLASS_COUNT = 3
export const CNN_CLASS_COUNT = 4

export interface MLPTask {
  model: MLPModel
  classCount: number
  makeSample: (cls: number, rng: Rng) => number[]
}

export interface CNNTask {
  model: CNNModel
  classCount: number
  makeSample: (cls: number, rng: Rng) => Tensor3
}

export interface Models {
  mlp: MLPTask
  cnn: CNNTask
}

// ---------------------------------------------------------------- MLP task
// Three Gaussian clusters in a 4-dimensional feature space.

const MLP_MEANS: number[][] = [
  [0.8, 0.2, -0.6, 0.4],
  [-0.7, 0.5, 0.6, -0.3],
  [0.1, -0.8, 0.2, 0.7],
]

function mlpSample(cls: number, rng: Rng): number[] {
  return MLP_MEANS[cls].map((m) => m + gaussian(rng) * 0.25)
}

function oneHot(n: number, i: number): number[] {
  return Array.from({ length: n }, (_, k) => (k === i ? 1 : 0))
}

// ---------------------------------------------------------------- CNN task
// 8x8 grayscale patterns: vertical bar / horizontal bar / diagonal / ring.
// Conv kernels are classic hand-crafted edge detectors, so each feature map
// has a real interpretation; the dense head is trained on top of them.

function blankImage(rng: Rng): number[][] {
  return Array.from({ length: 8 }, () =>
    Array.from({ length: 8 }, () => clamp01(0.06 + gaussian(rng) * 0.04)),
  )
}

function cnnSample(cls: number, rng: Rng): Tensor3 {
  const img = blankImage(rng)
  const bright = () => clamp01(0.85 + gaussian(rng) * 0.08)
  if (cls === 0) {
    // vertical bar, 2 px wide
    const c0 = 1 + Math.floor(rng() * 5)
    for (let r = 0; r < 8; r++) {
      img[r][c0] = bright()
      img[r][c0 + 1] = bright()
    }
  } else if (cls === 1) {
    // horizontal bar
    const r0 = 1 + Math.floor(rng() * 5)
    for (let c = 0; c < 8; c++) {
      img[r0][c] = bright()
      img[r0 + 1][c] = bright()
    }
  } else if (cls === 2) {
    // diagonal stroke
    const o = Math.floor(rng() * 3) - 1
    for (let r = 0; r < 8; r++) {
      const c = r + o
      if (c >= 0 && c < 8) img[r][c] = bright()
      if (c + 1 >= 0 && c + 1 < 8) img[r][c + 1] = bright()
    }
  } else {
    // ring (square outline)
    const y0 = 1 + Math.floor(rng() * 2)
    const x0 = 1 + Math.floor(rng() * 2)
    for (let d = 0; d < 5; d++) {
      img[y0][x0 + d] = bright()
      img[y0 + 4][x0 + d] = bright()
      img[y0 + d][x0] = bright()
      img[y0 + d][x0 + 4] = bright()
    }
  }
  return [img]
}

function scaleKernel(k: number[][], s: number): number[][] {
  return k.map((row) => row.map((v) => v * s))
}

function buildConvLayer(): ConvLayerDef {
  // Sobel-x, Sobel-y and a diagonal detector, scaled to keep values readable.
  const kernels = [
    scaleKernel([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 0.25),
    scaleKernel([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 0.25),
    scaleKernel([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], 0.25),
  ]
  return { type: 'conv', kernels: kernels.map((k) => [k]), biases: [0, 0, 0] }
}

// ---------------------------------------------------------------- build + train

export function buildModels(): Models {
  const rng = mulberry32(20260806)

  // --- MLP: 4 -> 6 -> 5 -> 3
  const mlpModel = createMLP(rng, [4, 6, 5, 3], ['relu', 'relu', 'softmax'])
  const mlpData: TrainSample[] = []
  for (let c = 0; c < MLP_CLASS_COUNT; c++) {
    for (let n = 0; n < 50; n++) mlpData.push({ x: mlpSample(c, rng), y: oneHot(MLP_CLASS_COUNT, c) })
  }
  trainMLP(mlpModel, mlpData, { lr: 0.05, epochs: 200, rng })

  // --- CNN: 1x8x8 -> conv3x3(3) -> pool2 -> flatten(27) -> 10 -> 4
  const conv = buildConvLayer()
  const featuresOf = (input: Tensor3) =>
    flattenTensor(poolForward(convForward(input, conv).out, 2))

  const head = createMLP(rng, [27, 10, 4], ['relu', 'softmax'])
  const headData: TrainSample[] = []
  for (let c = 0; c < CNN_CLASS_COUNT; c++) {
    for (let n = 0; n < 40; n++) {
      headData.push({ x: featuresOf(cnnSample(c, rng)), y: oneHot(CNN_CLASS_COUNT, c) })
    }
  }
  trainMLP(head, headData, { lr: 0.05, epochs: 140, rng })

  const cnnModel: CNNModel = {
    inputShape: [1, 8, 8],
    layers: [
      conv,
      { type: 'pool', size: 2 },
      { type: 'flatten' },
      { type: 'dense', layer: head.layers[0] },
      { type: 'dense', layer: head.layers[1] },
    ],
  }

  return {
    mlp: { model: mlpModel, classCount: MLP_CLASS_COUNT, makeSample: mlpSample },
    cnn: { model: cnnModel, classCount: CNN_CLASS_COUNT, makeSample: cnnSample },
  }
}

export const MODELS: Models = buildModels()
