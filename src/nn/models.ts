// Builds the demo networks. All of them are genuinely trained at load time
// (tens of milliseconds to ~1s for the largest CNN) so the predictions you
// watch are real. MLP and CNN come in three scales (s/m/l) and can be
// rebuilt + retrained live.

import { Rng, mulberry32, gaussian, clamp01 } from './rng'
import { MLPModel, createMLP, trainMLP, TrainSample, forwardMLP, argmax } from './mlp'
import {
  CNNModel,
  CNNTrainSample,
  ConvLayerDef,
  Tensor3,
  convForward,
  flattenTensor,
  poolForward,
  trainCNNEndToEnd,
} from './cnn'
import { LLMTask, LLMVariant, buildLLMTask } from './transformer'
import { LSTMTask, RNNTask, buildLSTMTask, buildRNNTask } from './rnn'
import { trainMLPMSE } from './mlp'

export type KernelMode = 'hand' | 'learned'

export type Scale = 's' | 'm' | 'l'

export const MLP_CLASS_COUNT = 3
export const CNN_CLASS_COUNT = 4
export const TEXT_CLASS_COUNT = 3
export const TEXT_FEATURE_COUNT = 8

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

export interface TextTask {
  model: MLPModel
  classCount: number
  sampleText: (cls: number, rng: Rng) => string
}

export interface AETask {
  /** dense autoencoder over flattened n×n patterns */
  model: MLPModel
  n: number
  latent: number
  classCount: number
  makeSample: (cls: number, rng: Rng) => Tensor3
  finalMSE: number
}

export interface Models {
  mlp: MLPTask
  cnn: CNNTask
  text: TextTask
  llm: LLMTask
  rnn: RNNTask
  lstm: LSTMTask
  ae: AETask
}

function oneHot(n: number, i: number): number[] {
  return Array.from({ length: n }, (_, k) => (k === i ? 1 : 0))
}

// ---------------------------------------------------------------- MLP task
// Gaussian clusters in a d-dimensional feature space; d grows with scale.

const MLP_SIZES_BY_SCALE: Record<Scale, number[]> = {
  s: [4, 6, 5, 3],
  m: [6, 10, 8, 3],
  l: [10, 14, 10, 3],
}

export function buildMLPTask(scale: Scale): MLPTask {
  const rng = mulberry32(0x51ab + scale.charCodeAt(0))
  const sizes = MLP_SIZES_BY_SCALE[scale]
  const d = sizes[0]
  const means = Array.from({ length: MLP_CLASS_COUNT }, () =>
    Array.from({ length: d }, () => gaussian(rng) * 0.65),
  )
  const makeSample = (cls: number, r: Rng) => means[cls].map((m) => m + gaussian(r) * 0.25)

  const model = createMLP(rng, sizes, ['relu', 'relu', 'softmax'])
  const data: TrainSample[] = []
  for (let c = 0; c < MLP_CLASS_COUNT; c++) {
    for (let i = 0; i < 50; i++) data.push({ x: makeSample(c, rng), y: oneHot(MLP_CLASS_COUNT, c) })
  }
  trainMLP(model, data, { lr: 0.05, epochs: 200, rng })
  return { model, classCount: MLP_CLASS_COUNT, makeSample }
}

// ---------------------------------------------------------------- CNN task
// n×n grayscale patterns: vertical bar / horizontal bar / diagonal / ring.
// Conv kernels are classic hand-crafted detectors; the dense head is trained.

interface CNNScaleCfg {
  n: number
  kernels: number
  hidden: number
  epochs: number
  perClass: number
}

const CNN_SCALE_CFG: Record<Scale, CNNScaleCfg> = {
  s: { n: 8, kernels: 3, hidden: 10, epochs: 140, perClass: 40 },
  m: { n: 12, kernels: 4, hidden: 14, epochs: 110, perClass: 35 },
  l: { n: 16, kernels: 6, hidden: 20, epochs: 80, perClass: 30 },
}

function blankImage(n: number, rng: Rng): number[][] {
  return Array.from({ length: n }, () =>
    Array.from({ length: n }, () => clamp01(0.06 + gaussian(rng) * 0.04)),
  )
}

function cnnSampleOfSize(n: number, cls: number, rng: Rng): Tensor3 {
  const img = blankImage(n, rng)
  const bright = () => clamp01(0.85 + gaussian(rng) * 0.08)
  if (cls === 0) {
    const c0 = 1 + Math.floor(rng() * (n - 3))
    for (let r = 0; r < n; r++) {
      img[r][c0] = bright()
      img[r][c0 + 1] = bright()
    }
  } else if (cls === 1) {
    const r0 = 1 + Math.floor(rng() * (n - 3))
    for (let c = 0; c < n; c++) {
      img[r0][c] = bright()
      img[r0 + 1][c] = bright()
    }
  } else if (cls === 2) {
    const o = Math.floor(rng() * 3) - 1
    for (let r = 0; r < n; r++) {
      const c = r + o
      if (c >= 0 && c < n) img[r][c] = bright()
      if (c + 1 >= 0 && c + 1 < n) img[r][c + 1] = bright()
    }
  } else {
    const side = Math.round(n * 0.6)
    const range = Math.max(1, n - side - 2)
    const y0 = 1 + Math.floor(rng() * range)
    const x0 = 1 + Math.floor(rng() * range)
    for (let d = 0; d < side; d++) {
      img[y0][x0 + d] = bright()
      img[y0 + side - 1][x0 + d] = bright()
      img[y0 + d][x0] = bright()
      img[y0 + d][x0 + side - 1] = bright()
    }
  }
  return [img]
}

function scaleKernel(k: number[][], s: number): number[][] {
  return k.map((row) => row.map((v) => v * s))
}

// Pool of hand-crafted 3×3 detectors; a scale uses the first k of them.
const KERNEL_POOL: number[][][] = [
  scaleKernel([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 0.25), // vertical edges (Sobel-x)
  scaleKernel([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 0.25), // horizontal edges (Sobel-y)
  scaleKernel([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], 0.25), // diagonal ↘
  scaleKernel([[2, 1, 0], [1, 0, -1], [0, -1, -2]], 0.25), // diagonal ↗
  scaleKernel([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 0.3), // laplacian (spots/lines)
  scaleKernel([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 1 / 9), // box blur (brightness)
]

export function buildCNNTask(scale: Scale, kernelMode: KernelMode = 'hand'): CNNTask {
  const cfg = CNN_SCALE_CFG[scale]
  const rng = mulberry32(0xc44 + scale.charCodeAt(0) + (kernelMode === 'learned' ? 977 : 0))
  const learned = kernelMode === 'learned'
  const conv: ConvLayerDef = {
    type: 'conv',
    kernels: learned
      ? Array.from({ length: cfg.kernels }, () => [
          Array.from({ length: 3 }, () => Array.from({ length: 3 }, () => gaussian(rng) * 0.35)),
        ])
      : KERNEL_POOL.slice(0, cfg.kernels).map((k) => [k]),
    biases: Array.from({ length: cfg.kernels }, () => 0),
  }
  const makeSample = (cls: number, r: Rng) => cnnSampleOfSize(cfg.n, cls, r)

  const flatSize = cfg.kernels * Math.floor((cfg.n - 2) / 2) ** 2
  const head = createMLP(rng, [flatSize, cfg.hidden, CNN_CLASS_COUNT], ['relu', 'softmax'])
  const model: CNNModel = {
    inputShape: [1, cfg.n, cfg.n],
    layers: [
      conv,
      { type: 'pool', size: 2 },
      { type: 'flatten' },
      { type: 'dense', layer: head.layers[0] },
      { type: 'dense', layer: head.layers[1] },
    ],
  }

  if (learned) {
    // kernels start as noise and are trained end-to-end through conv backprop
    const data: CNNTrainSample[] = []
    for (let c = 0; c < CNN_CLASS_COUNT; c++) {
      for (let i = 0; i < cfg.perClass; i++) {
        data.push({ x: makeSample(c, rng), y: oneHot(CNN_CLASS_COUNT, c) })
      }
    }
    trainCNNEndToEnd(model, data, { lr: 0.025, epochs: Math.min(50, cfg.epochs), rng })
  } else {
    // classic hand-crafted detectors; only the dense head is trained
    const featuresOf = (input: Tensor3) => flattenTensor(poolForward(convForward(input, conv).out, 2))
    const data: TrainSample[] = []
    for (let c = 0; c < CNN_CLASS_COUNT; c++) {
      for (let i = 0; i < cfg.perClass; i++) {
        data.push({ x: featuresOf(makeSample(c, rng)), y: oneHot(CNN_CLASS_COUNT, c) })
      }
    }
    trainMLP(head, data, { lr: 0.05, epochs: cfg.epochs, rng })
  }

  return { model, classCount: CNN_CLASS_COUNT, makeSample }
}

// ---------------------------------------------------------------- Text task
// Type any text → 8 interpretable statistics → MLP detects the language.

export function extractTextFeatures(text: string): number[] {
  const chars = [...text]
  const nonSpace = chars.filter((c) => !/\s/.test(c))
  const n = nonSpace.length || 1
  const count = (re: RegExp) => nonSpace.filter((c) => re.test(c)).length
  const latin = count(/[a-zA-Z]/)
  const cjk = count(/[一-鿿]/)
  const kana = count(/[぀-ヿ]/)
  const digit = count(/[0-9０-９]/)
  const punct = count(/[.,!?;:'"()[\]\-–—。、，！？：；「」『』（）．・…]/)
  const spaces = chars.filter((c) => c === ' ').length
  const words = text.split(/[^a-zA-Z]+/).filter(Boolean)
  const avgWord = words.length ? words.reduce((s, w) => s + w.length, 0) / words.length : 0
  const vowels = count(/[aeiouAEIOU]/)
  return [
    latin / n,
    cjk / n,
    kana / n,
    digit / n,
    punct / n,
    Math.min(1, (spaces / Math.max(1, chars.length)) * 4),
    Math.min(1, avgWord / 8),
    latin ? vowels / latin : 0,
  ]
}

const EN_WORDS =
  'the quick brown fox neural network learns deep patterns hello world data model light room space mind time good we explore computation inside every signal flows'.split(' ')
const ZH_TEXT = '神经网络学习模式你好世界时间数据模型光空间思维今天天气很好我们一起探索计算过程走进内部每个信号在流动'
const JA_KANA = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわん'
const JA_KATA = ['ニューラル', 'ネットワーク', 'データ', 'モデル', 'システム']
const JA_TAIL = ['です', 'ます', 'した', 'ですね']
const JA_KANJI = '学習時間世界計算信号'

function pick(s: string, rng: Rng): string {
  const arr = [...s]
  return arr[Math.floor(rng() * arr.length)]
}

function sampleTextOf(cls: number, rng: Rng): string {
  if (cls === 0) {
    const len = 6 + Math.floor(rng() * 10)
    let s = ''
    for (let i = 0; i < len; i++) s += pick(ZH_TEXT, rng)
    return rng() < 0.3 ? s + '。' : s
  }
  if (cls === 1) {
    const len = 4 + Math.floor(rng() * 6)
    const words: string[] = []
    for (let i = 0; i < len; i++) words.push(EN_WORDS[Math.floor(rng() * EN_WORDS.length)])
    let s = words.join(' ')
    s = s.charAt(0).toUpperCase() + s.slice(1)
    return rng() < 0.5 ? s + '.' : s
  }
  const tokens = 3 + Math.floor(rng() * 5)
  let s = ''
  for (let i = 0; i < tokens; i++) {
    const r = rng()
    if (r < 0.5) s += pick(JA_KANA, rng) + pick(JA_KANA, rng)
    else if (r < 0.7) s += JA_KATA[Math.floor(rng() * JA_KATA.length)]
    else if (r < 0.85) s += pick(JA_KANJI, rng)
    else s += JA_TAIL[Math.floor(rng() * JA_TAIL.length)]
  }
  return s
}

export function buildTextTask(): TextTask {
  const rng = mulberry32(0x7e47)
  const model = createMLP(rng, [TEXT_FEATURE_COUNT, 10, 8, TEXT_CLASS_COUNT], ['relu', 'relu', 'softmax'])
  const data: TrainSample[] = []
  for (let c = 0; c < TEXT_CLASS_COUNT; c++) {
    for (let i = 0; i < 60; i++) {
      data.push({ x: extractTextFeatures(sampleTextOf(c, rng)), y: oneHot(TEXT_CLASS_COUNT, c) })
    }
  }
  trainMLP(model, data, { lr: 0.06, epochs: 180, rng })
  return { model, classCount: TEXT_CLASS_COUNT, sampleText: sampleTextOf }
}

// ---------------------------------------------------------------- Autoencoder
// Compress an 8×8 pattern through a 4-number bottleneck and reconstruct it.

export function buildAETask(): AETask {
  const rng = mulberry32(0xae0e)
  const n = 8
  const latent = 6
  const model = createMLP(rng, [n * n, 24, latent, 24, n * n], ['relu', 'tanh', 'relu', 'sigmoid'])
  const makeSample = (cls: number, r: Rng) => cnnSampleOfSize(n, cls, r)
  const data: TrainSample[] = []
  for (let c = 0; c < CNN_CLASS_COUNT; c++) {
    for (let i = 0; i < 30; i++) {
      const img = makeSample(c, rng)[0].flat()
      data.push({ x: img, y: img })
    }
  }
  const finalMSE = trainMLPMSE(model, data, { lr: 0.1, epochs: 80, rng })
  return { model, n, latent, classCount: CNN_CLASS_COUNT, makeSample, finalMSE }
}

// ---------------------------------------------------------------- registry

export let MODELS: Models = undefined as unknown as Models

export interface BootLine {
  label: string
  status: 'pending' | 'running' | 'done'
  detail?: string
}

/**
 * Train all seven networks, one at a time, yielding to the browser between
 * each so a boot screen can paint real progress. MODELS is assigned when
 * everything is ready.
 */
export async function initModels(onUpdate: (lines: BootLine[]) => void): Promise<void> {
  const partial: Partial<Models> = {}
  const steps: { label: string; build: () => string }[] = [
    {
      label: 'MLP',
      build: () => {
        partial.mlp = buildMLPTask('s')
        return '4 → 6 → 5 → 3'
      },
    },
    {
      label: 'CNN',
      build: () => {
        partial.cnn = buildCNNTask('s')
        return 'conv 3×3 · pool · dense'
      },
    },
    {
      label: 'RNN',
      build: () => {
        partial.rnn = buildRNNTask()
        return `loss ${partial.rnn.finalLoss.toFixed(2)}`
      },
    },
    {
      label: 'LSTM',
      build: () => {
        partial.lstm = buildLSTMTask()
        return `loss ${partial.lstm.finalLoss.toFixed(2)}`
      },
    },
    {
      label: 'Autoencoder',
      build: () => {
        partial.ae = buildAETask()
        return `MSE ${partial.ae.finalMSE.toFixed(4)}`
      },
    },
    {
      label: 'Transformer',
      build: () => {
        partial.llm = buildLLMTask()
        return `loss ${partial.llm.finalLoss.toFixed(2)}`
      },
    },
    {
      label: 'Lang ID',
      build: () => {
        partial.text = buildTextTask()
        return '8 features → 3 langs'
      },
    },
  ]

  const lines: BootLine[] = steps.map((s) => ({ label: s.label, status: 'pending' }))
  const paint = () => onUpdate(lines.map((l) => ({ ...l })))
  paint()
  for (let i = 0; i < steps.length; i++) {
    lines[i].status = 'running'
    paint()
    await new Promise((r) => setTimeout(r, 30))
    const t0 = performance.now()
    const detail = steps[i].build()
    lines[i].status = 'done'
    lines[i].detail = `${detail} · ${Math.round(performance.now() - t0)}ms`
    paint()
  }
  MODELS = partial as Models
}

/** Rebuild + retrain one architecture at a new scale / kernel mode. */
export function rebuildArch(arch: 'mlp' | 'cnn', scale: Scale, kernelMode: KernelMode = 'hand'): void {
  if (arch === 'mlp') MODELS = { ...MODELS, mlp: buildMLPTask(scale) }
  else MODELS = { ...MODELS, cnn: buildCNNTask(scale, kernelMode) }
}

/** Rebuild + retrain the transformer with a dense or mixture-of-experts FFN. */
export function rebuildLLM(variant: LLMVariant): void {
  MODELS = { ...MODELS, llm: buildLLMTask(variant) }
}

/** Quick self-check used by scripts/sanity.ts. */
export function evalTextAccuracy(samples = 60): number {
  const rng = mulberry32(31337)
  let ok = 0
  for (let i = 0; i < samples; i++) {
    const cls = i % TEXT_CLASS_COUNT
    const traces = forwardMLP(MODELS.text.model, extractTextFeatures(MODELS.text.sampleText(cls, rng)))
    if (argmax(traces[traces.length - 1].a) === cls) ok++
  }
  return ok / samples
}
