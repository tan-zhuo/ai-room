import { create } from 'zustand'
import { KernelMode, MODELS, Scale, extractTextFeatures, rebuildArch } from './nn/models'
import { DenseTrace, forwardMLP } from './nn/mlp'
import { CNNStep, Tensor3, forwardCNN } from './nn/cnn'
import { clamp01 } from './nn/rng'
import { LLMTrace, encodeLLM, forwardLLM } from './nn/transformer'
import { mulberry32 } from './nn/rng'
import { Lang, LANGS, detectLang, translate } from './i18n'

export type Arch = 'mlp' | 'cnn' | 'text' | 'llm'

export type NodeRef =
  | { space: 'vector'; layer: number; index: number }
  | { space: 'grid'; layer: number; channel: number; row: number; col: number }

export function sameRef(a: NodeRef | null, b: NodeRef | null): boolean {
  if (!a || !b) return false
  if (a.space === 'vector' && b.space === 'vector') return a.layer === b.layer && a.index === b.index
  if (a.space === 'grid' && b.space === 'grid')
    return a.layer === b.layer && a.channel === b.channel && a.row === b.row && a.col === b.col
  return false
}

/** Mutable per-frame playback state, read inside useFrame without re-rendering React. */
export const flow = { phase: 0, hold: 0 }

const sampleRng = mulberry32(0xc0ffee)

const LLM_STEP_COUNT = 9

export function totalSteps(arch: Arch): number {
  if (arch === 'llm') return LLM_STEP_COUNT
  return MODELS[arch].model.layers.length
}

export interface HoverInfo {
  ref: NodeRef
  value: number
}

/** Pad to the model's full context length so the layout stays fixed. */
function llmIdsFor(text: string): number[] {
  const model = MODELS.llm.model
  const ids = encodeLLM(model, text)
  const space = model.vocab.indexOf(' ')
  while (ids.length < model.T) ids.unshift(space)
  return ids
}

interface AppState {
  arch: Arch
  lang: Lang
  scale: { mlp: Scale; cnn: Scale }
  cnnKernels: KernelMode
  modelsVersion: number
  menuOpen: boolean
  /** paint-on-the-input mode for the CNN */
  drawMode: boolean
  llmTemp: number
  step: number
  playing: boolean
  transitioning: boolean
  speed: number
  mlpInput: number[]
  mlpClass: number
  mlpTrace: DenseTrace[]
  cnnInput: Tensor3
  cnnClass: number
  cnnTrace: CNNStep[]
  textRaw: string
  textClass: number
  textFeatures: number[]
  textTrace: DenseTrace[]
  llmText: string
  llmClass: number
  llmTrace: LLMTrace
  /** characters produced so far in autoregressive generation mode */
  llmGenerated: string
  llmGenerating: boolean
  selected: NodeRef | null
  /** layer index whose module explanation is open (-1 = input), for the current arch */
  explain: number | null
  hoverInfo: HoverInfo | null
  focusTarget: [number, number, number] | null
  focusDistance: number
  focusNonce: number
  helpOpen: boolean
  toast: string | null

  setArch: (a: Arch) => void
  setScale: (size: Scale) => void
  setKernelMode: (mode: KernelMode) => void
  toggleMenu: () => void
  toggleDraw: () => void
  paintPixel: (row: number, col: number) => void
  clearCnnInput: () => void
  setLLMTemp: (temp: number) => void
  togglePlay: () => void
  nextStep: () => void
  prevStep: () => void
  reset: () => void
  cycleSpeed: () => void
  finishStep: () => void
  select: (ref: NodeRef | null) => void
  setExplain: (layer: number | null) => void
  setHover: (h: HoverInfo | null) => void
  setLang: (l: Lang) => void
  cycleLang: () => void
  newSample: (cls?: number) => void
  setTextInput: (raw: string) => void
  setLLMInput: (raw: string) => void
  toggleGenerate: () => void
  commitGeneratedChar: () => void
  requestFocus: (pos: [number, number, number] | null, distance?: number) => void
  toggleHelp: () => void
  showToast: (key: string) => void
}

function makeInputs(arch: Arch, cls: number) {
  if (arch === 'mlp') {
    const input = MODELS.mlp.makeSample(cls, sampleRng)
    return { mlpInput: input, mlpClass: cls, mlpTrace: forwardMLP(MODELS.mlp.model, input) }
  }
  if (arch === 'cnn') {
    const input = MODELS.cnn.makeSample(cls, sampleRng)
    return { cnnInput: input, cnnClass: cls, cnnTrace: forwardCNN(MODELS.cnn.model, input) }
  }
  if (arch === 'text') {
    const raw = MODELS.text.sampleText(cls, sampleRng)
    const features = extractTextFeatures(raw)
    return {
      textRaw: raw,
      textClass: cls,
      textFeatures: features,
      textTrace: forwardMLP(MODELS.text.model, features),
    }
  }
  const raw = MODELS.llm.samples[cls % MODELS.llm.samples.length]
  return { llmText: raw, llmClass: cls, llmTrace: forwardLLM(MODELS.llm.model, llmIdsFor(raw)) }
}

function classCountOf(arch: Arch): number {
  if (arch === 'llm') return MODELS.llm.samples.length
  return MODELS[arch].classCount
}

const restart = { step: 0, playing: true, transitioning: false }

let toastTimer: ReturnType<typeof setTimeout> | null = null

export const useStore = create<AppState>((set, get) => ({
  arch: 'mlp',
  lang: detectLang(),
  scale: { mlp: 's', cnn: 's' },
  cnnKernels: 'hand',
  modelsVersion: 0,
  menuOpen: false,
  drawMode: false,
  llmTemp: 0.7,
  step: 0,
  playing: true,
  transitioning: false,
  speed: 1,
  ...(makeInputs('mlp', 0) as { mlpInput: number[]; mlpClass: number; mlpTrace: DenseTrace[] }),
  ...(makeInputs('cnn', 0) as { cnnInput: Tensor3; cnnClass: number; cnnTrace: CNNStep[] }),
  ...(makeInputs('text', 1) as {
    textRaw: string
    textClass: number
    textFeatures: number[]
    textTrace: DenseTrace[]
  }),
  ...(makeInputs('llm', 0) as { llmText: string; llmClass: number; llmTrace: LLMTrace }),
  llmGenerated: '',
  llmGenerating: false,
  selected: null,
  explain: null,
  hoverInfo: null,
  focusTarget: null,
  focusDistance: 4.5,
  focusNonce: 0,
  helpOpen: false,
  toast: null,

  setArch: (a) => {
    if (get().arch === a) return
    flow.phase = 0
    flow.hold = 0
    set((s) => ({
      arch: a,
      ...restart,
      selected: null,
      explain: null,
      hoverInfo: null,
      focusTarget: null,
      focusNonce: s.focusNonce + 1,
      llmGenerated: '',
      llmGenerating: false,
      drawMode: false,
      menuOpen: false,
    }))
  },

  setScale: (size) => {
    const s = get()
    if (s.arch !== 'mlp' && s.arch !== 'cnn') return
    const archKey = s.arch
    if (s.scale[archKey] === size) return
    get().showToast('toast.training')
    // let the toast paint before the (synchronous) retraining burst
    setTimeout(() => {
      rebuildArch(archKey, size, get().cnnKernels)
      // refreshLayout is imported lazily to keep layout.ts free of runtime store deps
      import('./scene/layout').then(({ refreshLayout }) => {
        refreshLayout()
        const cls = archKey === 'mlp' ? get().mlpClass : get().cnnClass
        flow.phase = 0
        flow.hold = 0
        set((st) => ({
          scale: { ...st.scale, [archKey]: size },
          modelsVersion: st.modelsVersion + 1,
          ...makeInputs(archKey, cls),
          ...restart,
          selected: null,
          explain: null,
          hoverInfo: null,
        }))
      })
    }, 60)
  },

  setKernelMode: (mode) => {
    const s = get()
    if (s.arch !== 'cnn' || s.cnnKernels === mode) return
    get().showToast('toast.training')
    setTimeout(() => {
      rebuildArch('cnn', get().scale.cnn, mode)
      import('./scene/layout').then(({ refreshLayout }) => {
        refreshLayout()
        flow.phase = 0
        flow.hold = 0
        set((st) => ({
          cnnKernels: mode,
          modelsVersion: st.modelsVersion + 1,
          ...makeInputs('cnn', get().cnnClass),
          ...restart,
          selected: null,
          explain: null,
          hoverInfo: null,
        }))
      })
    }, 60)
  },

  toggleMenu: () => set((s) => ({ menuOpen: !s.menuOpen })),

  toggleDraw: () => {
    const s = get()
    if (s.arch !== 'cnn') return
    if (s.drawMode) {
      set({ drawMode: false })
      return
    }
    // entering draw mode: pause and reveal the whole network for live inference
    set({ drawMode: true, playing: false, transitioning: false, step: totalSteps('cnn') })
  },

  paintPixel: (row, col) => {
    const s = get()
    if (s.arch !== 'cnn' || !s.drawMode) return
    if (s.cnnInput[0][row][col] >= 0.9) return
    const img = s.cnnInput.map((ch) => ch.map((r) => [...r]))
    img[0][row][col] = clamp01(0.95)
    set({
      cnnInput: img,
      cnnClass: -1,
      cnnTrace: forwardCNN(MODELS.cnn.model, img),
      step: totalSteps('cnn'),
      playing: false,
      transitioning: false,
    })
  },

  clearCnnInput: () => {
    const s = get()
    if (s.arch !== 'cnn') return
    const n = MODELS.cnn.model.inputShape[1]
    const img: Tensor3 = [Array.from({ length: n }, () => Array.from({ length: n }, () => 0.05))]
    set({
      cnnInput: img,
      cnnClass: -1,
      cnnTrace: forwardCNN(MODELS.cnn.model, img),
      step: totalSteps('cnn'),
      playing: false,
      transitioning: false,
    })
  },

  setLLMTemp: (temp) => set({ llmTemp: temp }),

  togglePlay: () => {
    const s = get()
    if (s.step >= totalSteps(s.arch) && !s.playing) {
      flow.phase = 0
      set(restart)
      return
    }
    set({ playing: !s.playing, transitioning: false })
  },

  nextStep: () => {
    const s = get()
    if (s.step >= totalSteps(s.arch)) return
    set({ playing: false, transitioning: true })
  },

  prevStep: () => {
    const s = get()
    flow.phase = 0
    set({ step: Math.max(0, s.step - 1), playing: false, transitioning: false })
  },

  reset: () => {
    flow.phase = 0
    flow.hold = 0
    set({ step: 0, playing: false, transitioning: false, llmGenerating: false })
  },

  cycleSpeed: () => {
    const order = [0.5, 1, 1.5, 2]
    const cur = order.indexOf(get().speed)
    set({ speed: order[(cur + 1) % order.length] })
  },

  finishStep: () => {
    const s = get()
    const total = totalSteps(s.arch)
    set({ step: Math.min(total, s.step + 1), transitioning: false })
  },

  select: (ref) => set({ selected: ref, ...(ref ? { explain: null } : {}) }),
  setExplain: (layer) => set({ explain: layer, ...(layer !== null ? { selected: null } : {}) }),
  setHover: (h) => set({ hoverInfo: h }),

  setLang: (l) => set({ lang: l }),
  cycleLang: () => {
    const cur = LANGS.indexOf(get().lang)
    set({ lang: LANGS[(cur + 1) % LANGS.length] })
  },

  newSample: (cls) => {
    const s = get()
    const chosen = cls ?? Math.floor(sampleRng() * classCountOf(s.arch))
    flow.phase = 0
    flow.hold = 0
    set({ ...makeInputs(s.arch, chosen), ...restart, llmGenerated: '', llmGenerating: false })
  },

  setTextInput: (raw) => {
    const features = extractTextFeatures(raw)
    flow.phase = 0
    flow.hold = 0
    set({
      textRaw: raw,
      textClass: -1,
      textFeatures: features,
      textTrace: forwardMLP(MODELS.text.model, features),
      ...restart,
    })
  },

  setLLMInput: (raw) => {
    flow.phase = 0
    flow.hold = 0
    set({
      llmText: raw,
      llmClass: -1,
      llmTrace: forwardLLM(MODELS.llm.model, llmIdsFor(raw)),
      ...restart,
      selected: null,
      llmGenerated: '',
      llmGenerating: false,
    })
  },

  toggleGenerate: () => {
    const s = get()
    if (s.llmGenerating) {
      set({ llmGenerating: false })
      return
    }
    flow.phase = 0
    flow.hold = 0
    set({
      llmGenerated: '',
      llmGenerating: true,
      llmTrace: forwardLLM(MODELS.llm.model, llmIdsFor(s.llmText)),
      ...restart,
      selected: null,
    })
  },

  /** Sample the next character from the current distribution, append it to the
   *  context and start the next forward pass — one autoregressive step. */
  commitGeneratedChar: () => {
    const s = get()
    const model = MODELS.llm.model
    // temperature sampling keeps the text lively without going random
    const temp = s.llmTemp
    const logits = s.llmTrace.probs.map((p) => Math.log(Math.max(p, 1e-9)) / temp)
    const m = Math.max(...logits)
    const exps = logits.map((v) => Math.exp(v - m))
    const sum = exps.reduce((acc, v) => acc + v, 0)
    let r = sampleRng() * sum
    let idx = 0
    for (; idx < exps.length - 1; idx++) {
      r -= exps[idx]
      if (r <= 0) break
    }
    const generated = s.llmGenerated + model.vocab[idx]
    const keepGoing = s.llmGenerating && generated.length < 30
    flow.phase = 0
    flow.hold = 0
    set({
      llmGenerated: generated,
      llmGenerating: keepGoing,
      llmTrace: forwardLLM(model, llmIdsFor(s.llmText + generated)),
      step: 0,
      playing: keepGoing,
      transitioning: false,
    })
  },

  requestFocus: (pos, distance = 4.5) =>
    set((s) => ({ focusTarget: pos, focusDistance: distance, focusNonce: s.focusNonce + 1 })),

  toggleHelp: () => set((s) => ({ helpOpen: !s.helpOpen })),

  showToast: (key) => {
    if (toastTimer) clearTimeout(toastTimer)
    set({ toast: key })
    toastTimer = setTimeout(() => set({ toast: null }), 2600)
  },
}))

/** Translation hook bound to the current language. */
export function useT() {
  const lang = useStore((s) => s.lang)
  return (key: string, params?: Record<string, string | number>) => translate(lang, key, params)
}

/** Seconds a step transition takes at 1x speed. */
export function stepDuration(arch: Arch, step: number): number {
  if (arch === 'mlp' || arch === 'text') return 1.6
  if (arch === 'llm') {
    const durations = [1.3, 1.3, 1.8, 3.0, 1.8, 1.5, 1.6, 1.5, 1.5]
    return durations[step] ?? 1.4
  }
  const model = MODELS.cnn.model
  const def = model.layers[step]
  if (!def) return 1.2
  const n = model.inputShape[1]
  if (def.type === 'conv') return Math.min(6, 1.5 + (n - 2) ** 2 * 0.028)
  if (def.type === 'pool') return Math.min(4.5, 1.1 + Math.floor((n - 2) / 2) ** 2 * 0.14)
  if (def.type === 'flatten') return 1.2
  return step === model.layers.length - 1 ? 1.4 : 1.6
}
