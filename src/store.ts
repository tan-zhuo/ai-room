import { create } from 'zustand'
import { MODELS } from './nn/models'
import { DenseTrace, forwardMLP } from './nn/mlp'
import { CNNStep, Tensor3, forwardCNN } from './nn/cnn'
import { mulberry32 } from './nn/rng'
import { Lang, LANGS, detectLang, translate } from './i18n'

export type Arch = 'mlp' | 'cnn'

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

export function totalSteps(arch: Arch): number {
  return arch === 'mlp' ? MODELS.mlp.model.layers.length : MODELS.cnn.model.layers.length
}

export interface HoverInfo {
  ref: NodeRef
  value: number
}

interface AppState {
  arch: Arch
  lang: Lang
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
  requestFocus: (pos: [number, number, number] | null, distance?: number) => void
  toggleHelp: () => void
  showToast: (key: string) => void
}

function makeInputs(arch: Arch, cls: number) {
  if (arch === 'mlp') {
    const input = MODELS.mlp.makeSample(cls, sampleRng)
    return { mlpInput: input, mlpClass: cls, mlpTrace: forwardMLP(MODELS.mlp.model, input) }
  }
  const input = MODELS.cnn.makeSample(cls, sampleRng)
  return { cnnInput: input, cnnClass: cls, cnnTrace: forwardCNN(MODELS.cnn.model, input) }
}

const initialMLP = makeInputs('mlp', 0) as { mlpInput: number[]; mlpClass: number; mlpTrace: DenseTrace[] }
const initialCNN = makeInputs('cnn', 0) as { cnnInput: Tensor3; cnnClass: number; cnnTrace: CNNStep[] }

let toastTimer: ReturnType<typeof setTimeout> | null = null

export const useStore = create<AppState>((set, get) => ({
  arch: 'mlp',
  lang: detectLang(),
  step: 0,
  playing: true,
  transitioning: false,
  speed: 1,
  ...initialMLP,
  ...initialCNN,
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
      step: 0,
      playing: true,
      transitioning: false,
      selected: null,
      explain: null,
      hoverInfo: null,
      focusTarget: null,
      focusNonce: s.focusNonce + 1,
    }))
  },

  togglePlay: () => {
    const s = get()
    if (s.step >= totalSteps(s.arch) && !s.playing) {
      flow.phase = 0
      set({ step: 0, playing: true, transitioning: false })
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
    set({ step: 0, playing: false, transitioning: false })
  },

  cycleSpeed: () => {
    const order = [0.5, 1, 1.5, 2]
    const cur = order.indexOf(get().speed)
    set({ speed: order[(cur + 1) % order.length] })
  },

  finishStep: () => {
    const s = get()
    const total = totalSteps(s.arch)
    const next = Math.min(total, s.step + 1)
    set({ step: next, transitioning: false, playing: s.playing })
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
    const count = s.arch === 'mlp' ? MODELS.mlp.classCount : MODELS.cnn.classCount
    const chosen = cls ?? Math.floor(sampleRng() * count)
    flow.phase = 0
    flow.hold = 0
    set({ ...makeInputs(s.arch, chosen), step: 0, playing: true, transitioning: false })
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
  if (arch === 'mlp') return 1.6
  const durations = [3.6, 2.4, 1.2, 1.6, 1.4]
  return durations[step] ?? 1.2
}
