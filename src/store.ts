import { create } from 'zustand'
import {
  AEVariant,
  KernelMode,
  MODELS,
  Scale,
  extractTextFeatures,
  getVAETask,
  rebuildArch,
  rebuildLLM,
  setAEVariantFlag,
} from './nn/models'
import { VAETrace, forwardVAE, generateVAE } from './nn/vae'
import { DiffusionTrace, sampleDiffusion } from './nn/diffusion'
import { GANTrace, forwardGAN } from './nn/gan'
import { GNNTrace, forwardGNN, sampleGraph } from './nn/gnn'
import { ViTTrace, forwardViT } from './nn/vit'
import { MambaTrace, forwardMamba } from './nn/mamba'
import { AgentLang, AgentTrace, buildAgentTrace } from './nn/agent'
import { LLMVariant } from './nn/transformer'
import { DenseTrace, forwardMLP } from './nn/mlp'
import { CNNStep, Tensor3, forwardCNN } from './nn/cnn'
import { clamp01 } from './nn/rng'
import { LLMTrace, encodeLLM, forwardLLM } from './nn/transformer'
import { LSTMTrace, RNNTrace, encodeSeq, forwardLSTM, forwardRNN } from './nn/rnn'
import { mulberry32 } from './nn/rng'
import { Lang, LANGS, detectLang, translate } from './i18n'
import { refreshLayout } from './scene/layout'

export type Arch = 'mlp' | 'cnn' | 'text' | 'llm' | 'rnn' | 'lstm' | 'ae' | 'diff' | 'gan' | 'gnn' | 'vit' | 'mamba' | 'giant' | 'agent'

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

const ARCH_KEYS: Arch[] = ['mlp', 'cnn', 'rnn', 'lstm', 'mamba', 'llm', 'gnn', 'vit', 'ae', 'diff', 'gan', 'text', 'agent', 'giant']

function urlParam(name: string): string | null {
  if (typeof location === 'undefined') return null
  return new URLSearchParams(location.search).get(name)
}

function syncUrl(arch: Arch, lang: Lang): void {
  if (typeof history === 'undefined') return
  const url = new URL(location.href)
  url.searchParams.set('arch', arch)
  url.searchParams.set('lang', lang)
  history.replaceState(null, '', url)
}

const initialArch: Arch = ARCH_KEYS.includes(urlParam('arch') as Arch)
  ? (urlParam('arch') as Arch)
  : 'mlp'
const initialLang: Lang = LANGS.includes(urlParam('lang') as Lang)
  ? (urlParam('lang') as Lang)
  : detectLang()

const sampleRng = mulberry32(0xc0ffee)

/** current UI language, readable from makeInputs before the store exists */
let currentLang: Lang = initialLang

const FIXED_STEPS: Partial<Record<Arch, number>> = { rnn: 3, lstm: 5, ae: 4, gan: 4, gnn: 3, vit: 5, giant: 1, mamba: 5, agent: 6 }

export function totalSteps(arch: Arch): number {
  if (arch === 'llm') return MODELS.llm.model.moe ? 11 : 9
  if (arch === 'ae') return useStore.getState().aeVariant === 'vae' ? 5 : 4
  if (arch === 'diff') return MODELS.diff.model.T
  const fixed = FIXED_STEPS[arch]
  if (fixed !== undefined) return fixed
  return (MODELS[arch as 'mlp' | 'cnn' | 'text'].model as { layers: unknown[] }).layers.length
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
  scale: Record<Arch, Scale>
  cnnKernels: KernelMode
  llmVariant: LLMVariant
  modelsVersion: number
  menuOpen: boolean
  /** whole-architecture overview panel */
  overviewOpen: boolean
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
  rnnText: string
  rnnClass: number
  rnnTrace: RNNTrace
  lstmText: string
  lstmClass: number
  lstmTrace: LSTMTrace
  aeInput: Tensor3
  aeClass: number
  aeTrace: DenseTrace[]
  aeVariant: AEVariant
  vaeTrace: VAETrace | null
  diffTrace: DiffusionTrace
  ganTrace: GANTrace
  gnnTrace: GNNTrace
  vitInput: Tensor3
  vitClass: number
  vitTrace: ViTTrace
  mambaText: string
  mambaClass: number
  mambaTrace: MambaTrace
  /** which production model the True-Scale page highlights */
  giantSel: number
  agentTrace: AgentTrace
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
  setLLMVariant: (variant: LLMVariant) => void
  setAEVariant: (variant: AEVariant) => void
  resampleVAE: () => void
  generateFromPrior: () => void
  toggleMenu: () => void
  toggleOverview: () => void
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
  setRNNInput: (raw: string) => void
  setLSTMInput: (raw: string) => void
  setMambaInput: (raw: string) => void
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
  if (arch === 'llm') {
    const raw = MODELS.llm.samples[cls % MODELS.llm.samples.length]
    return { llmText: raw, llmClass: cls, llmTrace: forwardLLM(MODELS.llm.model, llmIdsFor(raw)) }
  }
  if (arch === 'rnn') {
    const raw = MODELS.rnn.samples[cls % MODELS.rnn.samples.length]
    const m = MODELS.rnn.model
    return { rnnText: raw, rnnClass: cls, rnnTrace: forwardRNN(m, encodeSeq(m.vocab, m.T, raw)) }
  }
  if (arch === 'lstm') {
    const raw = MODELS.lstm.samples[cls % MODELS.lstm.samples.length]
    const m = MODELS.lstm.model
    return { lstmText: raw, lstmClass: cls, lstmTrace: forwardLSTM(m, encodeSeq(m.vocab, m.T, raw)) }
  }
  if (arch === 'diff') {
    return { diffTrace: sampleDiffusion(MODELS.diff.model, sampleRng) }
  }
  if (arch === 'gan') {
    return { ganTrace: forwardGAN(MODELS.gan.model, MODELS.gan.realBank, sampleRng) }
  }
  if (arch === 'gnn') {
    return { gnnTrace: forwardGNN(MODELS.gnn.model, sampleGraph(MODELS.gnn.model, sampleRng)) }
  }
  if (arch === 'vit') {
    const input = MODELS.vit.makeSample(cls, sampleRng)
    return { vitInput: input, vitClass: cls, vitTrace: forwardViT(MODELS.vit.model, input[0]) }
  }
  if (arch === 'mamba') {
    const raw = MODELS.mamba.samples[cls % MODELS.mamba.samples.length]
    const m = MODELS.mamba.model
    return { mambaText: raw, mambaClass: cls, mambaTrace: forwardMamba(m, encodeSeq(m.vocab, m.T, raw)) }
  }
  if (arch === 'giant') {
    return { giantSel: cls }
  }
  if (arch === 'agent') {
    return { agentTrace: buildAgentTrace(MODELS.agent, currentLang as AgentLang, cls % 4) }
  }
  const input = MODELS.ae.makeSample(cls, sampleRng)
  return { aeInput: input, aeClass: cls, aeTrace: forwardMLP(MODELS.ae.model, input[0].flat()) }
}

function classCountOf(arch: Arch): number {
  if (arch === 'giant') return 5
  if (arch === 'agent') return 4
  if (arch === 'diff' || arch === 'gan' || arch === 'gnn') return 0
  if (arch === 'llm' || arch === 'rnn' || arch === 'lstm' || arch === 'mamba') return MODELS[arch].samples.length
  return MODELS[arch].classCount
}

const restart = { step: 0, playing: true, transitioning: false }

let toastTimer: ReturnType<typeof setTimeout> | null = null

export const useStore = create<AppState>((set, get) => ({
  arch: initialArch,
  lang: initialLang,
  scale: { mlp: 's', cnn: 's', rnn: 's', lstm: 's', llm: 's', gnn: 's', vit: 's', mamba: 's', agent: 's', ae: 's', diff: 's', gan: 's', text: 's', giant: 's' },
  cnnKernels: 'hand',
  llmVariant: 'dense',
  modelsVersion: 0,
  menuOpen: false,
  overviewOpen: false,
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
  ...(makeInputs('rnn', 0) as { rnnText: string; rnnClass: number; rnnTrace: RNNTrace }),
  ...(makeInputs('lstm', 0) as { lstmText: string; lstmClass: number; lstmTrace: LSTMTrace }),
  ...(makeInputs('ae', 0) as { aeInput: Tensor3; aeClass: number; aeTrace: DenseTrace[] }),
  aeVariant: 'ae',
  vaeTrace: null,
  ...(makeInputs('diff', 0) as { diffTrace: DiffusionTrace }),
  ...(makeInputs('gan', 0) as { ganTrace: GANTrace }),
  ...(makeInputs('gnn', 0) as { gnnTrace: GNNTrace }),
  ...(makeInputs('vit', 0) as { vitInput: Tensor3; vitClass: number; vitTrace: ViTTrace }),
  ...(makeInputs('mamba', 0) as { mambaText: string; mambaClass: number; mambaTrace: MambaTrace }),
  giantSel: 2,
  ...(makeInputs('agent', 0) as { agentTrace: AgentTrace }),
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
      overviewOpen: false,
    }))
    syncUrl(a, get().lang)
  },

  setScale: (size) => {
    const s = get()
    const archKey = s.arch
    if (archKey === 'giant' || archKey === 'agent') return
    if (s.scale[archKey] === size) return
    get().showToast('toast.training')
    // let the toast paint before the (synchronous) retraining burst
    setTimeout(() => {
      rebuildArch(archKey, size, { kernelMode: get().cnnKernels, llmVariant: get().llmVariant })
      refreshLayout()
      flow.phase = 0
      flow.hold = 0

      // re-run the forward pass on the freshly trained model
      let inputs: Partial<AppState>
      if (archKey === 'text') {
        const features = extractTextFeatures(get().textRaw)
        inputs = {
          textFeatures: features,
          textTrace: forwardMLP(MODELS.text.model, features),
        }
      } else if (archKey === 'llm') {
        inputs = {
          llmTrace: forwardLLM(MODELS.llm.model, llmIdsFor(get().llmText)),
          llmGenerated: '',
          llmGenerating: false,
        }
      } else if (archKey === 'rnn') {
        const m = MODELS.rnn.model
        inputs = { rnnTrace: forwardRNN(m, encodeSeq(m.vocab, m.T, get().rnnText)) }
      } else if (archKey === 'lstm') {
        const m = MODELS.lstm.model
        inputs = { lstmTrace: forwardLSTM(m, encodeSeq(m.vocab, m.T, get().lstmText)) }
      } else if (archKey === 'mamba') {
        const m = MODELS.mamba.model
        inputs = { mambaTrace: forwardMamba(m, encodeSeq(m.vocab, m.T, get().mambaText)) }
      } else if (archKey === 'ae') {
        // the input grid size may have changed — draw a fresh sample
        const cls = Math.max(0, get().aeClass)
        inputs = makeInputs('ae', cls)
        if (get().aeVariant === 'vae') {
          const img = (inputs as { aeInput: Tensor3 }).aeInput
          inputs = {
            ...inputs,
            vaeTrace: forwardVAE(getVAETask().model, img[0].flat(), undefined, sampleRng),
          }
        }
      } else {
        const cls =
          archKey === 'mlp'
            ? get().mlpClass
            : archKey === 'cnn'
              ? Math.max(0, get().cnnClass)
              : archKey === 'vit'
                ? Math.max(0, get().vitClass)
                : 0
        inputs = makeInputs(archKey, cls)
      }

      set((st) => ({
        scale: { ...st.scale, [archKey]: size },
        modelsVersion: st.modelsVersion + 1,
        ...inputs,
        ...restart,
        selected: null,
        explain: null,
        hoverInfo: null,
        drawMode: false,
      }))
    }, 60)
  },

  setKernelMode: (mode) => {
    const s = get()
    if (s.arch !== 'cnn' || s.cnnKernels === mode) return
    get().showToast('toast.training')
    setTimeout(() => {
      rebuildArch('cnn', get().scale.cnn, { kernelMode: mode })
      {
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
      }
    }, 60)
  },

  setLLMVariant: (variant) => {
    const s = get()
    if (s.arch !== 'llm' || s.llmVariant === variant) return
    get().showToast('toast.training')
    setTimeout(() => {
      rebuildLLM(variant, get().scale.llm)
      {
        refreshLayout()
        flow.phase = 0
        flow.hold = 0
        set((st) => ({
          llmVariant: variant,
          modelsVersion: st.modelsVersion + 1,
          llmTrace: forwardLLM(MODELS.llm.model, llmIdsFor(get().llmText)),
          ...restart,
          selected: null,
          explain: null,
          hoverInfo: null,
          llmGenerated: '',
          llmGenerating: false,
        }))
      }
    }, 60)
  },

  setAEVariant: (variant) => {
    const s = get()
    if (s.arch !== 'ae' || s.aeVariant === variant) return
    if (variant === 'vae') get().showToast('toast.training')
    setTimeout(() => {
      setAEVariantFlag(variant) // trains the VAE lazily on first use
      refreshLayout()
      flow.phase = 0
      flow.hold = 0
      set((st) => ({
        aeVariant: variant,
        modelsVersion: st.modelsVersion + 1,
        vaeTrace:
          variant === 'vae'
            ? forwardVAE(getVAETask().model, get().aeInput[0].flat(), undefined, sampleRng)
            : null,
        ...restart,
        selected: null,
        explain: null,
        hoverInfo: null,
      }))
    }, 60)
  },

  /** VAE: draw a fresh ε through the same input — shows the stochastic bottleneck. */
  resampleVAE: () => {
    const s = get()
    if (s.arch !== 'ae' || s.aeVariant !== 'vae') return
    flow.phase = 0
    set({
      vaeTrace: forwardVAE(getVAETask().model, s.aeInput[0].flat(), undefined, sampleRng),
      step: totalSteps('ae'),
      playing: false,
      transitioning: false,
    })
  },

  /** VAE: sample z straight from the prior and decode — pure generation. */
  generateFromPrior: () => {
    const s = get()
    if (s.arch !== 'ae' || s.aeVariant !== 'vae') return
    flow.phase = 0
    set({
      vaeTrace: generateVAE(getVAETask().model, sampleRng),
      aeClass: -1,
      step: totalSteps('ae'),
      playing: false,
      transitioning: false,
      selected: null,
    })
  },

  toggleMenu: () => set((s) => ({ menuOpen: !s.menuOpen })),

  toggleOverview: () =>
    set((s) => ({
      overviewOpen: !s.overviewOpen,
      ...(s.overviewOpen ? {} : { selected: null, explain: null }),
    })),

  toggleDraw: () => {
    const s = get()
    if (s.arch !== 'cnn' && s.arch !== 'ae') return
    if (s.drawMode) {
      set({ drawMode: false })
      return
    }
    // entering draw mode: pause and reveal the whole network for live inference
    set({ drawMode: true, playing: false, transitioning: false, step: totalSteps(s.arch) })
  },

  paintPixel: (row, col) => {
    const s = get()
    if (!s.drawMode) return
    if (s.arch === 'cnn') {
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
      return
    }
    if (s.arch === 'ae') {
      if (s.aeInput[0][row][col] >= 0.9) return
      const img = s.aeInput.map((ch) => ch.map((r) => [...r]))
      img[0][row][col] = clamp01(0.95)
      set({
        aeInput: img,
        aeClass: -1,
        aeTrace: forwardMLP(MODELS.ae.model, img[0].flat()),
        vaeTrace:
          s.aeVariant === 'vae'
            ? forwardVAE(getVAETask().model, img[0].flat(), get().vaeTrace?.eps, sampleRng)
            : null,
        step: totalSteps('ae'),
        playing: false,
        transitioning: false,
      })
    }
  },

  clearCnnInput: () => {
    const s = get()
    if (s.arch === 'cnn') {
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
      return
    }
    if (s.arch === 'ae') {
      const n = MODELS.ae.n
      const img: Tensor3 = [Array.from({ length: n }, () => Array.from({ length: n }, () => 0.05))]
      set({
        aeInput: img,
        aeClass: -1,
        aeTrace: forwardMLP(MODELS.ae.model, img[0].flat()),
        vaeTrace:
          s.aeVariant === 'vae'
            ? forwardVAE(getVAETask().model, img[0].flat(), undefined, sampleRng)
            : null,
        step: totalSteps('ae'),
        playing: false,
        transitioning: false,
      })
    }
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

  select: (ref) => set({ selected: ref, ...(ref ? { explain: null, overviewOpen: false } : {}) }),
  setExplain: (layer) =>
    set({ explain: layer, ...(layer !== null ? { selected: null, overviewOpen: false } : {}) }),
  setHover: (h) => set({ hoverInfo: h }),

  setLang: (l) => {
    currentLang = l
    set({ lang: l })
    if (get().arch === 'agent') {
      set({ agentTrace: buildAgentTrace(MODELS.agent, l as AgentLang, get().agentTrace.scenario) })
    }
    syncUrl(get().arch, l)
  },
  cycleLang: () => {
    const next = LANGS[(LANGS.indexOf(get().lang) + 1) % LANGS.length]
    currentLang = next
    set({ lang: next })
    if (get().arch === 'agent') {
      set({ agentTrace: buildAgentTrace(MODELS.agent, next as AgentLang, get().agentTrace.scenario) })
    }
    syncUrl(get().arch, next)
  },

  newSample: (cls) => {
    const s = get()
    const chosen = cls ?? Math.floor(sampleRng() * classCountOf(s.arch))
    flow.phase = 0
    flow.hold = 0
    const inputs = makeInputs(s.arch, chosen)
    set({ ...inputs, ...restart, llmGenerated: '', llmGenerating: false })
    if (s.arch === 'ae' && s.aeVariant === 'vae') {
      const img = (inputs as { aeInput: Tensor3 }).aeInput
      set({ vaeTrace: forwardVAE(getVAETask().model, img[0].flat(), undefined, sampleRng) })
    }
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

  setRNNInput: (raw) => {
    const m = MODELS.rnn.model
    flow.phase = 0
    flow.hold = 0
    set({
      rnnText: raw,
      rnnClass: -1,
      rnnTrace: forwardRNN(m, encodeSeq(m.vocab, m.T, raw)),
      ...restart,
      selected: null,
    })
  },

  setLSTMInput: (raw) => {
    const m = MODELS.lstm.model
    flow.phase = 0
    flow.hold = 0
    set({
      lstmText: raw,
      lstmClass: -1,
      lstmTrace: forwardLSTM(m, encodeSeq(m.vocab, m.T, raw)),
      ...restart,
      selected: null,
    })
  },

  setMambaInput: (raw) => {
    const m = MODELS.mamba.model
    flow.phase = 0
    flow.hold = 0
    set({
      mambaText: raw,
      mambaClass: -1,
      mambaTrace: forwardMamba(m, encodeSeq(m.vocab, m.T, raw)),
      ...restart,
      selected: null,
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
    const durations = MODELS.llm.model.moe
      ? [1.3, 1.3, 1.8, 3.0, 1.8, 1.5, 1.8, 2.4, 1.6, 1.5, 1.5]
      : [1.3, 1.3, 1.8, 3.0, 1.8, 1.5, 1.6, 1.5, 1.5]
    return durations[step] ?? 1.4
  }
  if (arch === 'rnn') return [1.4, 3.4, 1.5][step] ?? 1.4
  if (arch === 'lstm') return [1.4, 2.6, 2.2, 2.0, 1.5][step] ?? 1.4
  if (arch === 'ae') return [1.5, 1.4, 1.5, 1.6][step] ?? 1.4
  if (arch === 'diff') return 0.75
  if (arch === 'gan') return [1.5, 1.6, 1.6, 1.5][step] ?? 1.4
  if (arch === 'gnn') return [2.4, 2.4, 1.6][step] ?? 1.4
  if (arch === 'vit') return [1.7, 2.6, 1.6, 1.7, 1.4][step] ?? 1.4
  if (arch === 'giant') return 0.6
  if (arch === 'agent') return [1.6, 2.0, 2.4, 1.8, 1.8, 2.0][step] ?? 1.6
  if (arch === 'mamba') return [1.4, 1.8, 3.2, 1.6, 1.5][step] ?? 1.4
  const model = MODELS.cnn.model
  const def = model.layers[step]
  if (!def) return 1.2
  const n = model.inputShape[1]
  if (def.type === 'conv') return Math.min(6, 1.5 + (n - 2) ** 2 * 0.028)
  if (def.type === 'pool') return Math.min(4.5, 1.1 + Math.floor((n - 2) / 2) ** 2 * 0.14)
  if (def.type === 'flatten') return 1.2
  return step === model.layers.length - 1 ? 1.4 : 1.6
}
