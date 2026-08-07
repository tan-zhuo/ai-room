import { AE_VARIANT, MODELS } from '../nn/models'
import { Arch } from '../store'
import { LLMStageKind, llmStageKind } from '../scene/layout'

type T = (key: string, params?: Record<string, string | number>) => string

const LLM_KIND_NAMES: Record<LLMStageKind, string> = {
  tokens: 'layer.tokens',
  embed: 'layer.embed',
  posenc: 'layer.posenc',
  qkv: '',
  attn: 'layer.attn',
  attnout: 'layer.attnout',
  addnorm1: 'layer.addnorm',
  ffn: 'layer.ffn',
  router: 'layer.router',
  experts: 'layer.experts',
  combine: 'layer.combine',
  addnorm2: 'layer.addnorm',
  output: 'layer.output',
}

const LLM_KIND_EXPLAIN: Record<LLMStageKind, string> = {
  tokens: 'tokens',
  embed: 'embed',
  posenc: 'posenc',
  qkv: 'qkv',
  attn: 'attn',
  attnout: 'attnout',
  addnorm1: 'addnorm',
  ffn: 'ffn',
  router: 'router',
  experts: 'experts',
  combine: 'combine',
  addnorm2: 'addnorm',
  output: 'llmOutput',
}

const RNN_LAYER_KEYS = ['layer.embed', 'layer.rnnHidden', 'layer.output']
const LSTM_LAYER_KEYS = ['layer.embed', 'layer.gates', 'layer.cell', 'layer.rnnHidden', 'layer.output']
const AE_LAYER_KEYS = ['layer.encoder', 'layer.latent', 'layer.decoder', 'layer.recon']
const VAE_LAYER_KEYS = ['layer.encoder', '', 'layer.sampleZ', 'layer.decoder', 'layer.recon']

/** Localized display name for a layer (-1 = input). */
export function layerNameOf(arch: Arch, layer: number, t: T): string {
  if (arch === 'giant') return t('arch.giant')
  if (arch === 'agent') {
    return t(
      ['layer.agentDialog', 'layer.agentPlanner', 'layer.agentRetrieve', 'layer.agentCode', 'layer.agentRun', 'layer.agentAnswer', 'layer.agentSummarize'][
        layer + 1
      ] ?? 'layer.agentDialog',
    )
  }
  if (arch === 'llm') {
    const kind = llmStageKind(layer)
    if (kind === 'qkv') return 'Q · K · V'
    if (kind === 'router') {
      const m = MODELS.llm.model
      return t('layer.router', { n: m.nExperts, k: m.topK })
    }
    return t(LLM_KIND_NAMES[kind])
  }
  if (arch === 'mamba') {
    if (layer === -1) return t('layer.tokens')
    return t(
      ['layer.embed', 'layer.selectivity', 'layer.ssmState', 'layer.gatedOut', 'layer.output'][layer] ??
        'layer.output',
    )
  }
  if (arch === 'rnn' || arch === 'lstm') {
    if (layer === -1) return t('layer.tokens')
    const keys = arch === 'rnn' ? RNN_LAYER_KEYS : LSTM_LAYER_KEYS
    return t(keys[layer] ?? 'layer.output')
  }
  if (arch === 'diff') {
    return t(['layer.xt', 'layer.denoiser', 'layer.x0hat', 'layer.ddpmStep'][layer + 1] ?? 'layer.ddpmStep')
  }
  if (arch === 'gan') {
    return t(
      ['layer.zLatent', 'layer.generator', 'layer.fakeImg', 'layer.discriminator', 'layer.verdict', 'layer.realImg'][
        layer + 1
      ] ?? 'layer.verdict',
    )
  }
  if (arch === 'gnn') {
    return t(['layer.graphIn', 'layer.msgpass1', 'layer.msgpass2', 'layer.nodeCls'][layer + 1] ?? 'layer.nodeCls')
  }
  if (arch === 'vit') {
    return t(
      ['layer.input', 'layer.patchEmbed', 'layer.attn', 'layer.vitResid', 'layer.vitFfn', 'layer.clsHead'][
        layer + 1
      ] ?? 'layer.clsHead',
    )
  }
  if (arch === 'ae') {
    if (layer === -1) return t('layer.input')
    if (AE_VARIANT === 'vae') {
      const key = VAE_LAYER_KEYS[layer]
      return key === '' ? 'μ · log σ²' : t(key ?? 'layer.recon')
    }
    return t(AE_LAYER_KEYS[layer] ?? 'layer.recon')
  }
  if (layer === -1) return t('layer.input')
  if (arch === 'mlp' || arch === 'text') {
    const last = MODELS[arch].model.layers.length - 1
    return layer === last ? t('layer.output') : t('layer.hidden', { n: layer + 1 })
  }
  const last = MODELS.cnn.model.layers.length - 1
  if (layer === last) return t('layer.output')
  const def = MODELS.cnn.model.layers[layer]
  switch (def.type) {
    case 'conv':
      return t('layer.conv')
    case 'pool':
      return t('layer.pool')
    case 'flatten':
      return t('layer.flatten')
    default:
      return t('layer.dense')
  }
}

/** i18n key prefix (explain.<key>) for a layer's module explanation. */
export function explainKeyOf(arch: Arch, layer: number): string {
  if (arch === 'giant') return 'giantScale'
  if (arch === 'agent') {
    return (
      ['agentDialog', 'agentPlanner', 'agentMemory', 'agentCode', 'agentRun', 'agentAnswer', 'agentWriteback'][layer + 1] ??
      'agentDialog'
    )
  }
  if (arch === 'llm') {
    return LLM_KIND_EXPLAIN[llmStageKind(layer)]
  }
  if (arch === 'rnn') {
    return ['tokens', 'embed', 'rnnHidden', 'llmOutput'][layer + 1] ?? 'llmOutput'
  }
  if (arch === 'mamba') {
    return ['tokens', 'embed', 'selectivity', 'ssmScan', 'gatedOut', 'llmOutput'][layer + 1] ?? 'llmOutput'
  }
  if (arch === 'lstm') {
    return ['tokens', 'embed', 'gates', 'cellstate', 'lstmHidden', 'llmOutput'][layer + 1] ?? 'llmOutput'
  }
  if (arch === 'diff') {
    return ['xt', 'denoiserH', 'x0hat', 'ddpmStep'][layer + 1] ?? 'ddpmStep'
  }
  if (arch === 'gan') {
    return (
      ['zLatent', 'generator', 'fakeImg', 'discriminator', 'ganVerdict', 'realSample'][layer + 1] ?? 'ganVerdict'
    )
  }
  if (arch === 'gnn') {
    return ['graphIn', 'msgpass', 'msgpass', 'nodeCls'][layer + 1] ?? 'nodeCls'
  }
  if (arch === 'vit') {
    return ['vitInput', 'patchEmbed', 'vitAttn', 'vitResid', 'vitFfn', 'clsHead'][layer + 1] ?? 'clsHead'
  }
  if (arch === 'ae') {
    if (AE_VARIANT === 'vae') {
      return ['aeInput', 'encoder', 'muSigma', 'sampleZ', 'decoder', 'recon'][layer + 1] ?? 'recon'
    }
    return ['aeInput', 'encoder', 'latent', 'decoder', 'recon'][layer + 1] ?? 'recon'
  }
  if (arch === 'text') {
    if (layer === -1) return 'textInput'
    return layer === MODELS.text.model.layers.length - 1 ? 'output' : 'hidden'
  }
  if (arch === 'mlp') {
    if (layer === -1) return 'input'
    return layer === MODELS.mlp.model.layers.length - 1 ? 'output' : 'hidden'
  }
  if (layer === -1) return 'cnnInput'
  if (layer === MODELS.cnn.model.layers.length - 1) return 'output'
  const def = MODELS.cnn.model.layers[layer]
  switch (def.type) {
    case 'conv':
      return 'conv'
    case 'pool':
      return 'pool'
    case 'flatten':
      return 'flatten'
    default:
      return 'dense'
  }
}
