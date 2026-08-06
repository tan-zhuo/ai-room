import { MODELS } from '../nn/models'
import { Arch } from '../store'

type T = (key: string, params?: Record<string, string | number>) => string

const LLM_LAYER_KEYS = [
  'layer.embed',
  'layer.posenc',
  'layer.qkv',
  'layer.attn',
  'layer.attnout',
  'layer.addnorm',
  'layer.ffn',
  'layer.addnorm',
  'layer.output',
]

const RNN_LAYER_KEYS = ['layer.embed', 'layer.rnnHidden', 'layer.output']
const LSTM_LAYER_KEYS = ['layer.embed', 'layer.gates', 'layer.cell', 'layer.rnnHidden', 'layer.output']
const AE_LAYER_KEYS = ['layer.encoder', 'layer.latent', 'layer.decoder', 'layer.recon']

/** Localized display name for a layer (-1 = input). */
export function layerNameOf(arch: Arch, layer: number, t: T): string {
  if (arch === 'llm') {
    if (layer === -1) return t('layer.tokens')
    return layer === 2 ? 'Q · K · V' : t(LLM_LAYER_KEYS[layer] ?? 'layer.output')
  }
  if (arch === 'rnn' || arch === 'lstm') {
    if (layer === -1) return t('layer.tokens')
    const keys = arch === 'rnn' ? RNN_LAYER_KEYS : LSTM_LAYER_KEYS
    return t(keys[layer] ?? 'layer.output')
  }
  if (arch === 'ae') {
    if (layer === -1) return t('layer.input')
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
  if (arch === 'llm') {
    const keys = ['tokens', 'embed', 'posenc', 'qkv', 'attn', 'attnout', 'addnorm', 'ffn', 'addnorm', 'llmOutput']
    return keys[layer + 1] ?? 'llmOutput'
  }
  if (arch === 'rnn') {
    return ['tokens', 'embed', 'rnnHidden', 'llmOutput'][layer + 1] ?? 'llmOutput'
  }
  if (arch === 'lstm') {
    return ['tokens', 'embed', 'gates', 'cellstate', 'lstmHidden', 'llmOutput'][layer + 1] ?? 'llmOutput'
  }
  if (arch === 'ae') {
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
