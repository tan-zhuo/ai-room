import { MODELS } from '../nn/models'
import { Arch } from '../store'

type T = (key: string, params?: Record<string, string | number>) => string

const LLM_LAYER_KEYS = ['layer.embed', 'layer.qkv', 'layer.attn', 'layer.attnout', 'layer.ffn', 'layer.output']

/** Localized display name for a layer (-1 = input). */
export function layerNameOf(arch: Arch, layer: number, t: T): string {
  if (arch === 'llm') {
    if (layer === -1) return t('layer.tokens')
    return layer === 1 ? 'Q · K · V' : t(LLM_LAYER_KEYS[layer] ?? 'layer.output')
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
    const keys = ['tokens', 'embed', 'qkv', 'attn', 'attnout', 'ffn', 'llmOutput']
    return keys[layer + 1] ?? 'llmOutput'
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
