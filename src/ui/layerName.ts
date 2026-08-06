import { MODELS } from '../nn/models'
import { Arch } from '../store'

type T = (key: string, params?: Record<string, string | number>) => string

/** Localized display name for a layer (-1 = input). */
export function layerNameOf(arch: Arch, layer: number, t: T): string {
  if (layer === -1) return t('layer.input')
  if (arch === 'mlp') {
    const last = MODELS.mlp.model.layers.length - 1
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
