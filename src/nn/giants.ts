// Real production-scale model specs — DISPLAY ONLY, nothing here is computed.
// All numbers are from the public papers / model cards; the point of this
// page is honest scale comparison against the tiny nets trained in-browser.

import { MODELS } from './models'

export interface GiantSpec {
  key: string
  name: string
  year: number
  /** total parameters */
  params: number
  /** activated per token (MoE) */
  active?: number
  layers: number
  d: number
  heads: number
  ctx: number
  /** training tokens (approx, public) */
  tokens?: number
  moe?: { experts: number; topK: number; shared: number; expertDff: number }
  /** FFN hidden width (per expert for MoE) */
  dff: number
  vocab: number
  /** 'gelu2' = 2-matrix FFN, 'swiglu3' = gate/up/down, 'moe' = expert grid */
  ffn: 'gelu2' | 'swiglu3' | 'moe'
  /** KV projection width when grouped-query attention shrinks it */
  kvDims?: number
  /** i18n key suffix for the one-line description */
  desc: string
}

export const GIANTS: GiantSpec[] = [
  {
    key: 'mini',
    name: 'AI ROOM mini',
    year: 2026,
    params: 0, // filled at runtime from the actually-trained transformer
    layers: 1,
    d: 12,
    heads: 2,
    ctx: 8,
    dff: 24,
    vocab: 28,
    ffn: 'gelu2',
    desc: 'mini',
  },
  {
    key: 'gpt2',
    name: 'GPT-2 XL',
    year: 2019,
    params: 1.5e9,
    layers: 48,
    d: 1600,
    heads: 25,
    ctx: 1024,
    tokens: 9e9,
    dff: 6400,
    vocab: 50257,
    ffn: 'gelu2',
    desc: 'gpt2',
  },
  {
    key: 'gpt3',
    name: 'GPT-3',
    year: 2020,
    params: 175e9,
    layers: 96,
    d: 12288,
    heads: 96,
    ctx: 2048,
    tokens: 300e9,
    dff: 49152,
    vocab: 50257,
    ffn: 'gelu2',
    desc: 'gpt3',
  },
  {
    key: 'llama',
    name: 'Llama 3.1 405B',
    year: 2024,
    params: 405e9,
    layers: 126,
    d: 16384,
    heads: 128,
    ctx: 128_000,
    tokens: 15e12,
    dff: 53248,
    vocab: 128_256,
    ffn: 'swiglu3',
    kvDims: 1024,
    desc: 'llama',
  },
  {
    key: 'dsv3',
    name: 'DeepSeek-V3',
    year: 2024,
    params: 671e9,
    active: 37e9,
    layers: 61,
    d: 7168,
    heads: 128,
    ctx: 128_000,
    tokens: 14.8e12,
    moe: { experts: 256, topK: 8, shared: 1, expertDff: 2048 },
    dff: 2048,
    vocab: 129_280,
    ffn: 'moe',
    desc: 'dsv3',
  },
]

/** Attention + FFN parameters of ONE layer (approx, biases ignored). */
export function perLayerParams(g: GiantSpec): number {
  const kv = g.kvDims ?? g.d
  const attn = g.d * g.d * 2 + g.d * kv * 2 // Q,O full width; K,V possibly grouped
  if (g.ffn === 'moe' && g.moe) {
    return attn + (g.moe.experts + g.moe.shared) * 2 * g.d * g.moe.expertDff
  }
  const ffn = (g.ffn === 'swiglu3' ? 3 : 2) * g.d * g.dff
  return attn + ffn
}

export function embedParams(g: GiantSpec): number {
  return g.vocab * g.d
}

/** Count every trainable number in the in-browser transformer. */
export function countMiniParams(): number {
  const m = MODELS.llm.model
  const mat = (w: number[][]) => w.length * (w[0]?.length ?? 0)
  let n = mat(m.E) + mat(m.P)
  n += mat(m.Wq) + mat(m.Wk) + mat(m.Wv) + mat(m.Wo) + m.bo.length
  n += mat(m.W1) + m.b1.length + mat(m.W2) + m.b2.length
  n += m.g1.length + m.be1.length + m.g2.length + m.be2.length
  n += mat(m.Wout) + m.bout.length
  if (m.moe) {
    n += mat(m.Wr) + m.br.length
    for (let e = 0; e < m.nExperts; e++) {
      n += mat(m.We1[e]) + m.bE1[e].length + mat(m.We2[e]) + m.bE2[e].length
    }
  }
  return n
}

/** "1750亿" / "175B" style formatting. */
export function fmtParams(n: number, lang: string): string {
  const f = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 1 })
  if (n >= 1e9) return lang === 'zh' ? `${f(n / 1e8)} 亿` : `${f(n / 1e9)}B`
  if (n >= 1e6) return lang === 'zh' ? `${f(n / 1e8)} 亿` : `${f(n / 1e6)}M`
  return n.toLocaleString()
}

export function fmtTokens(n: number | undefined, lang: string): string {
  if (!n) return '—'
  if (n >= 1e12) return lang === 'zh' ? `${(n / 1e12).toLocaleString()} 万亿 token` : `${(n / 1e12).toLocaleString()}T tokens`
  return lang === 'zh' ? `${(n / 1e8).toLocaleString()} 亿 token` : `${(n / 1e9).toLocaleString()}B tokens`
}
