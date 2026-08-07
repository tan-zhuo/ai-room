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
  moe?: { experts: number; topK: number }
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
    moe: { experts: 256, topK: 8 },
    desc: 'dsv3',
  },
]

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
  if (n >= 1e9) {
    return lang === 'zh' ? `${(n / 1e8).toLocaleString()} 亿` : `${(n / 1e9).toLocaleString()}B`
  }
  if (n >= 1e6) return lang === 'zh' ? `${(n / 1e4).toLocaleString()} 万` : `${(n / 1e6).toLocaleString()}M`
  return n.toLocaleString()
}

export function fmtTokens(n: number | undefined, lang: string): string {
  if (!n) return '—'
  if (n >= 1e12) return lang === 'zh' ? `${(n / 1e12).toLocaleString()} 万亿 token` : `${(n / 1e12).toLocaleString()}T tokens`
  return lang === 'zh' ? `${(n / 1e8).toLocaleString()} 亿 token` : `${(n / 1e9).toLocaleString()}B tokens`
}
