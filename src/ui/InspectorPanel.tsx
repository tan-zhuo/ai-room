import { ReactNode } from 'react'
import { MODELS } from '../nn/models'
import { ActivationKind } from '../nn/mlp'
import { convAt, poolAt, unflattenIndex } from '../nn/cnn'
import { NodeRef, useStore, useT } from '../store'
import { FlattenSlot, cnnSlot } from '../scene/layout'
import { NumGrid, cellStyle, fmt } from './valueCell'

function layerNameOf(arch: 'mlp' | 'cnn', layer: number, t: (k: string, p?: Record<string, string | number>) => string): string {
  if (layer === -1) return t('layer.input')
  if (arch === 'mlp') {
    const last = MODELS.mlp.model.layers.length - 1
    return layer === last ? t('layer.output') : t('layer.hidden', { n: layer + 1 })
  }
  const def = MODELS.cnn.model.layers[layer]
  const last = MODELS.cnn.model.layers.length - 1
  if (layer === last) return t('layer.output')
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

const ACT_LABEL: Record<ActivationKind, string> = {
  relu: 'ReLU',
  tanh: 'tanh',
  sigmoid: 'sigmoid',
  softmax: 'softmax',
  linear: 'linear',
}

export function InspectorPanel() {
  const arch = useStore((s) => s.arch)
  const sel = useStore((s) => s.selected)
  const step = useStore((s) => s.step)
  const select = useStore((s) => s.select)
  const t = useT()

  if (!sel) return null

  const title = layerNameOf(arch, sel.layer, t)
  const subtitle =
    sel.space === 'vector'
      ? `${t('panel.neuron')} #${sel.index + 1}`
      : `${t('panel.channel')} ${sel.channel + 1} · (${sel.row}, ${sel.col})`
  const notYet = sel.layer >= 0 && step <= sel.layer

  return (
    <aside className="inspector">
      <header className="inspector-head">
        <div>
          <div className="inspector-title">{title}</div>
          <div className="inspector-sub">{subtitle}</div>
        </div>
        <button className="icon-btn" onClick={() => select(null)} title={t('panel.close')}>
          ×
        </button>
      </header>
      {notYet && <div className="not-yet">{t('panel.notYet')}</div>}
      <div className="inspector-body">
        {arch === 'mlp' ? <MLPDetail sel={sel} /> : <CNNDetail sel={sel} />}
      </div>
    </aside>
  )
}

// ---------------------------------------------------------------- shared pieces

function DenseComputation({
  prev,
  prevLabel,
  weights,
  bias,
  z,
  a,
  activation,
  probs,
  probLabels,
}: {
  prev: number[]
  prevLabel: (i: number) => string
  weights: number[]
  bias: number
  z: number
  a: number
  activation: ActivationKind
  probs?: number[]
  probLabels?: string[]
}) {
  const t = useT()
  const scaleX = Math.max(...prev.map(Math.abs), 0.001)
  const scaleW = Math.max(...weights.map(Math.abs), 0.001)
  const sum = z - bias
  return (
    <>
      <section>
        <h4>
          {t('panel.inputs')} × {t('panel.weight')}
        </h4>
        <div className="dense-table">
          <div className="dense-row dense-head">
            <span />
            <span>x</span>
            <span>w</span>
            <span>w·x</span>
          </div>
          <div className="dense-rows">
            {prev.map((x, i) => (
              <div className="dense-row" key={i}>
                <span className="dense-label">{prevLabel(i)}</span>
                <span className="num" style={cellStyle(x, scaleX)}>
                  {fmt(x)}
                </span>
                <span className="num" style={cellStyle(weights[i], scaleW)}>
                  {fmt(weights[i])}
                </span>
                <span className="num strong" style={cellStyle(weights[i] * x, scaleW * scaleX)}>
                  {fmt(weights[i] * x)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>
      <section>
        <h4>{t('panel.sum')}</h4>
        <div className="calc-chain">
          <span>
            Σ w·x = <b className="num">{fmt(sum)}</b>
          </span>
          <span>
            + {t('panel.bias')} <b className="num">{fmt(bias)}</b>
          </span>
          <span>
            → z = <b className="num">{fmt(z)}</b>
          </span>
        </div>
      </section>
      <section>
        <h4>
          {t('panel.activation')} · {ACT_LABEL[activation]}
        </h4>
        {activation === 'softmax' ? (
          <div className="calc-chain">
            <span>
              p = e<sup>z</sup> / Σ e<sup>z</sup> = <b className="num">{fmt(a)}</b>
            </span>
            <span className="muted">({t('panel.softmaxNote')})</span>
          </div>
        ) : (
          <div className="calc-chain">
            <span>
              {ACT_LABEL[activation]}({fmt(z)}) = <b className="num accent">{fmt(a)}</b>
            </span>
          </div>
        )}
      </section>
      {probs && probLabels && (
        <section>
          <h4>{t('panel.probabilities')}</h4>
          <div className="prob-list">
            {probs.map((p, i) => (
              <div className="prob-row" key={i}>
                <span className="prob-name">{probLabels[i]}</span>
                <span className="prob-bar">
                  <span style={{ width: `${Math.max(2, p * 100)}%` }} />
                </span>
                <span className="num">{(p * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </>
  )
}

// ---------------------------------------------------------------- MLP

function MLPDetail({ sel }: { sel: NodeRef }): ReactNode {
  const input = useStore((s) => s.mlpInput)
  const trace = useStore((s) => s.mlpTrace)
  const t = useT()
  if (sel.space !== 'vector') return null
  const model = MODELS.mlp.model

  if (sel.layer === -1) {
    return (
      <section>
        <h4>{t('feature.n', { n: sel.index + 1 })}</h4>
        <div className="big-value num">{fmt(input[sel.index])}</div>
      </section>
    )
  }

  const layer = model.layers[sel.layer]
  const prev = sel.layer === 0 ? input : trace[sel.layer - 1].a
  const isOutput = sel.layer === model.layers.length - 1
  return (
    <DenseComputation
      prev={prev}
      prevLabel={(i) => (sel.layer === 0 ? t('feature.n', { n: i + 1 }) : `#${i + 1}`)}
      weights={layer.weights[sel.index]}
      bias={layer.biases[sel.index]}
      z={trace[sel.layer].z[sel.index]}
      a={trace[sel.layer].a[sel.index]}
      activation={layer.activation}
      probs={isOutput ? trace[sel.layer].a : undefined}
      probLabels={isOutput ? trace[sel.layer].a.map((_, i) => t(`class.mlp.${i}`)) : undefined}
    />
  )
}

// ---------------------------------------------------------------- CNN

function CNNDetail({ sel }: { sel: NodeRef }): ReactNode {
  const input = useStore((s) => s.cnnInput)
  const trace = useStore((s) => s.cnnTrace)
  const t = useT()
  const model = MODELS.cnn.model

  // input pixel
  if (sel.space === 'grid' && sel.layer === -1) {
    return (
      <section>
        <h4>
          {t('panel.pixel')} ({sel.row}, {sel.col})
        </h4>
        <div className="big-value num">{fmt(input[sel.channel][sel.row][sel.col])}</div>
      </section>
    )
  }

  // conv pixel: the full convolution
  if (sel.space === 'grid' && sel.layer === 0) {
    const def = model.layers[0]
    if (def.type !== 'conv') return null
    const d = convAt(input, def, sel.channel, sel.row, sel.col)
    const kScale = Math.max(...d.kernel.flat(2).map(Math.abs), 0.001)
    return (
      <>
        <section>
          <h4>
            {t('panel.patch')} · ({sel.row}..{sel.row + 2}, {sel.col}..{sel.col + 2})
          </h4>
          <NumGrid values={d.patch[0]} scale={1} />
        </section>
        <section>
          <h4>
            {t('panel.kernel')} · {t('panel.channel')} {sel.channel + 1}
          </h4>
          <NumGrid values={d.kernel[0]} scale={kScale} />
        </section>
        <section>
          <h4>{t('panel.products')}</h4>
          <NumGrid values={d.products[0]} scale={kScale} />
        </section>
        <section>
          <h4>{t('panel.sum')}</h4>
          <div className="calc-chain">
            <span>
              Σ = <b className="num">{fmt(d.sum)}</b>
            </span>
            <span>
              + {t('panel.bias')} <b className="num">{fmt(d.bias)}</b>
            </span>
            <span>
              → z = <b className="num">{fmt(d.z)}</b>
            </span>
            <span>
              ReLU → <b className="num accent">{fmt(d.a)}</b>
            </span>
          </div>
        </section>
      </>
    )
  }

  // pool pixel: max over its window
  if (sel.space === 'grid' && sel.layer === 1) {
    const conv = trace[0]
    if (conv.kind !== 'tensor') return null
    const def = model.layers[1]
    const size = def.type === 'pool' ? def.size : 2
    const d = poolAt(conv.out, size, sel.channel, sel.row, sel.col)
    return (
      <>
        <section>
          <h4>{t('panel.window')}</h4>
          <NumGrid
            values={d.values}
            scale={Math.max(...d.values.flat().map(Math.abs), 0.001)}
            highlight={{ row: d.argRow, col: d.argCol }}
          />
        </section>
        <section>
          <h4>{t('panel.max')}</h4>
          <div className="big-value num accent">{fmt(d.max)}</div>
        </section>
      </>
    )
  }

  // flatten node: pure copy
  if (sel.space === 'vector' && sel.layer === 2) {
    const flat = trace[2]
    if (flat.kind !== 'vector') return null
    const src = unflattenIndex((cnnSlot(2) as FlattenSlot).srcShape, sel.index)
    return (
      <section>
        <h4>{t('panel.mapsTo')}</h4>
        <div className="calc-chain">
          <span>
            {t('panel.featureMap')} {src.channel + 1} · ({src.row}, {src.col})
          </span>
          <span>
            → <b className="num accent">{fmt(flat.a[sel.index])}</b>
          </span>
        </div>
      </section>
    )
  }

  // dense layers
  if (sel.space === 'vector' && (sel.layer === 3 || sel.layer === 4)) {
    const def = model.layers[sel.layer]
    if (def.type !== 'dense') return null
    const prevStep = trace[sel.layer - 1]
    const curStep = trace[sel.layer]
    if (prevStep.kind !== 'vector' || curStep.kind !== 'vector' || !curStep.z) return null
    const isOutput = sel.layer === 4
    return (
      <DenseComputation
        prev={prevStep.a}
        prevLabel={(i) => `#${i + 1}`}
        weights={def.layer.weights[sel.index]}
        bias={def.layer.biases[sel.index]}
        z={curStep.z[sel.index]}
        a={curStep.a[sel.index]}
        activation={def.layer.activation}
        probs={isOutput ? curStep.a : undefined}
        probLabels={isOutput ? curStep.a.map((_, i) => t(`class.cnn.${i}`)) : undefined}
      />
    )
  }

  return null
}
