import { ReactNode } from 'react'
import { MODELS } from '../nn/models'
import { ActivationKind } from '../nn/mlp'
import { convAt, poolAt, unflattenIndex } from '../nn/cnn'
import { NodeRef, useStore, useT } from '../store'
import { DenseArch, FlattenSlot, cnnSlot } from '../scene/layout'
import { layerNameOf } from './layerName'
import { NumGrid, cellStyle, fmt } from './valueCell'

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
  let subtitle: string
  if (sel.space === 'vector') {
    subtitle = `${t('panel.neuron')} #${sel.index + 1}`
  } else if (arch === 'llm') {
    const chName = sel.layer === 1 ? ['Q', 'K', 'V'][sel.channel] + ' · ' : ''
    subtitle = `${chName}(${sel.row}, ${sel.col})`
  } else {
    subtitle = `${t('panel.channel')} ${sel.channel + 1} · (${sel.row}, ${sel.col})`
  }
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
        {arch === 'cnn' ? (
          <CNNDetail sel={sel} />
        ) : arch === 'llm' ? (
          <LLMDetail sel={sel} />
        ) : (
          <DenseDetail arch={arch} sel={sel} />
        )}
      </div>
    </aside>
  )
}

// ---------------------------------------------------------------- shared pieces

function ProductRows({
  prev,
  prevLabel,
  weights,
  weightHeader = 'w',
}: {
  prev: number[]
  prevLabel: (i: number) => string
  weights: number[]
  weightHeader?: string
}) {
  const scaleX = Math.max(...prev.map(Math.abs), 0.001)
  const scaleW = Math.max(...weights.map(Math.abs), 0.001)
  return (
    <div className="dense-table">
      <div className="dense-row dense-head">
        <span />
        <span>x</span>
        <span>{weightHeader}</span>
        <span>{weightHeader}·x</span>
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
  )
}

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
  const sum = z - bias
  return (
    <>
      <section>
        <h4>
          {t('panel.inputs')} × {t('panel.weight')}
        </h4>
        <ProductRows prev={prev} prevLabel={prevLabel} weights={weights} />
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

// ---------------------------------------------------------------- MLP / TEXT

function DenseDetail({ arch, sel }: { arch: DenseArch; sel: NodeRef }): ReactNode {
  const mlpInput = useStore((s) => s.mlpInput)
  const mlpTrace = useStore((s) => s.mlpTrace)
  const textFeatures = useStore((s) => s.textFeatures)
  const textTrace = useStore((s) => s.textTrace)
  const textRaw = useStore((s) => s.textRaw)
  const t = useT()
  if (sel.space !== 'vector') return null
  const model = MODELS[arch].model
  const input = arch === 'mlp' ? mlpInput : textFeatures
  const trace = arch === 'mlp' ? mlpTrace : textTrace
  const inputName = (i: number) => (arch === 'mlp' ? t('feature.n', { n: i + 1 }) : t(`textfeat.${i}`))

  if (sel.layer === -1) {
    return (
      <>
        {arch === 'text' && (
          <section>
            <h4>{t('panel.rawText')}</h4>
            <div className="raw-text-quote">“{textRaw}”</div>
          </section>
        )}
        <section>
          <h4>{inputName(sel.index)}</h4>
          <div className="big-value num">{fmt(input[sel.index])}</div>
        </section>
      </>
    )
  }

  const layer = model.layers[sel.layer]
  const prev = sel.layer === 0 ? input : trace[sel.layer - 1].a
  const isOutput = sel.layer === model.layers.length - 1
  return (
    <DenseComputation
      prev={prev}
      prevLabel={(i) => (sel.layer === 0 ? inputName(i) : `#${i + 1}`)}
      weights={layer.weights[sel.index]}
      bias={layer.biases[sel.index]}
      z={trace[sel.layer].z[sel.index]}
      a={trace[sel.layer].a[sel.index]}
      activation={layer.activation}
      probs={isOutput ? trace[sel.layer].a : undefined}
      probLabels={
        isOutput ? trace[sel.layer].a.map((_, i) => t(`class.${arch}.${i}`)) : undefined
      }
    />
  )
}

// ---------------------------------------------------------------- CNN

function CNNDetail({ sel }: { sel: NodeRef }): ReactNode {
  const input = useStore((s) => s.cnnInput)
  const trace = useStore((s) => s.cnnTrace)
  const t = useT()
  const model = MODELS.cnn.model

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

  if (sel.space === 'vector' && (sel.layer === 3 || sel.layer === 4)) {
    const def = model.layers[sel.layer]
    if (def.type !== 'dense') return null
    const prevStep = trace[sel.layer - 1]
    const curStep = trace[sel.layer]
    if (prevStep.kind !== 'vector' || curStep.kind !== 'vector' || !curStep.z) return null
    const isOutput = sel.layer === model.layers.length - 1
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

// ---------------------------------------------------------------- LLM

function LLMDetail({ sel }: { sel: NodeRef }): ReactNode {
  const trace = useStore((s) => s.llmTrace)
  const t = useT()
  const model = MODELS.llm.model
  const show = (c: string) => (c === ' ' ? '␣' : c)

  // token tile
  if (sel.space === 'grid' && sel.layer === -1) {
    return (
      <section>
        <h4>
          {t('panel.token')} #{sel.col + 1}
        </h4>
        <div className="big-value num">
          ‘{show(trace.chars[sel.col])}’ <span className="muted">(id {trace.ids[sel.col]})</span>
        </div>
      </section>
    )
  }

  // embedding cell: E[token][k] + P[pos][k]
  if (sel.space === 'grid' && sel.layer === 0) {
    const id = trace.ids[sel.row]
    const e = model.E[id][sel.col]
    const p = model.P[sel.row][sel.col]
    return (
      <section>
        <h4>
          x = E[‘{show(trace.chars[sel.row])}’][{sel.col}] + P[{sel.row}][{sel.col}]
        </h4>
        <div className="calc-chain">
          <span>
            E = <b className="num">{fmt(e)}</b>
          </span>
          <span>
            P = <b className="num">{fmt(p)}</b>
          </span>
          <span>
            → x = <b className="num accent">{fmt(trace.X[sel.row][sel.col])}</b>
          </span>
        </div>
      </section>
    )
  }

  // Q/K/V cell: x_i · W column
  if (sel.space === 'grid' && sel.layer === 1) {
    const W = [model.Wq, model.Wk, model.Wv][sel.channel]
    const name = ['Q', 'K', 'V'][sel.channel]
    const mats = [trace.Q, trace.K, trace.V][sel.channel]
    const weights = model.E[0].map((_, m) => W[m][sel.col])
    return (
      <>
        <section>
          <h4>
            {name}[{sel.row}][{sel.col}] = x{sel.row} · W{name.toLowerCase()}[:, {sel.col}]
          </h4>
          <ProductRows
            prev={trace.X[sel.row]}
            prevLabel={(m) => `x${sel.row}[${m}]`}
            weights={weights}
          />
        </section>
        <section>
          <h4>{t('panel.sum')}</h4>
          <div className="calc-chain">
            <span>
              Σ = <b className="num accent">{fmt(mats[sel.row][sel.col])}</b>
            </span>
          </div>
        </section>
      </>
    )
  }

  // attention cell (i, j)
  if (sel.space === 'grid' && sel.layer === 2) {
    const i = sel.row
    const j = sel.col
    if (j > i) {
      return (
        <section>
          <h4>{t('llm.masked')}</h4>
          <p className="explain-text">{t('llm.maskedNote')}</p>
        </section>
      )
    }
    const products = trace.Q[i].map((q, m) => q * trace.K[j][m])
    const rowScores = trace.S[i].slice(0, i + 1)
    return (
      <>
        <section>
          <h4>
            q{i} · k{j} / √{model.d}
          </h4>
          <ProductRows
            prev={trace.Q[i]}
            prevLabel={(m) => `dim ${m}`}
            weights={trace.K[j]}
            weightHeader="k"
          />
          <div className="calc-chain" style={{ marginTop: 8 }}>
            <span>
              Σ = <b className="num">{fmt(products.reduce((s, v) => s + v, 0))}</b>
            </span>
            <span>
              ÷ √{model.d} → {t('llm.score')} = <b className="num">{fmt(trace.S[i][j])}</b>
            </span>
          </div>
        </section>
        <section>
          <h4>
            softmax → {t('llm.attnWeight')} (‘{show(trace.chars[i])}’ → ‘{show(trace.chars[j])}’)
          </h4>
          <div className="calc-chain">
            <span>
              A[{i}][{j}] = e<sup>{fmt(trace.S[i][j], 2)}</sup> / Σ<sub>j≤{i}</sub> ={' '}
              <b className="num accent">{fmt(trace.A[i][j])}</b>
            </span>
          </div>
          <div className="prob-list" style={{ marginTop: 8 }}>
            {rowScores.map((_, jj) => (
              <div className="prob-row" key={jj}>
                <span className="prob-name mono-char">‘{show(trace.chars[jj])}’</span>
                <span className="prob-bar">
                  <span style={{ width: `${Math.max(2, trace.A[i][jj] * 100)}%` }} />
                </span>
                <span className="num">{(trace.A[i][jj] * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      </>
    )
  }

  // z cell: Σ_j A[i][j] · V[j][k]
  if (sel.space === 'grid' && sel.layer === 3) {
    const i = sel.row
    const k = sel.col
    const aRow = trace.A[i].slice(0, i + 1)
    const vCol = aRow.map((_, j) => trace.V[j][k])
    return (
      <>
        <section>
          <h4>
            z[{i}][{k}] = Σ<sub>j</sub> A[{i}][j] · V[j][{k}]
          </h4>
          <ProductRows
            prev={aRow}
            prevLabel={(j) => `A[${i}][${j}]`}
            weights={vCol}
            weightHeader="V"
          />
        </section>
        <section>
          <h4>{t('panel.sum')}</h4>
          <div className="calc-chain">
            <span>
              Σ = <b className="num accent">{fmt(trace.Z[i][k])}</b>
            </span>
          </div>
        </section>
      </>
    )
  }

  // ffn cell
  if (sel.space === 'grid' && sel.layer === 4) {
    const i = sel.row
    const k = sel.col
    const weights = trace.Z[i].map((_, m) => model.W1[m][k])
    return (
      <DenseComputation
        prev={trace.Z[i]}
        prevLabel={(m) => `z[${i}][${m}]`}
        weights={weights}
        bias={model.b1[k]}
        z={trace.Hpre[i][k]}
        a={trace.H[i][k]}
        activation="relu"
      />
    )
  }

  // output logit / next-char probability
  if (sel.space === 'vector' && sel.layer === 5) {
    const T = trace.ids.length
    const vIdx = sel.index
    const weights = trace.H[T - 1].map((_, k) => model.W2[k][vIdx])
    const top = [...trace.probs.keys()].sort((a, b) => trace.probs[b] - trace.probs[a]).slice(0, 6)
    return (
      <>
        <section>
          <h4>
            {t('llm.next')}: ‘{show(model.vocab[vIdx])}’
          </h4>
        </section>
        <DenseComputation
          prev={trace.H[T - 1]}
          prevLabel={(k) => `h[${k}]`}
          weights={weights}
          bias={model.b2[vIdx]}
          z={trace.U[T - 1][vIdx]}
          a={trace.probs[vIdx]}
          activation="softmax"
        />
        <section>
          <h4>{t('llm.topCandidates')}</h4>
          <div className="prob-list">
            {top.map((i) => (
              <div className="prob-row" key={i}>
                <span className="prob-name mono-char">‘{show(model.vocab[i])}’</span>
                <span className="prob-bar">
                  <span style={{ width: `${Math.max(2, trace.probs[i] * 100)}%` }} />
                </span>
                <span className="num">{(trace.probs[i] * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      </>
    )
  }

  return null
}
