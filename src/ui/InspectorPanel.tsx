import { ReactNode } from 'react'
import { MODELS, getVAETask } from '../nn/models'
import { ActivationKind } from '../nn/mlp'
import { convAt, poolAt, unflattenIndex } from '../nn/cnn'
import { NodeRef, useStore, useT } from '../store'
import { DenseArch, FlattenSlot, cnnSlot, diffSlot, llmStageKind } from '../scene/layout'
import { layerNameOf } from './layerName'
import { NumGrid, cellStyle, fmt } from './valueCell'

const ACT_LABEL: Record<ActivationKind, string> = {
  relu: 'ReLU',
  leaky: 'LeakyReLU',
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
    const kind = llmStageKind(sel.layer)
    let chName = ''
    if (kind === 'qkv') chName = ['Q', 'K', 'V'][sel.channel] + ' · '
    else if (kind === 'attn') chName = t('llm.head', { n: sel.channel + 1 }) + ' · '
    else if (kind === 'experts') chName = `E${sel.channel + 1} · `
    subtitle = `${chName}(${sel.row}, ${sel.col})`
  } else if (arch === 'lstm' && sel.space === 'grid' && sel.layer === 1) {
    subtitle = `${['f', 'i', 'g', 'o'][sel.channel]} · (${sel.row}, ${sel.col})`
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
        ) : arch === 'diff' ? (
          <DiffDetail sel={sel} />
        ) : arch === 'gan' ? (
          <GanDetail sel={sel} />
        ) : arch === 'gnn' ? (
          <GnnDetail sel={sel} />
        ) : arch === 'vit' ? (
          <VitDetail sel={sel} />
        ) : arch === 'mamba' ? (
          <MambaDetail sel={sel} />
        ) : arch === 'rnn' ? (
          <RNNDetail sel={sel} />
        ) : arch === 'lstm' ? (
          <LSTMDetail sel={sel} />
        ) : arch === 'ae' ? (
          <AEDetail sel={sel} />
        ) : arch === 'giant' ? null : (
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

// ---------------------------------------------------------------- RNN / LSTM / AE

function TopChars({ probs, vocab }: { probs: number[]; vocab: string[] }) {
  const t = useT()
  const show = (c: string) => (c === ' ' ? '␣' : c)
  const top = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 6)
  return (
    <section>
      <h4>{t('llm.topCandidates')}</h4>
      <div className="prob-list">
        {top.map((i) => (
          <div className="prob-row" key={i}>
            <span className="prob-name mono-char">‘{show(vocab[i])}’</span>
            <span className="prob-bar">
              <span style={{ width: `${Math.max(2, probs[i] * 100)}%` }} />
            </span>
            <span className="num">{(probs[i] * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function RNNDetail({ sel }: { sel: NodeRef }): ReactNode {
  const trace = useStore((s) => s.rnnTrace)
  const t = useT()
  const model = MODELS.rnn.model
  const show = (c: string) => (c === ' ' ? '␣' : c)

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
  if (sel.space === 'grid' && sel.layer === 0) {
    return (
      <section>
        <h4>
          E[‘{show(trace.chars[sel.row])}’][{sel.col}]
        </h4>
        <div className="big-value num">{fmt(trace.X[sel.row][sel.col])}</div>
      </section>
    )
  }
  if (sel.space === 'grid' && sel.layer === 1) {
    const tstep = sel.row
    const j = sel.col
    const wx = Array.from({ length: model.d }, (_, i) => model.Wx[i][j])
    const wh = Array.from({ length: model.h }, (_, i) => model.Wh[i][j])
    const hPrev = tstep > 0 ? trace.H[tstep - 1] : Array.from({ length: model.h }, () => 0)
    return (
      <>
        <section>
          <h4>x{tstep} · Wx[:, {j}]</h4>
          <ProductRows prev={trace.X[tstep]} prevLabel={(m) => `x[${m}]`} weights={wx} />
        </section>
        <section>
          <h4>
            h{tstep - 1} · Wh[:, {j}] {tstep === 0 && <span className="muted">(h₋₁ = 0)</span>}
          </h4>
          <ProductRows prev={hPrev} prevLabel={(m) => `h[${m}]`} weights={wh} />
        </section>
        <section>
          <h4>{t('panel.activation')} · tanh</h4>
          <div className="calc-chain">
            <span>
              Σ + b = <b className="num">{fmt(trace.Hpre[tstep][j])}</b>
            </span>
            <span>
              tanh → <b className="num accent">{fmt(trace.H[tstep][j])}</b>
            </span>
          </div>
        </section>
      </>
    )
  }
  if (sel.space === 'vector' && sel.layer === 2) {
    const T = trace.ids.length
    const weights = Array.from({ length: model.h }, (_, jj) => model.Wy[jj][sel.index])
    return (
      <>
        <DenseComputation
          prev={trace.H[T - 1]}
          prevLabel={(jj) => `h[${jj}]`}
          weights={weights}
          bias={model.by[sel.index]}
          z={trace.U[sel.index]}
          a={trace.probs[sel.index]}
          activation="softmax"
        />
        <TopChars probs={trace.probs} vocab={model.vocab} />
      </>
    )
  }
  return null
}

function LSTMDetail({ sel }: { sel: NodeRef }): ReactNode {
  const trace = useStore((s) => s.lstmTrace)
  const t = useT()
  const model = MODELS.lstm.model
  const show = (c: string) => (c === ' ' ? '␣' : c)
  const { d, h } = model

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
  if (sel.space === 'grid' && sel.layer === 0) {
    return (
      <section>
        <h4>
          E[‘{show(trace.chars[sel.row])}’][{sel.col}]
        </h4>
        <div className="big-value num">{fmt(trace.X[sel.row][sel.col])}</div>
      </section>
    )
  }
  if (sel.space === 'grid' && sel.layer === 1) {
    const tstep = sel.row
    const j = sel.col
    const gate = sel.channel
    const name = ['f', 'i', 'g', 'o'][gate]
    const act = gate === 2 ? 'tanh' : 'σ'
    const W = [model.Wf, model.Wi, model.Wg, model.Wo][gate]
    const b = [model.bf, model.bi, model.bg, model.bo][gate][j]
    const hPrev = tstep > 0 ? trace.H[tstep - 1] : Array.from({ length: h }, () => 0)
    const z = [...trace.X[tstep], ...hPrev]
    const weights = Array.from({ length: d + h }, (_, k) => W[k][j])
    const val = [trace.F, trace.I, trace.G, trace.O][gate][tstep][j]
    return (
      <>
        <section>
          <h4>
            {name} = {act}([x{tstep}, h{tstep - 1}] · W{name}[:, {j}] + b)
          </h4>
          <ProductRows
            prev={z}
            prevLabel={(k) => (k < d ? `x[${k}]` : `h[${k - d}]`)}
            weights={weights}
          />
        </section>
        <section>
          <h4>{t('panel.activation')}</h4>
          <div className="calc-chain">
            <span>
              + b <b className="num">{fmt(b)}</b>
            </span>
            <span>
              {act} → <b className="num accent">{fmt(val)}</b>
            </span>
          </div>
        </section>
      </>
    )
  }
  if (sel.space === 'grid' && sel.layer === 2) {
    const tstep = sel.row
    const j = sel.col
    const cPrev = tstep > 0 ? trace.C[tstep - 1][j] : 0
    const f = trace.F[tstep][j]
    const i = trace.I[tstep][j]
    const g = trace.G[tstep][j]
    return (
      <section>
        <h4>c = f ⊙ c′ + i ⊙ g</h4>
        <div className="calc-chain">
          <span>
            f({fmt(f)}) × c′({fmt(cPrev)}) = <b className="num">{fmt(f * cPrev)}</b>
          </span>
          <span>
            i({fmt(i)}) × g({fmt(g)}) = <b className="num">{fmt(i * g)}</b>
          </span>
          <span>
            → c = <b className="num accent">{fmt(trace.C[tstep][j])}</b>
          </span>
        </div>
      </section>
    )
  }
  if (sel.space === 'grid' && sel.layer === 3) {
    const tstep = sel.row
    const j = sel.col
    return (
      <section>
        <h4>h = o ⊙ tanh(c)</h4>
        <div className="calc-chain">
          <span>
            o = <b className="num">{fmt(trace.O[tstep][j])}</b>
          </span>
          <span>
            tanh(c) = <b className="num">{fmt(trace.Ct[tstep][j])}</b>
          </span>
          <span>
            → h = <b className="num accent">{fmt(trace.H[tstep][j])}</b>
          </span>
        </div>
      </section>
    )
  }
  if (sel.space === 'vector' && sel.layer === 4) {
    const T = trace.ids.length
    const weights = Array.from({ length: h }, (_, jj) => model.Wy[jj][sel.index])
    return (
      <>
        <DenseComputation
          prev={trace.H[T - 1]}
          prevLabel={(jj) => `h[${jj}]`}
          weights={weights}
          bias={model.by[sel.index]}
          z={trace.U[sel.index]}
          a={trace.probs[sel.index]}
          activation="softmax"
        />
        <TopChars probs={trace.probs} vocab={model.vocab} />
      </>
    )
  }
  return null
}

function VAEDetail({ sel }: { sel: NodeRef }): ReactNode {
  const input = useStore((s) => s.aeInput)
  const trace = useStore((s) => s.vaeTrace)
  const t = useT()
  const task = getVAETask()
  const model = task.model
  if (!trace) return null

  if (sel.space === 'grid' && sel.layer === -1) {
    return (
      <section>
        <h4>
          {t('panel.pixel')} ({sel.row}, {sel.col})
        </h4>
        <div className="big-value num">{fmt(input[0][sel.row][sel.col])}</div>
      </section>
    )
  }
  if (sel.space === 'vector' && sel.layer === 0 && trace.h.a.length) {
    return (
      <DenseComputation
        prev={input[0].flat()}
        prevLabel={(i) => `px(${Math.floor(i / task.n)},${i % task.n})`}
        weights={model.enc.weights[sel.index]}
        bias={model.enc.biases[sel.index]}
        z={trace.h.z[sel.index]}
        a={trace.h.a[sel.index]}
        activation="relu"
      />
    )
  }
  if (sel.space === 'grid' && sel.layer === 1 && trace.h.a.length) {
    const head = sel.col === 0 ? model.muHead : model.lvHead
    const val = sel.col === 0 ? trace.mu[sel.row] : trace.logvar[sel.row]
    return (
      <>
        <section>
          <h4>
            {sel.col === 0 ? 'μ' : 'log σ²'}[{sel.row}]
          </h4>
          <ProductRows prev={trace.h.a} prevLabel={(i) => `#${i + 1}`} weights={head.weights[sel.row]} />
        </section>
        <section>
          <h4>{t('panel.sum')}</h4>
          <div className="calc-chain">
            <span>
              Σ + b = <b className="num accent">{fmt(val)}</b>
            </span>
            {sel.col === 1 && (
              <span>
                σ = e^(z/2) = <b className="num">{fmt(trace.sigma[sel.row])}</b>
              </span>
            )}
          </div>
        </section>
      </>
    )
  }
  if (sel.space === 'vector' && sel.layer === 2) {
    const i = sel.index
    return (
      <>
        <section>
          <h4>z = μ + ε · σ</h4>
          <div className="calc-chain">
            <span>
              μ = <b className="num">{fmt(trace.mu[i])}</b>
            </span>
            <span>
              σ = <b className="num">{fmt(trace.sigma[i])}</b>
            </span>
            <span>
              ε = <b className="num">{fmt(trace.eps[i])}</b>
            </span>
            <span>
              → z = <b className="num accent">{fmt(trace.z[i])}</b>
            </span>
          </div>
        </section>
        <section>
          <h4>{t('panel.activation')}</h4>
          <p className="explain-text">{t('vae.reparamNote')}</p>
        </section>
      </>
    )
  }
  if (sel.space === 'vector' && sel.layer === 3) {
    return (
      <DenseComputation
        prev={trace.z}
        prevLabel={(i) => `z[${i}]`}
        weights={model.dec.weights[sel.index]}
        bias={model.dec.biases[sel.index]}
        z={trace.d.z[sel.index]}
        a={trace.d.a[sel.index]}
        activation="relu"
      />
    )
  }
  if (sel.space === 'grid' && sel.layer === 4) {
    const idx = sel.row * task.n + sel.col
    const original = input[0][sel.row][sel.col]
    const recon = trace.o.a[idx]
    return (
      <>
        <DenseComputation
          prev={trace.d.a}
          prevLabel={(i) => `#${i + 1}`}
          weights={model.out.weights[idx]}
          bias={model.out.biases[idx]}
          z={trace.o.z[idx]}
          a={recon}
          activation="sigmoid"
        />
        {!trace.generated && (
          <section>
            <h4>{t('ae.compare')}</h4>
            <div className="calc-chain">
              <span>
                {t('ae.original')}: <b className="num">{fmt(original)}</b>
              </span>
              <span>
                {t('ae.recon')}: <b className="num accent">{fmt(recon)}</b>
              </span>
            </div>
          </section>
        )}
      </>
    )
  }
  return null
}

function AEDetail({ sel }: { sel: NodeRef }): ReactNode {
  const input = useStore((s) => s.aeInput)
  const trace = useStore((s) => s.aeTrace)
  const variant = useStore((s) => s.aeVariant)
  const t = useT()
  const task = MODELS.ae
  const model = task.model
  if (variant === 'vae') return <VAEDetail sel={sel} />

  if (sel.space === 'grid' && sel.layer === -1) {
    return (
      <section>
        <h4>
          {t('panel.pixel')} ({sel.row}, {sel.col})
        </h4>
        <div className="big-value num">{fmt(input[0][sel.row][sel.col])}</div>
      </section>
    )
  }
  if (sel.space === 'vector' && sel.layer >= 0 && sel.layer <= 2) {
    const layer = model.layers[sel.layer]
    const prev = sel.layer === 0 ? input[0].flat() : trace[sel.layer - 1].a
    return (
      <DenseComputation
        prev={prev}
        prevLabel={(i) =>
          sel.layer === 0 ? `px(${Math.floor(i / task.n)},${i % task.n})` : `#${i + 1}`
        }
        weights={layer.weights[sel.index]}
        bias={layer.biases[sel.index]}
        z={trace[sel.layer].z[sel.index]}
        a={trace[sel.layer].a[sel.index]}
        activation={layer.activation}
      />
    )
  }
  if (sel.space === 'grid' && sel.layer === 3) {
    const idx = sel.row * task.n + sel.col
    const layer = model.layers[3]
    const original = input[0][sel.row][sel.col]
    const reconstructed = trace[3].a[idx]
    return (
      <>
        <DenseComputation
          prev={trace[2].a}
          prevLabel={(i) => `#${i + 1}`}
          weights={layer.weights[idx]}
          bias={layer.biases[idx]}
          z={trace[3].z[idx]}
          a={reconstructed}
          activation={layer.activation}
        />
        <section>
          <h4>{t('ae.compare')}</h4>
          <div className="calc-chain">
            <span>
              {t('ae.original')}: <b className="num">{fmt(original)}</b>
            </span>
            <span>
              {t('ae.recon')}: <b className="num accent">{fmt(reconstructed)}</b>
            </span>
            <span>
              Δ = <b className="num">{fmt(Math.abs(original - reconstructed))}</b>
            </span>
          </div>
        </section>
      </>
    )
  }
  return null
}

// ---------------------------------------------------------------- Diffusion

function DiffDetail({ sel }: { sel: NodeRef }): ReactNode {
  const trace = useStore((s) => s.diffTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const task = MODELS.diff
  const model = task.model
  if (sel.space !== 'grid') return null

  const total = model.T
  const si = Math.min(step, total - 1)
  const cur = trace.steps[si]
  const idx = sel.row * task.n + sel.col

  if (sel.layer === -1) {
    return (
      <section>
        <h4>
          x_t({sel.row}, {sel.col}) · t = {cur.t}
        </h4>
        <div className="big-value num">{fmt(trace.xs[Math.min(step, total)][idx])}</div>
        <p className="explain-text" style={{ marginTop: 8 }}>
          {t('diff.xtNote')}
        </p>
      </section>
    )
  }
  if (sel.layer === 0) {
    const hS = diffSlot(0)
    const hIdx = sel.row * (hS.kind === 'grid' ? hS.cols : 8) + sel.col
    if (hIdx >= cur.h2.a.length) return <p className="explain-text">—</p>
    const weights = cur.h1.a.map((_, m) => model.l2.weights[hIdx][m])
    return (
      <DenseComputation
        prev={cur.h1.a}
        prevLabel={(m) => `h₁[${m}]`}
        weights={weights}
        bias={model.l2.biases[hIdx]}
        z={cur.h2.z[hIdx]}
        a={cur.h2.a[hIdx]}
        activation="relu"
      />
    )
  }
  if (sel.layer === 1) {
    return (
      <DenseComputation
        prev={cur.h2.a}
        prevLabel={(m) => `h₂[${m}]`}
        weights={model.out.weights[idx]}
        bias={model.out.biases[idx]}
        z={cur.pred.z[idx]}
        a={cur.pred.a[idx]}
        activation="tanh"
      />
    )
  }
  if (sel.layer === 2) {
    const beta = model.betas[cur.t - 1]
    const alpha = model.alphas[cur.t - 1]
    const aBar = model.alphaBars[cur.t - 1]
    const aBarPrev = cur.t > 1 ? model.alphaBars[cur.t - 2] : 1
    const c0 = (Math.sqrt(aBarPrev) * beta) / (1 - aBar)
    const ct = (Math.sqrt(alpha) * (1 - aBarPrev)) / (1 - aBar)
    const sigma = cur.t > 1 ? Math.sqrt(((1 - aBarPrev) / (1 - aBar)) * beta) : 0
    return (
      <>
        <section>
          <h4>x_(t−1) = c₀·x̂₀ + cₜ·x_t + σ·z</h4>
          <div className="calc-chain">
            <span>
              c₀({fmt(c0)}) × x̂₀({fmt(cur.pred.a[idx])}) = <b className="num">{fmt(c0 * cur.pred.a[idx])}</b>
            </span>
            <span>
              cₜ({fmt(ct)}) × x_t({fmt(cur.x[idx])}) = <b className="num">{fmt(ct * cur.x[idx])}</b>
            </span>
            <span>
              σ({fmt(sigma)}) × z({fmt(cur.z[idx])}) = <b className="num">{fmt(sigma * cur.z[idx])}</b>
            </span>
            <span>
              → <b className="num accent">{fmt(cur.xNext[idx])}</b>
            </span>
          </div>
        </section>
        <section>
          <h4>t = {cur.t}</h4>
          <p className="explain-text">{t('diff.stepNote')}</p>
        </section>
      </>
    )
  }
  return null
}

// ---------------------------------------------------------------- Mamba

function MambaDetail({ sel }: { sel: NodeRef }): ReactNode {
  const trace = useStore((s) => s.mambaTrace)
  const t = useT()
  const model = MODELS.mamba.model
  const showChar = (c: string) => (c === ' ' ? '␣' : c)

  if (sel.space === 'vector') {
    // output logits from the last gated output
    const last = trace.O[trace.O.length - 1]
    return (
      <DenseComputation
        prev={last}
        prevLabel={(m) => `o[${m}]`}
        weights={last.map((_, m) => model.Wout[m][sel.index])}
        bias={model.bout[sel.index]}
        z={0}
        a={trace.probs[sel.index]}
        activation="softmax"
      />
    )
  }
  if (sel.layer === -1) {
    return (
      <section>
        <h4>
          t = {sel.col} · “{showChar(trace.chars[sel.col])}”
        </h4>
        <p className="explain-text">{t('mamba.tokNote')}</p>
      </section>
    )
  }
  const ts = sel.row
  const i = sel.col
  if (sel.layer === 0) {
    return (
      <section>
        <h4>
          E[“{showChar(trace.chars[ts])}”][{i}]
        </h4>
        <div className="big-value num">{fmt(trace.X[ts][i])}</div>
      </section>
    )
  }
  if (sel.layer === 1) {
    return (
      <section>
        <h4>Δ[{ts}][{i}] = softplus(x·WΔ + b)</h4>
        <div className="big-value num">{fmt(trace.Delta[ts][i])}</div>
        <p className="explain-text" style={{ marginTop: 8 }}>
          {t('mamba.deltaNote')}
        </p>
      </section>
    )
  }
  if (sel.layer === 2) {
    const hRow = trace.H[ts][i]
    const hPrev = ts > 0 ? trace.H[ts - 1][i] : hRow.map(() => 0)
    const delta = trace.Delta[ts][i]
    return (
      <>
        <section>
          <h4>h[{i}][·] = ā ⊙ h′ + Δ·B·u</h4>
          <p className="explain-text">{t('mamba.scanNote', { d: fmt(delta) })}</p>
          <div style={{ height: 6 }} />
          <NumGrid values={[hPrev]} scale={Math.max(...hPrev.map(Math.abs), 1)} />
          <div style={{ height: 4 }} />
          <NumGrid values={[hRow]} scale={Math.max(...hRow.map(Math.abs), 1)} />
        </section>
        <section>
          <h4>y = h·C + D·u</h4>
          <div className="calc-chain">
            <span>
              Σ h·C = <b className="num">{fmt(trace.Y[ts][i] - model.Dskip[i] * trace.U[ts][i])}</b>
            </span>
            <span>
              + D·u = <b className="num">{fmt(model.Dskip[i] * trace.U[ts][i])}</b>
            </span>
            <span>
              → <b className="num accent">{fmt(trace.Y[ts][i])}</b>
            </span>
          </div>
        </section>
      </>
    )
  }
  if (sel.layer === 3) {
    return (
      <section>
        <h4>o = y ⊙ silu(z)</h4>
        <div className="calc-chain">
          <span>
            y = <b className="num">{fmt(trace.Y[ts][i])}</b>
          </span>
          <span>
            × silu(z) = <b className="num">{fmt(trace.G[ts][i])}</b>
          </span>
          <span>
            → <b className="num accent">{fmt(trace.O[ts][i])}</b>
          </span>
        </div>
        <p className="explain-text" style={{ marginTop: 8 }}>
          {t('mamba.gateNote')}
        </p>
      </section>
    )
  }
  return null
}

// ---------------------------------------------------------------- ViT

function VitDetail({ sel }: { sel: NodeRef }): ReactNode {
  const input = useStore((s) => s.vitInput)
  const trace = useStore((s) => s.vitTrace)
  const t = useT()
  const model = MODELS.vit.model

  if (sel.space === 'vector') {
    // classification head
    const c = sel.index
    return (
      <DenseComputation
        prev={trace.clsLn}
        prevLabel={(k) => `LN(CLS)[${k}]`}
        weights={trace.clsLn.map((_, k) => model.Wh[k][c])}
        bias={model.bh[c]}
        z={trace.logits[c]}
        a={trace.probs[c]}
        activation="softmax"
        probs={trace.probs}
        probLabels={[0, 1, 2, 3].map((i) => t(`class.cnn.${i}`))}
      />
    )
  }

  if (sel.layer === -1) {
    const pr = Math.floor(sel.row / model.p)
    const pc = Math.floor(sel.col / model.p)
    return (
      <section>
        <h4>
          px({sel.row}, {sel.col})
        </h4>
        <div className="big-value num">{fmt(input[0][sel.row][sel.col])}</div>
        <p className="explain-text" style={{ marginTop: 8 }}>
          {t('vit.pixelNote', { p: pr * model.grid + pc + 1 })}
        </p>
      </section>
    )
  }
  if (sel.layer === 0) {
    const tk = sel.row
    const j = sel.col
    if (tk === 0) {
      return (
        <section>
          <h4>[CLS] · dim {j}</h4>
          <div className="calc-chain">
            <span>
              cls[{j}] = <b className="num">{fmt(model.cls[j])}</b>
            </span>
            <span>
              + pos[0][{j}] = <b className="num">{fmt(model.pos[0][j])}</b>
            </span>
            <span>
              → <b className="num accent">{fmt(trace.X[0][j])}</b>
            </span>
          </div>
          <p className="explain-text" style={{ marginTop: 8 }}>
            {t('vit.clsNote')}
          </p>
        </section>
      )
    }
    const patch = trace.patches[tk - 1]
    return (
      <>
        <section>
          <h4>
            {t('vit.patchTitle', { n: tk })} · dim {j}
          </h4>
          <ProductRows prev={patch} prevLabel={(k) => `px${k}`} weights={patch.map((_, k) => model.Wp[k][j])} />
        </section>
        <section>
          <h4>{t('panel.sum')}</h4>
          <div className="calc-chain">
            <span>
              Σ + b = <b className="num">{fmt(trace.X[tk][j] - model.pos[tk][j])}</b>
            </span>
            <span>
              + pos[{tk}][{j}] {fmt(model.pos[tk][j])}
            </span>
            <span>
              → X = <b className="num accent">{fmt(trace.X[tk][j])}</b>
            </span>
          </div>
        </section>
      </>
    )
  }
  if (sel.layer === 1) {
    const h = sel.channel
    const i = sel.row
    const j = sel.col
    const dh = model.d / model.heads
    const o = h * dh
    let dot = 0
    for (let k = 0; k < dh; k++) dot += trace.Q[i][o + k] * trace.K[j][o + k]
    return (
      <section>
        <h4>
          A[{i}][{j}] · {t('llm.head', { n: h + 1 })}
        </h4>
        <div className="calc-chain">
          <span>
            q{i}·k{j} = <b className="num">{fmt(dot)}</b>
          </span>
          <span>/ √{dh} → softmax →</span>
          <span>
            <b className="num accent">{fmt(trace.att[h][i][j])}</b>
          </span>
        </div>
        <p className="explain-text" style={{ marginTop: 8 }}>
          {t('vit.attnNote')}
        </p>
      </section>
    )
  }
  if (sel.layer === 2) {
    const i = sel.row
    const j = sel.col
    return (
      <section>
        <h4>X₁ = X + MHA(LN(X))</h4>
        <div className="calc-chain">
          <span>
            X[{i}][{j}] = <b className="num">{fmt(trace.X[i][j])}</b>
          </span>
          <span>
            + Z[{i}][{j}] = <b className="num">{fmt(trace.Z[i][j])}</b>
          </span>
          <span>
            → <b className="num accent">{fmt(trace.X1[i][j])}</b>
          </span>
        </div>
      </section>
    )
  }
  if (sel.layer === 3) {
    const i = sel.row
    const j = sel.col
    return (
      <section>
        <h4>X₂ = X₁ + FFN(LN(X₁))</h4>
        <div className="calc-chain">
          <span>
            X₁[{i}][{j}] = <b className="num">{fmt(trace.X1[i][j])}</b>
          </span>
          <span>
            + F[{i}][{j}] = <b className="num">{fmt(trace.F[i][j])}</b>
          </span>
          <span>
            → <b className="num accent">{fmt(trace.X2[i][j])}</b>
          </span>
        </div>
        {i === 0 && (
          <p className="explain-text" style={{ marginTop: 8 }}>
            {t('vit.clsRowNote')}
          </p>
        )}
      </section>
    )
  }
  return null
}

// ---------------------------------------------------------------- GNN

function GnnDetail({ sel }: { sel: NodeRef }): ReactNode {
  const trace = useStore((s) => s.gnnTrace)
  const t = useT()
  const g = trace.graph
  if (sel.space !== 'vector') return null
  const i = sel.index
  const cls = (c: number) => `${t('gnn.community')} ${['A', 'B', 'C'][c]}`

  const neighbours = g.ahat[i]
    .map((w, j) => ({ j, w }))
    .filter(({ w }) => w > 0)

  const aggTable = (source: number[][], label: string) => (
    <section>
      <h4>{t('gnn.aggTitle')}</h4>
      <p className="explain-text">{t('gnn.aggNote')}</p>
      <div className="dense-table" style={{ marginTop: 6 }}>
        <div className="dense-row dense-head">
          <span />
          <span>Â</span>
          <span>{label}</span>
        </div>
        <div className="dense-rows">
          {neighbours.map(({ j, w }) => (
            <div className="dense-row" key={j}>
              <span className="dense-label">{j === i ? `${t('gnn.self')} #${j + 1}` : `#${j + 1}`}</span>
              <span className="num" style={cellStyle(w, 1)}>
                {fmt(w)}
              </span>
              <span className="num">{fmt(meanRow(source[j]))}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )

  if (sel.layer === -1) {
    return (
      <>
        <section>
          <h4>
            {t('gnn.nodeTitle')} #{i + 1} · {cls(g.labels[i])}
          </h4>
          <p className="explain-text">{t('gnn.featNote')}</p>
          <NumGrid values={[g.X[i]]} scale={Math.max(...g.X[i].map(Math.abs), 1)} />
        </section>
        <section>
          <h4>{t('gnn.degree')}</h4>
          <div className="big-value num">{neighbours.length - 1}</div>
        </section>
      </>
    )
  }
  if (sel.layer === 0) {
    return (
      <>
        {aggTable(g.X.map((r) => r), 'x̄')}
        <section>
          <h4>agg = Σ Â·x → z₁ = agg·W₁+b₁ → ReLU</h4>
          <NumGrid values={[trace.agg1[i]]} scale={Math.max(...trace.agg1[i].map(Math.abs), 1)} />
          <div style={{ height: 6 }} />
          <NumGrid values={[trace.H1[i]]} scale={Math.max(...trace.H1[i].map(Math.abs), 1)} />
        </section>
      </>
    )
  }
  if (sel.layer === 1) {
    return (
      <>
        {aggTable(trace.H1, 'h̄')}
        <section>
          <h4>agg = Σ Â·h → Z = agg·W₂+b₂</h4>
          <NumGrid values={[trace.agg2[i]]} scale={Math.max(...trace.agg2[i].map(Math.abs), 1)} />
          <div style={{ height: 6 }} />
          <NumGrid values={[trace.logits[i]]} scale={Math.max(...trace.logits[i].map(Math.abs), 1)} />
        </section>
      </>
    )
  }
  const correct = trace.pred[i] === g.labels[i]
  return (
    <>
      <section>
        <h4>{t('panel.probabilities')}</h4>
        <div className="prob-list">
          {trace.probs[i].map((p, c) => (
            <div className="prob-row" key={c}>
              <span className="prob-name">{cls(c)}</span>
              <span className="prob-bar">
                <span style={{ width: `${Math.max(2, p * 100)}%` }} />
              </span>
              <span className="num">{(p * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </section>
      <section>
        <h4>{t('gnn.verdict')}</h4>
        <p className="explain-text">
          {t('gnn.predIs')} <b>{cls(trace.pred[i])}</b> · {t('gnn.truthIs')} <b>{cls(g.labels[i])}</b> —{' '}
          {correct ? '✓' : '✗'}
        </p>
      </section>
    </>
  )
}

function meanRow(row: number[]): number {
  return row.reduce((s, v) => s + v, 0) / row.length
}

// ---------------------------------------------------------------- GAN

function GanDetail({ sel }: { sel: NodeRef }): ReactNode {
  const trace = useStore((s) => s.ganTrace)
  const t = useT()
  const task = MODELS.gan
  const model = task.model

  if (sel.space === 'vector') {
    if (sel.layer === -1) {
      return (
        <section>
          <h4>z[{sel.index}]</h4>
          <div className="big-value num">{fmt(trace.z[sel.index])}</div>
          <p className="explain-text" style={{ marginTop: 8 }}>
            {t('gan.zNote')}
          </p>
        </section>
      )
    }
    if (sel.layer === 0) {
      return (
        <DenseComputation
          prev={trace.z}
          prevLabel={(m) => `z[${m}]`}
          weights={model.g1.weights[sel.index]}
          bias={model.g1.biases[sel.index]}
          z={trace.g1.z[sel.index]}
          a={trace.g1.a[sel.index]}
          activation="relu"
        />
      )
    }
    if (sel.layer === 2) {
      return (
        <DenseComputation
          prev={trace.img.a}
          prevLabel={(m) => `G(z)[${m}]`}
          weights={model.d1.weights[sel.index]}
          bias={model.d1.biases[sel.index]}
          z={trace.d1.z[sel.index]}
          a={trace.d1.a[sel.index]}
          activation="leaky"
        />
      )
    }
    if (sel.layer === 3) {
      const isFake = sel.index === 0
      const d1 = isFake ? trace.d1 : trace.realD1
      const out = isFake ? trace.out : trace.realOut
      return (
        <>
          <DenseComputation
            prev={d1.a}
            prevLabel={(m) => `h[${m}]`}
            weights={model.d2.weights[0]}
            bias={model.d2.biases[0]}
            z={out.z[0]}
            a={out.a[0]}
            activation="sigmoid"
          />
          <section>
            <h4>{isFake ? 'D(G(z))' : 'D(x)'}</h4>
            <p className="explain-text">{t(isFake ? 'gan.fakeVerdictNote' : 'gan.realVerdictNote')}</p>
          </section>
        </>
      )
    }
    return null
  }
  const idx = sel.row * task.n + sel.col
  if (sel.layer === 1) {
    return (
      <DenseComputation
        prev={trace.g1.a}
        prevLabel={(m) => `h[${m}]`}
        weights={model.g2.weights[idx]}
        bias={model.g2.biases[idx]}
        z={trace.img.z[idx]}
        a={trace.img.a[idx]}
        activation="tanh"
      />
    )
  }
  if (sel.layer === 4) {
    return (
      <section>
        <h4>
          x({sel.row}, {sel.col})
        </h4>
        <div className="big-value num">{fmt(trace.real[idx])}</div>
        <p className="explain-text" style={{ marginTop: 8 }}>
          {t('gan.realNote')}
        </p>
      </section>
    )
  }
  return null
}

// ---------------------------------------------------------------- LLM

function AddNormDetail({
  which,
  row,
  col,
}: {
  which: 1 | 2
  row: number
  col: number
}): ReactNode {
  const trace = useStore((s) => s.llmTrace)
  const t = useT()
  const model = MODELS.llm.model
  const a = which === 1 ? trace.X[row][col] : trace.R1[row][col]
  const b =
    which === 1
      ? trace.Z[row][col]
      : MODELS.llm.model.moe
        ? trace.Y[row][col]
        : trace.F[row][col]
  const res = which === 1 ? trace.res1[row][col] : trace.res2[row][col]
  const mu = which === 1 ? trace.mu1[row] : trace.mu2[row]
  const sig = which === 1 ? trace.sig1[row] : trace.sig2[row]
  const g = which === 1 ? model.g1[col] : model.g2[col]
  const be = which === 1 ? model.be1[col] : model.be2[col]
  const out = which === 1 ? trace.R1[row][col] : trace.R2[row][col]
  const inputName = which === 1 ? 'x' : 'r₁'
  const subName = which === 1 ? 'attn' : MODELS.llm.model.moe ? 'moe' : 'ffn'
  return (
    <>
      <section>
        <h4>{t('panel.residual')}</h4>
        <div className="calc-chain">
          <span>
            {inputName} = <b className="num">{fmt(a)}</b>
          </span>
          <span>
            + {subName} = <b className="num">{fmt(b)}</b>
          </span>
          <span>
            → <b className="num">{fmt(res)}</b>
          </span>
        </div>
      </section>
      <section>
        <h4>LayerNorm</h4>
        <div className="calc-chain">
          <span>
            {t('panel.mean')} = <b className="num">{fmt(mu)}</b>
          </span>
          <span>
            {t('panel.std')} = <b className="num">{fmt(sig)}</b>
          </span>
        </div>
        <div className="calc-chain" style={{ marginTop: 6 }}>
          <span>
            ({fmt(res)} − {fmt(mu)}) / {fmt(sig)} × γ<sub>{col}</sub>({fmt(g)}) + β<sub>{col}</sub>(
            {fmt(be)}) = <b className="num accent">{fmt(out)}</b>
          </span>
        </div>
      </section>
    </>
  )
}

function LLMDetail({ sel }: { sel: NodeRef }): ReactNode {
  const trace = useStore((s) => s.llmTrace)
  const t = useT()
  const model = MODELS.llm.model
  const dh = model.d / model.heads
  const show = (c: string) => (c === ' ' ? '␣' : c)
  const kind = llmStageKind(sel.layer)

  // ---- MoE stages
  if (sel.space === 'grid' && kind === 'router') {
    const tk = sel.row
    const e = sel.col
    const weights = Array.from({ length: model.d }, (_, m) => model.Wr[m][e])
    const selected = (trace.topIdx[tk] ?? []).includes(e)
    return (
      <>
        <section>
          <h4>
            r₁[{tk}] · Wr[:, E{e + 1}]
          </h4>
          <ProductRows prev={trace.R1[tk]} prevLabel={(m) => `r₁[${m}]`} weights={weights} />
        </section>
        <section>
          <h4>softmax → {t('llm.gate')}</h4>
          <div className="calc-chain">
            <span>
              score = <b className="num">{fmt(trace.Sr[tk][e])}</b>
            </span>
            <span>
              g = <b className="num accent">{fmt(trace.G[tk][e])}</b>
            </span>
            <span className={selected ? 'accent' : 'muted'}>
              {selected ? `✓ ${t('llm.selected')}` : t('llm.notSelected')}
            </span>
          </div>
          <div className="prob-list" style={{ marginTop: 8 }}>
            {trace.G[tk].map((g, ee) => (
              <div className="prob-row" key={ee}>
                <span className="prob-name mono-char">E{ee + 1}</span>
                <span className="prob-bar">
                  <span style={{ width: `${Math.max(2, g * 100)}%` }} />
                </span>
                <span className="num">{(g * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      </>
    )
  }
  if (sel.space === 'grid' && kind === 'experts') {
    const e = sel.channel
    const tk = sel.row
    const routed = (trace.topIdx[tk] ?? []).includes(e)
    if (!routed) {
      return (
        <section>
          <h4>E{e + 1}</h4>
          <p className="explain-text">{t('llm.notRouted')}</p>
        </section>
      )
    }
    const weights = Array.from({ length: model.d }, (_, m) => model.We1[e][m][sel.col])
    const h = trace.expertH[e][tk][sel.col]
    return (
      <>
        <section>
          <h4>
            E{e + 1} · r₁[{tk}] · W[:, {sel.col}]
          </h4>
          <ProductRows prev={trace.R1[tk]} prevLabel={(m) => `r₁[${m}]`} weights={weights} />
        </section>
        <section>
          <h4>{t('panel.activation')} · ReLU</h4>
          <div className="calc-chain">
            <span>
              + b <b className="num">{fmt(model.bE1[e][sel.col])}</b>
            </span>
            <span>
              ReLU → <b className="num accent">{fmt(h)}</b>
            </span>
          </div>
        </section>
      </>
    )
  }
  if (sel.space === 'grid' && kind === 'combine') {
    const tk = sel.row
    const k = sel.col
    const parts = (trace.topIdx[tk] ?? []).map((e) => {
      const h = trace.expertH[e][tk]
      const out = model.bE2[e][k] + h.reduce((s, hv, j) => s + hv * model.We2[e][j][k], 0)
      return { e, g: trace.G[tk][e], out }
    })
    return (
      <>
        <section>
          <h4>
            y[{tk}][{k}] = Σ gₑ · Eₑ(x)[{k}]
          </h4>
          <div className="dense-table">
            <div className="dense-row dense-head">
              <span />
              <span>g</span>
              <span>Eₑ(x)</span>
              <span>g·Eₑ</span>
            </div>
            {parts.map((p) => (
              <div className="dense-row" key={p.e}>
                <span className="dense-label">E{p.e + 1}</span>
                <span className="num">{fmt(p.g)}</span>
                <span className="num">{fmt(p.out)}</span>
                <span className="num strong">{fmt(p.g * p.out)}</span>
              </div>
            ))}
          </div>
        </section>
        <section>
          <h4>{t('panel.sum')}</h4>
          <div className="calc-chain">
            <span>
              Σ = <b className="num accent">{fmt(trace.Y[tk][k])}</b>
            </span>
          </div>
        </section>
      </>
    )
  }

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

  // embedding lookup
  if (sel.space === 'grid' && sel.layer === 0) {
    return (
      <section>
        <h4>
          E[‘{show(trace.chars[sel.row])}’][{sel.col}]
        </h4>
        <div className="big-value num">{fmt(trace.E[sel.row][sel.col])}</div>
      </section>
    )
  }

  // + positional encoding
  if (sel.space === 'grid' && sel.layer === 1) {
    const e = trace.E[sel.row][sel.col]
    const p = model.P[sel.row][sel.col]
    const fn = sel.col % 2 === 0 ? 'sin' : 'cos'
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
            P ({fn}) = <b className="num">{fmt(p)}</b>
          </span>
          <span>
            → x = <b className="num accent">{fmt(trace.X[sel.row][sel.col])}</b>
          </span>
        </div>
      </section>
    )
  }

  // Q/K/V cell: x_i · W column
  if (sel.space === 'grid' && sel.layer === 2) {
    const W = [model.Wq, model.Wk, model.Wv][sel.channel]
    const name = ['Q', 'K', 'V'][sel.channel]
    const mats = [trace.Q, trace.K, trace.V][sel.channel]
    const weights = Array.from({ length: model.d }, (_, m) => W[m][sel.col])
    const head = Math.floor(sel.col / dh)
    return (
      <>
        <section>
          <h4>
            {name}[{sel.row}][{sel.col}] = x{sel.row} · W{name.toLowerCase()}[:, {sel.col}] ·{' '}
            {t('llm.head', { n: head + 1 })}
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

  // attention cell (head, i, j)
  if (sel.space === 'grid' && sel.layer === 3) {
    const h = sel.channel
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
    const off = h * dh
    const qSlice = Array.from({ length: dh }, (_, m) => trace.Q[i][off + m])
    const kSlice = Array.from({ length: dh }, (_, m) => trace.K[j][off + m])
    const S = trace.S[h]
    const A = trace.A[h]
    return (
      <>
        <section>
          <h4>
            {t('llm.head', { n: h + 1 })} · q{i} · k{j} / √{dh}
          </h4>
          <ProductRows prev={qSlice} prevLabel={(m) => `dim ${off + m}`} weights={kSlice} weightHeader="k" />
          <div className="calc-chain" style={{ marginTop: 8 }}>
            <span>
              Σ ÷ √{dh} → {t('llm.score')} = <b className="num">{fmt(S[i][j])}</b>
            </span>
          </div>
        </section>
        <section>
          <h4>
            softmax → {t('llm.attnWeight')} (‘{show(trace.chars[i])}’ → ‘{show(trace.chars[j])}’)
          </h4>
          <div className="calc-chain">
            <span>
              A[{i}][{j}] = e<sup>{fmt(S[i][j], 2)}</sup> / Σ<sub>j≤{i}</sub> ={' '}
              <b className="num accent">{fmt(A[i][j])}</b>
            </span>
          </div>
          <div className="prob-list" style={{ marginTop: 8 }}>
            {Array.from({ length: i + 1 }, (_, jj) => (
              <div className="prob-row" key={jj}>
                <span className="prob-name mono-char">‘{show(trace.chars[jj])}’</span>
                <span className="prob-bar">
                  <span style={{ width: `${Math.max(2, A[i][jj] * 100)}%` }} />
                </span>
                <span className="num">{(A[i][jj] * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      </>
    )
  }

  // attention output cell: per-head A·V concat, then the W_O projection
  if (sel.space === 'grid' && sel.layer === 4) {
    const i = sel.row
    const k = sel.col
    const woCol = Array.from({ length: model.d }, (_, m) => model.Wo[m][k])
    return (
      <>
        <section>
          <h4>
            z[{i}][{k}] = concat(heads)[{i}] · W<sub>O</sub>[:, {k}] + b
          </h4>
          <ProductRows
            prev={trace.Zc[i]}
            prevLabel={(m) => `${t('llm.head', { n: Math.floor(m / dh) + 1 })} · av[${m}]`}
            weights={woCol}
          />
        </section>
        <section>
          <h4>{t('panel.sum')}</h4>
          <div className="calc-chain">
            <span>
              Σ = <b className="num">{fmt(trace.Z[i][k] - model.bo[k])}</b>
            </span>
            <span>
              + b<sub>{k}</sub> <b className="num">{fmt(model.bo[k])}</b>
            </span>
            <span>
              → <b className="num accent">{fmt(trace.Z[i][k])}</b>
            </span>
          </div>
          <p className="muted" style={{ fontSize: 11.5, marginTop: 6 }}>
            av[m] = Σ<sub>j</sub> A<sub>h</sub>[{i}][j] · V[j][m]
          </p>
        </section>
      </>
    )
  }

  // Add & Norm stages
  if (sel.space === 'grid' && kind === 'addnorm1') {
    return <AddNormDetail which={1} row={sel.row} col={sel.col} />
  }
  if (sel.space === 'grid' && kind === 'addnorm2') {
    return <AddNormDetail which={2} row={sel.row} col={sel.col} />
  }

  // ffn cell
  if (sel.space === 'grid' && kind === 'ffn') {
    const i = sel.row
    const k = sel.col
    const weights = trace.R1[i].map((_, m) => model.W1[m][k])
    return (
      <DenseComputation
        prev={trace.R1[i]}
        prevLabel={(m) => `r₁[${i}][${m}]`}
        weights={weights}
        bias={model.b1[k]}
        z={trace.Hpre[i][k]}
        a={trace.H[i][k]}
        activation="relu"
      />
    )
  }

  // output logit / next-char probability
  if (sel.space === 'vector' && kind === 'output') {
    const T = trace.ids.length
    const vIdx = sel.index
    const weights = trace.R2[T - 1].map((_, k) => model.Wout[k][vIdx])
    const top = [...trace.probs.keys()].sort((a, b) => trace.probs[b] - trace.probs[a]).slice(0, 6)
    return (
      <>
        <section>
          <h4>
            {t('llm.next')}: ‘{show(model.vocab[vIdx])}’
          </h4>
        </section>
        <DenseComputation
          prev={trace.R2[T - 1]}
          prevLabel={(k) => `r₂[${k}]`}
          weights={weights}
          bias={model.bout[vIdx]}
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
