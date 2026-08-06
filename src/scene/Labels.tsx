import { Html } from '@react-three/drei'
import { argmax } from '../nn/mlp'
import { MODELS } from '../nn/models'
import { Arch, totalSteps, useStore, useT } from '../store'
import { Vec3, mlpPos, cnnPos } from './layout'

export function LayerLabel({
  position,
  title,
  sub,
  layer,
}: {
  position: Vec3
  title: string
  sub?: string
  /** layer index (-1 = input); when provided, clicking opens the module explanation */
  layer?: number
}) {
  const explain = useStore((s) => s.explain)
  const setExplain = useStore((s) => s.setExplain)
  const clickable = layer !== undefined
  const active = clickable && explain === layer
  return (
    <Html position={position} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
      <button
        className={`layer-label${clickable ? ' clickable' : ''}${active ? ' active' : ''}`}
        style={{ pointerEvents: clickable ? 'auto' : 'none' }}
        onClick={clickable ? () => setExplain(active ? null : layer) : undefined}
        tabIndex={-1}
      >
        <div className="layer-label-title">{title}</div>
        {sub && <div className="layer-label-sub">{sub}</div>}
      </button>
    </Html>
  )
}

/** Class name + live probability next to each output neuron. */
export function OutputLabels({ arch }: { arch: Arch }) {
  const t = useT()
  const step = useStore((s) => s.step)
  const mlpTrace = useStore((s) => s.mlpTrace)
  const cnnTrace = useStore((s) => s.cnnTrace)

  const lastLayer = totalSteps(arch) - 1
  const done = step > lastLayer
  let probs: number[]
  if (arch === 'mlp') {
    probs = mlpTrace[mlpTrace.length - 1].a
  } else {
    const last = cnnTrace[cnnTrace.length - 1]
    probs = last.kind === 'vector' ? last.a : []
  }
  const pred = argmax(probs)

  return (
    <>
      {probs.map((p, i) => {
        const pos =
          arch === 'mlp'
            ? mlpPos(MODELS.mlp.model.layers.length - 1, i)
            : cnnPos(lastLayer, { index: i })
        return (
          <Html
            key={i}
            position={[pos[0] + 0.55, pos[1], pos[2]]}
            zIndexRange={[20, 0]}
            style={{ pointerEvents: 'none' }}
          >
            <div className={`out-label${done ? ' done' : ''}${done && i === pred ? ' pred' : ''}`}>
              <span className="out-name">{t(`class.${arch}.${i}`)}</span>
              {done && <span className="out-prob">{(p * 100).toFixed(1)}%</span>}
              {done && (
                <span className="out-bar">
                  <span style={{ width: `${Math.max(2, p * 100)}%` }} />
                </span>
              )}
            </div>
          </Html>
        )
      })}
    </>
  )
}

/** Feature name + current value beside each MLP input node. */
export function InputLabels() {
  const t = useT()
  const input = useStore((s) => s.mlpInput)
  return (
    <>
      {input.map((v, i) => {
        const pos = mlpPos(-1, i)
        return (
          <Html
            key={i}
            position={[pos[0] - 0.55, pos[1], pos[2]]}
            zIndexRange={[20, 0]}
            style={{ pointerEvents: 'none' }}
          >
            <div className="in-label">
              <span className="in-name">{t('feature.n', { n: i + 1 })}</span>
              <span className="in-val">{v.toFixed(2)}</span>
            </div>
          </Html>
        )
      })}
    </>
  )
}
