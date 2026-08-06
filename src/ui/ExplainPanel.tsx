import { useStore, useT } from '../store'
import { explainKeyOf, layerNameOf } from './layerName'

/** Module explanation: what a layer does, why it exists, and a plain-words analogy. */
export function ExplainPanel() {
  const arch = useStore((s) => s.arch)
  const explain = useStore((s) => s.explain)
  const setExplain = useStore((s) => s.setExplain)
  const t = useT()

  if (explain === null) return null
  const key = explainKeyOf(arch, explain)

  const sections: { title: string; body: string; cls: string }[] = [
    { title: t('explain.what'), body: t(`explain.${key}.what`), cls: 'what' },
    { title: t('explain.why'), body: t(`explain.${key}.why`), cls: 'why' },
    { title: t('explain.simple'), body: t(`explain.${key}.simple`), cls: 'simple' },
  ]

  return (
    <aside className="inspector explain">
      <header className="inspector-head">
        <div>
          <div className="inspector-title">{layerNameOf(arch, explain, t)}</div>
          <div className="inspector-sub">{t(`arch.${arch}Full`)}</div>
        </div>
        <button className="icon-btn" onClick={() => setExplain(null)} title={t('panel.close')}>
          ×
        </button>
      </header>
      <div className="inspector-body">
        {sections.map((s) => (
          <section key={s.cls} className={`explain-section ${s.cls}`}>
            <h4>{s.title}</h4>
            <p className="explain-text">{s.body}</p>
          </section>
        ))}
        <div className="explain-tip">{t('explain.tip')}</div>
      </div>
    </aside>
  )
}
