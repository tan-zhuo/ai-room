import { useStore, useT } from '../store'

/** Whole-architecture overview: what it is, what problems it solves,
 *  where it is used and which industries it powers. */
export function OverviewPanel() {
  const arch = useStore((s) => s.arch)
  const open = useStore((s) => s.overviewOpen)
  const toggleOverview = useStore((s) => s.toggleOverview)
  const t = useT()

  if (!open) return null

  const bullets = (key: string) =>
    t(key)
      .split('\n')
      .filter(Boolean)
      .map((line, i) => <li key={i}>{line}</li>)

  return (
    <aside className="inspector overview">
      <header className="inspector-head">
        <div>
          <div className="inspector-title">{t(`arch.${arch}Full`)}</div>
          <div className="inspector-sub">{t('overview.title')}</div>
        </div>
        <button className="icon-btn" onClick={toggleOverview} title={t('panel.close')}>
          ×
        </button>
      </header>
      <div className="inspector-body">
        <section className="explain-section what">
          <h4>{t('overview.secIntro')}</h4>
          <p className="explain-text">{t(`overview.${arch}.intro`)}</p>
        </section>
        <section className="explain-section why">
          <h4>{t('overview.secProblems')}</h4>
          <p className="explain-text">{t(`overview.${arch}.problems`)}</p>
        </section>
        <section className="explain-section simple">
          <h4>{t('overview.secDomains')}</h4>
          <ul className="overview-list">{bullets(`overview.${arch}.domains`)}</ul>
        </section>
        <section className="explain-section industries">
          <h4>{t('overview.secIndustries')}</h4>
          <ul className="overview-list">{bullets(`overview.${arch}.industries`)}</ul>
        </section>
        <div className="explain-tip">{t('overview.tip')}</div>
      </div>
    </aside>
  )
}
