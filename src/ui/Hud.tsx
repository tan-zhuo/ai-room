import { useEffect, useState } from 'react'
import { LANGS, LANG_LABEL } from '../i18n'
import { MODELS } from '../nn/models'
import { Arch, totalSteps, useStore, useT } from '../store'
import { InspectorPanel } from './InspectorPanel'
import { ExplainPanel } from './ExplainPanel'
import { layerNameOf } from './layerName'
import { IconNext, IconPause, IconPlay, IconPrev, IconReset, IconShuffle } from './icons'

export function Hud() {
  return (
    <div className="hud">
      <TopBar />
      <Hint />
      <BottomBar />
      <Legend />
      <InspectorPanel />
      <ExplainPanel />
      <Tooltip />
      <Toast />
      <HelpOverlay />
    </div>
  )
}

// ---------------------------------------------------------------- top bar

function TopBar() {
  const t = useT()
  const arch = useStore((s) => s.arch)
  const lang = useStore((s) => s.lang)
  const setArch = useStore((s) => s.setArch)
  const setLang = useStore((s) => s.setLang)
  const toggleHelp = useStore((s) => s.toggleHelp)

  const archBtn = (a: Arch, kbd: string, full: string) => (
    <button
      className={`tab${arch === a ? ' active' : ''}`}
      onClick={() => setArch(a)}
      title={full}
    >
      {t(`arch.${a}`)}
      <kbd>{kbd}</kbd>
    </button>
  )

  return (
    <div className="topbar">
      <div className="brand">
        <span className="logo-dot" />
        <h1>{t('app.title')}</h1>
        <span className="tagline">{t('app.tagline')}</span>
      </div>
      <div className="top-controls">
        <div className="seg">
          {archBtn('mlp', '1', t('arch.mlpFull'))}
          {archBtn('cnn', '2', t('arch.cnnFull'))}
        </div>
        <div className="seg">
          {LANGS.map((l) => (
            <button key={l} className={`tab${lang === l ? ' active' : ''}`} onClick={() => setLang(l)}>
              {LANG_LABEL[l]}
            </button>
          ))}
        </div>
        <button className="tab round" onClick={toggleHelp} title={t('help.title')}>
          ?
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- bottom bar

function BottomBar() {
  const t = useT()
  const arch = useStore((s) => s.arch)
  const step = useStore((s) => s.step)
  const playing = useStore((s) => s.playing)
  const transitioning = useStore((s) => s.transitioning)
  const speed = useStore((s) => s.speed)
  const mlpClass = useStore((s) => s.mlpClass)
  const cnnClass = useStore((s) => s.cnnClass)
  const { togglePlay, nextStep, prevStep, reset, cycleSpeed, newSample } = useStore.getState()

  const total = totalSteps(arch)
  const running = playing || transitioning

  let statusText: string
  if (running && step < total) statusText = `${layerNameOf(arch, step, t)} · ${t('layer.computing')}`
  else if (step === 0) statusText = t('step.inputLoaded')
  else statusText = layerNameOf(arch, step - 1, t)

  const classCount = arch === 'mlp' ? MODELS.mlp.classCount : MODELS.cnn.classCount
  const currentClass = arch === 'mlp' ? mlpClass : cnnClass

  return (
    <div className="bottombar">
      <div className="transport">
        <button className="icon-btn" onClick={reset} title={`${t('controls.reset')} (R)`}>
          <IconReset />
        </button>
        <button className="icon-btn" onClick={prevStep} title={`${t('controls.prev')} (←)`}>
          <IconPrev />
        </button>
        <button
          className="icon-btn primary"
          onClick={togglePlay}
          title={`${playing ? t('controls.pause') : t('controls.play')} (Space)`}
        >
          {playing ? <IconPause /> : <IconPlay />}
        </button>
        <button className="icon-btn" onClick={nextStep} title={`${t('controls.next')} (→)`}>
          <IconNext />
        </button>
        <button className="icon-btn wide" onClick={cycleSpeed} title={t('controls.speed')}>
          ×{speed}
        </button>
      </div>

      <div className="progress">
        <div className="progress-label">
          <span>
            {t('controls.step')} {Math.min(step, total)}/{total}
          </span>
          <span className="status">{statusText}</span>
        </div>
        <div className="progress-track">
          {Array.from({ length: total }, (_, i) => (
            <span key={i} className={`progress-seg${step > i ? ' filled' : ''}${running && step === i ? ' current' : ''}`} />
          ))}
        </div>
      </div>

      <div className="samples">
        <span className="samples-label">{t('controls.input')}</span>
        {Array.from({ length: classCount }, (_, i) => (
          <button
            key={i}
            className={`chip${currentClass === i ? ' active' : ''}`}
            onClick={() => newSample(i)}
          >
            {t(`class.${arch}.${i}`)}
          </button>
        ))}
        <button className="chip" onClick={() => newSample()} title={t('controls.randomize')}>
          <IconShuffle />
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- misc overlays

function Hint() {
  const t = useT()
  const selected = useStore((s) => s.selected)
  if (selected) return null
  return <div className="hint">{t('hint.click')}</div>
}

function Legend() {
  const t = useT()
  return (
    <div className="legend">
      <div className="legend-title">{t('legend.title')}</div>
      <div className="legend-row">
        <span className="swatch pos" />
        {t('legend.posWeight')}
      </div>
      <div className="legend-row">
        <span className="swatch neg" />
        {t('legend.negWeight')}
      </div>
      <div className="legend-row">
        <span className="swatch flow" />
        {t('legend.flow')}
      </div>
      <div className="legend-row">
        <span className="swatch glow" />
        {t('legend.glow')}
      </div>
    </div>
  )
}

function Tooltip() {
  const t = useT()
  const arch = useStore((s) => s.arch)
  const hover = useStore((s) => s.hoverInfo)
  const [pos, setPos] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const fn = (e: MouseEvent) => setPos({ x: e.clientX, y: e.clientY })
    window.addEventListener('mousemove', fn)
    return () => window.removeEventListener('mousemove', fn)
  }, [])

  if (!hover) return null

  const layerName = layerNameOf(arch, hover.ref.layer, t)

  const where =
    hover.ref.space === 'vector'
      ? `#${hover.ref.index + 1}`
      : `ch${hover.ref.channel + 1} (${hover.ref.row}, ${hover.ref.col})`

  return (
    <div className="tooltip" style={{ left: pos.x + 14, top: pos.y + 16 }}>
      <span className="tooltip-name">
        {layerName} {where}
      </span>
      <span className="tooltip-val num">{hover.value.toFixed(3)}</span>
    </div>
  )
}

function Toast() {
  const t = useT()
  const toast = useStore((s) => s.toast)
  if (!toast) return null
  return <div className="toast">{t(toast)}</div>
}

function HelpOverlay() {
  const t = useT()
  const open = useStore((s) => s.helpOpen)
  const toggleHelp = useStore((s) => s.toggleHelp)
  if (!open) return null
  const rows: [string, string][] = [
    ['Space', t('help.space')],
    ['← →', t('help.arrows')],
    ['R', t('help.r')],
    ['1 / 2 / 3', t('help.digits')],
    ['L', t('help.l')],
    ['F', t('help.f')],
    ['Esc', t('help.esc')],
  ]
  return (
    <div className="modal-backdrop" onClick={toggleHelp}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header>
          <h3>{t('help.title')}</h3>
          <button className="icon-btn" onClick={toggleHelp}>
            ×
          </button>
        </header>
        <div className="help-rows">
          {rows.map(([k, v]) => (
            <div className="help-row" key={k}>
              <kbd>{k}</kbd>
              <span>{v}</span>
            </div>
          ))}
        </div>
        <p className="muted">{t('help.mouse')}</p>
      </div>
    </div>
  )
}
