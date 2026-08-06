import { useEffect, useState } from 'react'
import { LANGS, LANG_LABEL } from '../i18n'
import { MODELS, Scale } from '../nn/models'
import { Arch, totalSteps, useStore, useT } from '../store'
import { InspectorPanel } from './InspectorPanel'
import { ExplainPanel } from './ExplainPanel'
import { layerNameOf } from './layerName'
import { IconNext, IconPause, IconPlay, IconPrev, IconReset, IconShuffle } from './icons'

export function Hud() {
  return (
    <div className="hud">
      <TopBar />
      <Drawer />
      <Hint />
      <GenBanner />
      <BottomBar />
      <Legend />
      <FooterLinks />
      <InspectorPanel />
      <ExplainPanel />
      <Tooltip />
      <Toast />
      <HelpOverlay />
    </div>
  )
}

// ---------------------------------------------------------------- top bar

const MODEL_ARCHS: { arch: Arch; kbd: string }[] = [
  { arch: 'mlp', kbd: '1' },
  { arch: 'cnn', kbd: '2' },
  { arch: 'llm', kbd: '3' },
]

const APP_ARCHS: { arch: Arch; kbd: string }[] = [{ arch: 'text', kbd: '4' }]

const SCALES: Scale[] = ['s', 'm', 'l']

function TopBar() {
  const t = useT()
  const arch = useStore((s) => s.arch)
  const lang = useStore((s) => s.lang)
  const scale = useStore((s) => s.scale)
  const setScale = useStore((s) => s.setScale)
  const setLang = useStore((s) => s.setLang)
  const toggleHelp = useStore((s) => s.toggleHelp)
  const toggleMenu = useStore((s) => s.toggleMenu)

  const scalable = arch === 'mlp' || arch === 'cnn'

  return (
    <div className="topbar">
      <div className="brand">
        <button className="menu-btn" onClick={toggleMenu} title={t('nav.menu')}>
          <svg viewBox="0 0 24 24" width="17" height="17" fill="currentColor">
            <path d="M3 6h18v2H3zM3 11h18v2H3zM3 16h18v2H3z" />
          </svg>
        </button>
        <span className="logo-dot" />
        <h1>{t('app.title')}</h1>
        <button className="current-arch" onClick={toggleMenu} title={t(`arch.${arch}Full`)}>
          {t(`arch.${arch}`)} <span className="caret">▾</span>
        </button>
        <span className="tagline">{t('app.tagline')}</span>
      </div>
      <div className="top-controls">
        {scalable && (
          <div className="seg" title={t('scale.tooltip')}>
            {SCALES.map((sz) => (
              <button
                key={sz}
                className={`tab${scale[arch] === sz ? ' active' : ''}`}
                onClick={() => setScale(sz)}
              >
                {t(`scale.${sz}`)}
              </button>
            ))}
          </div>
        )}
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

// ---------------------------------------------------------------- drawer

function Drawer() {
  const t = useT()
  const open = useStore((s) => s.menuOpen)
  const arch = useStore((s) => s.arch)
  const setArch = useStore((s) => s.setArch)
  const toggleMenu = useStore((s) => s.toggleMenu)

  const item = ({ arch: a, kbd }: { arch: Arch; kbd: string }) => (
    <button
      key={a}
      className={`drawer-item${arch === a ? ' active' : ''}`}
      onClick={() => {
        setArch(a)
        if (arch !== a) return
        toggleMenu()
      }}
    >
      <span className="drawer-dot" />
      <span className="drawer-texts">
        <span className="drawer-name">{t(`arch.${a}`)}</span>
        <span className="drawer-sub">{t(`arch.${a}Full`)}</span>
      </span>
      <kbd>{kbd}</kbd>
    </button>
  )

  return (
    <>
      <div className={`drawer-backdrop${open ? ' open' : ''}`} onClick={toggleMenu} />
      <aside className={`drawer${open ? ' open' : ''}`}>
        <header className="drawer-head">
          <span className="logo-dot" />
          <h2>{t('app.title')}</h2>
          <button className="icon-btn" onClick={toggleMenu} title={t('panel.close')}>
            ×
          </button>
        </header>
        <div className="drawer-section">{t('nav.models')}</div>
        {MODEL_ARCHS.map(item)}
        <div className="drawer-section">{t('nav.apps')}</div>
        {APP_ARCHS.map(item)}
        <p className="drawer-note">{t('nav.more')}</p>
        <ExternalLinks />
      </aside>
    </>
  )
}

// ---------------------------------------------------------------- bottom bar

function TextEntry() {
  const t = useT()
  const arch = useStore((s) => s.arch)
  const textRaw = useStore((s) => s.textRaw)
  const llmText = useStore((s) => s.llmText)
  const setTextInput = useStore((s) => s.setTextInput)
  const setLLMInput = useStore((s) => s.setLLMInput)
  const stored = arch === 'text' ? textRaw : llmText
  const [val, setVal] = useState(stored)

  useEffect(() => setVal(stored), [stored])

  const submit = () => {
    const trimmed = val.trim()
    if (!trimmed) return
    if (arch === 'text') setTextInput(trimmed)
    else setLLMInput(trimmed)
  }

  return (
    <div className="text-entry">
      <input
        className="text-input"
        value={val}
        maxLength={arch === 'llm' ? 24 : 60}
        placeholder={t('controls.typeText')}
        onChange={(e) => setVal(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            submit()
            e.currentTarget.blur()
          }
        }}
      />
      <button className="chip run" onClick={submit}>
        {t('controls.run')}
      </button>
      {arch === 'llm' && <GenerateButton />}
    </div>
  )
}

function GenerateButton() {
  const t = useT()
  const generating = useStore((s) => s.llmGenerating)
  const toggleGenerate = useStore((s) => s.toggleGenerate)
  return (
    <button className={`chip gen${generating ? ' stop' : ''}`} onClick={toggleGenerate}>
      {generating ? t('llm.stop') : t('llm.generate')}
    </button>
  )
}

/** Streaming output: prompt + characters generated so far, with a caret. */
function GenBanner() {
  const arch = useStore((s) => s.arch)
  const prompt = useStore((s) => s.llmText)
  const generated = useStore((s) => s.llmGenerated)
  const generating = useStore((s) => s.llmGenerating)
  if (arch !== 'llm' || (!generated && !generating)) return null
  return (
    <div className="gen-banner">
      <span className="gen-prompt">{prompt}</span>
      <span className="gen-out">{generated}</span>
      {generating && <span className="gen-caret">▊</span>}
    </div>
  )
}

function BottomBar() {
  const t = useT()
  const arch = useStore((s) => s.arch)
  const step = useStore((s) => s.step)
  const playing = useStore((s) => s.playing)
  const transitioning = useStore((s) => s.transitioning)
  const speed = useStore((s) => s.speed)
  const mlpClass = useStore((s) => s.mlpClass)
  const cnnClass = useStore((s) => s.cnnClass)
  const textClass = useStore((s) => s.textClass)
  const llmClass = useStore((s) => s.llmClass)
  const { togglePlay, nextStep, prevStep, reset, cycleSpeed, newSample } = useStore.getState()

  const total = totalSteps(arch)
  const running = playing || transitioning

  let statusText: string
  if (running && step < total) statusText = `${layerNameOf(arch, step, t)} · ${t('layer.computing')}`
  else if (step === 0) statusText = t('step.inputLoaded')
  else statusText = layerNameOf(arch, step - 1, t)

  const currentClass = { mlp: mlpClass, cnn: cnnClass, text: textClass, llm: llmClass }[arch]
  const chips: string[] =
    arch === 'llm'
      ? MODELS.llm.samples.map((s) => `“${s}”`)
      : Array.from({ length: MODELS[arch].classCount }, (_, i) => t(`class.${arch}.${i}`))

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
            <span
              key={i}
              className={`progress-seg${step > i ? ' filled' : ''}${running && step === i ? ' current' : ''}`}
            />
          ))}
        </div>
      </div>

      <div className="samples">
        <span className="samples-label">{t('controls.input')}</span>
        {(arch === 'text' || arch === 'llm') && <TextEntry />}
        {chips.map((label, i) => (
          <button
            key={i}
            className={`chip${currentClass === i ? ' active' : ''}${arch === 'llm' ? ' mono' : ''}`}
            onClick={() => newSample(i)}
          >
            {label}
          </button>
        ))}
        {arch !== 'llm' && (
          <button className="chip" onClick={() => newSample()} title={t('controls.randomize')}>
            <IconShuffle />
          </button>
        )}
        {arch === 'cnn' && <CNNControls />}
        {arch === 'llm' && <TempSlider />}
      </div>
    </div>
  )
}

/** CNN extras: paint-your-own input + hand-crafted vs learned kernels. */
function CNNControls() {
  const t = useT()
  const drawMode = useStore((s) => s.drawMode)
  const cnnKernels = useStore((s) => s.cnnKernels)
  const toggleDraw = useStore((s) => s.toggleDraw)
  const clearCnnInput = useStore((s) => s.clearCnnInput)
  const setKernelMode = useStore((s) => s.setKernelMode)
  return (
    <>
      <span className="chip-divider" />
      <button className={`chip${drawMode ? ' active' : ''}`} onClick={toggleDraw}>
        ✎ {t('controls.draw')}
      </button>
      {drawMode && (
        <button className="chip" onClick={clearCnnInput}>
          {t('controls.clear')}
        </button>
      )}
      <span className="chip-divider" />
      <button
        className={`chip${cnnKernels === 'hand' ? ' active' : ''}`}
        onClick={() => setKernelMode('hand')}
        title={t('cnn.handFull')}
      >
        {t('cnn.hand')}
      </button>
      <button
        className={`chip${cnnKernels === 'learned' ? ' active' : ''}`}
        onClick={() => setKernelMode('learned')}
        title={t('cnn.learnedFull')}
      >
        {t('cnn.learned')}
      </button>
    </>
  )
}

/** Sampling temperature for autoregressive generation. */
function TempSlider() {
  const t = useT()
  const temp = useStore((s) => s.llmTemp)
  const setLLMTemp = useStore((s) => s.setLLMTemp)
  return (
    <div className="temp-ctl" title={t('llm.tempTip')}>
      <span>{t('llm.temp')}</span>
      <input
        type="range"
        min={0.2}
        max={1.4}
        step={0.1}
        value={temp}
        onChange={(e) => setLLMTemp(parseFloat(e.target.value))}
      />
      <span className="num">{temp.toFixed(1)}</span>
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

const BlogIcon = () => (
  <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor">
    <path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm7.93 9h-3.02a15.7 15.7 0 0 0-1.4-6.14A8.02 8.02 0 0 1 19.93 11zM12 4.06c.94 1.24 1.84 3.6 2.08 6.94H9.92c.24-3.34 1.14-5.7 2.08-6.94zM8.49 4.86A15.7 15.7 0 0 0 7.09 11H4.07a8.02 8.02 0 0 1 4.42-6.14zM4.07 13h3.02c.14 2.35.63 4.48 1.4 6.14A8.02 8.02 0 0 1 4.07 13zM12 19.94c-.94-1.24-1.84-3.6-2.08-6.94h4.16c-.24 3.34-1.14 5.7-2.08 6.94zm3.51-.8a15.7 15.7 0 0 0 1.4-6.14h3.02a8.02 8.02 0 0 1-4.42 6.14z" />
  </svg>
)

const GitHubIcon = () => (
  <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor">
    <path d="M12 2C6.48 2 2 6.58 2 12.25c0 4.53 2.87 8.37 6.84 9.73.5.09.68-.22.68-.49 0-.24-.01-.88-.01-1.73-2.78.62-3.37-1.37-3.37-1.37-.45-1.18-1.11-1.5-1.11-1.5-.91-.63.07-.62.07-.62 1 .07 1.53 1.06 1.53 1.06.89 1.57 2.34 1.12 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.7 0 0 .84-.28 2.75 1.05a9.36 9.36 0 0 1 5 0c1.91-1.33 2.75-1.05 2.75-1.05.55 1.4.2 2.44.1 2.7.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.8-4.57 5.06.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.59.69.49A10.05 10.05 0 0 0 22 12.25C22 6.58 17.52 2 12 2z" />
  </svg>
)

/** Labeled external links: what they are, then where they go. */
function ExternalLinks({ compact = false }: { compact?: boolean }) {
  const t = useT()
  return (
    <div className={compact ? 'footer-links' : 'drawer-links'}>
      <a href="https://tanzhuo.xyz" target="_blank" rel="noopener noreferrer">
        <BlogIcon />
        <span className="link-label">{t('footer.blog')}</span>
        <span className="link-addr">tanzhuo.xyz</span>
      </a>
      <a href="https://github.com/tan-zhuo/ai-room" target="_blank" rel="noopener noreferrer">
        <GitHubIcon />
        <span className="link-label">{t('footer.source')}</span>
        <span className="link-addr">tan-zhuo/ai-room</span>
      </a>
    </div>
  )
}

function FooterLinks() {
  return <ExternalLinks compact />
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
    ['1 / 2 / 3 / 4', t('help.digits')],
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
