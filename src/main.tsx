import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { inject } from '@vercel/analytics'
import { injectSpeedInsights } from '@vercel/speed-insights'
import './styles.css'
import { BootLine, initModels } from './nn/models'

inject()
injectSpeedInsights()

const root = createRoot(document.getElementById('root')!)

const zh = typeof navigator !== 'undefined' && navigator.language.startsWith('zh')

function BootScreen({ lines }: { lines: BootLine[] }) {
  return (
    <div className="boot">
      <div className="boot-brand">
        <span className="logo-dot" />
        <h1>AI ROOM</h1>
      </div>
      <p className="boot-sub">
        {zh
          ? '正在你的浏览器中真实训练 11 个神经网络…'
          : 'Training 11 neural networks live in your browser…'}
      </p>
      <div className="boot-lines">
        {lines.map((l) => (
          <div key={l.label} className={`boot-line ${l.status}`}>
            <span className="boot-mark">
              {l.status === 'done' ? '✓' : l.status === 'running' ? '◌' : '·'}
            </span>
            <span className="boot-label">{l.label}</span>
            {l.detail && <span className="boot-detail">{l.detail}</span>}
          </div>
        ))}
      </div>
      <p className="boot-note">
        {zh ? '真实计算 · 无后端 · 无预制动画' : 'Real computation · no backend · no canned animation'}
      </p>
    </div>
  )
}

async function start() {
  await initModels((lines) => root.render(<BootScreen lines={lines} />))
  const { default: App } = await import('./App')
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
}

start()
