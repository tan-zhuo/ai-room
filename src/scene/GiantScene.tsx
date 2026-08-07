import { useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import {
  GIANTS,
  countMiniParams,
  embedParams,
  fmtParams,
  fmtTokens,
  perLayerParams,
} from '../nn/giants'
import { useStore, useT } from '../store'

const TOWER_X = [-21, -13, -5, 3, 11]
const FOOT = 3.6
/** world units per parameter (height): DeepSeek-V3 ends up ~33 units tall */
const H_PER_PARAM = 1 / 2e10

/** One million instanced points — the "scale ruler" the GPU can still draw. */
function ParamCloud({ position }: { position: [number, number, number] }) {
  const count = typeof window !== 'undefined' && window.innerWidth < 700 ? 400_000 : 1_000_000
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry()
    const pos = new Float32Array(count * 3)
    let seed = 42
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) % 4294967296
      return seed / 4294967296
    }
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (rnd() - 0.5) * 6
      pos[i * 3 + 1] = (rnd() - 0.5) * 6
      pos[i * 3 + 2] = (rnd() - 0.5) * 6
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    return g
  }, [count])
  const mat = useMemo(
    () =>
      new THREE.PointsMaterial({
        color: '#59c7f5',
        size: 0.017,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.32,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    [],
  )
  const ref = useRef<THREE.Points>(null)
  useFrame((state) => {
    if (ref.current) ref.current.rotation.y = state.clock.elapsedTime * 0.12
  })
  return <points ref={ref} geometry={geo} material={mat} position={[position[0], position[1], position[2]]} />
}

function Beacon({ x }: { x: number }) {
  const ref = useRef<THREE.Mesh>(null)
  useFrame((state) => {
    const m = ref.current
    if (!m) return
    const t = state.clock.elapsedTime
    const s = 1 + 0.5 * (0.5 + 0.5 * Math.sin(t * 2.2))
    m.scale.setScalar(s)
    ;(m.material as THREE.MeshBasicMaterial).opacity = 0.55 - 0.3 * (0.5 + 0.5 * Math.sin(t * 2.2))
  })
  return (
    <mesh ref={ref} position={[x, 0.05, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[0.7, 0.9, 40]} />
      <meshBasicMaterial color="#ffd166" transparent opacity={0.5} toneMapped={false} side={THREE.DoubleSide} />
    </mesh>
  )
}

/** The selected model's layer stack: real layer count, footprint ∝ hidden width. */
function LayerStack({ x }: { x: number }) {
  const sel = useStore((s) => s.giantSel)
  const t = useT()
  const g = GIANTS[sel]
  const mesh = useRef<THREE.InstancedMesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const N = g.layers
  const w = Math.max(0.5, Math.min(8, g.d / 2300))
  const totalH = 13
  const slabH = (totalH / N) * 0.72
  const gap = (totalH / N) * 0.28

  useMemo(() => {
    // positions are set in the effect below via useFrame once
  }, [])

  useFrame(() => {
    const m = mesh.current
    if (!m) return
    for (let i = 0; i < N; i++) {
      dummy.position.set(x, i * (slabH + gap) + slabH / 2, 0)
      dummy.scale.set(w, slabH, w)
      dummy.updateMatrix()
      m.setMatrixAt(i, dummy.matrix)
    }
    m.count = N
    m.instanceMatrix.needsUpdate = true
  })

  return (
    <group>
      <instancedMesh ref={mesh} args={[undefined, undefined, 160]} key={g.key}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="#2d89b8" emissive="#2d89b8" emissiveIntensity={0.35} toneMapped={false} />
      </instancedMesh>
      <Html position={[x, totalH + 1.6, 0]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
        <div className="next-char-banner">
          {g.name} · {t('giant.layersLabel', { n: g.layers, d: g.d.toLocaleString() })}
        </div>
      </Html>
    </group>
  )
}

// ---------------------------------------------------------------- token journey
// A single token rides up through ALL the real layers, its representation
// "growing" — the classic deep-transformer story (word → syntax → meaning →
// context → whole-sentence fusion), with milestones scaled to the model.

const STACK_H = 13
const JOURNEY_STAGES = [0, 0.15, 0.4, 0.7, 1] as const
const TOK_CASES = 3
const LOOP_RISE = 16 // seconds to climb the stack
const LOOP_HOLD = 4.5 // pause at the top before the next case starts

interface TokSeg {
  text: string
  token: boolean
  /** milestone stages at which this context word gets absorbed */
  stages: number[]
}

/** Sentence encoding: segments split by '|', '[word]' marks the token itself,
 *  'text*1,3' marks context absorbed at stages 1 and 3. */
function parseTokSentence(raw: string): TokSeg[] {
  return raw.split('|').map((seg) => {
    const token = seg.startsWith('[')
    const [text, st] = seg.replace(/[[\]]/g, '').split('*')
    return { text, token, stages: st ? st.split(',').map(Number) : [] }
  })
}

function TokenJourney({ x }: { x: number }) {
  const sel = useStore((s) => s.giantSel)
  const t = useT()
  const g = GIANTS[sel]
  const N = g.layers
  const orb = useRef<THREE.Mesh>(null)
  const halo = useRef<THREE.Mesh>(null)
  const sats = useRef<THREE.InstancedMesh>(null)
  const glowSlab = useRef<THREE.Mesh>(null)
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const [stage, setStage] = useState(0)
  const [layerNow, setLayerNow] = useState(1)
  const [caseIdx, setCaseIdx] = useState(0)
  const SAT_MAX = 22

  useFrame((state) => {
    const loop = LOOP_RISE + LOOP_HOLD
    const ci = Math.floor(state.clock.elapsedTime / loop) % TOK_CASES
    if (ci !== caseIdx) setCaseIdx(ci)
    const tt = state.clock.elapsedTime % loop
    const p = Math.min(1, tt / LOOP_RISE)
    const y = 0.4 + p * (STACK_H - 0.4)
    const o = orb.current
    if (o) {
      o.position.set(x, y, 0)
      const s = 0.4 + p * 0.5
      o.scale.setScalar(s)
    }
    if (halo.current) {
      halo.current.position.set(x, y, 0)
      halo.current.scale.setScalar(0.7 + p * 1.1 + 0.08 * Math.sin(state.clock.elapsedTime * 3))
    }
    // information satellites accumulate with depth
    const m = sats.current
    if (m) {
      const count = Math.max(1, Math.round(p * SAT_MAX))
      for (let i = 0; i < count; i++) {
        const a = state.clock.elapsedTime * (0.8 + (i % 5) * 0.22) + i * 2.4
        const r = 0.75 + (i % 4) * 0.22 + p * 0.5
        dummy.position.set(x + Math.cos(a) * r, y + Math.sin(a * 0.7 + i) * 0.55, Math.sin(a) * r)
        dummy.scale.setScalar(0.07 + 0.03 * (i % 3))
        dummy.updateMatrix()
        m.setMatrixAt(i, dummy.matrix)
      }
      m.count = count
      m.instanceMatrix.needsUpdate = true
    }
    // light up the slab being crossed
    const li = Math.min(N - 1, Math.floor(p * N))
    if (glowSlab.current) {
      const slabH = (STACK_H / N) * 0.72
      const gap = (STACK_H / N) * 0.28
      const w = Math.max(0.5, Math.min(8, g.d / 2300)) + 0.25
      glowSlab.current.position.set(x, li * (slabH + gap) + slabH / 2, 0)
      glowSlab.current.scale.set(w, slabH + 0.06, w)
    }
    const st =
      p >= 1
        ? 4
        : JOURNEY_STAGES.findIndex((f, i) => i < 4 && p >= f && p < JOURNEY_STAGES[i + 1])
    const stClamped = st === -1 ? 4 : st
    if (stClamped !== stage) setStage(stClamped)
    const ln = Math.min(N, li + 1)
    if (ln !== layerNow) setLayerNow(ln)
  })

  if (sel === 0) {
    return (
      <Html position={[x, STACK_H + 2.6, 0]} center zIndexRange={[22, 0]} style={{ pointerEvents: 'none' }}>
        <div className="giant-tag mini">{t('giant.tok.mini')}</div>
      </Html>
    )
  }

  const milestoneLayer = (f: number) => Math.max(1, Math.round(f * N))

  return (
    <group>
      <mesh ref={orb}>
        <sphereGeometry args={[1, 24, 18]} />
        <meshStandardMaterial color="#ffd166" emissive="#ffd166" emissiveIntensity={0.9} toneMapped={false} />
      </mesh>
      <mesh ref={halo}>
        <sphereGeometry args={[1, 18, 14]} />
        <meshBasicMaterial color="#ffd166" transparent opacity={0.12} toneMapped={false} />
      </mesh>
      <instancedMesh ref={sats} args={[undefined, undefined, SAT_MAX]}>
        <sphereGeometry args={[1, 8, 6]} />
        <meshBasicMaterial color="#8fe3ff" toneMapped={false} />
      </instancedMesh>
      <mesh ref={glowSlab}>
        <boxGeometry args={[1, 1, 1]} />
        <meshBasicMaterial color="#ffd166" transparent opacity={0.35} toneMapped={false} />
      </mesh>

      {/* journey caption card */}
      <Html position={[x + 1.5, STACK_H + 3.4, 0]} center zIndexRange={[22, 0]} style={{ pointerEvents: 'none' }}>
        <div className="tok-card" key={`${caseIdx}:${stage}`}>
          <div className="tok-head">
            <span className="tok-word">{t(`giant.tok.case${caseIdx}.word`)}</span>
            <span className="tok-layer">
              Layer {stage === 4 ? N : layerNow} / {N}
            </span>
          </div>
          {/* the sentence: context words LIGHT UP as they are absorbed, and stay lit */}
          <p className="tok-sentence-live">
            {parseTokSentence(t(`giant.tok.case${caseIdx}.sentence`)).map((seg, i) => {
              const absorbedNow = seg.stages.includes(stage) || (stage === 4 && !seg.token)
              const absorbedBefore = seg.stages.some((s) => s < stage)
              const cls = seg.token
                ? 'tok-self'
                : absorbedNow
                  ? 'tok-ctx now'
                  : absorbedBefore
                    ? 'tok-ctx on'
                    : 'tok-ctx'
              return (
                <span key={i} className={cls}>
                  {seg.text}
                </span>
              )
            })}
          </p>
          <p className="tok-text">{t(`giant.tok.case${caseIdx}.s${stage}`)}</p>
          <div className="tok-milestones">
            {JOURNEY_STAGES.map((f, i) => (
              <span key={i} className={`tok-dot${i <= stage ? ' on' : ''}`} title={`L${milestoneLayer(f)}`} />
            ))}
          </div>
          <p className="tok-sentence">{t('giant.tok.caseNote', { c: caseIdx + 1, n: TOK_CASES })}</p>
        </div>
      </Html>
    </group>
  )
}

/** dims per world unit inside the block-anatomy exhibit */
const ANAT_U = 3000

interface Plate {
  name: string
  /** matrix input dims (depth, z) and output dims (height, y) */
  inD: number
  outD: number
  x: number
  color: string
  params: number
  /** show even when sub-pixel thin (LayerNorm) */
  sliver?: boolean
}

/** One transformer layer of the selected model, every weight matrix drawn at
 *  its TRUE shape (face = out-dims × in-dims on the same ruler). */
function BlockAnatomy({ startX }: { startX: number }) {
  const sel = useStore((s) => s.giantSel)
  const t = useT()
  const lang = useStore((s) => s.lang)
  const g = GIANTS[sel]

  const { plates, moeX, endX } = useMemo(() => {
    const kv = g.kvDims ?? g.d
    const d = g.d
    const list: Plate[] = []
    let x = startX
    const push = (name: string, inD: number, outD: number, color: string, sliver = false) => {
      const depth = Math.max(0.25, inD / ANAT_U)
      x += depth / 2
      list.push({ name, inD, outD, x, color, params: sliver ? 2 * d : inD * outD, sliver })
      x += depth / 2 + 3.6
    }
    push('LN', d, d, '#5f6f85', true)
    push('W_Q', d, d, '#39b8e8')
    push('W_K', d, kv, '#39b8e8')
    push('W_V', d, kv, '#39b8e8')
    push('W_O', d, d, '#39b8e8')
    push('LN', d, d, '#5f6f85', true)
    let mX = 0
    if (g.ffn === 'moe' && g.moe) {
      mX = x + 5.5
      x = mX + 8
    } else if (g.ffn === 'swiglu3') {
      push('W_gate', d, g.dff, '#ffb74d')
      push('W_up', d, g.dff, '#ffb74d')
      push('W_down', g.dff, d, '#ffb74d')
    } else {
      push('W_1', d, g.dff, '#ffb74d')
      push('W_2', g.dff, d, '#ffb74d')
    }
    return { plates: list, moeX: mX, endX: x }
  }, [g, startX])

  if (sel === 0) {
    return (
      <Html position={[startX + 6, 4, 26]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
        <div className="giant-tag mini">{t('giant.anat.miniHint')}</div>
      </Html>
    )
  }

  const carpetZ = Math.min(46, g.vocab / ANAT_U)
  const miniRatio = Math.round(g.d / 12)

  return (
    <group position={[0, 0, 26]}>
      {/* title + layer-count reminder */}
      <Html position={[(startX + endX) / 2, 21.5, 0]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
        <div className="giant-tag active">
          <b>{t('giant.anat.title')}</b>
          <span>{t('giant.anat.times', { n: g.layers })} · {t('giant.anat.same', { r: miniRatio.toLocaleString() })}</span>
        </div>
      </Html>

      {plates.map((p, i) => {
        const h = Math.max(0.25, p.outD / ANAT_U)
        const depth = Math.max(0.25, p.inD / ANAT_U)
        const thick = p.sliver ? 0.06 : 0.5
        return (
          <group key={i}>
            <mesh position={[p.x, h / 2, 0]}>
              <boxGeometry args={[thick, h, depth]} />
              <meshStandardMaterial
                color={p.color}
                emissive={p.color}
                emissiveIntensity={p.sliver ? 0.15 : 0.4}
                transparent
                opacity={p.sliver ? 0.65 : 0.92}
                toneMapped={false}
              />
            </mesh>
            <Html
              position={[p.x, i % 2 === 0 ? -1.4 : -3.1, 0]}
              center
              zIndexRange={[18, 0]}
              style={{ pointerEvents: 'none' }}
            >
              <div className="anat-tag">
                <b>{p.name}</b>
                {p.sliver ? (
                  <span>{t('giant.anat.lnNote')}</span>
                ) : (
                  <span>
                    {p.outD.toLocaleString()} × {p.inD.toLocaleString()} · {fmtParams(p.params, lang)}
                  </span>
                )}
              </div>
            </Html>
          </group>
        )
      })}

      {/* MoE expert grid in place of the FFN */}
      {g.ffn === 'moe' && g.moe && (
        <group>
          {Array.from({ length: g.moe.experts }, (_, e) => {
            const gr = Math.floor(e / 16)
            const gc = e % 16
            const lit = [3, 41, 77, 106, 133, 172, 201, 244].includes(e)
            const side = 0.58
            return (
              <mesh
                key={e}
                position={[moeX, 0.6 + gr * (side + 0.16) + side / 2, (gc - 7.5) * (side + 0.16)]}
              >
                <boxGeometry args={[0.5, side, side]} />
                <meshStandardMaterial
                  color={lit ? '#ffb74d' : '#27405a'}
                  emissive={lit ? '#ffb74d' : '#1b3048'}
                  emissiveIntensity={lit ? 0.85 : 0.18}
                  toneMapped={false}
                />
              </mesh>
            )
          })}
          <mesh position={[moeX, 0.25, 8.9]}>
            <boxGeometry args={[0.5, 0.58, 0.58]} />
            <meshStandardMaterial color="#58d68d" emissive="#58d68d" emissiveIntensity={0.85} toneMapped={false} />
          </mesh>
          <Html position={[moeX, -1.5, 0]} center zIndexRange={[18, 0]} style={{ pointerEvents: 'none' }}>
            <div className="anat-tag">
              <b>MoE FFN</b>
              <span>
                {t('giant.anat.expertGrid', {
                  e: g.moe.experts,
                  k: g.moe.topK,
                  s: g.moe.shared,
                  d: (2 * g.d * g.moe.expertDff / 1e6).toFixed(0),
                })}
              </span>
            </div>
          </Html>
        </group>
      )}

      {/* token-embedding matrix as a floor carpet at the same ruler */}
      <mesh position={[(startX + endX) / 2, 0.04, -carpetZ / 2 - 3]}>
        <boxGeometry args={[endX - startX + 2, 0.08, carpetZ]} />
        <meshStandardMaterial color="#1d4d70" emissive="#1d4d70" emissiveIntensity={0.25} transparent opacity={0.8} toneMapped={false} />
      </mesh>
      <Html
        position={[(startX + endX) / 2, 1.1, -carpetZ - 4]}
        center
        zIndexRange={[18, 0]}
        style={{ pointerEvents: 'none' }}
      >
        <div className="anat-tag">
          <b>{t('giant.anat.embed')}</b>
          <span>
            {g.vocab.toLocaleString()} × {g.d.toLocaleString()} · {fmtParams(embedParams(g), lang)}
          </span>
        </div>
      </Html>
    </group>
  )
}

/** Production-scale comparison: NOTHING here is computed — the volumes are
 *  drawn to true parameter-count proportion against the in-browser nets. */
export function GiantScene() {
  const sel = useStore((s) => s.giantSel)
  const lang = useStore((s) => s.lang)
  const t = useT()
  const miniParams = useMemo(() => countMiniParams(), [])

  const specs = GIANTS.map((g, i) => ({
    ...g,
    params: i === 0 ? miniParams : g.params,
  }))
  const cur = specs[sel]

  return (
    <group>
      {/* parameter towers, height ∝ true parameter count */}
      {specs.map((g, i) => {
        const h = Math.max(0.03, g.params * H_PER_PARAM)
        const active = i === sel
        return (
          <group key={g.key}>
            <mesh
              position={[TOWER_X[i], h / 2, 0]}
              onClick={(e) => {
                e.stopPropagation()
                useStore.getState().newSample(i)
              }}
            >
              <boxGeometry args={[FOOT, h, FOOT]} />
              <meshStandardMaterial
                color={active ? '#39b8e8' : '#27618a'}
                emissive={active ? '#39b8e8' : '#1d4d70'}
                emissiveIntensity={active ? 0.6 : 0.35}
                toneMapped={false}
              />
            </mesh>
            <Html
              position={[TOWER_X[i], h + 1.7, 0]}
              center
              zIndexRange={[20, 0]}
              style={{ pointerEvents: 'none' }}
            >
              <div className={`giant-tag${active ? ' active' : ''}`}>
                <b>{g.name}</b>
                <span>{fmtParams(g.params, lang)}</span>
              </div>
            </Html>
          </group>
        )
      })}
      <Beacon x={TOWER_X[0]} />
      <Html position={[TOWER_X[0], 2.6, 0]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
        <div className="giant-tag mini">↓ {t('giant.youAreHere')}</div>
      </Html>

      {/* 1M-parameter light cloud as a ruler */}
      <ParamCloud position={[-30, 5, 0]} />
      <Html position={[-30, 10.4, 0]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
        <div className="giant-tag">
          <b>{t('giant.cloud')}</b>
          <span>{t('giant.cloudSub', { n: Math.round(cur.params / 1e6).toLocaleString(), name: cur.name })}</span>
        </div>
      </Html>

      {/* front row: dissected layer (left) + layer stack with token journey (right) */}
      <group position={[0, 0, 26]}>
        <LayerStack x={25} />
        <TokenJourney x={25} />
      </group>
      <BlockAnatomy startX={-33} />

      {/* stats card — screen-anchored so it never clips */}
      <Html fullscreen zIndexRange={[30, 0]} style={{ pointerEvents: 'none' }}>
        <div className="giant-stats floating">
          <h3>{cur.name}</h3>
          <div className="giant-row">
            <span>{t('giant.params')}</span>
            <b>{fmtParams(cur.params, lang)}</b>
          </div>
          {cur.active && (
            <div className="giant-row">
              <span>{t('giant.active')}</span>
              <b>{fmtParams(cur.active, lang)}</b>
            </div>
          )}
          <div className="giant-row">
            <span>{t('giant.layers')}</span>
            <b>{cur.layers}</b>
          </div>
          <div className="giant-row">
            <span>{t('giant.width')}</span>
            <b>d = {cur.d.toLocaleString()}</b>
          </div>
          <div className="giant-row">
            <span>{t('giant.heads')}</span>
            <b>{cur.heads}</b>
          </div>
          <div className="giant-row">
            <span>{t('giant.ctx')}</span>
            <b>{cur.ctx.toLocaleString()}</b>
          </div>
          <div className="giant-row">
            <span>{t('giant.tokens')}</span>
            <b>{fmtTokens(cur.tokens, lang)}</b>
          </div>
          {cur.moe && (
            <div className="giant-row">
              <span>MoE</span>
              <b>
                {cur.moe.experts} experts · top-{cur.moe.topK}
              </b>
            </div>
          )}
          <div className="giant-row ratio">
            <span>{t('giant.ratio')}</span>
            <b>×{Math.round(cur.params / Math.max(miniParams, 1)).toLocaleString()}</b>
          </div>
          {sel > 0 && (
            <div className="giant-row">
              <span>{t('giant.audit')}</span>
              <b>
                {cur.layers} × {fmtParams(perLayerParams(cur), lang)} + {fmtParams(embedParams(cur), lang)} ≈{' '}
                {fmtParams(cur.params, lang)}
              </b>
            </div>
          )}
          <p className="giant-note">{t(sel === 0 ? 'giant.miniNote' : 'giant.trainNote')}</p>
          <p className="giant-disclaimer">{t('giant.disclaimer')}</p>
        </div>
      </Html>
    </group>
  )
}
