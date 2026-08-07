import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { GIANTS, countMiniParams, fmtParams, fmtTokens } from '../nn/giants'
import { useStore, useT } from '../store'

const TOWER_X = [-17, -9, -1, 7, 15]
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

      {/* real layer structure of the selected model */}
      <LayerStack x={27} />

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
          <p className="giant-note">{t(sel === 0 ? 'giant.miniNote' : 'giant.trainNote')}</p>
          <p className="giant-disclaimer">{t('giant.disclaimer')}</p>
        </div>
      </Html>
    </group>
  )
}
