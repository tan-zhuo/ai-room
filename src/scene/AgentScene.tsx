import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { AGENT_SCENARIOS, AgentLang } from '../nn/agent'
import { useStore, useT } from '../store'
import { agentModulePos } from './layout'
import { Segment } from './common'
import { FlowParticles } from './FlowParticles'
import { LayerLabel } from './Labels'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

const CLUSTER_COLORS = ['#4fc3f7', '#ffb74d', '#ba68c8', '#58d68d']
/** memory cloud placement */
const MEM_C: [number, number] = [-4.5, -5]
const MEM_SCALE: [number, number] = [7.5, 4.2]

function Station({
  layer,
  color,
  label,
  sub,
  intensity = 1,
  radius = 0.55,
}: {
  layer: number
  color: string
  label: string
  sub?: string
  intensity?: number
  radius?: number
}) {
  const step = useStore((s) => s.step)
  const computed = layer === -1 || step > layer
  const p = agentModulePos(layer)
  return (
    <group>
      <mesh position={p}>
        <sphereGeometry args={[radius, 22, 16]} />
        <meshStandardMaterial
          color={computed ? color : '#16283c'}
          emissive={computed ? color : '#10202f'}
          emissiveIntensity={computed ? 0.25 + 0.55 * intensity : 0.1}
          toneMapped={false}
        />
      </mesh>
      <LayerLabel position={[p[0], p[1] + 1.5, p[2]]} title={label} sub={sub} layer={layer} />
    </group>
  )
}

function memPos(pos: [number, number]): [number, number, number] {
  return [MEM_C[0] + pos[0] * MEM_SCALE[0], MEM_C[1] + pos[1] * MEM_SCALE[1], 0]
}

const FLOOR_Y = -7.2
const RACK_H = 1.9

/** One vector-DB shard: a little server rack with breathing status LEDs. */
function ServerRack({ x, color, tag, count }: { x: number; color: string; tag: string; count: number }) {
  const leds = useRef<THREE.Mesh[]>([])
  const t = useT()
  useFrame((state) => {
    leds.current.forEach((m, i) => {
      if (!m) return
      const tt = state.clock.elapsedTime * (1.6 + i * 0.7) + i * 2.1
      ;(m.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.25 + 0.75 * (0.5 + 0.5 * Math.sin(tt))
    })
  })
  const yC = FLOOR_Y + RACK_H / 2
  return (
    <group>
      {/* cabinet */}
      <mesh position={[x, yC, 0]}>
        <boxGeometry args={[1.5, RACK_H, 1.05]} />
        <meshStandardMaterial color="#0c1624" emissive={color} emissiveIntensity={0.07} toneMapped={false} />
      </mesh>
      {/* front slats */}
      {[0, 1, 2, 3].map((k) => (
        <mesh key={k} position={[x, FLOOR_Y + 0.35 + k * 0.42, 0.56]}>
          <boxGeometry args={[1.26, 0.16, 0.04]} />
          <meshStandardMaterial color="#182c44" emissive="#182c44" emissiveIntensity={0.35} toneMapped={false} />
        </mesh>
      ))}
      {/* status LEDs */}
      {[0, 1, 2].map((k) => (
        <mesh
          key={`l${k}`}
          position={[x - 0.45 + k * 0.45, FLOOR_Y + RACK_H - 0.18, 0.56]}
          ref={(m) => {
            if (m) leds.current[k] = m
          }}
        >
          <sphereGeometry args={[0.055, 8, 6]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.6} toneMapped={false} />
        </mesh>
      ))}
      {/* glowing top plate the cables plug into */}
      <mesh position={[x, FLOOR_Y + RACK_H + 0.03, 0]}>
        <boxGeometry args={[1.5, 0.06, 1.05]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} toneMapped={false} />
      </mesh>
      <Html position={[x, FLOOR_Y - 0.55, 0.6]} center zIndexRange={[18, 0]} style={{ pointerEvents: 'none' }}>
        <div className="agent-hit" style={{ borderColor: color, color }}>
          {t('agent.shard', { c: tag, n: count })}
        </div>
      </Html>
    </group>
  )
}

/** A thin data cable between a rack top and a memory point. */
function Cable({ a, b, color }: { a: [number, number, number]; b: [number, number, number]; color: string }) {
  const { pos, quat, len } = useMemo(() => {
    const va = new THREE.Vector3(...a)
    const vb = new THREE.Vector3(...b)
    const dir = vb.clone().sub(va)
    const length = dir.length()
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize())
    const mid = va.clone().add(vb).multiplyScalar(0.5)
    return { pos: mid, quat: q, len: length }
  }, [a, b])
  return (
    <mesh position={pos} quaternion={quat}>
      <cylinderGeometry args={[0.014, 0.014, len, 5]} />
      <meshBasicMaterial color={color} transparent opacity={0.3} toneMapped={false} />
    </mesh>
  )
}

/** A complete agent loop: dialogue → planner (trained router) → sub-agents →
 *  tools/terminal → answer → memory summarization → vector memory write-back.
 *  Retrieval scores, route probabilities, clusters and the write-back target
 *  are all REAL computation (see nn/agent.ts). */
export function AgentScene() {
  const trace = useStore((s) => s.agentTrace)
  const step = useStore((s) => s.step)
  const lang = useStore((s) => s.lang)
  const t = useT()
  const sc = AGENT_SCENARIOS[lang as AgentLang][trace.scenario]

  const routeNames = [t('agent.route0'), t('agent.route1'), t('agent.route2'), t('agent.route3')]

  const flows = useMemo(() => {
    const seg = (a: [number, number, number], b: [number, number, number], w = 0.6): Segment => ({
      a: v(a),
      b: v(b),
      w,
      target: 0,
    })
    const P = agentModulePos
    const f0 = [seg(P(-1), P(0))]
    // planner fans out to the three sub-agents, weighted by REAL route probs
    const f1 = [
      seg(P(0), P(1), 0.3 + trace.route[0]),
      seg(P(0), P(2), 0.3 + trace.route[1]),
      seg(P(0), P(3), 0.3 + trace.route[2] + trace.route[3]),
    ]
    // retrieval reaches into the memory cloud
    const f2 = trace.top.map((hit) => seg(P(1), memPos(trace.positions[hit.idx]), 0.4 + hit.score))
    const f3 = [seg(P(2), P(4), 0.8)]
    const f4 = [seg(P(3), P(4), 0.8), seg(P(4), P(-1), 0.7)]
    const f5 = [seg(P(-1), P(5), 0.7), seg(P(5), memPos(trace.newPosition), 0.9)]
    return [f0, f1, f2, f3, f4, f5]
  }, [trace])

  const showAnswer = step > 4
  const showWrite = step > 5

  // one shard server per k-means cluster, standing on the floor under its points
  const racks = useMemo(() => {
    const list: { c: number; x: number; n: number }[] = []
    for (let c = 0; c < 4; c++) {
      const members = trace.positions.filter((_, i) => trace.clusters[i] === c)
      if (!members.length) continue
      const mx = members.reduce((sum, p) => sum + memPos(p)[0], 0) / members.length
      list.push({ c, x: mx, n: members.length })
    }
    list.sort((a, b) => a.x - b.x)
    for (let i = 1; i < list.length; i++) {
      if (list[i].x - list[i - 1].x < 3.4) list[i].x = list[i - 1].x + 3.4
    }
    const xOf: Record<number, number> = {}
    list.forEach((e) => (xOf[e.c] = e.x))
    return { list, xOf }
  }, [trace])

  return (
    <group>
      <Station layer={-1} color="#39b8e8" label={t('layer.agentDialog')} sub={t('agent.dialogSub')} />
      <Station layer={0} color="#ffd166" label={t('layer.agentPlanner')} sub={t('agent.plannerSub')} />
      <Station
        layer={1}
        color="#4fc3f7"
        label={t('layer.agentRetrieve')}
        sub="cosine top-3"
        intensity={trace.route[0]}
        radius={0.45}
      />
      <Station
        layer={2}
        color="#ba68c8"
        label={t('layer.agentCode')}
        sub={t('agent.codeSub')}
        intensity={trace.route[1]}
        radius={0.45}
      />
      <Station
        layer={3}
        color="#ff8a3c"
        label={t('layer.agentRun')}
        sub={t('agent.runSub')}
        intensity={trace.route[2]}
        radius={0.45}
      />
      <Station layer={4} color="#58d68d" label={t('layer.agentAnswer')} sub={t('agent.answerSub')} radius={0.45} />
      <Station layer={5} color="#c39bd3" label={t('layer.agentSummarize')} sub={t('agent.summarizeSub')} radius={0.45} />

      {/* long-term vector memory: real embeddings, clusters and PCA layout */}
      {trace.positions.map((p, i) => {
        const hit = step > 1 && trace.top.some((h) => h.idx === i)
        const c = CLUSTER_COLORS[trace.clusters[i]]
        return (
          <mesh key={i} position={memPos(p)}>
            <sphereGeometry args={[hit ? 0.34 : 0.22, 14, 10]} />
            <meshStandardMaterial
              color={c}
              emissive={c}
              emissiveIntensity={hit ? 1.0 : 0.35}
              toneMapped={false}
            />
          </mesh>
        )
      })}
      {/* shard servers + data cables binding each cluster to its rack */}
      {racks.list.map((r) => (
        <ServerRack key={`rack${r.c}`} x={r.x} color={CLUSTER_COLORS[r.c]} tag={'ABCD'[r.c]} count={r.n} />
      ))}
      {trace.positions.map((p, i) => (
        <Cable
          key={`cab${i}`}
          a={[racks.xOf[trace.clusters[i]] ?? 0, FLOOR_Y + RACK_H, 0]}
          b={memPos(p)}
          color={CLUSTER_COLORS[trace.clusters[i]]}
        />
      ))}
      {showWrite && (
        <Cable
          a={[racks.xOf[trace.newCluster] ?? 0, FLOOR_Y + RACK_H, 0]}
          b={memPos(trace.newPosition)}
          color={CLUSTER_COLORS[trace.newCluster]}
        />
      )}
      {showWrite && (
        <mesh position={memPos(trace.newPosition)}>
          <sphereGeometry args={[0.34, 14, 10]} />
          <meshStandardMaterial
            color={CLUSTER_COLORS[trace.newCluster]}
            emissive={CLUSTER_COLORS[trace.newCluster]}
            emissiveIntensity={1.1}
            toneMapped={false}
          />
        </mesh>
      )}
      <LayerLabel
        position={[MEM_C[0] - 8.5, MEM_C[1] - 1.2, 0]}
        title={t('layer.agentMemory')}
        sub={t('agent.memorySub')}
        layer={6}
      />

      {/* retrieval hits: real cosine scores + memory text */}
      {step > 1 &&
        trace.top.map((hit, k) => {
          const p = memPos(trace.positions[hit.idx])
          return (
            <Html key={`hit${k}`} position={[p[0], p[1] + 0.55 + k * 0.55, p[2]]} center zIndexRange={[19, 0]} style={{ pointerEvents: 'none' }}>
              <div className="agent-hit">
                <span className="num">{hit.score.toFixed(2)}</span> {trace.memories[hit.idx]}
              </div>
            </Html>
          )
        })}
      {showWrite && (
        <Html
          position={[memPos(trace.newPosition)[0], memPos(trace.newPosition)[1] + 0.75, 0]}
          center
          zIndexRange={[19, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="agent-hit write">+ {trace.newMemory}</div>
        </Html>
      )}

      {flows.map((segs, k) => (
        <FlowParticles key={`f${k}`} segments={segs} layerIndex={k} size={0.09} />
      ))}

      {/* dialogue card */}
      <Html position={[agentModulePos(-1)[0] - 0.6, agentModulePos(-1)[1] - 2.6, 0]} center zIndexRange={[22, 0]}>
        <div className="agent-card" style={{ pointerEvents: 'none' }}>
          <div className="agent-user"><b>{t('agent.userTag')}</b> {sc.user}</div>
          {showAnswer && <div className="agent-answer"><b>{t('agent.botTag')}</b> {sc.answer}</div>}
        </div>
      </Html>

      {/* planner card: REAL router softmax */}
      {step > 0 && (
        <Html position={[agentModulePos(0)[0] + 2.2, agentModulePos(0)[1] - 3.4, 0]} center zIndexRange={[21, 0]}>
          <div className="agent-card slim" style={{ pointerEvents: 'none' }}>
            <div className="agent-card-title">{t('agent.routeTitle')}</div>
            {trace.route.map((p, i) => (
              <div className="agent-route-row" key={i}>
                <span>{routeNames[i]}</span>
                <span className="agent-route-bar">
                  <span style={{ width: `${Math.max(3, p * 100)}%` }} />
                </span>
                <span className="num">{(p * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </Html>
      )}

      {/* terminal card */}
      {step > 0 && (
        <Html position={[9.6, -0.6, 0]} center zIndexRange={[21, 0]}>
          <div className="agent-term" style={{ pointerEvents: 'none' }}>
            <div className="agent-term-head">{t('agent.termTitle')}</div>
            <pre className="agent-term-body">
              {sc.plan}
              {step > 2 && sc.code ? `\n\n${sc.code}` : ''}
              {step > 3 ? `\n\n${sc.command}\n${sc.output}` : ''}
            </pre>
          </div>
        </Html>
      )}
    </group>
  )
}
