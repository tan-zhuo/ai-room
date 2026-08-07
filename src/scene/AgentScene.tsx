import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
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
      <Html
        position={[MEM_C[0] - 7.5, MEM_C[1] - 1.4, 0]}
        center
        zIndexRange={[18, 0]}
        style={{ pointerEvents: 'none' }}
      >
        <div className="anat-tag">
          <b>{t('layer.agentMemory')}</b>
          <span>{t('agent.memorySub')}</span>
        </div>
      </Html>

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
