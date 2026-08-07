import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { totalSteps, useStore, useT } from '../store'
import { GNN_XS, gnnLabelAnchor, gnnNodePos } from './layout'
import { Segment } from './common'
import { Connections } from './Connections'
import { FlowParticles } from './FlowParticles'
import { LayerLabel } from './Labels'
import { SelectionMarker } from './SelectionMarker'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

/** community colors shared by the input (truth) and output (prediction) copies */
export const GNN_CLASS_COLORS = ['#4fc3f7', '#ffb74d', '#ba68c8']

const IDLE = '#15202f'

function meanAbs(row: number[]): number {
  return row.reduce((s, x) => s + Math.abs(x), 0) / row.length
}

/** One x-shifted copy of the graph. mode decides how nodes are painted. */
function GraphCopy({
  layer,
  colors,
  intensities,
}: {
  layer: number
  colors: (string | null)[]
  /** 0..1 brightness per node */
  intensities: number[]
}) {
  const step = useStore((s) => s.step)
  const select = useStore((s) => s.select)
  const computed = layer === -1 || step > layer
  return (
    <group>
      {intensities.map((int, i) => {
        const p = gnnNodePos(layer, i)
        const c = computed ? colors[i] ?? IDLE : IDLE
        const glow = computed ? 0.25 + 0.75 * int : 0.08
        return (
          <mesh
            key={i}
            position={p}
            onClick={(e) => {
              e.stopPropagation()
              select({ space: 'vector', layer, index: i })
            }}
          >
            <sphereGeometry args={[0.34, 20, 14]} />
            <meshStandardMaterial
              color={c}
              emissive={c}
              emissiveIntensity={glow}
              toneMapped={false}
            />
          </mesh>
        )
      })}
    </group>
  )
}

/** GCN: the same graph four times — input features, hidden after message
 *  passing 1, logits after message passing 2, per-node prediction. Particles
 *  travel along the actual graph edges: that IS the message passing. */
export function GnnScene() {
  const trace = useStore((s) => s.gnnTrace)
  const step = useStore((s) => s.step)
  const t = useT()
  const model = MODELS.gnn.model
  const g = trace.graph

  // intra-copy graph edges (drawn in every copy, weighted by Â)
  const edgeSegs = useMemo(() => {
    const perLayer: Segment[][] = []
    for (let layer = -1; layer <= 2; layer++) {
      const segs: Segment[] = []
      for (const [i, j] of g.edges) {
        segs.push({
          a: v(gnnNodePos(layer, i)),
          b: v(gnnNodePos(layer, j)),
          w: g.ahat[i][j] * 2.2,
          target: j,
        })
      }
      perLayer.push(segs)
    }
    return perLayer
  }, [g])

  // message passing: previous copy's neighbours (and self) → next copy's node
  const passSegs = useMemo(() => {
    const mk = (from: number, to: number): Segment[] => {
      const segs: Segment[] = []
      for (let i = 0; i < g.n; i++)
        for (let j = 0; j < g.n; j++) {
          if (g.ahat[i][j] > 0)
            segs.push({ a: v(gnnNodePos(from, j)), b: v(gnnNodePos(to, i)), w: g.ahat[i][j] * 2.2, target: i })
        }
      return segs
    }
    const out: Segment[] = []
    for (let i = 0; i < g.n; i++)
      out.push({ a: v(gnnNodePos(1, i)), b: v(gnnNodePos(2, i)), w: 0.8, target: i })
    return [mk(-1, 0), mk(0, 1), out]
  }, [g])

  const inputColors = g.labels.map((c) => GNN_CLASS_COLORS[c])
  const inputInt = g.X.map((row) => Math.min(1, meanAbs(row)))
  const h1Int = trace.H1.map((row) => Math.min(1, meanAbs(row) * 1.4))
  const logitInt = trace.logits.map((row) => Math.min(1, meanAbs(row)))
  const predColors = trace.pred.map((c) => GNN_CLASS_COLORS[c])
  const predInt = trace.probs.map((row, i) => row[trace.pred[i]])

  const done = step >= totalSteps('gnn')

  return (
    <group>
      <GraphCopy layer={-1} colors={inputColors} intensities={inputInt} />
      <GraphCopy layer={0} colors={inputInt.map(() => '#3ea8d8')} intensities={h1Int} />
      <GraphCopy layer={1} colors={inputInt.map(() => '#3ea8d8')} intensities={logitInt} />
      <GraphCopy layer={2} colors={predColors} intensities={predInt} />

      {edgeSegs.map((segs, k) => (
        <Connections key={`e${k}`} segments={segs} layerIndex={k - 1} maxRadius={0.022} />
      ))}
      {passSegs.map((segs, k) => (
        <FlowParticles key={`p${k}`} segments={segs} layerIndex={k} size={0.09} />
      ))}

      {done && (
        <Html
          position={[GNN_XS[3], gnnLabelAnchor(2)[1] - 0.9, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="next-char-banner">
            {t('gnn.accBadge')}: <b>{(trace.acc * 100).toFixed(0)}%</b>
          </div>
        </Html>
      )}

      <LayerLabel position={gnnLabelAnchor(-1)} title={t('layer.graphIn')} sub={`${g.n} nodes · d=${model.d}`} layer={-1} />
      <LayerLabel position={gnnLabelAnchor(0)} title={t('layer.msgpass1')} sub="H₁ = ReLU(Â X W₁)" layer={0} />
      <LayerLabel position={gnnLabelAnchor(1)} title={t('layer.msgpass2')} sub="Z = Â H₁ W₂" layer={1} />
      <LayerLabel position={gnnLabelAnchor(2)} title={t('layer.nodeCls')} sub="softmax" layer={2} />

      <SelectionMarker />
    </group>
  )
}
