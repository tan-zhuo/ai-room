import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { totalSteps, useStore, useT } from '../store'
import {
  GridSlot,
  LLMStageKind,
  VecSlot,
  gridPos,
  llmLabelAnchor,
  llmLayerOf,
  llmSlot,
  llmStageKind,
  llmSteps,
  vecPos,
} from './layout'
import { Segment, ease } from './common'
import { GridNodes } from './GridNodes'
import { VectorNodes } from './VectorNodes'
import { FlowParticles } from './FlowParticles'
import { LayerLabel } from './Labels'
import { SelectionMarker } from './SelectionMarker'

const v = (p: [number, number, number]) => new THREE.Vector3(p[0], p[1], p[2])

function tensorMax(t: Tensor3): number {
  let m = 0
  for (const ch of t) for (const row of ch) for (const val of row) m = Math.max(m, Math.abs(val))
  return m || 1
}

const FLOW_COUNT = 7

/**
 * The autoregressive feedback loop, made visible: a return wire with
 * direction-flow particles and an arrowhead, a loop label, the committed
 * character flying back, and a landing flash on the token queue.
 */
function FeedbackLoop() {
  const t = useT()
  const generating = useStore((s) => s.llmGenerating)
  const generated = useStore((s) => s.llmGenerated)
  const model = MODELS.llm.model
  const tokSlot = llmSlot(-1) as GridSlot
  const outSlot = llmSlot(llmSteps() - 1) as VecSlot
  const flyer = useRef<THREE.Group>(null)
  const flowMesh = useRef<THREE.InstancedMesh>(null)
  const flash = useRef<THREE.Mesh>(null)
  const anim = useRef<{ curve: THREE.CubicBezierCurve3; t: number } | null>(null)
  const flashT = useRef(-1)
  const dummy = useMemo(() => new THREE.Object3D(), [])

  const lastTokenPos = useMemo(
    () => new THREE.Vector3(...gridPos(tokSlot, 0, 0, tokSlot.cols - 1)),
    [tokSlot],
  )

  const { wire, curve, arrowPos, arrowQuat, labelPos } = useMemo(() => {
    const from = new THREE.Vector3(outSlot.x, -((outSlot.size - 1) / 2) * outSlot.gapY - 1, 0)
    const to = lastTokenPos.clone().add(new THREE.Vector3(0, -0.7, 0))
    const c = new THREE.CubicBezierCurve3(
      from,
      new THREE.Vector3(outSlot.x * 0.55, -8.5, 6),
      new THREE.Vector3(tokSlot.x * 0.55, -8.5, 6),
      to,
    )
    const geom = new THREE.BufferGeometry().setFromPoints(c.getPoints(72))
    const mat = new THREE.LineBasicMaterial({ color: '#38d6ff', transparent: true, opacity: 0.15 })
    mat.toneMapped = false
    const tangent = c.getTangent(0.99)
    const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent)
    return {
      wire: new THREE.Line(geom, mat),
      curve: c,
      arrowPos: c.getPoint(0.985),
      arrowQuat: quat,
      labelPos: c.getPoint(0.9).add(new THREE.Vector3(0, -1.2, 0)),
    }
  }, [outSlot, tokSlot, lastTokenPos])

  useEffect(() => {
    if (!generated) {
      anim.current = null
      return
    }
    const ch = generated[generated.length - 1]
    const idx = Math.max(0, model.vocab.indexOf(ch))
    const from = new THREE.Vector3(...vecPos(outSlot, idx))
    anim.current = {
      curve: new THREE.CubicBezierCurve3(
        from,
        new THREE.Vector3(outSlot.x * 0.55, -8.5, 6),
        new THREE.Vector3(tokSlot.x * 0.55, -8.5, 6),
        lastTokenPos,
      ),
      t: 0,
    }
  }, [generated, model, outSlot, tokSlot, lastTokenPos])

  useFrame((state, dt) => {
    ;(wire.material as THREE.LineBasicMaterial).opacity = generating ? 0.5 : 0.15

    const fm = flowMesh.current
    if (fm) {
      fm.visible = generating
      if (generating) {
        for (let i = 0; i < FLOW_COUNT; i++) {
          const tt = (state.clock.elapsedTime * 0.16 + i / FLOW_COUNT) % 1
          dummy.position.copy(curve.getPoint(tt))
          dummy.scale.setScalar(0.06 * (0.6 + 0.4 * Math.sin(tt * Math.PI)))
          dummy.updateMatrix()
          fm.setMatrixAt(i, dummy.matrix)
        }
        fm.instanceMatrix.needsUpdate = true
      }
    }

    const g = flyer.current
    if (g) {
      const a = anim.current
      if (!a) {
        g.visible = false
      } else {
        a.t += dt / 0.55
        if (a.t >= 1) {
          anim.current = null
          g.visible = false
          flashT.current = 0
        } else {
          g.visible = true
          g.position.copy(a.curve.getPoint(ease(a.t)))
        }
      }
    }

    const fl = flash.current
    if (fl) {
      if (flashT.current >= 0) {
        flashT.current += dt
        const k = flashT.current / 0.4
        if (k >= 1) {
          flashT.current = -1
          fl.visible = false
        } else {
          fl.visible = true
          fl.position.copy(lastTokenPos)
          fl.scale.setScalar(tokSlot.cell * (0.8 + k * 1.6))
          ;(fl.material as THREE.MeshBasicMaterial).opacity = 0.85 * (1 - k)
        }
      } else {
        fl.visible = false
      }
    }
  })

  const lastChar = generated ? generated[generated.length - 1] : ''
  return (
    <group>
      <primitive object={wire} />
      <mesh position={arrowPos} quaternion={arrowQuat}>
        <coneGeometry args={[0.14, 0.4, 8]} />
        <meshBasicMaterial color="#38d6ff" transparent opacity={0.7} toneMapped={false} />
      </mesh>
      <instancedMesh ref={flowMesh} args={[undefined, undefined, FLOW_COUNT]} frustumCulled={false} visible={false}>
        <sphereGeometry args={[1, 8, 8]} />
        <meshBasicMaterial
          color="#7fe3ff"
          transparent
          opacity={0.85}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </instancedMesh>
      {generating && (
        <Html position={labelPos} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
          <div className="loop-label">{t('llm.loop')}</div>
        </Html>
      )}
      <mesh ref={flash} visible={false}>
        <sphereGeometry args={[1, 12, 12]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          toneMapped={false}
        />
      </mesh>
      <group ref={flyer} visible={false}>
        <mesh>
          <sphereGeometry args={[0.16, 12, 12]} />
          <meshBasicMaterial
            color="#9beaff"
            transparent
            opacity={0.9}
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            toneMapped={false}
          />
        </mesh>
        <Html center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
          <div className="fly-char">{lastChar === ' ' ? '␣' : lastChar}</div>
        </Html>
      </group>
    </group>
  )
}

/** Map every cell of the target sheet to a source cell — decorative flow routes. */
function sheetFlow(
  src: GridSlot,
  dst: GridSlot,
  opts: { srcCh?: (dstCh: number) => number; srcCol?: (dstCol: number) => number; cap?: number } = {},
): Segment[] {
  const segs: Segment[] = []
  const chOf = opts.srcCh ?? (() => 0)
  const colOf = opts.srcCol ?? ((c: number) => Math.min(c, src.cols - 1))
  for (let ch = 0; ch < dst.channels; ch++) {
    for (let r = 0; r < dst.rows; r++) {
      for (let c = 0; c < dst.cols; c++) {
        const srcRow = src.rows === 1 ? 0 : Math.min(r, src.rows - 1)
        const srcColIdx = src.rows === 1 ? Math.min(r, src.cols - 1) : colOf(c)
        segs.push({
          a: v(gridPos(src, chOf(ch), srcRow, srcColIdx)),
          b: v(gridPos(dst, ch, r, c)),
          w: 0.5,
          target: 0,
        })
      }
    }
  }
  const cap = opts.cap ?? 300
  if (segs.length <= cap) return segs
  const step = segs.length / cap
  return Array.from({ length: cap }, (_, i) => segs[Math.floor(i * step)])
}

/** Faithful transformer scene (dense FFN or mixture-of-experts). */
export function LLMScene() {
  const trace = useStore((s) => s.llmTrace)
  const step = useStore((s) => s.step)
  const generated = useStore((s) => s.llmGenerated)
  const t = useT()
  const model = MODELS.llm.model

  const steps = llmSteps()
  const kinds = useMemo(
    () => Array.from({ length: steps + 1 }, (_, i) => llmStageKind(i - 1)),
    [steps],
  )

  // conveyor effect: when a char is committed the whole token queue slides one tile
  const tokGroup = useRef<THREE.Group>(null)
  const slideT = useRef(1)
  useEffect(() => {
    if (generated) slideT.current = 0
  }, [generated])

  const tokSlot = llmSlot(-1) as GridSlot
  const outSlot = llmSlot(steps - 1) as VecSlot
  const grid = (layer: number) => llmSlot(layer) as GridSlot

  const T = model.T
  const vocabN = model.vocab.length

  const tokVals = useMemo<Tensor3>(
    () => [[trace.ids.map((id) => 0.25 + (0.7 * id) / vocabN)]],
    [trace, vocabN],
  )

  const valuesOf = (kind: LLMStageKind): Tensor3 => {
    switch (kind) {
      case 'embed':
        return [trace.E]
      case 'posenc':
        return [trace.X]
      case 'qkv':
        return [trace.Q, trace.K, trace.V]
      case 'attn':
        return trace.A
      case 'attnout':
        return [trace.Z]
      case 'addnorm1':
        return [trace.R1]
      case 'ffn':
        return [trace.H]
      case 'router':
        return [trace.G]
      case 'experts':
        return trace.expertH
      case 'combine':
        return [trace.Y]
      default:
        return [trace.R2]
    }
  }

  const labelOf = (kind: LLMStageKind): { title: string; sub: string } => {
    switch (kind) {
      case 'tokens':
        return { title: t('layer.tokens'), sub: `${T}` }
      case 'embed':
        return { title: t('layer.embed'), sub: `${T}×${model.d}` }
      case 'posenc':
        return { title: t('layer.posenc'), sub: 'X = E + P' }
      case 'qkv':
        return { title: 'Q · K · V', sub: `3 × ${T}×${model.d}` }
      case 'attn':
        return { title: t('layer.attn'), sub: `${model.heads} × ${T}×${T} · softmax` }
      case 'attnout':
        return { title: t('layer.attnout'), sub: 'concat · W_O' }
      case 'addnorm1':
        return { title: t('layer.addnorm'), sub: 'LN(x + attn)' }
      case 'ffn':
        return { title: t('layer.ffn'), sub: `${model.dff} · ReLU` }
      case 'router':
        return { title: t('layer.router'), sub: `softmax → top-${model.topK}` }
      case 'experts':
        return { title: t('layer.experts'), sub: `${model.nExperts} × FFN ${model.dffE}` }
      case 'combine':
        return { title: t('layer.combine'), sub: 'Σ gₑ · Eₑ(x)' }
      case 'addnorm2':
        return { title: t('layer.addnorm'), sub: model.moe ? 'LN(x + moe)' : 'LN(x + ffn)' }
      default:
        return { title: t('layer.output'), sub: `${vocabN} · softmax` }
    }
  }

  const outPositions = useMemo(
    () => Array.from({ length: vocabN }, (_, i) => vecPos(outSlot, i)),
    [outSlot, vocabN],
  )

  const flows = useMemo(() => {
    const dh = model.d / model.heads
    const byLayer: Segment[][] = []
    for (let layer = 0; layer < steps; layer++) {
      const kind = kinds[layer + 1]
      if (kind === 'output') {
        const outSegs: Segment[] = []
        const prev = grid(layer - 1)
        for (let k = 0; k < model.d; k++)
          for (let j = 0; j < vocabN; j += 2)
            outSegs.push({ a: v(gridPos(prev, 0, T - 1, k)), b: v(outPositions[j]), w: 0.5, target: j })
        byLayer.push(outSegs)
        continue
      }
      if (kind === 'attn') {
        const attnSegs: Segment[] = []
        const qkv = grid(layer - 1)
        const att = grid(layer)
        for (let h = 0; h < model.heads; h++) {
          const off = h * dh + Math.floor(dh / 2)
          for (let i = 0; i < T; i++) {
            for (let j = 0; j <= i; j++) {
              attnSegs.push({ a: v(gridPos(qkv, 0, i, off)), b: v(gridPos(att, h, i, j)), w: 0.5, target: 0 })
              attnSegs.push({ a: v(gridPos(qkv, 1, j, off)), b: v(gridPos(att, h, i, j)), w: -0.5, target: 0 })
            }
          }
        }
        byLayer.push(attnSegs)
        continue
      }
      if (kind === 'attnout') {
        const zSegs: Segment[] = []
        const att = grid(layer - 1)
        const zSlot = grid(layer)
        for (let r = 0; r < T; r++) {
          for (let c = 0; c < model.d; c++) {
            const head = Math.floor(c / dh)
            zSegs.push({ a: v(gridPos(att, head, r, Math.min(r, T - 1))), b: v(gridPos(zSlot, 0, r, c)), w: 0.5, target: 0 })
          }
        }
        byLayer.push(zSegs)
        continue
      }
      if (kind === 'addnorm1') {
        byLayer.push([
          ...sheetFlow(grid(layer - 1), grid(layer), { cap: 160 }),
          ...sheetFlow(grid(llmLayerOf('posenc')), grid(layer), { cap: 140 }), // residual skip
        ])
        continue
      }
      if (kind === 'experts') {
        // THE MoE picture: each token flows only to its routed experts
        const segs: Segment[] = []
        const routerSlot = grid(layer - 1)
        const expSlot = grid(layer)
        for (let tk = 0; tk < T; tk++) {
          for (const e of trace.topIdx[tk] ?? []) {
            for (let j = 0; j < model.dffE; j += 2) {
              segs.push({ a: v(gridPos(routerSlot, 0, tk, e)), b: v(gridPos(expSlot, e, tk, j)), w: 0.5, target: 0 })
            }
          }
        }
        byLayer.push(segs)
        continue
      }
      if (kind === 'combine') {
        const segs: Segment[] = []
        const expSlot = grid(layer - 1)
        const combSlot = grid(layer)
        for (let tk = 0; tk < T; tk++) {
          for (const e of trace.topIdx[tk] ?? []) {
            for (let k = 0; k < model.d; k += 2) {
              segs.push({ a: v(gridPos(expSlot, e, tk, k % model.dffE)), b: v(gridPos(combSlot, 0, tk, k)), w: 0.5, target: 0 })
            }
          }
        }
        byLayer.push(segs)
        continue
      }
      if (kind === 'addnorm2') {
        byLayer.push([
          ...sheetFlow(grid(layer - 1), grid(layer), { cap: 160 }),
          ...sheetFlow(grid(llmLayerOf('addnorm1')), grid(layer), { cap: 140 }), // residual skip
        ])
        continue
      }
      // tokens→embed, embed→posenc, posenc→qkv, addnorm1→ffn/router
      const src = layer === 0 ? tokSlot : grid(layer - 1)
      byLayer.push(sheetFlow(src, grid(layer), { srcCol: (c) => c % Math.max(1, src.cols) }))
    }
    return byLayer
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, steps, kinds, trace, tokSlot, outPositions, T, vocabN])

  useFrame((_, dt) => {
    const g = tokGroup.current
    if (!g) return
    if (slideT.current < 1) {
      slideT.current = Math.min(1, slideT.current + dt / 0.35)
      g.position.z = (1 - ease(slideT.current)) * tokSlot.cell
    } else {
      g.position.z = 0
    }
  })

  const done = step >= totalSteps('llm')
  const pred = trace.probs.indexOf(Math.max(...trace.probs))
  const topIdx = [...trace.probs.keys()].sort((a, b) => trace.probs[b] - trace.probs[a]).slice(0, 5)
  const showChar = (c: string) => (c === ' ' ? '␣' : c)

  const expertsLayer = llmLayerOf('experts')
  const utilization = useMemo(() => {
    if (!model.moe) return []
    const u = Array.from({ length: model.nExperts }, () => 0)
    trace.topIdx.forEach((sel) => sel.forEach((e) => u[e]++))
    return u
  }, [model, trace])

  return (
    <group>
      <group ref={tokGroup}>
        <GridNodes slot={tokSlot} values={tokVals} scale={1} />
        {trace.chars.map((ch, i) => {
          const p = gridPos(tokSlot, 0, 0, i)
          return (
            <Html key={`tk${i}`} position={[p[0], p[1], p[2]]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
              <div className="token-char">{showChar(ch)}</div>
            </Html>
          )
        })}
      </group>

      {kinds.slice(1, -1).map((kind, idx) => {
        const layer = idx
        const vals = valuesOf(kind)
        return <GridNodes key={`${kind}${layer}`} slot={grid(layer)} values={vals} scale={kind === 'attn' || kind === 'router' ? 1 : tensorMax(vals)} />
      })}
      <VectorNodes
        positions={outPositions}
        values={trace.probs}
        scale={Math.max(...trace.probs)}
        layerIndex={steps - 1}
        radius={Math.min(0.22, outSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: steps - 1, index: i })}
      />

      {flows.map((segs, k) => (
        <FlowParticles key={`p${k}`} segments={segs} layerIndex={k} size={0.06} />
      ))}

      {/* expert badges: name + live utilization */}
      {model.moe &&
        Array.from({ length: model.nExperts }, (_, e) => {
          const slot = grid(expertsLayer)
          const p = gridPos(slot, e, 0, Math.floor(model.dffE / 2))
          const computed = step > expertsLayer
          return (
            <Html key={`ex${e}`} position={[p[0], p[1] + 0.7, p[2]]} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
              <div className="gate-letter">
                E{e + 1}
                {computed && <span className="expert-util"> {utilization[e]}/{T}</span>}
              </div>
            </Html>
          )
        })}

      {/* every vocab character labels its node; top candidates also show probability */}
      {outPositions.map((p, i) => {
        const isTop = topIdx.includes(i)
        return (
          <Html key={`ov${i}`} position={[p[0] + 0.4, p[1], p[2]]} zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
            <div className={`out-label${done ? ' done' : ''}${done && i === pred ? ' pred' : ''}`}>
              <span className={`out-name mono-char${isTop ? '' : ' dim'}`}>{showChar(model.vocab[i])}</span>
              {done && isTop && <span className="out-prob">{(trace.probs[i] * 100).toFixed(1)}%</span>}
            </div>
          </Html>
        )
      })}
      {done && (
        <Html
          position={[outSlot.x, ((outSlot.size - 1) / 2) * outSlot.gapY + 2.4, 0]}
          center
          zIndexRange={[20, 0]}
          style={{ pointerEvents: 'none' }}
        >
          <div className="next-char-banner">
            {t('llm.next')}: <b>{showChar(model.vocab[pred])}</b> ({(trace.probs[pred] * 100).toFixed(1)}%)
          </div>
        </Html>
      )}

      {kinds.map((kind, i) => {
        const layer = i - 1
        const { title, sub } = labelOf(kind)
        return <LayerLabel key={`lb${kind}${layer}`} position={llmLabelAnchor(layer)} title={title} sub={sub} layer={layer} />
      })}

      <FeedbackLoop />
      <SelectionMarker />
    </group>
  )
}
