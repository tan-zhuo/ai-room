import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import { MODELS } from '../nn/models'
import { Tensor3 } from '../nn/cnn'
import { totalSteps, useStore, useT } from '../store'
import { GridSlot, LLM_STEPS, VecSlot, gridPos, llmLabelAnchor, llmSlot, vecPos } from './layout'
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
  const outSlot = llmSlot(LLM_STEPS - 1) as VecSlot
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

  // the permanent return wire (kept together with its curve for the flow particles)
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

  // launch a flight whenever a character is committed
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

    // continuous direction-flow particles while generating
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

    // committed character flying home
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
          flashT.current = 0 // landed: flash the queue tail
        } else {
          g.visible = true
          g.position.copy(a.curve.getPoint(ease(a.t)))
        }
      }
    }

    // landing flash
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
      {/* direction arrowhead where the wire re-enters the token queue */}
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

/** Faithful transformer scene:
 *  tokenizer → embedding → +positional encoding → Q/K/V → multi-head attention
 *  → concat A·V → Add&Norm → feed-forward → Add&Norm → next-char output. */
export function LLMScene() {
  const trace = useStore((s) => s.llmTrace)
  const step = useStore((s) => s.step)
  const generated = useStore((s) => s.llmGenerated)
  const t = useT()
  const model = MODELS.llm.model

  // conveyor effect: when a char is committed the whole token queue slides one tile
  const tokGroup = useRef<THREE.Group>(null)
  const slideT = useRef(1)
  useEffect(() => {
    if (generated) slideT.current = 0
  }, [generated])

  const slots = useMemo(
    () => Array.from({ length: LLM_STEPS + 1 }, (_, i) => llmSlot(i - 1)),
    [],
  )
  const tokSlot = slots[0] as GridSlot
  const outSlot = slots[LLM_STEPS] as VecSlot
  const grid = (layer: number) => slots[layer + 1] as GridSlot

  const T = model.T
  const vocabN = model.vocab.length

  const tokVals = useMemo<Tensor3>(
    () => [[trace.ids.map((id) => 0.25 + (0.7 * id) / vocabN)]],
    [trace, vocabN],
  )
  const sheets = useMemo(
    () =>
      [
        { layer: 0, values: [trace.E] },
        { layer: 1, values: [trace.X] },
        { layer: 2, values: [trace.Q, trace.K, trace.V] },
        { layer: 3, values: trace.A },
        { layer: 4, values: [trace.Z] },
        { layer: 5, values: [trace.R1] },
        { layer: 6, values: [trace.H] },
        { layer: 7, values: [trace.R2] },
      ] as { layer: number; values: Tensor3 }[],
    [trace],
  )

  const outPositions = useMemo(
    () => Array.from({ length: vocabN }, (_, i) => vecPos(outSlot, i)),
    [outSlot, vocabN],
  )

  const flows = useMemo(() => {
    const dh = model.d / model.heads
    const all: Segment[][] = []
    all.push(sheetFlow(tokSlot, grid(0)))
    all.push(sheetFlow(grid(0), grid(1)))
    all.push(sheetFlow(grid(1), grid(2)))
    // attention: from Q (ch 0) and K (ch 1) rows into each head's matrix
    const attnSegs: Segment[] = []
    for (let h = 0; h < model.heads; h++) {
      const off = h * dh + Math.floor(dh / 2)
      for (let i = 0; i < T; i++) {
        for (let j = 0; j <= i; j++) {
          attnSegs.push({ a: v(gridPos(grid(2), 0, i, off)), b: v(gridPos(grid(3), h, i, j)), w: 0.5, target: 0 })
          attnSegs.push({ a: v(gridPos(grid(2), 1, j, off)), b: v(gridPos(grid(3), h, i, j)), w: -0.5, target: 0 })
        }
      }
    }
    all.push(attnSegs)
    // concat A·V: each z column is fed from its own head's attention row
    const zSegs: Segment[] = []
    for (let r = 0; r < T; r++) {
      for (let c = 0; c < model.d; c++) {
        const head = Math.floor(c / dh)
        zSegs.push({
          a: v(gridPos(grid(3), head, r, Math.min(r, T - 1))),
          b: v(gridPos(grid(4), 0, r, c)),
          w: 0.5,
          target: 0,
        })
      }
    }
    all.push(zSegs)
    // Add&Norm 1: from Z plus the residual skip from X
    all.push([
      ...sheetFlow(grid(4), grid(5), { cap: 160 }),
      ...sheetFlow(grid(1), grid(5), { cap: 140 }),
    ])
    all.push(sheetFlow(grid(5), grid(6), { srcCol: (c) => c % model.d }))
    // Add&Norm 2: from FFN plus the residual skip from R1
    all.push([
      ...sheetFlow(grid(6), grid(7), { srcCol: (c) => Math.min(model.dff - 1, c * 2) , cap: 160 }),
      ...sheetFlow(grid(5), grid(7), { cap: 140 }),
    ])
    // output: last row of R2 to every vocab node
    const outSegs: Segment[] = []
    for (let k = 0; k < model.d; k++)
      for (let j = 0; j < vocabN; j += 2)
        outSegs.push({ a: v(gridPos(grid(7), 0, T - 1, k)), b: v(outPositions[j]), w: 0.5, target: j })
    all.push(outSegs)
    return all
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, tokSlot, outPositions, T, vocabN])

  const done = step >= totalSteps('llm')
  const pred = trace.probs.indexOf(Math.max(...trace.probs))
  const topIdx = [...trace.probs.keys()].sort((a, b) => trace.probs[b] - trace.probs[a]).slice(0, 5)
  const showChar = (c: string) => (c === ' ' ? '␣' : c)

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

  return (
    <group>
      <group ref={tokGroup}>
        <GridNodes slot={tokSlot} values={tokVals} scale={1} />
        {trace.chars.map((ch, i) => {
          const p = gridPos(tokSlot, 0, 0, i)
          return (
            <Html
              key={`tk${i}`}
              position={[p[0], p[1], p[2]]}
              center
              zIndexRange={[20, 0]}
              style={{ pointerEvents: 'none' }}
            >
              <div className="token-char">{showChar(ch)}</div>
            </Html>
          )
        })}
      </group>
      {sheets.map((sh) => (
        <GridNodes key={sh.layer} slot={grid(sh.layer)} values={sh.values} scale={tensorMax(sh.values)} />
      ))}
      <VectorNodes
        positions={outPositions}
        values={trace.probs}
        scale={Math.max(...trace.probs)}
        layerIndex={LLM_STEPS - 1}
        radius={Math.min(0.22, outSlot.gapY * 0.42)}
        refFor={(i) => ({ space: 'vector', layer: LLM_STEPS - 1, index: i })}
      />

      {flows.map((segs, k) => (
        <FlowParticles key={`p${k}`} segments={segs} layerIndex={k} size={0.06} />
      ))}

      {/* every vocab character labels its node; top candidates also show their probability */}
      {outPositions.map((p, i) => {
        const isTop = topIdx.includes(i)
        return (
          <Html
            key={`ov${i}`}
            position={[p[0] + 0.4, p[1], p[2]]}
            zIndexRange={[20, 0]}
            style={{ pointerEvents: 'none' }}
          >
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

      <LayerLabel position={llmLabelAnchor(-1)} title={t('layer.tokens')} sub={`${T}`} layer={-1} />
      <LayerLabel position={llmLabelAnchor(0)} title={t('layer.embed')} sub={`${T}×${model.d}`} layer={0} />
      <LayerLabel position={llmLabelAnchor(1)} title={t('layer.posenc')} sub="X = E + P" layer={1} />
      <LayerLabel position={llmLabelAnchor(2)} title="Q · K · V" sub={`3 × ${T}×${model.d}`} layer={2} />
      <LayerLabel
        position={llmLabelAnchor(3)}
        title={t('layer.attn')}
        sub={`${model.heads} × ${T}×${T} · softmax`}
        layer={3}
      />
      <LayerLabel position={llmLabelAnchor(4)} title={t('layer.attnout')} sub="concat · W_O" layer={4} />
      <LayerLabel position={llmLabelAnchor(5)} title={t('layer.addnorm')} sub="LN(x + attn)" layer={5} />
      <LayerLabel position={llmLabelAnchor(6)} title={t('layer.ffn')} sub={`${model.dff} · ReLU`} layer={6} />
      <LayerLabel position={llmLabelAnchor(7)} title={t('layer.addnorm')} sub="LN(x + ffn)" layer={7} />
      <LayerLabel position={llmLabelAnchor(8)} title={t('layer.output')} sub={`${vocabN} · softmax`} layer={8} />

      <FeedbackLoop />
      <SelectionMarker />
    </group>
  )
}
