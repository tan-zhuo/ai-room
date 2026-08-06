// Engine sanity check: run `npm run sanity`.
import { MODELS, buildCNNTask, buildMLPTask, evalTextAccuracy } from '../src/nn/models'
import { forwardMLP, argmax } from '../src/nn/mlp'
import { forwardCNN } from '../src/nn/cnn'
import { evalLLMAccuracy, forwardLLM, encodeLLM } from '../src/nn/transformer'
import { mulberry32 } from '../src/nn/rng'

const rng = mulberry32(999)
const failures: string[] = []

// --- MLP + CNN across all scales
for (const scale of ['s', 'm', 'l'] as const) {
  const t0 = Date.now()
  const mlp = buildMLPTask(scale)
  const cnn = buildCNNTask(scale)
  const buildMs = Date.now() - t0

  let mlpOk = 0
  for (let i = 0; i < 60; i++) {
    const cls = i % mlp.classCount
    const traces = forwardMLP(mlp.model, mlp.makeSample(cls, rng))
    if (argmax(traces[traces.length - 1].a) === cls) mlpOk++
  }
  let cnnOk = 0
  for (let i = 0; i < 60; i++) {
    const cls = i % cnn.classCount
    const steps = forwardCNN(cnn.model, cnn.makeSample(cls, rng))
    const last = steps[steps.length - 1]
    if (last.kind === 'vector' && argmax(last.a) === cls) cnnOk++
  }
  console.log(`scale=${scale}: MLP ${mlpOk}/60, CNN ${cnnOk}/60 (build+train ${buildMs}ms)`)
  if (mlpOk < 51 || cnnOk < 51) failures.push(`scale ${scale} accuracy too low`)
}

// --- learned conv kernels (end-to-end conv backprop)
{
  const t0 = Date.now()
  const cnn = buildCNNTask('s', 'learned')
  let ok = 0
  for (let i = 0; i < 60; i++) {
    const cls = i % cnn.classCount
    const steps = forwardCNN(cnn.model, cnn.makeSample(cls, rng))
    const last = steps[steps.length - 1]
    if (last.kind === 'vector' && argmax(last.a) === cls) ok++
  }
  console.log(`CNN learned kernels (s): ${ok}/60 (train ${Date.now() - t0}ms)`)
  if (ok < 51) failures.push('learned-kernel CNN accuracy too low')
}

// --- text language detector
const textAcc = evalTextAccuracy(90)
console.log(`TEXT language detector accuracy: ${(textAcc * 100).toFixed(1)}%`)
if (textAcc < 0.85) failures.push('text accuracy too low')

// --- tiny transformer
const llm = MODELS.llm
const llmAcc = evalLLMAccuracy(llm)
console.log(`LLM final training loss: ${llm.finalLoss.toFixed(3)}, next-char top-1: ${(llmAcc * 100).toFixed(1)}%`)
const demo = forwardLLM(llm.model, encodeLLM(llm.model, 'attentio'))
const top = [...demo.probs.keys()].sort((a, b) => demo.probs[b] - demo.probs[a]).slice(0, 3)
console.log(`  "attentio" → next char candidates: ${top.map((i) => `'${llm.model.vocab[i]}' ${(demo.probs[i] * 100).toFixed(0)}%`).join(', ')}`)
if (llm.finalLoss > 2.2 || llmAcc < 0.25) failures.push('llm did not learn')

if (failures.length > 0) {
  throw new Error('Sanity failures: ' + failures.join('; '))
}
console.log('Sanity OK')
