// Engine sanity check: run `npm run sanity`.
import { MODELS, buildCNNTask, buildMLPTask, evalTextAccuracy, initModels } from '../src/nn/models'

await initModels(() => {})
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

// --- RNN / LSTM / Autoencoder
console.log(`RNN final loss: ${MODELS.rnn.finalLoss.toFixed(3)}, LSTM final loss: ${MODELS.lstm.finalLoss.toFixed(3)}`)
if (MODELS.rnn.finalLoss > 2.0) failures.push('rnn did not learn')
if (MODELS.lstm.finalLoss > 2.0) failures.push('lstm did not learn')
console.log(`Autoencoder training MSE: ${MODELS.ae.finalMSE.toFixed(4)}`)
if (MODELS.ae.finalMSE > 0.03) failures.push('autoencoder reconstruction too poor')

// --- diffusion
{
  const diff = MODELS.diff
  console.log(`Diffusion x0-pred MSE: ${diff.finalLoss.toFixed(4)}`)
  if (diff.finalLoss > 0.3) failures.push('diffusion did not learn')
}

// --- GAN
{
  const gan = MODELS.gan
  console.log(
    `GAN: D(real) ${gan.dReal.toFixed(3)}, D(fake) ${gan.dFake.toFixed(3)}, sample quality ${gan.quality.toFixed(4)}, modes ${gan.modes}/4`,
  )
  if (gan.quality > 0.08) failures.push('gan samples too far from real patterns')
  if (gan.modes < 2) failures.push('gan mode-collapsed to a single pattern class')
}

// --- GNN
{
  const gnn = MODELS.gnn
  console.log(`GNN held-out node accuracy: ${(gnn.accuracy * 100).toFixed(1)}%`)
  if (gnn.accuracy < 0.8) failures.push('gnn accuracy too low')
  const gnnL = (await import('../src/nn/gnn')).buildGNNTask('l')
  console.log(`GNN L held-out node accuracy: ${(gnnL.accuracy * 100).toFixed(1)}%`)
  if (gnnL.accuracy < 0.8) failures.push('gnn L accuracy too low')
}

// --- Agent router
{
  const agent = MODELS.agent
  console.log(`Agent router accuracy: ${(agent.accuracy * 100).toFixed(1)}%`)
  if (agent.accuracy < 0.9) failures.push('agent router accuracy too low')
}

// --- Mamba (same task as RNN/LSTM — the comparison is the point)
{
  const mamba = MODELS.mamba
  console.log(
    `Mamba final loss: ${mamba.finalLoss.toFixed(3)} (vs RNN ${MODELS.rnn.finalLoss.toFixed(3)}, LSTM ${MODELS.lstm.finalLoss.toFixed(3)})`,
  )
  if (mamba.finalLoss > 2.0) failures.push('mamba did not learn')
  const mambaL = (await import('../src/nn/mamba')).buildMambaTask('l')
  console.log(`Mamba L final loss: ${mambaL.finalLoss.toFixed(3)}`)
  if (mambaL.finalLoss > 1.5) failures.push('mamba L did not learn')
}

// --- ViT
{
  const vit = MODELS.vit
  console.log(`ViT held-out accuracy: ${(vit.accuracy * 100).toFixed(1)}% (loss ${vit.finalLoss.toFixed(3)})`)
  if (vit.accuracy < 0.85) failures.push('vit accuracy too low')
}

// --- large-scale variants (every arch supports s/m/l; L must still learn)
{
  const { buildRNNTask, buildLSTMTask } = await import('../src/nn/rnn')
  const { buildDiffusionTask } = await import('../src/nn/diffusion')
  const { buildGANTask } = await import('../src/nn/gan')
  const { buildAETask, buildTextTask } = await import('../src/nn/models')
  const { cnnSampleOfSize } = await import('../src/nn/models')

  const t0 = Date.now()
  const rnnL = buildRNNTask('l')
  const lstmL = buildLSTMTask('l')
  const aeL = buildAETask('l')
  const diffL = buildDiffusionTask(cnnSampleOfSize, 'l')
  const ganL = buildGANTask(cnnSampleOfSize, 'l')
  buildTextTask('l')
  console.log(
    `L-scale: RNN ${rnnL.finalLoss.toFixed(2)}, LSTM ${lstmL.finalLoss.toFixed(2)}, AE ${aeL.finalMSE.toFixed(4)}, ` +
      `DIFF ${diffL.finalLoss.toFixed(3)}, GAN q ${ganL.quality.toFixed(4)}/${ganL.modes} modes (${Date.now() - t0}ms)`,
  )
  if (rnnL.finalLoss > 2.0) failures.push('rnn L did not learn')
  if (lstmL.finalLoss > 2.0) failures.push('lstm L did not learn')
  if (aeL.finalMSE > 0.03) failures.push('ae L reconstruction too poor')
  if (diffL.finalLoss > 0.35) failures.push('diffusion L did not learn')
  if (ganL.quality > 0.09 || ganL.modes < 2) failures.push('gan L failed')

  const llmL = (await import('../src/nn/transformer')).buildLLMTask('dense', 'l')
  const llmAccL = evalLLMAccuracy(llmL)
  console.log(`L-scale LLM: loss ${llmL.finalLoss.toFixed(3)}, top-1 ${(llmAccL * 100).toFixed(1)}%`)
  if (llmL.finalLoss > 2.0 || llmAccL < 0.4) failures.push('llm L did not learn')
}

// --- tiny transformer
{
  const moe = (await import('../src/nn/transformer')).buildLLMTask('moe')
  const moeAcc = evalLLMAccuracy(moe)
  console.log(`MoE transformer loss: ${moe.finalLoss.toFixed(3)}, next-char top-1: ${(moeAcc * 100).toFixed(1)}%`)
  if (moe.finalLoss > 2.0 || moeAcc < 0.25) failures.push('moe llm did not learn')
}

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
