// Quick engine sanity check: run `npm run sanity`.
import { MODELS } from '../src/nn/models'
import { forwardMLP, argmax } from '../src/nn/mlp'
import { forwardCNN } from '../src/nn/cnn'
import { mulberry32 } from '../src/nn/rng'

const rng = mulberry32(999)

let mlpOk = 0
const MLP_N = 60
for (let i = 0; i < MLP_N; i++) {
  const cls = i % MODELS.mlp.classCount
  const x = MODELS.mlp.makeSample(cls, rng)
  const traces = forwardMLP(MODELS.mlp.model, x)
  if (argmax(traces[traces.length - 1].a) === cls) mlpOk++
}
console.log(`MLP accuracy on fresh samples: ${mlpOk}/${MLP_N}`)

let cnnOk = 0
const CNN_N = 80
for (let i = 0; i < CNN_N; i++) {
  const cls = i % MODELS.cnn.classCount
  const x = MODELS.cnn.makeSample(cls, rng)
  const steps = forwardCNN(MODELS.cnn.model, x)
  const last = steps[steps.length - 1]
  if (last.kind === 'vector' && argmax(last.a) === cls) cnnOk++
}
console.log(`CNN accuracy on fresh samples: ${cnnOk}/${CNN_N}`)

const probs = (() => {
  const steps = forwardCNN(MODELS.cnn.model, MODELS.cnn.makeSample(0, rng))
  const last = steps[steps.length - 1]
  return last.kind === 'vector' ? last.a.map((v) => v.toFixed(3)).join(', ') : ''
})()
console.log(`Sample CNN output distribution: [${probs}]`)

if (mlpOk / MLP_N < 0.85 || cnnOk / CNN_N < 0.85) {
  throw new Error('Accuracy too low — tune training.')
}
console.log('Sanity OK')
