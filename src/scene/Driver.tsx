import { useFrame } from '@react-three/fiber'
import { flow, stepDuration, totalSteps, useStore } from '../store'

/** Advances playback: fills flow.phase each frame, commits steps when it wraps. */
export function Driver() {
  useFrame((_, dt) => {
    const s = useStore.getState()
    const total = totalSteps(s.arch)
    const generating = s.arch === 'llm' && s.llmGenerating
    if (s.playing && s.step >= total) {
      flow.hold += dt
      if (generating) {
        // brief pause on the prediction, then commit the char and run again
        if (flow.hold > 0.45) {
          flow.hold = 0
          useStore.getState().commitGeneratedChar()
        }
        return
      }
      // hold on the finished network, then loop
      if (flow.hold > 1.6) {
        flow.hold = 0
        flow.phase = 0
        useStore.setState({ step: 0 })
      }
      return
    }
    if (!s.playing && !s.transitioning) return
    if (s.step >= total) {
      useStore.setState({ transitioning: false })
      return
    }
    // generation mode runs the pipeline much faster so characters stream out
    const dur = stepDuration(s.arch, s.step) * (generating ? 0.22 : 1)
    flow.phase += (dt / dur) * s.speed * (s.transitioning ? 1.7 : 1)
    if (flow.phase >= 1) {
      flow.phase = 0
      useStore.getState().finishStep()
    }
  })
  return null
}
