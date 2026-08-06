import { useFrame } from '@react-three/fiber'
import { flow, stepDuration, totalSteps, useStore } from '../store'

/** Advances playback: fills flow.phase each frame, commits steps when it wraps. */
export function Driver() {
  useFrame((_, dt) => {
    const s = useStore.getState()
    const total = totalSteps(s.arch)
    if (s.playing && s.step >= total) {
      // hold on the finished network, then loop
      flow.hold += dt
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
    const dur = stepDuration(s.arch, s.step)
    flow.phase += (dt / dur) * s.speed * (s.transitioning ? 1.7 : 1)
    if (flow.phase >= 1) {
      flow.phase = 0
      useStore.getState().finishStep()
    }
  })
  return null
}
