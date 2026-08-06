import { Canvas } from '@react-three/fiber'
import { SceneRoot } from './scene/SceneRoot'
import { DEFAULT_VIEW } from './scene/layout'
import { Hud } from './ui/Hud'
import { useKeyboard } from './hooks/useKeyboard'
import { useStore } from './store'

const aspect = typeof window !== 'undefined' ? window.innerWidth / window.innerHeight : 1.6
const camFactor = Math.min(2.8, Math.max(1, 1.15 / aspect))

const initialCamera = DEFAULT_VIEW.mlp.position.map((v) => v * camFactor) as [number, number, number]

export default function App() {
  useKeyboard()
  return (
    <div className="app">
      <Canvas
        dpr={[1, 2]}
        camera={{ position: initialCamera, fov: 42 }}
        onPointerMissed={() => useStore.getState().select(null)}
      >
        <SceneRoot />
      </Canvas>
      <Hud />
    </div>
  )
}
