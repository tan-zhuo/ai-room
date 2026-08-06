import { Canvas } from '@react-three/fiber'
import { SceneRoot } from './scene/SceneRoot'
import { DEFAULT_VIEW } from './scene/layout'
import { Hud } from './ui/Hud'
import { useKeyboard } from './hooks/useKeyboard'
import { useStore } from './store'

export default function App() {
  useKeyboard()
  return (
    <div className="app">
      <Canvas
        dpr={[1, 2]}
        camera={{ position: DEFAULT_VIEW.mlp.position, fov: 42 }}
        onPointerMissed={() => useStore.getState().select(null)}
      >
        <SceneRoot />
      </Canvas>
      <Hud />
    </div>
  )
}
