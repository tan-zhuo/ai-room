import { Grid, Stars } from '@react-three/drei'
import { Bloom, EffectComposer } from '@react-three/postprocessing'
import { useStore } from '../store'
import { DenseScene } from './DenseScene'
import { CNNScene } from './CNNScene'
import { LLMScene } from './LLMScene'
import { Driver } from './Driver'
import { CameraRig } from './CameraRig'
import { LayerHighlight } from './LayerHighlight'

export function SceneRoot() {
  const arch = useStore((s) => s.arch)
  const version = useStore((s) => s.modelsVersion)
  const llmText = useStore((s) => s.llmText)
  return (
    <>
      <color attach="background" args={['#04060c']} />
      <fog attach="fog" args={['#04060c', 26, 90]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[6, 10, 8]} intensity={0.6} />
      <Stars radius={70} depth={30} count={2400} factor={3.2} saturation={0} fade speed={0.5} />
      <Grid
        position={[0, -7.2, 0]}
        args={[90, 90]}
        cellSize={1.2}
        sectionSize={6}
        cellColor="#0d1a2b"
        sectionColor="#16324f"
        fadeDistance={55}
        fadeStrength={2.5}
        infiniteGrid
      />
      {(arch === 'mlp' || arch === 'text') && <DenseScene key={`${arch}${version}`} arch={arch} />}
      {arch === 'cnn' && <CNNScene key={`cnn${version}`} />}
      {arch === 'llm' && <LLMScene key={`llm${llmText}`} />}
      <LayerHighlight />
      <Driver />
      <CameraRig />
      <EffectComposer>
        <Bloom mipmapBlur intensity={0.85} luminanceThreshold={0.32} luminanceSmoothing={0.2} radius={0.75} />
      </EffectComposer>
    </>
  )
}
