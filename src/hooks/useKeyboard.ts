import { useEffect } from 'react'
import { useStore } from '../store'
import { positionOf } from '../scene/layout'

export function useKeyboard() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const s = useStore.getState()
      switch (e.code) {
        case 'Space':
          e.preventDefault()
          s.togglePlay()
          break
        case 'ArrowRight':
          e.preventDefault()
          s.nextStep()
          break
        case 'ArrowLeft':
          e.preventDefault()
          s.prevStep()
          break
        case 'KeyR':
          s.reset()
          break
        case 'Digit1':
          s.setArch('mlp')
          break
        case 'Digit2':
          s.setArch('cnn')
          break
        case 'Digit3':
          s.showToast('toast.comingSoon')
          break
        case 'KeyL':
          s.cycleLang()
          break
        case 'KeyF':
          if (s.selected) s.requestFocus(positionOf(s.arch, s.selected))
          break
        case 'Escape':
          if (s.helpOpen) s.toggleHelp()
          else s.select(null)
          break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
}
