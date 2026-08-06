import * as THREE from 'three'

export const COLOR_POS = new THREE.Color('#38d6ff')
export const COLOR_NEG = new THREE.Color('#ff8a3c')
export const COLOR_IDLE = new THREE.Color('#101a28')
export const COLOR_PARTICLE = new THREE.Color('#bdefff')

/** Smoothstep easing for animations. */
export function ease(t: number): number {
  const c = t < 0 ? 0 : t > 1 ? 1 : t
  return c * c * (3 - 2 * c)
}

/**
 * Map an activation value to a color: cyan for positive, orange for negative,
 * brightness by magnitude; goes overbright (>1) so bloom picks it up.
 */
export function activationColor(v: number, scale: number, target: THREE.Color): THREE.Color {
  const t = Math.min(1, Math.abs(v) / (scale || 1))
  const base = v >= 0 ? COLOR_POS : COLOR_NEG
  target.copy(COLOR_IDLE).lerp(base, 0.12 + 0.88 * t)
  if (t > 0.6) target.multiplyScalar(1 + (t - 0.6) * 1.8)
  return target
}

export function weightColor(w: number, maxW: number, brightness: number, target: THREE.Color): THREE.Color {
  const t = Math.min(1, Math.abs(w) / (maxW || 1))
  const base = w >= 0 ? COLOR_POS : COLOR_NEG
  target.copy(base).multiplyScalar((0.06 + 0.5 * t) * brightness)
  return target
}

const UP = new THREE.Vector3(0, 1, 0)
const tmpDir = new THREE.Vector3()

/**
 * Orient `obj` as a unit-Y cylinder stretched between a and b,
 * with the given radius.
 */
export function orientSegment(
  obj: THREE.Object3D,
  a: THREE.Vector3,
  b: THREE.Vector3,
  radius: number,
): void {
  tmpDir.subVectors(b, a)
  const len = tmpDir.length()
  obj.position.copy(a).addScaledVector(tmpDir, 0.5)
  obj.quaternion.setFromUnitVectors(UP, tmpDir.normalize())
  obj.scale.set(radius, len, radius)
}

export interface Segment {
  a: THREE.Vector3
  b: THREE.Vector3
  w: number
  /** index of target node in its layer (for selection highlighting) */
  target: number
}

export function segMaxW(segments: Segment[]): number {
  let m = 0
  for (const s of segments) m = Math.max(m, Math.abs(s.w))
  return m || 1
}
