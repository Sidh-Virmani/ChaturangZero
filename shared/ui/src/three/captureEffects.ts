import * as THREE from 'three'
import type { PieceKind, Side } from '../types/game'

/** Shared with the board choreography; all clocks are performance.now() milliseconds. */
export const CAPTURE_IMPACT_MS = 620
export const CAPTURE_FINISH_MS = 1450

const POOL_SIZE = 4
const TRAIL_SEGMENTS = 20
const SPARK_COUNT = 36
const GLYPH_COUNT = 16
const UP = new THREE.Vector3(0, 1, 0)
const TAU = Math.PI * 2
const clamp = (value: number) => Math.max(0, Math.min(1, value))
const smooth = (value: number) => value * value * (3 - 2 * value)

interface EffectSlot {
  group: THREE.Group
  active: boolean
  start: number
  kind: PieceKind
  from: THREE.Vector3
  to: THREE.Vector3
  direction: THREE.Vector3
  tangent: THREE.Vector3
  trail: THREE.InstancedMesh
  sparks: THREE.InstancedMesh
  glyphs: THREE.InstancedMesh
  petals: THREE.InstancedMesh
  rings: THREE.Mesh[]
  crescent: THREE.Mesh
  materials: {
    trail: THREE.MeshBasicMaterial
    sparks: THREE.MeshBasicMaterial
    glyphs: THREE.MeshBasicMaterial
    petals: THREE.MeshBasicMaterial
    rings: THREE.MeshBasicMaterial
    crescent: THREE.MeshBasicMaterial
  }
}

/**
 * Decorative capture choreography only. Nothing here reads or changes game state.
 * Every geometry, material, and instance buffer is built once and reused by a
 * bounded pool. Effects do not cast shadows, intercept pointer rays, or own timers.
 */
export function createCaptureEffects(parent: THREE.Group) {
  let disposed = false
  const root = new THREE.Group()
  root.name = 'Capture choreography'
  parent.add(root)

  const streakGeometry = new THREE.CylinderGeometry(1, 1, 1, 6, 1)
  const sparkGeometry = new THREE.OctahedronGeometry(1)
  const glyphGeometry = new THREE.ConeGeometry(1, 1, 4)
  const ringGeometry = new THREE.RingGeometry(.96, 1, 64)
  const crescentGeometry = new THREE.RingGeometry(.94, 1, 64, 1, Math.PI * .12, Math.PI * 1.3)
  const petalShape = new THREE.Shape()
  petalShape.moveTo(0, 0)
  petalShape.bezierCurveTo(-.18, .17, -.2, .39, 0, .62)
  petalShape.bezierCurveTo(.2, .39, .18, .17, 0, 0)
  const petalGeometry = new THREE.ShapeGeometry(petalShape, 10)
  const geometries = [streakGeometry, sparkGeometry, glyphGeometry, ringGeometry, crescentGeometry, petalGeometry]

  const dummy = new THREE.Object3D()
  const first = new THREE.Vector3()
  const second = new THREE.Vector3()
  const delta = new THREE.Vector3()
  const rotation = new THREE.Quaternion()
  const gold = new THREE.Color('#f5cc83')
  const mint = new THREE.Color('#93ead9')
  const pearl = new THREE.Color('#fff3d4')
  // Deterministic distributions prevent flickering and make replays reproducible.
  const seeds = Array.from({ length: SPARK_COUNT }, (_, index) => ({
    angle: index * 2.399963229728653,
    radial: .5 + ((index * 17) % 31) / 31,
    lift: .32 + ((index * 13) % 23) / 23,
    size: .018 + ((index * 7) % 11) / 650,
  }))

  const material = () => new THREE.MeshBasicMaterial({
    color: gold,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  })
  const makeInstances = (geometry: THREE.BufferGeometry, mat: THREE.Material, count: number) => {
    const mesh = new THREE.InstancedMesh(geometry, mat, count)
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    mesh.frustumCulled = false
    mesh.raycast = () => {}
    mesh.renderOrder = 5
    return mesh
  }
  const pool: EffectSlot[] = Array.from({ length: POOL_SIZE }, () => {
    const group = new THREE.Group()
    group.visible = false
    root.add(group)
    const materials = { trail: material(), sparks: material(), glyphs: material(), petals: material(), rings: material(), crescent: material() }
    const trail = makeInstances(streakGeometry, materials.trail, TRAIL_SEGMENTS)
    const sparks = makeInstances(sparkGeometry, materials.sparks, SPARK_COUNT)
    const glyphs = makeInstances(glyphGeometry, materials.glyphs, GLYPH_COUNT)
    const petals = makeInstances(petalGeometry, materials.petals, 8)
    const rings = Array.from({ length: 3 }, () => {
      const mesh = new THREE.Mesh(ringGeometry, materials.rings)
      mesh.rotation.x = -Math.PI / 2
      mesh.raycast = () => {}
      mesh.renderOrder = 4
      group.add(mesh)
      return mesh
    })
    const crescent = new THREE.Mesh(crescentGeometry, materials.crescent)
    crescent.raycast = () => {}
    crescent.renderOrder = 5
    group.add(trail, sparks, glyphs, petals, crescent)
    return {
      group, active: false, start: 0, kind: 'padati',
      from: new THREE.Vector3(), to: new THREE.Vector3(),
      direction: new THREE.Vector3(), tangent: new THREE.Vector3(),
      trail, sparks, glyphs, petals, rings, crescent, materials,
    }
  })

  const hide = (slot: EffectSlot) => {
    slot.active = false
    slot.group.visible = false
  }

  const clear = () => {
    for (const slot of pool) hide(slot)
  }

  const play = (kind: PieceKind, side: Side, from: THREE.Vector3, to: THREE.Vector3, now: number) => {
    if (disposed || !Number.isFinite(now)) return
    // A burst of fast inputs reuses the oldest slot; resources and draw calls stay bounded.
    let slot = pool.find(candidate => !candidate.active)
    if (!slot) slot = pool.reduce((oldest, candidate) => candidate.start < oldest.start ? candidate : oldest)
    slot.kind = kind
    slot.start = now
    slot.from.copy(from)
    slot.to.copy(to)
    slot.direction.copy(to).sub(from).setY(0)
    if (slot.direction.lengthSq() < .0001) slot.direction.set(0, 0, -1)
    slot.direction.normalize()
    slot.tangent.set(-slot.direction.z, 0, slot.direction.x)
    slot.active = true
    slot.group.visible = true
    const color = side === 'white' ? gold : mint
    slot.materials.trail.color.copy(color)
    slot.materials.glyphs.color.copy(color)
    slot.materials.petals.color.copy(color)
    slot.materials.rings.color.copy(color)
    slot.materials.crescent.color.copy(color)
    slot.materials.sparks.color.copy(color).lerp(pearl, .45)
    for (const mat of Object.values(slot.materials)) mat.opacity = 0
    slot.trail.visible = true
    slot.sparks.visible = false
    slot.glyphs.visible = false
    slot.petals.visible = false
    slot.crescent.visible = false
    for (const ring of slot.rings) ring.visible = false
  }

  // The trail follows the same from/to path as the actor, with a characteristic accent.
  const trajectory = (slot: EffectSlot, progress: number, out: THREE.Vector3) => {
    const arc = Math.sin(Math.PI * progress)
    out.lerpVectors(slot.from, slot.to, smooth(progress))
    out.y += .19
    switch (slot.kind) {
      case 'ashva': out.y += arc * .95; break
      case 'gaja': out.y += arc * .34; break
      case 'mantri': out.addScaledVector(slot.tangent, arc * .36); out.y += arc * .52; break
      case 'raja': out.y += arc * .42; break
      case 'ratha': out.y += .12; break
      case 'padati': out.y += arc * .18; break
    }
  }

  const drawSegment = (mesh: THREE.InstancedMesh, index: number, a: THREE.Vector3, b: THREE.Vector3, width: number) => {
    delta.copy(b).sub(a)
    const length = delta.length()
    dummy.position.copy(a).add(b).multiplyScalar(.5)
    dummy.scale.set(width, Math.max(.0001, length), width)
    if (length > .0001) rotation.setFromUnitVectors(UP, delta.multiplyScalar(1 / length))
    else rotation.identity()
    dummy.quaternion.copy(rotation)
    dummy.updateMatrix()
    mesh.setMatrixAt(index, dummy.matrix)
  }

  const updateTrail = (slot: EffectSlot, elapsed: number) => {
    const progress = clamp(elapsed / CAPTURE_IMPACT_MS)
    const tail = Math.max(0, progress - (slot.kind === 'ratha' ? .38 : .26))
    const afterImpact = clamp((elapsed - CAPTURE_IMPACT_MS) / 270)
    slot.trail.visible = elapsed < CAPTURE_IMPACT_MS + 270
    slot.materials.trail.opacity = .65 * smooth(clamp(progress * 4)) * (1 - progress * .18) * (1 - afterImpact)
    // Preserve a clean flash of the completed lance/arc immediately at impact.
    if (elapsed >= CAPTURE_IMPACT_MS) slot.materials.trail.opacity = .52 * (1 - afterImpact)
    for (let index = 0; index < TRAIL_SEGMENTS; index++) {
      const start = tail + (progress - tail) * index / TRAIL_SEGMENTS
      const end = tail + (progress - tail) * (index + 1) / TRAIL_SEGMENTS
      trajectory(slot, start, first)
      trajectory(slot, end, second)
      const taper = .006 + .016 * (index + 1) / TRAIL_SEGMENTS
      drawSegment(slot.trail, index, first, second, taper)
    }
    slot.trail.instanceMatrix.needsUpdate = true
  }

  const updateRings = (slot: EffectSlot, phase: number, fade: number) => {
    const isLotus = slot.kind === 'gaja'
    const isSigil = slot.kind === 'mantri'
    const isCrown = slot.kind === 'raja'
    slot.materials.rings.opacity = fade * (isLotus ? .5 : .34)
    const count = isLotus || isSigil ? 3 : isCrown ? 2 : 1
    for (let index = 0; index < slot.rings.length; index++) {
      const ring = slot.rings[index]
      ring.visible = index < count
      const delayed = Math.max(0, phase - index * .12)
      const radius = isSigil ? .46 + index * .16 + Math.sin(phase * Math.PI) * .1 : .15 + delayed * (isLotus ? 1.35 : .95)
      ring.position.copy(slot.to)
      ring.position.y += .028 + index * .006 + (isSigil ? index * .12 : 0)
      ring.scale.setScalar(Math.max(.01, radius))
      ring.rotation.set(-Math.PI / 2, 0, (index % 2 ? -1 : 1) * phase * .65)
    }
    slot.crescent.visible = slot.kind === 'ashva'
    slot.materials.crescent.opacity = fade * .7
    if (slot.crescent.visible) {
      slot.crescent.position.copy(slot.to)
      slot.crescent.position.y += .6
      slot.crescent.rotation.set(.18, Math.atan2(slot.direction.x, slot.direction.z), phase * .8)
      slot.crescent.scale.setScalar(.2 + smooth(phase) * .72)
    }
  }

  const updateSparks = (slot: EffectSlot, phase: number, fade: number) => {
    slot.sparks.visible = true
    slot.materials.sparks.opacity = fade * .9
    const travel = 1 - (1 - phase) ** 3
    for (let index = 0; index < SPARK_COUNT; index++) {
      const seed = seeds[index]
      const theta = seed.angle
      const radial = seed.radial * travel
      dummy.position.copy(slot.to)
      if (slot.kind === 'ratha' || slot.kind === 'padati') {
        // Forward fan suggests the force of a lance/spear, never obscuring a whole rank.
        dummy.position.addScaledVector(slot.direction, radial * (slot.kind === 'ratha' ? 1.2 : .8))
        dummy.position.addScaledVector(slot.tangent, Math.sin(theta) * radial * .48)
        dummy.position.y += .12 + Math.sin(Math.PI * phase) * seed.lift * .65
      } else if (slot.kind === 'mantri') {
        const orbit = theta + phase * TAU * .35
        dummy.position.x += Math.cos(orbit) * (.36 + radial * .38)
        dummy.position.z += Math.sin(orbit) * (.36 + radial * .38)
        dummy.position.y += .18 + seed.lift * Math.sin(Math.PI * phase) * .6
      } else {
        const spread = slot.kind === 'gaja' ? 1.05 : .82
        dummy.position.x += Math.cos(theta) * radial * spread
        dummy.position.z += Math.sin(theta) * radial * spread
        dummy.position.y += .08 + Math.sin(Math.PI * phase) * seed.lift * (slot.kind === 'raja' ? 1.35 : .85)
      }
      dummy.rotation.set(theta + phase * 2, theta * .6, phase * 3)
      dummy.scale.setScalar(seed.size * (.65 + fade * .7))
      dummy.updateMatrix()
      slot.sparks.setMatrixAt(index, dummy.matrix)
    }
    slot.sparks.instanceMatrix.needsUpdate = true
  }

  const updateGlyphs = (slot: EffectSlot, phase: number, fade: number) => {
    slot.glyphs.visible = true
    slot.materials.glyphs.opacity = fade * .62
    const opening = smooth(clamp(phase * 2.3))
    for (let index = 0; index < GLYPH_COUNT; index++) {
      const theta = index * TAU / GLYPH_COUNT
      const sin = Math.sin(theta), cos = Math.cos(theta)
      dummy.position.copy(slot.to)
      dummy.quaternion.identity()
      let length = .2 + .25 * opening
      let width = .025
      if (slot.kind === 'raja') {
        const radius = .2 + opening * .5
        dummy.position.x += sin * radius
        dummy.position.z += cos * radius
        dummy.position.y += .26 + opening * .28 + (index % 2 ? .08 : .23)
        length = index % 2 ? .24 : .42
        width = .038
        delta.set(sin * .3, 1, cos * .3).normalize()
        dummy.quaternion.setFromUnitVectors(UP, delta)
      } else if (slot.kind === 'mantri') {
        const orbit = theta + phase * .9
        dummy.position.x += Math.sin(orbit) * .61
        dummy.position.z += Math.cos(orbit) * .61
        dummy.position.y += .052
        delta.set(Math.sin(orbit), 0, Math.cos(orbit))
        dummy.quaternion.setFromUnitVectors(UP, delta)
        length = index % 2 ? .08 : .18
        width = .016
      } else if (slot.kind === 'gaja') {
        const radius = .2 + opening * .6
        dummy.position.x += sin * radius
        dummy.position.z += cos * radius
        dummy.position.y += .055
        delta.set(sin, .12, cos).normalize()
        dummy.quaternion.setFromUnitVectors(UP, delta)
        length = .14 + opening * .18
        width = .02
      } else if (slot.kind === 'ashva') {
        dummy.position.x += sin * opening * .58
        dummy.position.z += cos * opening * .58
        dummy.position.y += .25 + cos * .3 * opening
        delta.set(sin, .5, cos).normalize()
        dummy.quaternion.setFromUnitVectors(UP, delta)
        length = .16
        width = .018
      } else {
        // Two disciplined crossed slashes for Padati, a fan of lances for Ratha.
        const cross = slot.kind === 'padati' ? (index % 2 ? 1 : -1) : Math.sin(theta) * .75
        dummy.position.addScaledVector(slot.direction, .12 + opening * .28)
        dummy.position.addScaledVector(slot.tangent, cross * opening * .32)
        dummy.position.y += .3 + (index / GLYPH_COUNT) * .32
        delta.copy(slot.direction).multiplyScalar(slot.kind === 'ratha' ? 1 : .25)
        delta.addScaledVector(slot.tangent, cross).setY(.45).normalize()
        dummy.quaternion.setFromUnitVectors(UP, delta)
        length = .35 + opening * .3
        width = slot.kind === 'padati' ? .014 : .02
      }
      dummy.scale.set(width * fade, length * (.4 + fade * .6), width * fade)
      dummy.updateMatrix()
      slot.glyphs.setMatrixAt(index, dummy.matrix)
    }
    slot.glyphs.instanceMatrix.needsUpdate = true

    slot.petals.visible = slot.kind === 'gaja'
    slot.materials.petals.opacity = fade * .24
    if (slot.petals.visible) {
      for (let index = 0; index < 8; index++) {
        const theta = index * TAU / 8
        dummy.position.copy(slot.to)
        dummy.position.y += .035
        dummy.rotation.set(-Math.PI / 2, 0, theta)
        dummy.scale.setScalar(.42 + opening * .74)
        dummy.updateMatrix()
        slot.petals.setMatrixAt(index, dummy.matrix)
      }
      slot.petals.instanceMatrix.needsUpdate = true
    }
  }

  const update = (now: number, reduced: boolean) => {
    if (disposed) return
    if (reduced || !Number.isFinite(now)) { clear(); return }
    for (const slot of pool) {
      if (!slot.active) continue
      const elapsed = now - slot.start
      if (elapsed < 0) { slot.group.visible = false; continue }
      if (elapsed >= CAPTURE_FINISH_MS) { hide(slot); continue }
      slot.group.visible = true
      updateTrail(slot, elapsed)
      if (elapsed < CAPTURE_IMPACT_MS) continue
      const phase = clamp((elapsed - CAPTURE_IMPACT_MS) / (CAPTURE_FINISH_MS - CAPTURE_IMPACT_MS))
      const fade = (1 - phase) ** 1.7
      updateRings(slot, phase, fade)
      updateSparks(slot, phase, fade)
      updateGlyphs(slot, phase, fade)
    }
  }

  const dispose = () => {
    if (disposed) return
    disposed = true
    clear()
    parent.remove(root)
    for (const slot of pool) {
      slot.trail.dispose()
      slot.sparks.dispose()
      slot.glyphs.dispose()
      slot.petals.dispose()
      for (const mat of Object.values(slot.materials)) mat.dispose()
    }
    for (const geometry of geometries) geometry.dispose()
    root.clear()
  }

  return { play, update, clear, dispose }
}
