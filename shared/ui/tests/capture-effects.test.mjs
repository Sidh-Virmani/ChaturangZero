import { test } from 'node:test'
import assert from 'node:assert/strict'
import * as THREE from 'three'
import { createCaptureEffects, CAPTURE_IMPACT_MS, CAPTURE_FINISH_MS } from '../src/three/captureEffects.ts'
import { IMPACT_MS, CAPTURE_MS } from '../src/three/motion.ts'

const kinds = ['raja', 'mantri', 'ratha', 'gaja', 'ashva', 'padati']
const from = new THREE.Vector3(-2, .34, 1)
const to = new THREE.Vector3(1, .34, -2)

function fixture(t) {
  const parent = new THREE.Group()
  const effects = createCaptureEffects(parent)
  assert.equal(parent.children.length, 1)
  const root = parent.children[0]
  t.after(() => effects.dispose())
  return { parent, root, effects }
}

function resources(root) {
  const geometries = new Set(), materials = new Set(), instances = new Set()
  root.traverse(object => {
    if (object.isMesh) {
      geometries.add(object.geometry)
      materials.add(object.material)
    }
    if (object.isInstancedMesh) instances.add(object)
  })
  return { geometries, materials, instances }
}

function visibleEvents(root) {
  return root.children.filter(event => event.visible)
}

test('capture particles and actors use one impact and finish clock', () => {
  assert.equal(CAPTURE_IMPACT_MS, IMPACT_MS)
  assert.equal(CAPTURE_FINISH_MS, CAPTURE_MS)
  assert.ok(CAPTURE_IMPACT_MS > 0 && CAPTURE_FINISH_MS > CAPTURE_IMPACT_MS)
})

for (const kind of kinds) for (const side of ['white', 'black']) {
  test(`${side} ${kind}: approach and impact stay finite, reuse assets, and expire`, t => {
    const { root, effects } = fixture(t)
    const before = resources(root)
    const start = 100
    effects.play(kind, side, from, to, start)
    assert.equal(visibleEvents(root).length, 1)
    for (const elapsed of [0, 80, 240, CAPTURE_IMPACT_MS - 1, CAPTURE_IMPACT_MS, 760, 1020, CAPTURE_FINISH_MS - 1]) {
      effects.update(start + elapsed, false)
      assert.equal(visibleEvents(root).length, 1)
      let visibleInstances = 0
      root.traverseVisible(object => {
        assert.ok(object.position.toArray().every(Number.isFinite))
        assert.ok(object.scale.toArray().every(Number.isFinite))
        if (!object.isInstancedMesh) return
        visibleInstances++
        for (const scalar of object.instanceMatrix.array) assert.ok(Number.isFinite(scalar))
      })
      assert.ok(visibleInstances > 0, 'An active event must render its approach or impact instances')
    }
    const after = resources(root)
    assert.deepEqual(after.geometries, before.geometries, 'Frame updates must reuse geometry objects')
    assert.deepEqual(after.materials, before.materials, 'Frame updates must reuse material objects')
    assert.deepEqual(after.instances, before.instances, 'Frame updates must reuse instance buffers')
    assert.deepEqual(from.toArray(), [-2, .34, 1], 'Effects must not mutate the caller\'s source')
    assert.deepEqual(to.toArray(), [1, .34, -2], 'Effects must not mutate the caller\'s destination')
    effects.update(start + CAPTURE_FINISH_MS, false)
    assert.equal(visibleEvents(root).length, 0)
  })
}

test('rapid captures keep at most four events and reuse the oldest slot', t => {
  const { root, effects } = fixture(t)
  const before = resources(root)
  for (let index = 0; index < 40; index++) {
    effects.play(kinds[index % kinds.length], index % 2 ? 'black' : 'white', from, to, 2000 + index * 10)
    effects.update(2000 + index * 10 + CAPTURE_IMPACT_MS, false)
    assert.ok(visibleEvents(root).length <= 4)
  }
  assert.equal(root.children.length, 4, 'The pool must not grow during rapid inputs')
  assert.equal(visibleEvents(root).length, 4)
  assert.deepEqual(resources(root), before)
  effects.update(2390 + CAPTURE_FINISH_MS, false)
  assert.equal(visibleEvents(root).length, 0)
})

test('reduced motion and clear remove every active event, allowing a fresh capture', t => {
  const { root, effects } = fixture(t)
  for (const kind of kinds) effects.play(kind, 'white', from, to, 100)
  effects.update(100 + CAPTURE_IMPACT_MS, false)
  assert.equal(visibleEvents(root).length, 4)
  effects.update(100 + CAPTURE_IMPACT_MS + 1, true)
  assert.equal(visibleEvents(root).length, 0)
  effects.play('ashva', 'black', from, to, 2000)
  effects.update(2000 + CAPTURE_IMPACT_MS, false)
  assert.equal(visibleEvents(root).length, 1)
  effects.clear()
  effects.clear()
  assert.equal(visibleEvents(root).length, 0)
})

test('effects never intercept piece or square pointer rays', t => {
  const { root, effects } = fixture(t)
  effects.play('gaja', 'white', from, to, 100)
  effects.update(100 + CAPTURE_IMPACT_MS + 180, false)
  const raycaster = new THREE.Raycaster(new THREE.Vector3(to.x, 10, to.z), new THREE.Vector3(0, -1, 0))
  const hits = []
  root.traverse(object => { if (object.isMesh) object.raycast(raycaster, hits) })
  assert.deepEqual(hits, [])
  root.updateMatrixWorld(true)
  assert.deepEqual(raycaster.intersectObjects(root.children, true), [])
})

test('teardown removes the scene root and disposes each shared resource exactly once', t => {
  const { parent, root, effects } = fixture(t)
  const owned = resources(root)
  const disposals = new Map()
  const all = [...owned.geometries, ...owned.materials, ...owned.instances]
  assert.ok(owned.geometries.size > 0 && owned.materials.size > 0 && owned.instances.size > 0)
  for (const resource of all) {
    disposals.set(resource, 0)
    resource.addEventListener('dispose', () => disposals.set(resource, disposals.get(resource) + 1))
  }
  effects.play('raja', 'white', from, to, 100)
  effects.update(100 + CAPTURE_IMPACT_MS + 40, false)
  effects.dispose()
  effects.dispose()
  effects.play('raja', 'black', from, to, 2000)
  effects.update(2000 + CAPTURE_IMPACT_MS, false)
  assert.equal(parent.children.length, 0)
  assert.equal(root.children.length, 0)
  for (const resource of all) assert.equal(disposals.get(resource), 1)
})
