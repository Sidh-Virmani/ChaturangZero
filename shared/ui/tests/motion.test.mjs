import { test } from 'node:test'
import assert from 'node:assert/strict'
import { movePose, defeatPose, isSequentialMove, CAPTURE_MS, MOVE_MS, IMPACT_MS } from '../src/three/motion.ts'

const kinds = ['raja', 'mantri', 'ratha', 'gaja', 'ashva', 'padati']
const rest = { progress: 1, lift: 0, pitch: 0, yaw: 0, roll: 0, scale: 1 }
for (const kind of kinds) {
  test(`${kind}: movement lands exactly, capture contacts at the shared impact clock`, () => {
    for (const capture of [false, true]) {
      assert.equal(movePose(kind, 0, capture).progress, 0)
      const duration = capture ? CAPTURE_MS : MOVE_MS
      for (let elapsed = 0; elapsed < duration; elapsed += 17) {
        const pose = movePose(kind, elapsed, capture)
        assert.ok(Object.values(pose).every(Number.isFinite))
        assert.ok(pose.progress >= 0 && pose.progress <= 1)
        assert.ok(pose.lift >= 0 && pose.scale > 0)
      }
      assert.deepEqual(movePose(kind, duration, capture), rest)
      assert.deepEqual(movePose(kind, duration + 10000, capture), rest)
      assert.deepEqual(movePose(kind, 10, capture, true), rest)
    }
    assert.equal(movePose(kind, IMPACT_MS, true).progress, 1)
  })
  test(`${kind}: a victim stays intact until contact and disappears before the capture completes`, () => {
    assert.equal(defeatPose(kind, IMPACT_MS - 1).scale, 1)
    assert.equal(defeatPose(kind, 0).displacement, 0)
    assert.equal(defeatPose(kind, CAPTURE_MS).scale, 0)
    assert.equal(defeatPose(kind, 0, true).scale, 0)
  })
}
test('history continuity distinguishes reply, reset and a replaced history', () => {
  const first = { id: 'move-1', from: 'e2', to: 'e3', piece: 'padati', side: 'white' }
  const second = { id: 'move-2', from: 'd7', to: 'd6', piece: 'padati', side: 'black' }
  assert.equal(isSequentialMove({ moveHistory: [first] }, { moveHistory: [first, second], lastMove: second }), true)
  assert.equal(isSequentialMove({ moveHistory: [first] }, { moveHistory: [], lastMove: null }), false)
  assert.equal(isSequentialMove({ moveHistory: [first] }, { moveHistory: [{ ...first, to: 'e4' }, second], lastMove: second }), false)
  assert.equal(isSequentialMove({ moveHistory: [] }, { moveHistory: [first, second], lastMove: second }), false)
})
