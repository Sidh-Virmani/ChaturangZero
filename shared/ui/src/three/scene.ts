import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { createPiece, createPieceMaterials, disposeGroup } from './pieces'
import type { BoardState, LegalMove, Piece, Side, SquareId } from '../types/game'
import type { InspectCell, NeuralInspection, NeuralLayer } from '../types/neural'

export interface SceneSettings {
  state: BoardState
  orientation: Side
  selectedSquare: SquareId | null
  legalMoves: LegalMove[]
  disabled: boolean
  inspection: NeuralInspection | null
  exploded: boolean
  focus: string
  channel: number // -1: mean absolute activation
  replay: boolean
  resetCamera: number
}

interface Actor { mesh: THREE.Group; piece: Piece; from: THREE.Vector3; to: THREE.Vector3; started: number; retiring: boolean; hop: number }
interface Stage { layer: NeuralLayer; group: THREE.Group; bars: THREE.InstancedMesh; slab: THREE.Mesh; target: THREE.Vector3; scale: number; values: number[]; label: HTMLSpanElement }
const point = (square: string) => new THREE.Vector3(square.charCodeAt(0) - 97 - 3.5, .34, 4.5 - Number(square[1]))
const ease = (t: number) => t < .5 ? 16 * t ** 5 : 1 - (-2 * t + 2) ** 5 / 2

/** Imperative scene, owned by one React component. No game rules live here. */
export function createBoardScene(host: HTMLDivElement, onSquare: (square: SquareId) => void,
  onHover: (cell: InspectCell | null) => void, onFocus: (id: string) => void, onFailure: () => void) {
  let settings: SceneSettings | null = null
  let disposed = false, raf = 0, visible = true, lastTime = 0
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)')
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75))
  renderer.setClearColor('#151823')
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = .98
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFShadowMap
  renderer.domElement.setAttribute('aria-label', 'Interactive 3D Chaturanga board. Drag to orbit, scroll to zoom. Use the 2D board for keyboard play.')
  renderer.domElement.setAttribute('role', 'img')
  host.appendChild(renderer.domElement)
  const scene = new THREE.Scene()
  const environment = new RoomEnvironment()
  const pmrem = new THREE.PMREMGenerator(renderer)
  const environmentTarget = pmrem.fromScene(environment, .06)
  scene.environment = environmentTarget.texture
  scene.environmentIntensity = .55
  environment.dispose(); pmrem.dispose()
  const camera = new THREE.PerspectiveCamera(38, 1, .1, 150)
  camera.position.set(9, 11, 14)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true; controls.dampingFactor = .085
  controls.enablePan = false; controls.enableZoom = false
  controls.minPolarAngle = .12; controls.maxPolarAngle = Math.PI * .49
  controls.target.set(0, .1, 0)
  const hemi = new THREE.HemisphereLight('#e3e9ff', '#2a2135', 1.45)
  scene.add(hemi)
  const key = new THREE.DirectionalLight('#fff0db', 3)
  key.position.set(-4, 11, 5); key.castShadow = true
  key.shadow.mapSize.set(2048, 2048)
  Object.assign(key.shadow.camera, { left: -7, right: 7, top: 7, bottom: -7, near: .5, far: 28 })
  key.shadow.bias = -.0003; key.shadow.normalBias = .025
  scene.add(key)
  const rim = new THREE.DirectionalLight('#97d6e8', 1.6)
  rim.position.set(4, 5, -7); scene.add(rim)
  const board = new THREE.Group(); scene.add(board)
  const pieces = new THREE.Group(); board.add(pieces)
  const pieceMaterials = createPieceMaterials()
  const navy = new THREE.MeshStandardMaterial({ color: '#202b40', metalness: .42, roughness: .32 })
  const darkTile = new THREE.MeshPhysicalMaterial({ color: '#27364b', roughness: .36, metalness: .18, clearcoat: .3 })
  const lightTile = new THREE.MeshPhysicalMaterial({ color: '#c596a2', roughness: .38, metalness: .04, clearcoat: .3 })
  const gold = pieceMaterials.gold
  const glow = new THREE.MeshStandardMaterial({ color: '#529b9c', emissive: '#419eaa', emissiveIntensity: .9, roughness: .5 })
  const box = (width: number, height: number, depth: number, radius: number, material: THREE.Material, y: number) => {
    const mesh = new THREE.Mesh(new RoundedBoxGeometry(width, height, depth, 3, radius), material)
    mesh.position.y = y; mesh.receiveShadow = true; mesh.castShadow = true; board.add(mesh); return mesh
  }
  box(9.05, .36, 9.05, .11, navy, -.06)
  box(8.9, .11, 8.9, .04, glow, -.26)
  box(9.16, .12, 9.16, .045, navy, -.36)
  box(9.14, .15, 9.14, .045, navy, .17)
  box(8.85, .035, 8.85, .02, gold, .26)
  box(8.79, .045, 8.79, .015, navy, .283)
  const tileGeometry = new RoundedBoxGeometry(.997, .075, .997, 2, .016)
  const tiles: THREE.Mesh[] = []
  for (let rank = 1; rank <= 8; rank++) for (let file = 0; file < 8; file++) {
    const square = `${String.fromCharCode(97 + file)}${rank}` as SquareId
    const tile = new THREE.Mesh(tileGeometry, (file + rank) % 2 === 0 ? lightTile : darkTile)
    tile.position.copy(point(square)); tile.position.y = .298
    tile.receiveShadow = true; tile.userData.square = square; board.add(tile); tiles.push(tile)
  }
  // Fine corner rosettes and a restrained frame keep the board readable.
  for (const x of [-4.24, 4.24]) for (const z of [-4.24, 4.24]) {
    const ornament = new THREE.Mesh(new THREE.OctahedronGeometry(.05), gold)
    ornament.position.set(x, .32, z); board.add(ornament)
  }
  const makeText = (text: string, color = '#cbb48f', width = 256) => {
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = 64
    const context = canvas.getContext('2d')!
    context.font = '500 34px "DM Sans", sans-serif'; context.fillStyle = color
    context.textAlign = 'center'; context.textBaseline = 'middle'; context.fillText(text, width / 2, 32)
    const texture = new THREE.CanvasTexture(canvas)
    const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false })
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width === 64 ? .22 : 1, .25), material)
    mesh.rotation.x = -Math.PI / 2
    return mesh
  }
  for (let index = 0; index < 8; index++) {
    for (const z of [-4.21, 4.21]) {
      const letter = makeText(String.fromCharCode(97 + index), undefined, 64)
      letter.position.set(index - 3.5, .321, z); board.add(letter)
    }
    for (const x of [-4.21, 4.21]) {
      const number = makeText(String(8 - index), undefined, 64)
      number.position.set(x, .321, index - 3.5); board.add(number)
    }
  }
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(100, 100), new THREE.ShadowMaterial({ opacity: .28 }))
  floor.rotation.x = -Math.PI / 2; floor.position.y = -.43; floor.receiveShadow = true; scene.add(floor)
  const templates = new Map<string, THREE.Group>()
  const actors: Actor[] = []
  const markers = new THREE.Group(); board.add(markers)
  const ringGeometry = new THREE.RingGeometry(.36, .395, 48)
  const dotGeometry = new THREE.CircleGeometry(.105, 32)
  const markerMaterial = new THREE.MeshBasicMaterial({ color: '#86ddd4', transparent: true, opacity: .85, depthWrite: false, side: THREE.DoubleSide })
  const lastMaterial = new THREE.MeshBasicMaterial({ color: '#f0c786', transparent: true, opacity: .3, depthWrite: false, side: THREE.DoubleSide })
  const lastGeometry = new THREE.PlaneGeometry(.91, .91)
  const network = new THREE.Group(); scene.add(network)
  let stages: Stage[] = [], networkKey = '', positionKey = '', cameraKey = ''
  let boardOpacity = 1, activeStage = '', hoveredKey = ''
  let boardMaterials: THREE.Material[] = []
  let transition: { start: number; from: THREE.Vector3; to: THREE.Vector3; fromTarget: THREE.Vector3; target: THREE.Vector3 } | null = null
  let zoomVelocity = 0
  let measuredFrames = 0, measuredSince = 0
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2(), dummy = new THREE.Object3D()
  const tempColor = new THREE.Color(), cold = new THREE.Color('#1d3046'), warm = new THREE.Color('#4aa79e')
  const resources = new THREE.Group() // owns cached geometry even after captures / layer refreshes
  const pulseGeometry = new THREE.SphereGeometry(.06, 12, 8)
  const pulseMaterial = new THREE.MeshBasicMaterial({ color: '#bff9ed' })
  const pulses = [new THREE.Mesh(pulseGeometry, pulseMaterial), new THREE.Mesh(pulseGeometry, pulseMaterial)]
  pulses.forEach(pulse => { network.add(pulse); pulse.visible = false })
  const connectors = new THREE.Group(); network.add(connectors)
  const connectorMaterial = new THREE.LineBasicMaterial({ color: '#609397', transparent: true, opacity: .36 })
  const setCamera = (target: THREE.Vector3, offset: THREE.Vector3) => {
    const sign = settings?.orientation === 'black' ? -1 : 1
    offset.x *= sign; offset.z *= sign
    transition = { start: performance.now(), from: camera.position.clone(), to: target.clone().add(offset), fromTarget: controls.target.clone(), target }
    zoomVelocity = 0
  }
  const cameraDestination = () => {
    const ratio = Math.max(1, 1.35 / camera.aspect)
    if (!settings?.exploded) return setCamera(new THREE.Vector3(0, .15, 0), new THREE.Vector3(8, 10, 12.5).multiplyScalar(ratio))
    const focused = stages.find(stage => stage.layer.id === settings?.focus)
    if (focused) {
      const scalarZoom = focused.values.length === 1 ? .6 : 1
      setCamera(focused.target.clone(), new THREE.Vector3(7, 10, 11).multiplyScalar(focused.scale * ratio * scalarZoom))
    } else {
      const bottom = Math.min(-2, ...stages.map(stage => stage.target.y))
      setCamera(new THREE.Vector3(0, bottom * .48, 0), new THREE.Vector3(13, 7, 21).multiplyScalar(ratio * Math.max(1, -bottom / 13)))
    }
  }
  const resize = () => {
    if (disposed) return
    const { width, height } = host.getBoundingClientRect()
    renderer.setSize(width, height, false); camera.aspect = width / Math.max(height, 1); camera.updateProjectionMatrix()
    cameraDestination()
  }
  const observer = new ResizeObserver(resize); observer.observe(host)
  function reconcile(next: BoardState) {
    const key = next.positionKey ?? `${next.currentTurn}:${next.pieces.map(p => p.id).join(',')}`
    if (positionKey === key) return
    const animatedMove = settings && next.moveHistory.length === settings.state.moveHistory.length + 1 ? next.lastMove : null
    const unused = new Set(actors.filter(actor => !actor.retiring))
    const now = performance.now()
    for (const piece of next.pieces) {
      let actor: Actor | undefined
      if (animatedMove?.to === piece.square) actor = [...unused].find(a => a.piece.square === animatedMove.from && a.piece.side === piece.side && a.piece.kind === piece.kind)
      actor ??= [...unused].find(a => a.piece.square === piece.square && a.piece.side === piece.side && a.piece.kind === piece.kind && !(animatedMove?.from === a.piece.square))
      if (actor) {
        unused.delete(actor); actor.piece = piece
        actor.from.copy(actor.mesh.position); actor.to.copy(point(piece.square)); actor.started = now
        actor.hop = animatedMove?.to === piece.square ? (piece.kind === 'ashva' || piece.kind === 'gaja' ? .3 : .12) : 0
        actor.mesh.userData.square = piece.square
      } else {
        const key = `${piece.side}-${piece.kind}`
        let template = templates.get(key)
        if (!template) { template = createPiece(piece.kind, piece.side, pieceMaterials); templates.set(key, template); resources.add(template) }
        const promoted = animatedMove?.to === piece.square ? [...unused].find(a => a.piece.square === animatedMove.from && a.piece.side === piece.side && a.piece.kind === 'padati') : undefined
        const mesh = template.clone(true); mesh.position.copy(promoted?.mesh.position ?? point(piece.square)); pieces.add(mesh)
        mesh.userData.square = piece.square
        actors.push({ mesh, piece, from: mesh.position.clone(), to: point(piece.square), started: now, retiring: false, hop: promoted ? .12 : 0 })
      }
    }
    unused.forEach(actor => {
      if (animatedMove) { actor.retiring = true; actor.mesh.userData.retiring = true; actor.started = now; actor.from.copy(actor.mesh.position) }
      else { pieces.remove(actor.mesh); actors.splice(actors.indexOf(actor), 1) }
    })
    positionKey = key
  }
  function updateMarkers(next: SceneSettings) {
    markers.clear()
    const mark = (square: string, geometry: THREE.BufferGeometry, material: THREE.Material, y = .346) => {
      const mesh = new THREE.Mesh(geometry, material); mesh.rotation.x = -Math.PI / 2
      mesh.position.copy(point(square)); mesh.position.y = y; markers.add(mesh)
    }
    if (next.state.lastMove) { mark(next.state.lastMove.from, lastGeometry, lastMaterial); mark(next.state.lastMove.to, lastGeometry, lastMaterial) }
    if (next.selectedSquare) mark(next.selectedSquare, ringGeometry, markerMaterial, .354)
    next.legalMoves.forEach(move => mark(move.to, move.isCapture ? ringGeometry : dotGeometry, markerMaterial, .355))
  }
  function buildNetwork(inspection: NeuralInspection | null) {
    stages.forEach(stage => {
      network.remove(stage.group); stage.label.remove(); disposeGroup(stage.group, true)
    })
    stages = []
    edges.forEach(line => line.geometry.dispose()); edges = []; connectors.clear()
    if (!inspection) return
    const shared = inspection.layers.filter(layer => !layer.id.startsWith('policy') && !layer.id.startsWith('value'))
    for (const layer of inspection.layers) {
      const branch = layer.id.startsWith('policy') ? -1 : layer.id.startsWith('value') ? 1 : 0
      const level = branch ? shared.length + (layer.id === 'policy' || layer.id === 'value' ? 1 : 0) : shared.indexOf(layer)
      const target = new THREE.Vector3(branch * 3.1, -1.35 - level * 1.45, 0)
      const scale = branch ? .66 : .94
      const group = new THREE.Group(); group.position.set(target.x, -.65, target.z); group.scale.setScalar(scale)
      const cells = layer.channels[0].length
      const slab = new THREE.Mesh(new RoundedBoxGeometry(cells === 1 ? 2.8 : 8.14, .085, cells === 1 ? 2.8 : 8.14, 2, .055), new THREE.MeshStandardMaterial({ color: '#273349', metalness: .32, roughness: .43 }))
      slab.userData.layerId = layer.id; group.add(slab)
      const bars = new THREE.InstancedMesh(new THREE.BoxGeometry(cells === 1 ? 1.5 : .86, 1, cells === 1 ? 1.5 : .86), new THREE.MeshStandardMaterial({ roughness: .45, metalness: .2 }), cells)
      bars.instanceMatrix.setUsage(THREE.DynamicDrawUsage); group.add(bars)
      const label = document.createElement('span'); label.className = 'network-stage-label'; label.textContent = layer.label; label.setAttribute('aria-hidden', 'true'); host.appendChild(label)
      network.add(group); stages.push({ layer, group, bars, slab, target, scale, values: [], label })
    }
  }
  function updateChannels() {
    for (const stage of stages) {
      const index = stage.layer.channels.length === 1 ? 0 : settings?.focus === stage.layer.id ? settings.channel : -1
      const channels = stage.layer.channels
      const values = index >= 0 ? channels[Math.min(index, channels.length - 1)] : channels[0].map((_, square) => channels.reduce((sum, channel) => sum + Math.abs(channel[square]), 0) / channels.length)
      stage.values = values
      const maximum = Math.max(...values.map(Math.abs), .000001)
      values.forEach((value, square) => {
        const strength = Math.min(1, Math.abs(value) / maximum)
        const height = .018 + strength * .34
        dummy.position.set(values.length === 1 ? 0 : square % 8 - 3.5, .05 + height / 2, values.length === 1 ? 0 : 3.5 - Math.floor(square / 8))
        dummy.scale.set(1, height, 1); dummy.updateMatrix(); stage.bars.setMatrixAt(square, dummy.matrix)
        stage.bars.setColorAt(square, tempColor.copy(cold).lerp(warm, strength))
      })
      stage.bars.instanceMatrix.needsUpdate = true
      if (stage.bars.instanceColor) stage.bars.instanceColor.needsUpdate = true
      stage.bars.computeBoundingSphere()
    }
    hoveredKey = ''; onHover(null)
  }
  function update(next: SceneSettings) {
    reconcile(next.state)
    const nextNetworkKey = next.inspection ? `${next.inspection.positionKey}:${next.inspection.checkpoint}` : ''
    if (networkKey !== nextNetworkKey) { buildNetwork(next.inspection); networkKey = nextNetworkKey }
    const changedChannels = settings?.channel !== next.channel || settings?.focus !== next.focus || settings?.inspection !== next.inspection
    settings = next; updateMarkers(next)
    const materials = new Set<THREE.Material>()
    board.traverse(object => { if (object instanceof THREE.Mesh) (Array.isArray(object.material) ? object.material : [object.material]).forEach(material => materials.add(material)) })
    boardMaterials = [...materials]
    if (changedChannels) updateChannels()
    const key = `${next.exploded}:${next.focus}:${next.orientation}:${next.resetCamera}:${networkKey}`
    if (key !== cameraKey) { cameraKey = key; cameraDestination() }
  }
  function hit(event: PointerEvent) {
    const rect = renderer.domElement.getBoundingClientRect()
    pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1)
    raycaster.setFromCamera(pointer, camera)
    if (!settings?.exploded) {
      // The sculptures themselves are pickable: their heads can project over
      // another square, so raycasting only the tile would select the wrong piece.
      for (const found of raycaster.intersectObjects([...tiles, ...pieces.children], true)) {
        let object: THREE.Object3D | null = found.object
        while (object && !object.userData.square) object = object.parent
        if (object?.userData.square && !object.userData.retiring) return { square: object.userData.square as SquareId }
      }
      return null
    }
    const targets = stages.filter(stage => settings?.focus === 'overview' || stage.layer.id === settings?.focus).map(stage => stage.bars)
    const found = raycaster.intersectObjects(targets)[0]
    if (!found || found.instanceId === undefined) return null
    const stage = stages.find(stage => stage.bars === found.object)!
    const square = found.instanceId
    const channel = stage.layer.channels.length === 1 ? 0 : settings?.focus === stage.layer.id && settings.channel >= 0 ? settings.channel : -1
    return { layerId: stage.layer.id, square: stage.values.length === 1 ? 'scalar' : `${String.fromCharCode(97 + square % 8)}${Math.floor(square / 8) + 1}`, value: stage.values[square], channel: channel < 0 ? 'Mean absolute activation' : stage.layer.channelNames?.[channel] ?? `Channel ${channel + 1}` }
  }
  let down = { x: 0, y: 0 }, dragged = false
  const pointerDown = (event: PointerEvent) => { down = { x: event.clientX, y: event.clientY }; dragged = false }
  const pointerMove = (event: PointerEvent) => {
    if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5 && event.buttons) dragged = true
    if (dragged && event.buttons) return
    const found = hit(event)
    renderer.domElement.style.cursor = found ? (settings?.exploded ? 'crosshair' : settings?.disabled ? 'grab' : 'pointer') : 'grab'
    if (found && 'layerId' in found) {
      const key = `${found.layerId}:${found.square}`
      if (key !== hoveredKey) { hoveredKey = key; onHover(found as InspectCell) }
    } else if (hoveredKey) { hoveredKey = ''; onHover(null) }
  }
  const pointerUp = (event: PointerEvent) => {
    if (dragged || event.button !== 0 || Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5) return
    const found = hit(event)
    if (found && 'layerId' in found) { if (settings?.focus !== found.layerId) onFocus(found.layerId as string); else onHover(found as InspectCell) }
    else if (found && !settings?.disabled) onSquare(found.square as SquareId)
  }
  const leave = () => { hoveredKey = ''; onHover(null) }
  const wheel = (event: WheelEvent) => { event.preventDefault(); transition = null; zoomVelocity += Math.max(-.22, Math.min(.22, event.deltaY * .0008)) }
  const orbitStart = () => { transition = null; zoomVelocity = 0 }
  const contextLost = (event: Event) => { event.preventDefault(); onFailure() }
  controls.addEventListener('start', orbitStart)
  renderer.domElement.addEventListener('pointerdown', pointerDown)
  renderer.domElement.addEventListener('pointermove', pointerMove)
  renderer.domElement.addEventListener('pointerup', pointerUp)
  renderer.domElement.addEventListener('pointerleave', leave)
  renderer.domElement.addEventListener('wheel', wheel, { passive: false })
  renderer.domElement.addEventListener('webglcontextlost', contextLost)
  const visibility = () => { visible = !document.hidden; lastTime = 0; if (visible && !raf) raf = requestAnimationFrame(frame) }
  document.addEventListener('visibilitychange', visibility)
  function frame(now: number) {
    raf = 0
    if (disposed || !visible) return
    const delta = Math.min(.05, lastTime ? (now - lastTime) / 1000 : 1 / 60); lastTime = now
    if (transition) {
      const t = reduced.matches ? 1 : Math.min(1, (now - transition.start) / 950)
      camera.position.lerpVectors(transition.from, transition.to, ease(t))
      controls.target.lerpVectors(transition.fromTarget, transition.target, ease(t))
      if (t === 1) transition = null
    } else if (Math.abs(zoomVelocity) > .0001) {
      const distance = camera.position.distanceTo(controls.target)
      const next = THREE.MathUtils.clamp(distance * Math.exp(zoomVelocity * delta * 12), 4, 75)
      camera.position.sub(controls.target).multiplyScalar(next / distance).add(controls.target)
      zoomVelocity *= Math.exp(-delta * 10)
    }
    controls.update()
    for (let index = actors.length - 1; index >= 0; index--) {
      const actor = actors[index], t = reduced.matches ? 1 : Math.min(1, (now - actor.started) / (actor.retiring ? 300 : 540))
      if (actor.retiring) {
        actor.mesh.scale.setScalar(Math.max(.001, 1 - ease(t)))
        actor.mesh.position.y = actor.from.y - .22 * ease(t)
        if (t === 1) { pieces.remove(actor.mesh); actors.splice(index, 1) }
      } else {
        actor.mesh.position.lerpVectors(actor.from, actor.to, ease(t)); actor.mesh.position.y += Math.sin(Math.PI * t) * actor.hop
      }
    }
    const focused = settings?.exploded && settings.focus !== 'overview'
    const targetOpacity = focused ? .08 : 1
    boardOpacity = reduced.matches ? targetOpacity : THREE.MathUtils.damp(boardOpacity, targetOpacity, 7, delta)
    boardMaterials.forEach(material => {
      const transparent = boardOpacity < .995 || material === markerMaterial || material === lastMaterial || Boolean((material as THREE.MeshBasicMaterial).map)
      if (material.transparent !== transparent) { material.transparent = transparent; material.needsUpdate = true }
      material.opacity = material === markerMaterial ? boardOpacity * .85 : material === lastMaterial ? boardOpacity * .3 : boardOpacity
      material.depthWrite = boardOpacity > .9 && material !== markerMaterial && material !== lastMaterial
    })
    board.visible = !focused || boardOpacity > .09; floor.visible = !settings?.exploded
    network.visible = Boolean(settings?.exploded && settings.inspection)
    connectors.clear()
    // Connector geometry is reused, not allocated in the animation loop.
    for (const stage of stages) {
      const targetY = settings?.exploded ? stage.target.y : -.65
      stage.group.position.y = reduced.matches ? targetY : THREE.MathUtils.damp(stage.group.position.y, targetY, 6, delta)
      const selected = !focused || stage.layer.id === settings?.focus
      stage.group.visible = network.visible && selected
      if (selected && network.visible) {
        const scalar = stage.values.length === 1
        const projected = new THREE.Vector3((scalar ? -1 : -3.7) * stage.scale, .12, (scalar ? -1.7 : -4.2) * stage.scale).add(stage.group.position).project(camera)
        stage.label.style.transform = `translate(${(projected.x + 1) * host.clientWidth / 2}px, ${(-projected.y + 1) * host.clientHeight / 2}px)`
        stage.label.hidden = projected.z > 1 || projected.z < -1
      } else stage.label.hidden = true
    }
    const shared = stages.filter(stage => !stage.layer.id.startsWith('policy') && !stage.layer.id.startsWith('value'))
    const branches = [stages.filter(stage => stage.layer.id.startsWith('policy')), stages.filter(stage => stage.layer.id.startsWith('value'))]
    const paths = branches.map(branch => [...shared, ...branch])
    if (network.visible && !focused) {
      // Build edges once for each dataset; moving endpoints update the existing buffers.
      if (edges.length !== paths.reduce((sum, path) => sum + Math.max(0, path.length - 1), 0)) rebuildEdges(paths)
      let edgeIndex = 0
      paths.forEach(path => { for (let i = 1; i < path.length; i++) {
        const line = edges[edgeIndex++], positions = line.geometry.attributes.position as THREE.BufferAttribute
        const from = path[i - 1].group.position, to = path[i].group.position
        positions.setXYZ(0, from.x, from.y, from.z); positions.setXYZ(1, to.x, to.y, to.z); positions.needsUpdate = true
        line.geometry.computeBoundingSphere(); connectors.add(line)
      } })
    }
    pulses.forEach((pulse, index) => {
      pulse.visible = Boolean(network.visible && !focused && settings?.replay && !reduced.matches && paths[index].length > 1)
      if (pulse.visible) {
        const path = paths[index], t = ((now / 4200) % 1) * (path.length - 1), segment = Math.floor(t)
        pulse.position.lerpVectors(path[segment].group.position, path[Math.min(segment + 1, path.length - 1)].group.position, t - segment)
      }
    })
    if (activeStage !== (settings?.focus ?? '')) { activeStage = settings?.focus ?? ''; hoveredKey = '' }
    renderer.render(scene, camera)
    if (import.meta.env.DEV) {
      measuredFrames++
      if (now - measuredSince > 1000) {
        renderer.domElement.dataset.drawCalls = String(renderer.info.render.calls)
        renderer.domElement.dataset.triangles = String(renderer.info.render.triangles)
        renderer.domElement.dataset.observedFps = (measuredFrames * 1000 / Math.max(1, now - measuredSince)).toFixed(1)
        measuredFrames = 0; measuredSince = now
      }
    }
    raf = requestAnimationFrame(frame)
  }
  let edges: THREE.Line[] = []
  function rebuildEdges(paths: Stage[][]) {
    edges.forEach(line => line.geometry.dispose()); edges = []
    paths.forEach(path => { for (let i = 1; i < path.length; i++) {
      const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3))
      edges.push(new THREE.Line(geometry, connectorMaterial))
    } })
  }
  raf = requestAnimationFrame(frame)
  resize()
  return {
    update,
    dispose() {
      if (disposed) return
      disposed = true; cancelAnimationFrame(raf); observer.disconnect(); controls.dispose()
      document.removeEventListener('visibilitychange', visibility)
      renderer.domElement.removeEventListener('pointerdown', pointerDown)
      renderer.domElement.removeEventListener('pointermove', pointerMove)
      renderer.domElement.removeEventListener('pointerup', pointerUp)
      renderer.domElement.removeEventListener('pointerleave', leave)
      renderer.domElement.removeEventListener('wheel', wheel)
      renderer.domElement.removeEventListener('webglcontextlost', contextLost)
      stages.forEach(stage => stage.label.remove())
      board.traverse(object => { if (object instanceof THREE.Mesh) (Array.isArray(object.material) ? object.material : [object.material]).forEach(material => { const map = (material as THREE.MeshBasicMaterial).map; map?.dispose() }) })
      disposeGroup(scene, true); disposeGroup(resources, true)
      ringGeometry.dispose(); dotGeometry.dispose(); lastGeometry.dispose(); markerMaterial.dispose(); lastMaterial.dispose()
      edges.forEach(line => line.geometry.dispose()); connectorMaterial.dispose(); key.shadow.dispose()
      environmentTarget.dispose(); renderer.dispose(); renderer.domElement.remove()
    },
  }
}
