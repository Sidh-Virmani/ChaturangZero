import * as THREE from 'three'
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js'
import type { PieceKind, Side } from '../types/game'

// Original Chaturanga sculptures. The palette and layered plinths echo the
// supplied art direction; all geometry is local, interactive, and procedural.
export function createPieceMaterials() {
  return {
    ivory: new THREE.MeshPhysicalMaterial({ color: '#ead6bd', roughness: .25, metalness: .08, clearcoat: .72, clearcoatRoughness: .2 }),
    onyx: new THREE.MeshPhysicalMaterial({ color: '#172b35', roughness: .28, metalness: .26, clearcoat: .68, clearcoatRoughness: .23 }),
    gold: new THREE.MeshStandardMaterial({ color: '#cfa768', roughness: .25, metalness: .86 }),
    dark: new THREE.MeshStandardMaterial({ color: '#0d1820', roughness: .38, metalness: .25 }),
    gem: new THREE.MeshPhysicalMaterial({ color: '#70dbce', roughness: .16, metalness: .32, clearcoat: 1 }),
  }
}
export type PieceMaterials = ReturnType<typeof createPieceMaterials>

function add(group: THREE.Group, geometry: THREE.BufferGeometry, material: THREE.Material, x = 0, y = 0, z = 0) {
  const object = new THREE.Mesh(geometry, material)
  object.position.set(x, y, z); object.castShadow = true; object.receiveShadow = true
  group.add(object); return object
}
function ellipsoid(group: THREE.Group, material: THREE.Material, x: number, y: number, z: number, a: number, b: number, c: number) {
  const object = add(group, new THREE.SphereGeometry(1, 20, 14), material, x, y, z)
  object.scale.set(a, b, c); return object
}
function ring(group: THREE.Group, material: THREE.Material, radius: number, y: number, tube = .012) {
  const object = add(group, new THREE.TorusGeometry(radius, tube, 8, 48), material, 0, y)
  object.rotation.x = Math.PI / 2; return object
}
function lathe(group: THREE.Group, material: THREE.Material, points: number[][]) {
  return add(group, new THREE.LatheGeometry(points.map(([r, y]) => new THREE.Vector2(r, y)), 48), material)
}
function tube(group: THREE.Group, material: THREE.Material, points: number[][], radius: number) {
  const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p as [number,number,number])))
  return add(group, new THREE.TubeGeometry(curve, 20, radius, 6, false), material)
}
function jewel(group: THREE.Group, material: THREE.Material, x: number, y: number, z: number, size: number) {
  return add(group, new THREE.OctahedronGeometry(size), material, x, y, z)
}

// These small settings and incised lines read as hand-worked metal at a close
// camera distance. Low-poly jewels keep the detail cheaper than extra spheres.
function beadedCollar(group: THREE.Group, gold: THREE.Material, gem: THREE.Material, radius: number, y: number, count = 8) {
  ring(group, gold, radius, y, .008)
  for (let i = 0; i < count; i++) {
    const angle = i * Math.PI * 2 / count
    const setting = jewel(group, i % 2 === 0 ? gem : gold, Math.cos(angle) * radius, y, Math.sin(angle) * radius, .013)
    setting.rotation.y = -angle
  }
}
function pedestal(group: THREE.Group, material: THREE.Material, gold: THREE.Material, width: number, height: number) {
  lathe(group, material, [[0,0],[width*.86,0],[width,.025],[width,.07],[width*.96,.1],[width*.84,.115],
    [width*.84,.145],[width*.92,.165],[width*.91,.19],[width*.79,.215],
    [width*.67,.25],[width*.56,.3],[width*.46,.38],[width*.39,height*.74],
    [width*.37,height*.88],[width*.43,height*.94],[width*.61,height], [width*.63,height+.035],[0,height+.04]])
  ring(group,gold,width*.97,.078,.008)
  ring(group,gold,width*.85,.15,.008)
  ring(group,gold,width*.61,height+.025,.009)
  ring(group,gold,width*.8,.211,.004)
  for (let i = 0; i < 12; i++) {
    const a = i * Math.PI / 6
    jewel(group,gold,Math.cos(a)*width*.915,.173,Math.sin(a)*width*.915,.008)
  }
  // Fine longitudinal inlay on the turned stem, visible without changing its
  // silhouette or obscuring the square around the piece.
  for (let i = 0; i < 4; i++) {
    const a = Math.PI / 4 + i * Math.PI / 2
    tube(group,gold,[[Math.cos(a)*width*.47,.365,Math.sin(a)*width*.47],
      [Math.cos(a)*width*.395,height*.74,Math.sin(a)*width*.395],
      [Math.cos(a)*width*.375,height*.87,Math.sin(a)*width*.375]],.003)
  }
}

export function createPiece(kind: PieceKind, side: Side, materials: PieceMaterials): THREE.Group {
  const group = new THREE.Group()
  const material = side === 'white' ? materials.ivory : materials.onyx
  const { gold, dark, gem } = materials
  const height = kind === 'padati' ? .48 : kind === 'raja' ? .91 : kind === 'mantri' ? .83 : .5
  pedestal(group,material,gold,kind === 'padati' ? .265 : .345,height)
  if (kind === 'padati') {
    ellipsoid(group,material,0,.64,0,.148,.163,.148)
    lathe(group,material,[[0,.63],[.16,.63],[.165,.65],[.15,.72],[.1,.78],[0,.795]])
    ring(group,gold,.159,.641,.01)
    ellipsoid(group,gold,0,.813,0,.027,.044,.027)
    tube(group,gold,[[0,.79,.04],[0,.746,.119],[0,.65,.154]],.009)
    // A domed shield and leaf-shaped spear make the infantry readable even
    // from the default distant view; neither projects beyond its own square.
    ellipsoid(group,material,.105,.51,.17,.132,.149,.035)
    const rim = add(group,new THREE.TorusGeometry(.12,.009,6,36),gold,.105,.51,.2)
    rim.scale.y = 1.12
    ellipsoid(group,gold,.105,.51,.214,.032,.035,.015)
    for (let i = 0; i < 4; i++) {
      const a = i * Math.PI / 2
      jewel(group,gold,.105+Math.cos(a)*.091,.51+Math.sin(a)*.104,.202,.01)
    }
    add(group,new THREE.CylinderGeometry(.012,.014,.65,8),dark,-.205,.692,.018)
    for (const y of [.403,.92]) add(group,new THREE.CylinderGeometry(.021,.021,.044,8),gold,-.205,y,.018)
    const spear = jewel(group,gold,-.205,1.074,.018,.07)
    spear.scale.set(.5,1.45,.27)
    group.rotation.y = side === 'white' ? Math.PI : 0
  } else if (kind === 'raja') {
    lathe(group,gold,[[0,.96],[.21,.96],[.23,1.01],[.21,1.06],[.16,1.085],[0,1.085]])
    ellipsoid(group,material,0,1.155,0,.174,.145,.174)
    beadedCollar(group,gold,gem,.222,1.022)
    // Eight lotus-shaped crown leaves and a central faceted jewel.
    for (let i=0;i<8;i++) {
      const a=i*Math.PI/4
      const petal=ellipsoid(group,gold,Math.cos(a)*.18,1.25,Math.sin(a)*.18,.034,.15,.07)
      petal.rotation.y=-a; petal.rotation.z=Math.sin(a)*.22; petal.rotation.x=Math.cos(a)*.22
      jewel(group,gold,Math.cos(a)*.2,1.4,Math.sin(a)*.2,.024)
    }
    ring(group,gold,.204,1.14,.02)
    const heart = jewel(group,gem,0,1.4,0,.079)
    heart.scale.y = 1.2
    add(group,new THREE.OctahedronGeometry(.045),gold,0,1.51)
    for (let i = 0; i < 4; i++) {
      const a = Math.PI / 4 + i * Math.PI / 2
      tube(group,gold,[[Math.cos(a)*.162,1.1,Math.sin(a)*.162],
        [Math.cos(a)*.176,1.17,Math.sin(a)*.176],
        [Math.cos(a)*.13,1.24,Math.sin(a)*.13]],.005)
    }
  } else if (kind === 'mantri') {
    ellipsoid(group,material,0,1.018,0,.19,.205,.19)
    beadedCollar(group,gold,gem,.184,1.035)
    lathe(group,material,[[0,1.06],[.15,1.06],[.14,1.17],[.10,1.27],[.035,1.33],[0,1.34]])
    ellipsoid(group,gold,0,1.35,0,.035,.045,.035)
    ellipsoid(group,gem,0,1.08,side==='white'?-.173:.173,.038,.055,.018)
    // An advisor's fluted turban, shorter and enclosed compared with the
    // Raja's open crown. Four ribs follow its actual sculpted surface.
    for (let i = 0; i < 4; i++) {
      const a = Math.PI / 4 + i * Math.PI / 2
      tube(group,gold,[[Math.cos(a)*.153,1.085,Math.sin(a)*.153],
        [Math.cos(a)*.132,1.2,Math.sin(a)*.132],
        [Math.cos(a)*.039,1.32,Math.sin(a)*.039]],.005)
    }
    ring(group,gold,.142,1.17,.005)
  } else if (kind === 'ratha') {
    lathe(group,material,[[0,.5],[.215,.5],[.18,.61],[.17,.76],[.21,.9],[.28,.94],[.28,1.03],[.22,1.05],[.21,.99],[0,.99]])
    ring(group,gold,.267,.95,.014)
    ring(group,gold,.277,1.02,.006)
    for(let i=0;i<6;i++){
      const angle=i*Math.PI/3
      const tooth=add(group,new RoundedBoxGeometry(.14,.13,.12,2,.018),material,Math.cos(angle)*.228,1.078,Math.sin(angle)*.228)
      tooth.rotation.y=-angle
      jewel(group,gold,Math.cos(angle)*.276,1.078,Math.sin(angle)*.276,.018)
    }
    // Chariot wheels distinguish the ratha from a generic rook.
    for(const x of [-.254,.254]){
      const wheel=add(group,new THREE.TorusGeometry(.128,.015,8,32),gold,x,.68,0)
      wheel.rotation.y=Math.PI/2
      for(let i=0;i<6;i++){
        const spoke=add(group,new THREE.CylinderGeometry(.006,.006,.225,6),gold,x,.68,0)
        spoke.rotation.x=i*Math.PI/3
      }
      ellipsoid(group,gold,x,.68,0,.023,.029,.029)
      jewel(group,gem,x+Math.sign(x)*.012,.68,0,.018)
    }
    for (const z of [-.174,.174]) {
      tube(group,gold,[[-.09,.62,z],[-.1,.75,z],[0,.87,z],[.1,.75,z],[.09,.62,z]],.006)
    }
  } else if (kind === 'ashva') {
    const shape=new THREE.Shape()
    shape.moveTo(-.21,.51)
    shape.bezierCurveTo(-.26,.69,-.3,.92,-.19,1.13)
    shape.bezierCurveTo(-.15,1.22,-.1,1.29,-.015,1.3)
    shape.lineTo(.005,1.48);shape.quadraticCurveTo(.09,1.42,.11,1.3)
    shape.bezierCurveTo(.21,1.29,.22,1.19,.3,1.12)
    shape.bezierCurveTo(.43,1.06,.39,.96,.28,.96)
    shape.lineTo(.1,1.005);shape.quadraticCurveTo(.045,.89,.105,.79)
    shape.bezierCurveTo(.13,.67,.24,.59,.23,.51);shape.closePath()
    const carved=new THREE.ExtrudeGeometry(shape,{depth:.17,bevelEnabled:true,bevelThickness:.07,bevelSize:.055,bevelSegments:4,steps:1,curveSegments:22})
    add(group,carved,material,0,0,-.085)
    for(const z of [-.155,.155]){
      ellipsoid(group,gold,.135,1.19,z,.029,.033,.012)
      ellipsoid(group,dark,.139,1.191,z*1.025,.015,.019,.009)
      tube(group,gold,[[.05,1.055,z],[.15,1.01,z],[.28,.989,z]],.006)
      tube(group,gold,[[-.05,.87,z],[-.075,1.045,z],[.025,1.25,z],[.15,1.19,z],[.2,1.07,z]],.008)
      jewel(group,gem,-.054,1.042,z*1.04,.027)
    }
    for(let i=0;i<6;i++) tube(group,gold,[[-.23,.77+i*.068,-.06],[-.272,.79+i*.068,0],[-.23,.77+i*.068,.06]],.007)
    beadedCollar(group,gold,gem,.203,.544,6)
    group.rotation.y=side==='white'?Math.PI/2:-Math.PI/2
  } else {
    ellipsoid(group,material,0,.85,0,.217,.265,.215)
    for(const s of [-1,1]){
      const ear=ellipsoid(group,material,s*.24,.86,-.025,.137,.21,.075);ear.rotation.z=s*.14
      ellipsoid(group,gold,s*.247,.87,.043,.085,.14,.012)
      tube(group,gold,[[s*.27,1.004,.043],[s*.321,.94,.043],[s*.321,.81,.043],[s*.256,.749,.043]],.005)
      ellipsoid(group,dark,s*.111,.95,.192,.021,.024,.013)
      tube(group,materials.ivory,[[s*.132,.77,.15],[s*.17,.7,.27],[s*.18,.735,.355]],.028)
    }
    tube(group,material,[[0,.93,.185],[0,.81,.28],[0,.58,.29],[0,.56,.39],[0,.64,.42]],.061)
    beadedCollar(group,gold,gem,.17,1.046)
    ellipsoid(group,gold,0,1.08,0,.14,.035,.14)
    ellipsoid(group,gem,0,1.128,0,.034,.037,.034)
    // Forehead harness and trunk bands are sculptural details, not textures.
    tube(group,gold,[[-.132,1.015,.124],[0,.99,.193],[.132,1.015,.124]],.009)
    const forehead = jewel(group,gem,0,.992,.209,.038)
    forehead.scale.set(.8,1.15,.5)
    for (const [y,z] of [[.79,.286],[.7,.295],[.61,.301]]) {
      const band = add(group,new THREE.TorusGeometry(.06,.005,6,24),gold,0,y,z)
      band.rotation.x = Math.PI / 2
    }
    group.rotation.y=side==='white'?Math.PI:0
  }
  // Cache each sculpture merged by material to keep draw calls low.
  group.updateMatrixWorld(true)
  const parts = new Map<THREE.Material, THREE.BufferGeometry[]>()
  group.traverse(object => {
    if (!(object instanceof THREE.Mesh)) return
    const material = object.material as THREE.Material
    const geometry = object.geometry.index ? object.geometry.toNonIndexed() : object.geometry.clone()
    geometry.applyMatrix4(object.matrixWorld)
    if (!parts.has(material)) parts.set(material, [])
    parts.get(material)!.push(geometry)
  })
  const sculpture = new THREE.Group()
  parts.forEach((geometries, material) => {
    const geometry = mergeGeometries(geometries, false)!
    const mesh = new THREE.Mesh(geometry, material)
    mesh.castShadow = true; mesh.receiveShadow = true; sculpture.add(mesh)
    geometries.forEach(part => part.dispose())
  })
  disposeGroup(group)
  return sculpture
}

export function disposeGroup(group: THREE.Object3D, disposeMaterials = false) {
  const geometries=new Set<THREE.BufferGeometry>(), materials=new Set<THREE.Material>()
  group.traverse(object=>{
    if(object instanceof THREE.InstancedMesh)object.dispose()
    if(object instanceof THREE.Mesh || object instanceof THREE.Line){
      geometries.add(object.geometry)
      if(disposeMaterials) (Array.isArray(object.material)?object.material:[object.material]).forEach(m=>materials.add(m))
    }
  })
  geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose())
}
