import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, Focus, Minus, Pause, Play, Plus, ScanLine } from 'lucide-react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js'
import { createPiece, createPieceMaterials, disposeGroup } from '../three/pieces'
import { networkStages, planeNames, stageData } from '../three/network'
import type { Analysis, BoardState, LegalMove, SquareId } from '../types/game'

export interface Board3DProps {
  state: BoardState; selected: SquareId | null; legal: LegalMove[]; disabled: boolean
  analysis: Analysis | null; policy: boolean; layers: boolean; paths: boolean; flipped: boolean
  resetCamera: number; onSquare: (square: SquareId) => void; onFallback: () => void
}
const squarePosition = (square: string) => new THREE.Vector3('abcdefgh'.indexOf(square[0]) - 3.5, .065, 4.5 - Number(square[1]))
const ease = (t: number) => t * t * t * (t * (t * 6 - 15) + 10)
const pieceRules: Record<string,string> = {raja:'One square in any direction. Your raja must stay safe.',mantri:'One square diagonally.',ratha:'Any distance along an open rank or file.',ashva:'An L-shaped leap: two squares, then one.',gaja:'Exactly two squares diagonally. Leaps over intervening pieces.',padati:'One square forward; captures diagonally; promotes to mantri.'}
type CellInfo = {stage:number; square:string; raw:number; normalized:number}

export default function Board3D(props: Board3DProps) {
  const container = useRef<HTMLDivElement>(null)
  const [focus, setFocus] = useState(-1)
  const [plane, setPlane] = useState(5)
  const [channel, setChannel] = useState(-1)
  const [flowing, setFlowing] = useState(false)
  const [flowStep, setFlowStep] = useState(-1)
  const [cell, setCell] = useState<CellInfo | null>(null)
  const [hover, setHover] = useState('')
  const current = useRef({props,focus,plane,channel,flowing})
  current.current={props,focus,plane,channel,flowing}
  const sync=useRef<(() => void) | null>(null)
  const zoom=useRef<((factor:number) => void) | null>(null)
  const overview=()=>{setFocus(-1);setCell(null)}

  useEffect(()=>{
    if(!props.layers){setFocus(-1);setCell(null);setFlowing(false)}
  },[props.layers])

  useEffect(()=>{
    const host=container.current!
    let renderer: THREE.WebGLRenderer
    try {renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'})}
    catch {current.current.props.onFallback();return}
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2))
    renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFShadowMap
    renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=.98
    renderer.domElement.setAttribute('aria-label','Interactive 3D Chaturanga board. Click a piece to play, drag to orbit, and scroll to zoom. Neural layers support hover inspection. Use 2D for keyboard play.')
    host.prepend(renderer.domElement)
    const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(36,1,.1,100)
    const controls=new OrbitControls(camera,renderer.domElement)
    controls.enableDamping=true;controls.dampingFactor=.075;controls.enablePan=false
    controls.minDistance=7;controls.maxDistance=34;controls.minPolarAngle=.16;controls.maxPolarAngle=Math.PI/2.08
    const reducedMotion=matchMedia('(prefers-reduced-motion: reduce)').matches
    const pmrem=new THREE.PMREMGenerator(renderer),environment=new RoomEnvironment()
    const envTarget=pmrem.fromScene(environment)
    scene.environment=envTarget.texture;scene.environmentIntensity=.5;environment.dispose();pmrem.dispose()
    scene.add(new THREE.HemisphereLight('#e4d7f2','#171c31',1.45))
    const key=new THREE.DirectionalLight('#ffe3d2',2.7);key.position.set(-4,11,6);key.castShadow=true
    key.shadow.mapSize.set(2048,2048);key.shadow.camera.left=key.shadow.camera.bottom=-8
    key.shadow.camera.right=key.shadow.camera.top=8;key.shadow.bias=-.0004;key.shadow.normalBias=.025;scene.add(key)
    const rim=new THREE.DirectionalLight('#a4cddd',1.9);rim.position.set(7,6,-7);scene.add(rim)
    const rose=new THREE.DirectionalLight('#eab5c8',.7);rose.position.set(-7,3,-3);scene.add(rose)
    const board=new THREE.Group();scene.add(board)
    const pieceMaterials=createPieceMaterials()
    const obsidian=new THREE.MeshPhysicalMaterial({color:'#1b2435',roughness:.31,metalness:.28,clearcoat:.7})
    const gold=new THREE.MeshStandardMaterial({color:'#bc9361',metalness:.82,roughness:.27})
    const luminous=new THREE.MeshStandardMaterial({color:'#75c3c7',emissive:'#39b2be',emissiveIntensity:.52,roughness:.37,metalness:.23})
    const makeBox=(w:number,h:number,d:number,y:number,material:THREE.Material,r=.05)=>{
      const mesh=new THREE.Mesh(new RoundedBoxGeometry(w,h,d,3,r),material);mesh.position.y=y
      mesh.castShadow=mesh.receiveShadow=true;board.add(mesh);return mesh
    }
    makeBox(8.98,.43,8.98,-.49,luminous,.07)
    makeBox(9.16,.2,9.16,-.17,obsidian,.06)
    makeBox(9.14,.13,9.14,-.78,obsidian,.04)
    for(const y of [-.29,-.7])makeBox(9.035,.018,9.035,y,gold,.006)
    for(const x of [-4.31,4.31]){
      const line=new THREE.Mesh(new THREE.BoxGeometry(.019,.009,8.64),gold);line.position.set(x,-.061,0);board.add(line)
    }
    for(const z of [-4.31,4.31]){
      const line=new THREE.Mesh(new THREE.BoxGeometry(8.64,.009,.019),gold);line.position.set(0,-.061,z);board.add(line)
    }
    const tiles:THREE.Mesh<THREE.BoxGeometry,THREE.MeshPhysicalMaterial>[]=[]
    for(let rank=0;rank<8;rank++)for(let file=0;file<8;file++){
      const color=(rank+file)%2===0?'#354458':'#d9b6b0'
      const tile=new THREE.Mesh(new THREE.BoxGeometry(.991,.072,.991),new THREE.MeshPhysicalMaterial({color,roughness:.36,metalness:.14,clearcoat:.35}))
      tile.position.set(file-3.5,0,3.5-rank);tile.receiveShadow=true
      tile.userData={square:`${'abcdefgh'[file]}${rank+1}`,color};tiles.push(tile);board.add(tile)
    }
    const textures:THREE.Texture[]=[]
    for(let i=0;i<8;i++)for(const edge of [0,1,2,3]){
      const canvas=document.createElement('canvas');canvas.width=canvas.height=96
      const ctx=canvas.getContext('2d')!;ctx.fillStyle='#c6ad8c';ctx.font='34px sans-serif';ctx.textAlign='center';ctx.textBaseline='middle'
      ctx.fillText(edge%2?String(i+1):'abcdefgh'[i],48,48)
      const texture=new THREE.CanvasTexture(canvas);textures.push(texture)
      const label=new THREE.Mesh(new THREE.PlaneGeometry(.24,.24),new THREE.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false}))
      label.rotation.x=-Math.PI/2
      label.position.set(edge%2?(edge===1?-4.13:4.13):i-3.5,.045,edge%2?3.5-i:edge===0?4.13:-4.13)
      if(edge===2)label.rotation.z=Math.PI
      board.add(label)
    }
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(100,100),new THREE.ShadowMaterial({opacity:.3}))
    floor.rotation.x=-Math.PI/2;floor.position.y=-.89;floor.receiveShadow=true;scene.add(floor)
    const glowCanvas=document.createElement('canvas');glowCanvas.width=glowCanvas.height=128
    const gc=glowCanvas.getContext('2d')!,gradient=gc.createRadialGradient(64,64,15,64,64,64)
    gradient.addColorStop(0,'rgba(78,200,208,.3)');gradient.addColorStop(.6,'rgba(68,184,194,.16)');gradient.addColorStop(1,'rgba(60,150,170,0)')
    gc.fillStyle=gradient;gc.fillRect(0,0,128,128)
    const glowTexture=new THREE.CanvasTexture(glowCanvas);textures.push(glowTexture)
    const groundGlow=new THREE.Mesh(new THREE.PlaneGeometry(15,15),new THREE.MeshBasicMaterial({map:glowTexture,transparent:true,depthWrite:false}))
    groundGlow.rotation.x=-Math.PI/2;groundGlow.position.y=-.885;scene.add(groundGlow)
    const pieces=new Map<string,{object:THREE.Group;kind:string;target:THREE.Vector3;from:THREE.Vector3;started:number}>()
    const markers=new THREE.Group(),overlays=new THREE.Group();board.add(markers,overlays)
    const neural=new THREE.Group();scene.add(neural)
    const panels: {group:THREE.Group;bars:THREE.InstancedMesh;materials:THREE.Material[];stage:number;alpha:number;raw:number[];normalized:number[]}[]=[]
    const matrix=new THREE.Matrix4(),scale=new THREE.Vector3(),location=new THREE.Vector3(),rotation=new THREE.Quaternion()
    for(let stage=0;stage<networkStages.length;stage++){
      const group=new THREE.Group(),color=networkStages[stage].color
      const backingMaterial=new THREE.MeshPhysicalMaterial({color:'#1b2537',metalness:.3,roughness:.38,transparent:true,opacity:.94,side:THREE.DoubleSide})
      const backing=new THREE.Mesh(new RoundedBoxGeometry(8.25,.065,8.25,2,.045),backingMaterial);backing.userData={stage};group.add(backing)
      const outline=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(8.27,.07,8.27)),new THREE.LineBasicMaterial({color,transparent:true,opacity:.55}));group.add(outline)
      const material=new THREE.MeshStandardMaterial({color:'#ffffff',roughness:.4,metalness:.2,emissive:color,emissiveIntensity:.11,transparent:true})
      const bars=new THREE.InstancedMesh(new THREE.BoxGeometry(.9,1,.9),material,stage===5?1:64)
      bars.userData={stage};bars.instanceMatrix.setUsage(THREE.DynamicDrawUsage);group.add(bars)
      neural.add(group);panels.push({group,bars,materials:[backingMaterial,outline.material,material],stage,alpha:0,raw:[],normalized:[]})
    }
    const links=new THREE.Group();neural.add(links)
    const linkMaterial=new THREE.LineBasicMaterial({color:'#73b7c5',transparent:true,opacity:.18})
    for(const [x,z] of [[-4.18,-4.18],[4.18,-4.18]]){
      const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x,.65,z),new THREE.Vector3(x,-5.75,z)]),linkMaterial);links.add(line)
    }
    const pulseMaterial=new THREE.MeshBasicMaterial({color:'#b9f4ee',transparent:true})
    const pulse=new THREE.Mesh(new THREE.SphereGeometry(.075,12,10),pulseMaterial),valuePulse=pulse.clone();links.add(pulse,valuePulse)
    const hoverFrame=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(.94,1,.94)),new THREE.LineBasicMaterial({color:'#ffffff',transparent:true,opacity:.85}))
    hoverFrame.visible=false;scene.add(hoverFrame)
    let cameraTween:{from:THREE.Vector3;to:THREE.Vector3;lookFrom:THREE.Vector3;lookTo:THREE.Vector3;start:number}|null=null
    let boardAlpha=1,explosion=0,previousFocus=-99,previousFlip=false,previousReset=-1,previousLayers=false
    let priorAnalysis:Analysis|null=null,priorPlane=-1,priorChannel=-2,flowStart=0,lastFlow=-1,wasFlowing=false
    const boardMaterials=new Set<THREE.Material>()
    const collectBoardMaterials=()=>{boardMaterials.clear();board.traverse(object=>{if(object instanceof THREE.Mesh)(Array.isArray(object.material)?object.material:[object.material]).forEach(m=>{m.transparent=true;boardMaterials.add(m)})})}
    function transitionCamera(target:THREE.Vector3,look:THREE.Vector3,instant=false){
      if(reducedMotion||instant){camera.position.copy(target);controls.target.copy(look);controls.update();cameraTween=null}
      else cameraTween={from:camera.position.clone(),to:target,lookFrom:controls.target.clone(),lookTo:look,start:performance.now()}
    }
    function frameCamera(instant=false){
      const {props:p,focus:f}=current.current,active=p.layers&&!!p.analysis,sign=p.flipped?-1:1
      const y=f<0?-2.45:-1.75-f*1.04
      const aspect=host.clientWidth/Math.max(host.clientHeight,1)
      const fit=aspect<1?1+(1-aspect)*.5:1
      const target=active?(f<0?new THREE.Vector3(12.8*sign,10.2,20.7*sign):new THREE.Vector3(7.3*sign,y+11.2,12.9*sign)):new THREE.Vector3(8.5*sign,10.9,12*sign)
      const look=new THREE.Vector3(0,active?y:0,0)
      target.sub(look).multiplyScalar(fit).add(look);transitionCamera(target,look,instant)
    }
    controls.addEventListener('start',()=>{cameraTween=null})
    zoom.current=(factor)=>{const target=camera.position.clone().sub(controls.target).multiplyScalar(factor);target.setLength(THREE.MathUtils.clamp(target.length(),controls.minDistance,controls.maxDistance));transitionCamera(target.add(controls.target),controls.target.clone())}
    const clear=(group:THREE.Group)=>{disposeGroup(group,true);group.clear()}
    function update(){
      const {props:p,plane:pl,channel:ch,focus:f}=current.current,now=performance.now()
      for(const [id,entry] of pieces){if(!p.state.pieces.some(piece=>piece.id===id&&piece.kind===entry.kind)){board.remove(entry.object);disposeGroup(entry.object);pieces.delete(id)}}
      for(const piece of p.state.pieces){
        const target=squarePosition(piece.square);let entry=pieces.get(piece.id)
        if(!entry){const object=createPiece(piece.kind,piece.side,pieceMaterials);object.position.copy(target);board.add(object);entry={object,kind:piece.kind,target,from:target.clone(),started:now};pieces.set(piece.id,entry)}
        else if(!entry.target.equals(target)){entry.from.copy(entry.object.position);entry.target.copy(target);entry.started=now}
        entry.object.userData.square=piece.square
      }
      for(const tile of tiles){tile.material.color.set(tile.userData.color);tile.material.emissive.set('#000000');if(p.state.lastMove&&[p.state.lastMove.from,p.state.lastMove.to].includes(tile.userData.square))tile.material.color.lerp(new THREE.Color('#e0ba7d'),.32);if(tile.userData.square===p.selected){tile.material.emissive.set('#b78d55');tile.material.emissiveIntensity=.32}}
      clear(markers)
      for(const move of p.legal){const marker=new THREE.Mesh(move.isCapture?new THREE.RingGeometry(.36,.4,40):new THREE.CircleGeometry(.12,32),new THREE.MeshBasicMaterial({color:move.isCapture?'#f3ad9e':'#f9d79d',transparent:true,opacity:.95,side:THREE.DoubleSide}));marker.rotation.x=-Math.PI/2;marker.position.copy(squarePosition(move.to));marker.position.y=.051;markers.add(marker)}
      clear(overlays)
      const aligned=p.analysis?.positionPly===p.state.moveHistory.length&&p.analysis.positionTurn===p.state.currentTurn
      if(p.policy&&p.analysis&&aligned){const max=Math.max(...p.analysis.policyHeatmap,.01);p.analysis.policyHeatmap.forEach((v,i)=>{if(v<=0)return;const tile=new THREE.Mesh(new THREE.PlaneGeometry(.94,.94),new THREE.MeshBasicMaterial({color:'#7bded6',transparent:true,opacity:.12+v/max*.52,side:THREE.DoubleSide,depthWrite:false}));tile.rotation.x=-Math.PI/2;tile.position.set(i%8-3.5,.044,3.5-Math.floor(i/8));overlays.add(tile)})}
      if(p.paths&&p.analysis&&aligned)p.analysis.candidates.slice(0,3).forEach((candidate,i)=>{
        const from=squarePosition(candidate.from),to=squarePosition(candidate.to);from.y=to.y=.17
        const middle=from.clone().lerp(to,.5);middle.y=2.6-i*.2
        const curve=new THREE.QuadraticBezierCurve3(from,middle,to),material=new THREE.MeshBasicMaterial({color:['#edc58e','#88dad9','#b4a2e4'][i],transparent:true,opacity:.92-i*.16})
        overlays.add(new THREE.Mesh(new THREE.TubeGeometry(curve,40,.022-i*.003,8,false),material))
        const arrow=new THREE.Mesh(new THREE.ConeGeometry(.09,.2,12),material);arrow.position.copy(to);arrow.rotation.x=Math.PI;overlays.add(arrow)
      })
      collectBoardMaterials()
      if(p.analysis&&(p.analysis!==priorAnalysis||pl!==priorPlane||ch!==priorChannel)){
        for(const panel of panels){
          const data=stageData(p.analysis,panel.stage,pl,ch);panel.raw=data.raw;panel.normalized=data.normalized
          for(let i=0;i<panel.bars.count;i++){
            const v=data.normalized[i],h=panel.stage===5?.2+Math.abs(p.analysis.value)*1.4:.035+v*.34
            location.set(panel.stage===5?0:i%8-3.5,h/2+.045,panel.stage===5?0:3.5-Math.floor(i/8));scale.set(panel.stage===5?2.1:1,h,panel.stage===5?2.1:1)
            matrix.compose(location,rotation,scale);panel.bars.setMatrixAt(i,matrix)
            const c=new THREE.Color('#263149').lerp(new THREE.Color(networkStages[panel.stage].color),.08+v*.92);panel.bars.setColorAt(i,c)
          }
          panel.bars.instanceMatrix.needsUpdate=true;if(panel.bars.instanceColor)panel.bars.instanceColor.needsUpdate=true
          panel.bars.computeBoundingSphere()
        }
        priorAnalysis=p.analysis;priorPlane=pl;priorChannel=ch
      }
      if((p.layers&&!!p.analysis)!==previousLayers||f!==previousFocus||p.flipped!==previousFlip||p.resetCamera!==previousReset){
        setCell(null);hoverFrame.visible=false;frameCamera(previousFocus===-99)
        previousLayers=p.layers&&!!p.analysis;previousFocus=f;previousFlip=p.flipped;previousReset=p.resetCamera
      }
    }
    sync.current=update;update()
    const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2()
    const pick=(event:PointerEvent)=>{
      const rect=renderer.domElement.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera)
      const targets:THREE.Object3D[]=[]
      if(boardAlpha>.75)targets.push(...tiles,...Array.from(pieces.values(),v=>v.object))
      panels.filter(p=>p.alpha>.65).forEach(p=>targets.push(p.bars))
      return raycaster.intersectObjects(targets,true)[0]
    }
    let down={x:0,y:0},hoverKey='',hoverDescription=''
    const pointerDown=(e:PointerEvent)=>{down={x:e.clientX,y:e.clientY}}
    const pointerUp=(e:PointerEvent)=>{
      if(Math.hypot(e.clientX-down.x,e.clientY-down.y)>5)return
      const hit=pick(e);if(!hit)return
      if(hit.object.userData.stage!==undefined){const stage=hit.object.userData.stage as number,i=hit.instanceId??0;if(current.current.focus!==stage){setFocus(stage);setCell(null)}else{setCell({stage,square:stage===5?'Value':`${'abcdefgh'[i%8]}${Math.floor(i/8)+1}`,raw:panels[stage].raw[i],normalized:panels[stage].normalized[i]})}return}
      if(current.current.props.disabled)return
      let object:THREE.Object3D|null=hit.object;while(object&&!object.userData.square)object=object.parent
      if(object?.userData.square)current.current.props.onSquare(object.userData.square)
    }
    const pointerMove=(e:PointerEvent)=>{
      const hit=pick(e);let key='',description='';hoverFrame.visible=false
      if(hit&&hit.object.userData.stage!==undefined){
        const stage=hit.object.userData.stage as number,i=hit.instanceId??0,panel=panels[stage]
        const square=stage===5?'Value':`${'abcdefgh'[i%8]}${Math.floor(i/8)+1}`
        key=`${stage}:${i}`
        if(key!==hoverKey)setCell({stage,square,raw:panel.raw[i],normalized:panel.normalized[i]})
        if(stage<5){const h=.035+panel.normalized[i]*.34;hoverFrame.visible=true;hoverFrame.scale.set(1,h+.03,1);hoverFrame.position.set(i%8-3.5,panel.group.position.y+h/2+.045,3.5-Math.floor(i/8))}
      } else if(hit){
        let object:THREE.Object3D|null=hit.object;while(object&&!object.userData.square)object=object.parent
        const square=object?.userData.square as string|undefined
        const piece=current.current.props.state.pieces.find(p=>p.square===square)
        description=square?`${square.toUpperCase()}${piece?` · ${piece.side==='white'?'Ivory':'Midnight'} ${piece.kind} — ${pieceRules[piece.kind]}`:''}`:''
        key=description
      }
      if(!key.includes(':')&&hoverKey.includes(':'))setCell(null)
      if(description!==hoverDescription){setHover(description);hoverDescription=description}
      hoverKey=key;renderer.domElement.style.cursor=hit?'pointer':'grab'
    }
    const pointerLeave=()=>{setHover('');setCell(null);hoverFrame.visible=false;hoverKey=''}
    const lost=(event:Event)=>{event.preventDefault();current.current.props.onFallback()}
    renderer.domElement.addEventListener('pointerdown',pointerDown);renderer.domElement.addEventListener('pointerup',pointerUp)
    renderer.domElement.addEventListener('pointermove',pointerMove);renderer.domElement.addEventListener('pointerleave',pointerLeave)
    renderer.domElement.addEventListener('webglcontextlost',lost)
    let resizeTimer:ReturnType<typeof setTimeout>|undefined
    const observer=new ResizeObserver(()=>{const {width,height}=host.getBoundingClientRect();renderer.setSize(width,height);const nextAspect=width/Math.max(height,1);if(Math.abs(nextAspect-camera.aspect)>.015){clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>frameCamera(),180)}camera.aspect=nextAspect;camera.updateProjectionMatrix()})
    observer.observe(host)
    let frame=0,previousTime=performance.now()
    function animate(now:number){
      const dt=Math.min((now-previousTime)/1000,.05);previousTime=now
      const {props:p,focus:f,flowing:flow}=current.current,active=p.layers&&!!p.analysis
      explosion=reducedMotion?(active?1:0):THREE.MathUtils.damp(explosion,active?1:0,5.5,dt)
      board.position.y=explosion*1.28
      boardAlpha=reducedMotion?(active&&f>=0?0:1):THREE.MathUtils.damp(boardAlpha,active&&f>=0?0:1,6,dt)
      board.visible=boardAlpha>.015;boardMaterials.forEach(m=>{m.opacity=boardAlpha})
      floor.visible=explosion<.1;groundGlow.visible=explosion<.1;floor.material.opacity=.28*boardAlpha;groundGlow.material.opacity=.8*boardAlpha*(1-explosion)
      for(const panel of panels){
        const target=active&&(f<0||f===panel.stage)?1:0
        panel.alpha=reducedMotion?target:THREE.MathUtils.damp(panel.alpha,target,6,dt)
        panel.group.visible=panel.alpha>.015;panel.group.position.y=-.4-explosion*(.25+panel.stage*1.04)
        panel.materials.forEach((m,i)=>m.opacity=panel.alpha*(i===0?.94:i===1?.5:1))
      }
      links.visible=active&&f<0;linkMaterial.opacity=explosion*.17
      if(flow&&!wasFlowing){flowStart=now;lastFlow=-1}
      if(flow){const elapsed=(now-flowStart)/7800,step=Math.min(4,Math.floor(elapsed*5));if(step!==lastFlow){setFlowStep(step);lastFlow=step}
        pulse.visible=!reducedMotion;valuePulse.visible=!reducedMotion&&step===4;const headProgress=Math.max(0,(elapsed-.8)/.2);pulse.position.set(-4.18,step<4?-.65-elapsed/.8*3.12:-3.77-headProgress*1.04,-4.18);valuePulse.position.set(4.18,-3.77-headProgress*2.08,-4.18)
        panels.forEach(panel=>{(panel.bars.material as THREE.MeshStandardMaterial).emissiveIntensity=(panel.stage===step||(step===4&&panel.stage===5))?.38:.11})
        if(elapsed>=1){setFlowing(false);setFlowStep(-1)}
      }else{pulse.visible=false;valuePulse.visible=false;panels.forEach(panel=>(panel.bars.material as THREE.MeshStandardMaterial).emissiveIntensity=.11)}
      wasFlowing=flow
      for(const entry of pieces.values()){const t=reducedMotion?1:Math.min((now-entry.started)/520,1);entry.object.position.lerpVectors(entry.from,entry.target,ease(t));if(t<1&&!entry.from.equals(entry.target))entry.object.position.y+=Math.sin(t*Math.PI)*.35}
      if(cameraTween){const t=Math.min((now-cameraTween.start)/1100,1),a=ease(t);camera.position.lerpVectors(cameraTween.from,cameraTween.to,a);controls.target.lerpVectors(cameraTween.lookFrom,cameraTween.lookTo,a);if(t===1)cameraTween=null}
      controls.update();renderer.render(scene,camera);frame=requestAnimationFrame(animate)
    }
    frame=requestAnimationFrame(animate)
    return()=>{
      cancelAnimationFrame(frame);clearTimeout(resizeTimer);observer.disconnect();controls.dispose()
      renderer.domElement.removeEventListener('pointerdown',pointerDown);renderer.domElement.removeEventListener('pointerup',pointerUp)
      renderer.domElement.removeEventListener('pointermove',pointerMove);renderer.domElement.removeEventListener('pointerleave',pointerLeave);renderer.domElement.removeEventListener('webglcontextlost',lost)
      disposeGroup(scene,true);Object.values(pieceMaterials).forEach(m=>m.dispose());textures.forEach(t=>t.dispose());envTarget.dispose();renderer.dispose();renderer.domElement.remove();sync.current=null;zoom.current=null
    }
  },[])
  useEffect(()=>{sync.current?.()},[props.state,props.selected,props.legal,props.analysis,props.policy,props.layers,props.paths,props.flipped,props.resetCamera,focus,plane,channel])
  const active=props.layers&&!!props.analysis,stage=networkStages[cell?.stage??(focus>=0?focus:Math.max(0,flowStep))]
  return <div className={`board-3d ${active?'network-open':''}`} ref={container}>
    <div className="scene-zoom"><button aria-label="Zoom in" title="Zoom in" onClick={()=>zoom.current?.(.8)}><Plus size={15}/></button><button aria-label="Zoom out" title="Zoom out" onClick={()=>zoom.current?.(1.25)}><Minus size={15}/></button></div>
    {active&&<>
      <div className="network-toolbar"><div className="network-toolbar-title"><ScanLine size={14}/><span>NETWORK EXPLORER</span></div><label className="sr-only" htmlFor="network-focus">Inspect neural layer</label><select id="network-focus" value={focus} onChange={e=>{setFocus(Number(e.target.value));setCell(null)}}><option value={-1}>All layers · Overview</option>{networkStages.map((s,i)=><option key={s.name} value={i}>{String(i+1).padStart(2,'0')} · {s.name}</option>)}</select><button className={flowing?'is-active':''} aria-label={flowing?'Pause inference replay':'Replay inference'} onClick={()=>{setFlowing(!flowing);if(flowing)setFlowStep(-1)}}>{flowing?<Pause size={14}/>:<Play size={14}/>}<span>{flowing?'Pause':'Replay inference'}</span></button></div>
      {focus>=0&&<div className="network-focus-tools"><button onClick={overview}><ChevronLeft size={13}/>All layers</button><span>{networkStages[focus].shape}</span>{focus===0?<><label className="sr-only" htmlFor="network-plane">3D input plane</label><select id="network-plane" value={plane} onChange={e=>{setPlane(Number(e.target.value));setCell(null)}}>{planeNames.map((n,i)=><option key={n} value={i}>{n}</option>)}</select></>:focus<4?<><label className="sr-only" htmlFor="network-channel">Feature channel</label><select id="network-channel" value={channel} onChange={e=>{setChannel(Number(e.target.value));setCell(null)}}><option value={-1}>Mean of all 32 channels</option>{Array.from({length:32},(_,i)=><option value={i} key={i}>Channel {i+1}</option>)}</select></>:null}</div>}
      <div className="network-readout"><div className="network-readout-heading"><span style={{background:stage.color}}/><strong>{focus<0&&!cell&&flowStep<0?'A position becomes a decision.':focus<0&&flowStep===4?'Policy + value heads':stage.name}</strong>{cell&&<b>{cell.square} · {cell.stage===4?`${(cell.raw*100).toFixed(2)}%`:cell.raw.toFixed(4)}</b>}{focus===5&&!cell&&<b>{props.analysis!.value.toFixed(4)}</b>}</div><p>{cell?stage.meaning:focus<0&&flowStep<0?'Choose a layer to glide inside it, or hover a cell to inspect its measured value. Replay the signal flow to follow the computation.':focus<0&&flowStep===4?'The shared residual features feed two parallel heads. Policy supplies legal-move priors; value predicts the outcome. MCTS uses both.':stage.description}</p><small>{flowing?'Illustrative signal replay · ':''}Measured at ply {props.analysis!.positionPly} · {props.analysis!.positionTurn==='white'?'Ivory':'Midnight'} to move{focus>=1&&focus<=3?' · Height and color normalized for this view':''}{props.analysis!.positionPly!==props.state.moveHistory.length?' · Previous decision; refresh to inspect the new position':''}</small></div>
      <div className="network-stage-nav" aria-label="Neural layers">{networkStages.map((s,i)=><button key={s.name} title={s.description} aria-label={`Inspect ${s.name}`} aria-pressed={focus===i} className={flowStep===i||(flowStep===4&&i===5)?'flow-active':''} onClick={()=>{setFocus(i);setCell(null)}} style={{'--stage-color':s.color} as React.CSSProperties}><span>{String(i+1).padStart(2,'0')}</span>{s.short}</button>)}</div>
    </>}
    {!active&&<div className="scene-caption"><Focus size={11}/><span>{hover||'DRAG TO ORBIT · SCROLL TO ZOOM · CLICK TO PLAY'}</span></div>}
  </div>
}
