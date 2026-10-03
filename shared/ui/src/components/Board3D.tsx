import { useEffect, useRef } from 'react'
import { createBoardScene, type SceneSettings } from '../three/scene'
import type { SquareId } from '../types/game'
import type { InspectCell } from '../types/neural'

interface Props extends SceneSettings {
  onSquareClick: (square: SquareId) => void
  onHover: (cell: InspectCell | null) => void
  onFocus: (id: string) => void
  onFailure: () => void
}

export default function Board3D(props: Props) {
  const host = useRef<HTMLDivElement>(null)
  const scene = useRef<ReturnType<typeof createBoardScene> | null>(null)
  const current = useRef(props); current.current = props
  useEffect(() => {
    try {
      scene.current = createBoardScene(host.current!, square => current.current.onSquareClick(square), cell => current.current.onHover(cell), id => current.current.onFocus(id), () => current.current.onFailure())
      scene.current.update(current.current)
    } catch {
      current.current.onFailure()
    }
    return () => { scene.current?.dispose(); scene.current = null }
  }, [])
  useEffect(() => { scene.current?.update(props) }, [props])
  return <div ref={host} className="three-stage" />
}
