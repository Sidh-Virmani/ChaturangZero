import type { AnalyzedAction, Side } from './game'

export interface NeuralLayer {
  id: string
  label: string
  operation: string
  explanation: string
  shape: number[]
  channels: number[][]
  channelNames: string[] | null
  minimum: number
  maximum: number
}

export interface NeuralInspection {
  positionKey: string
  sideToMove: Side
  checkpoint: string | null
  smokeCheckpoint: boolean
  layers: NeuralLayer[]
  value: number | null
  legalActionCount: number
  parameterCount: number | null
  policy: AnalyzedAction[]
}

export interface InspectCell {
  layerId: string
  square: string
  value: number
  channel: string
}
