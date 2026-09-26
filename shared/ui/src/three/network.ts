import type { Analysis } from '../types/game'

export const planeNames = ['Own raja', 'Own mantri', 'Own ratha', 'Own ashva', 'Own gaja', 'Own padati', 'Enemy raja', 'Enemy mantri', 'Enemy ratha', 'Enemy ashva', 'Enemy gaja', 'Enemy padati', 'No-progress clock', 'Repetition count']
export const networkStages = [
  { name:'Input tensor', short:'Input', shape:'14 × 8 × 8', color:'#ddbaa9', description:'Fourteen planes encode the pieces, no-progress clock, and repetition. Select a plane to see exactly what enters the network.', meaning:'Piece planes contain 1 where that piece is present and 0 elsewhere. The clock and repetition planes carry normalized draw-rule context.' },
  { name:'Input convolution', short:'Convolution', shape:'32 × 8 × 8', color:'#72ccc9', description:'A learned 3 × 3 convolution combines nearby input squares into 32 feature channels. ReLU keeps positive activations.', meaning:'Brighter cells have larger activations. A channel is a learned feature, not a named concept such as danger or control.' },
  { name:'Residual block 1', short:'Residual 1', shape:'32 × 8 × 8', color:'#8fb3e0', description:'Two learned 3 × 3 convolutions refine the features. A skip connection adds the incoming features back before ReLU.', meaning:'Output = ReLU(x + F(x)). The skip connection helps preserve information while the block learns a correction.' },
  { name:'Residual block 2', short:'Residual 2', shape:'32 × 8 × 8', color:'#b6a2d8', description:'A second residual block combines a wider neighborhood. These features are shared by the policy and value heads.', meaning:'These are measured activations, not causal explanations of a decision. Compare channels without assigning them invented strategic meanings.' },
  { name:'Policy head', short:'Policy', shape:'4,096 actions → legal mask', color:'#d2b277', description:'A 1 × 1 convolution produces move logits. Illegal moves are masked, then softmax supplies the initial search probabilities.', meaning:'This board sums neural priors by destination. It is before MCTS; the Arena policy heatmap instead shows visit shares after search.' },
  { name:'Value head', short:'Value', shape:'4 × 8 × 8 → 64 → 1', color:'#dd969e', description:'A small convolution and dense layers predict one tanh-bounded value for the player to move. Search backs values up through its tree.', meaning:'−1 means predicted loss, 0 draw, +1 win. This raw network value is not a calibrated probability and can differ from the search value.' },
] as const

export function stageData(analysis: Analysis, stage: number, plane: number, channel: number) {
  let raw: number[]
  if (stage === 0) {
    const canonical = analysis.input[plane]
    raw = (analysis.positionTurn === 'black' ? canonical.slice().reverse() : canonical).flat()
  } else if (stage < 4) {
    const layer = analysis.layers[stage - 1]
    raw = channel < 0 ? layer.heatmap.map(v => v * layer.maxActivation) : layer.channels[channel]
  } else if (stage === 4) raw = analysis.policyPriorHeatmap
  else raw = [analysis.value]
  const max = Math.max(...raw.map(Math.abs), .000001)
  return { raw, normalized: raw.map(v => Math.abs(v) / (stage === 0 ? 1 : max)), max }
}
