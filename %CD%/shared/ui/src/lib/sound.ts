let audioContext: AudioContext | null = null

function context(): AudioContext | null {
  const AudioCtor = window.AudioContext || window.webkitAudioContext
  if (!AudioCtor) return null
  audioContext ??= new AudioCtor()
  return audioContext
}

function tone(frequency: number, duration: number, gainValue: number, type: OscillatorType) {
  const ctx = context()
  if (!ctx) return
  void ctx.resume()
  const oscillator = ctx.createOscillator()
  const gain = ctx.createGain()
  oscillator.type = type
  oscillator.frequency.setValueAtTime(frequency, ctx.currentTime)
  gain.gain.setValueAtTime(gainValue, ctx.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration)
  oscillator.connect(gain)
  gain.connect(ctx.destination)
  oscillator.start()
  oscillator.stop(ctx.currentTime + duration)
}

export function playMoveSound(isCapture: boolean): void {
  if (isCapture) {
    tone(220, 0.16, 0.08, 'square')
    window.setTimeout(() => tone(140, 0.18, 0.06, 'triangle'), 40)
    return
  }
  tone(392, 0.09, 0.05, 'sine')
  window.setTimeout(() => tone(523, 0.08, 0.04, 'sine'), 50)
}

declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext
  }
}
