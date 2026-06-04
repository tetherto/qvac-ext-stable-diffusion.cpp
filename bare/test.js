// bare/test.js — smoke test for the LTX-2 Bare addon.
//
// Needs a built addon (npm run build) and the LTX-2 model weights. Point it at
// the model dir with $LTX2_MODELS; if unset, the test skips rather than fails so
// CI without weights stays green.
//
//   LTX2_MODELS=/path/to/ltx2-models bare test.js

const env = require('bare-env')
const ltx2 = require('./index.js')

const MODELS = env.LTX2_MODELS
if (!MODELS) {
  console.log('SKIP: set $LTX2_MODELS to run the LTX-2 addon smoke test')
  Bare.exit(0)
}

const root = MODELS.replace(/\/+$/, '')
const ctx = ltx2.createContext({
  diffusionModel: `${root}/distilled/ltx-2.3-22b-distilled-Q4_0.gguf`,
  vae: `${root}/vae/ltx-2.3-22b-distilled_video_vae.safetensors`,
  llm: `${root}/gemma-3-12b-it-Q4_K_S.gguf`,
  connectors: `${root}/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors`
})

const out = ltx2.generateT2V(ctx, {
  prompt: 'a lovely cat sitting on a sunny windowsill, high quality',
  width: 512, height: 288, frames: 25, fps: 8, seed: 42
})

const expected = out.frames * out.width * out.height * out.channel
if (!(out.data instanceof ArrayBuffer)) throw new Error('expected ArrayBuffer of frames')
if (out.data.byteLength !== expected) {
  throw new Error(`frame buffer size ${out.data.byteLength} != expected ${expected}`)
}
if (out.frames !== 25) throw new Error(`expected 25 frames, got ${out.frames}`)

ctx.destroy()
console.log(`OK: T2V produced ${out.frames} frames @ ${out.width}x${out.height}x${out.channel}`)
