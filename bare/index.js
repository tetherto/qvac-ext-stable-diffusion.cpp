// bare/index.js — ergonomic JS API over the LTX-2 Bare addon.
//
//   const ltx2 = require('bare-ltx2')
//   const ctx = ltx2.createContext({ diffusionModel, vae, llm, connectors })
//   const out = ltx2.generateT2V(ctx, { prompt: 'a cat on a windowsill', frames: 25 })
//   // out: { width, height, channel, frames, data: ArrayBuffer }  (contiguous RGB)
//   ctx.destroy()
//
// `data` holds `frames` RGB images of width*height*channel bytes each; encode or
// save them with whatever pipeline you like (e.g. write PNGs, pipe to ffmpeg).

const binding = require('./binding')

const DEFAULTS = { width: 512, height: 288, frames: 25, fps: 8, seed: -1, negPrompt: null }

class LTX2Context {
  constructor (handle) {
    this._handle = handle
  }

  // The native sd_ctx_t is freed by the addon's finalizer when this object is
  // garbage-collected; destroy() drops the reference to make that eligible.
  destroy () {
    this._handle = null
  }

  _assert () {
    if (this._handle == null) throw new Error('LTX2 context has been destroyed')
    return this._handle
  }
}

function createContext (opts = {}) {
  const {
    diffusionModel, vae, llm, connectors,
    audioVae = null, threads = 0, backend = null
  } = opts

  for (const [k, v] of Object.entries({ diffusionModel, vae, llm, connectors })) {
    if (!v) throw new Error(`createContext: missing required option "${k}"`)
  }

  const handle = binding.createContext(diffusionModel, vae, audioVae, llm, connectors, threads, backend)
  return new LTX2Context(handle)
}

function generateT2V (ctx, opts = {}) {
  const handle = ctx._assert()
  const { prompt, negPrompt, width, height, frames, fps, seed } = { ...DEFAULTS, ...opts }
  if (!prompt) throw new Error('generateT2V: "prompt" is required')
  return binding.generateT2V(handle, prompt, negPrompt, width, height, frames, fps, seed)
}

function generateI2V (ctx, opts = {}) {
  const handle = ctx._assert()
  const { prompt, negPrompt, initImage, width, height, frames, fps, seed } = { ...DEFAULTS, ...opts }
  if (!prompt) throw new Error('generateI2V: "prompt" is required')
  if (!initImage || !initImage.data) {
    throw new Error('generateI2V: "initImage" { data, width, height, channel } is required')
  }

  // Accept either an ArrayBuffer or a TypedArray/Buffer for initImage.data.
  const data = initImage.data instanceof ArrayBuffer ? initImage.data : initImage.data.buffer
  const channel = initImage.channel ?? 3

  return binding.generateI2V(
    handle, prompt, negPrompt,
    data, initImage.width, initImage.height, channel,
    width, height, frames, fps, seed
  )
}

module.exports = { createContext, generateT2V, generateI2V, LTX2Context }
