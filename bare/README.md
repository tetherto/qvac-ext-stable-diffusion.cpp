# bare-ltx2 — Bare native addon for LTX-2 video generation

M3 of the [Tether LTX-2 bounty](https://tether.dev/grants/bounties/2885262943): a
[Bare](https://github.com/holepunchto/bare) native addon that exposes LTX-2
**T2V** and **I2V** to JavaScript in the QVAC ecosystem. It wraps the header-only
C API in [`../include/ltx2.h`](../include/ltx2.h) (`ltx2_new_ctx`,
`ltx2_generate_t2v`, `ltx2_generate_i2v`).

## Layout

| File | Role |
|------|------|
| `binding.c` | C addon — `js.h`/`bare.h`, registered via `BARE_MODULE` |
| `binding.js` | `module.exports = require.addon()` |
| `index.js` | ergonomic JS API (`createContext` / `generateT2V` / `generateI2V`) |
| `CMakeLists.txt` | `cmake-bare` build; links the repo's `stable-diffusion` lib |
| `package.json` | `"addon": true`, `bare-make` scripts |
| `test.js` | smoke test (needs model weights + a built addon) |

## Build

Requires the [Bare](https://github.com/holepunchto/bare) toolchain
(`npm i -g bare bare-make`), **clang + lld** (bare-make's CMake toolchain links
with `lld`; on Debian/Ubuntu: `apt install clang lld`), and CMake ≥ 3.25.

```sh
cd bare
npm install                 # pulls cmake-bare + bare-make
npm run generate            # bare-make generate  (configure)
npm run build               # bare-make build     (compiles binding.c + links sd lib)
npx bare-make install       # copies the addon into prebuilds/<platform>/ so require.addon() finds it
bare test.js                # smoke test (SKIPs unless $LTX2_MODELS is set)
```

> Verified building on linux-x64 (clang 18, gcc backend, CPU): the addon
> compiles, links `libstable-diffusion.a`, installs to `prebuilds/linux-x64/`,
> and `require()` exposes `createContext` / `generateT2V` / `generateI2V`.

For an accelerated backend, pass the ggml flag through to the embedded sd build,
e.g. `bare-make generate --define GGML_VULKAN=ON` (Vulkan) or `GGML_METAL=ON`
(Metal). The addon links the same `stable-diffusion` library the CLI uses, so
backend selection matches M2.

## Usage

```js
const ltx2 = require('bare-ltx2')

const ctx = ltx2.createContext({
  diffusionModel: '/models/distilled/ltx-2.3-22b-distilled-Q4_0.gguf',
  vae:            '/models/vae/ltx-2.3-22b-distilled_video_vae.safetensors',
  llm:            '/models/gemma-3-12b-it-Q4_K_S.gguf',
  connectors:     '/models/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors',
  threads: 0,        // 0 = auto
  backend: null      // null = default; or 'vulkan' / 'metal' if your build supports it
})

// Text-to-video
const t2v = ltx2.generateT2V(ctx, {
  prompt: 'a lovely cat on a sunny windowsill, high quality',
  width: 512, height: 288, frames: 25, fps: 8, seed: 42
})
// t2v: { width, height, channel, frames, data: ArrayBuffer }  — contiguous RGB

// Image-to-video (initImage.data = raw RGB bytes)
const i2v = ltx2.generateI2V(ctx, {
  prompt: 'gentle camera push-in',
  initImage: { data: rgbArrayBuffer, width: 512, height: 288, channel: 3 },
  frames: 25, fps: 8, seed: 42
})

ctx.destroy()
```

`data` is `frames × height × width × channel` bytes. Slice it into per-frame RGB
buffers and encode however you like (write PNGs, pipe to ffmpeg, etc.).

## Status / verify-points

Scaffolded against the Holepunch `bare-zlib` addon conventions but **not yet
compiled** against the real `bare.h`/`js.h` + `stable-diffusion` lib. Confirm on
first build:

1. `target_link_libraries(... stable-diffusion)` — match the exact library target
   name exported by the repo's top-level `CMakeLists.txt`.
2. `js_get_value_int64` / `js_get_arraybuffer_info` / `js_create_external`
   signatures against your installed `js.h` (these match N-API conventions but
   pin to the Bare version you build with).
3. `sd_image_t` field names (`width`/`height`/`channel`/`data`) and the
   `ltx2_generate_*` / `free_sd_ctx` signatures in `../include/`.
