// bare/binding.c — Bare native addon exposing LTX-2 video generation to JS.
//
// Wraps the header-only C API in include/ltx2.h (ltx2_new_ctx / ltx2_generate_t2v
// / ltx2_generate_i2v) so JavaScript running in the Bare runtime (QVAC) can drive
// LTX-2 T2V and I2V. Follows the Holepunch Bare addon conventions: js.h + bare.h,
// BARE_MODULE registration, built with cmake-bare / bare-make.
//
// JS-facing surface (see index.js for the ergonomic wrapper):
//   createContext(diffusion, vae, audioVae|null, llm, connectors, threads, backend|null) -> external
//   generateT2V(ctx, prompt, neg|null, w, h, frames, fps, seed) -> { width, height, channel, frames, data }
//   generateI2V(ctx, prompt, neg|null, initData, iw, ih, ic, w, h, frames, fps, seed) -> { ... }
//
// The returned `data` is an ArrayBuffer of contiguous RGB frames
// (frames * height * width * channel bytes); encode/save it on the JS side.

#include <assert.h>
#include <bare.h>
#include <js.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <utf.h>

#include "ltx2.h" // header-only LTX-2 C API; pulls in stable-diffusion.h

// --- helpers ---------------------------------------------------------------

// Read a JS string into a freshly malloc'd, NUL-terminated buffer.
// Returns NULL for null/undefined (so callers can pass an optional arg). Frees on caller.
static char *
ltx2_bare__strdup_arg(js_env_t *env, js_value_t *val) {
  js_value_type_t type;
  if (js_typeof(env, val, &type) != 0) return NULL;
  if (type == js_null || type == js_undefined) return NULL;

  size_t len = 0;
  if (js_get_value_string_utf8(env, val, NULL, 0, &len) != 0) return NULL;
  char *buf = malloc(len + 1);
  if (buf == NULL) return NULL;
  js_get_value_string_utf8(env, val, (utf8_t *) buf, len + 1, &len);
  buf[len] = '\0';
  return buf;
}

// Finalizer: release the sd_ctx_t when its JS external handle is garbage-collected.
static void
ltx2_bare__finalize_ctx(js_env_t *env, void *data, void *hint) {
  (void) env;
  (void) hint;
  if (data != NULL) free_sd_ctx((sd_ctx_t *) data);
}

// Pack an sd_image_t* frame array into { width, height, channel, frames, data:ArrayBuffer }.
// Frees the engine-allocated frames (and their .data) after copying.
static js_value_t *
ltx2_bare__pack_frames(js_env_t *env, sd_image_t *frames, int n) {
  uint32_t w = frames[0].width;
  uint32_t h = frames[0].height;
  uint32_t c = frames[0].channel;
  size_t frame_bytes = (size_t) w * h * c;
  size_t total = frame_bytes * (size_t) n;

  js_value_t *ab;
  void *data = NULL;
  assert(js_create_arraybuffer(env, total, &data, &ab) == 0);

  uint8_t *dst = (uint8_t *) data;
  for (int i = 0; i < n; i++) {
    memcpy(dst + (size_t) i * frame_bytes, frames[i].data, frame_bytes);
    free(frames[i].data);
  }
  free(frames);

  js_value_t *obj;
  js_create_object(env, &obj);
  js_value_t *jv;
  js_create_uint32(env, w, &jv);
  js_set_named_property(env, obj, "width", jv);
  js_create_uint32(env, h, &jv);
  js_set_named_property(env, obj, "height", jv);
  js_create_uint32(env, c, &jv);
  js_set_named_property(env, obj, "channel", jv);
  js_create_uint32(env, (uint32_t) n, &jv);
  js_set_named_property(env, obj, "frames", jv);
  js_set_named_property(env, obj, "data", ab);
  return obj;
}

// --- exported functions ----------------------------------------------------

static js_value_t *
ltx2_bare_create_context(js_env_t *env, js_callback_info_t *info) {
  size_t argc = 7;
  js_value_t *argv[7];
  js_get_callback_info(env, info, &argc, argv, NULL, NULL);

  char *diffusion = ltx2_bare__strdup_arg(env, argv[0]);
  char *vae = ltx2_bare__strdup_arg(env, argv[1]);
  char *audio_vae = ltx2_bare__strdup_arg(env, argv[2]);
  char *llm = ltx2_bare__strdup_arg(env, argv[3]);
  char *conn = ltx2_bare__strdup_arg(env, argv[4]);
  int32_t threads = 0;
  js_get_value_int32(env, argv[5], &threads);
  char *backend = ltx2_bare__strdup_arg(env, argv[6]);

  sd_ctx_t *ctx = ltx2_new_ctx(diffusion, vae, audio_vae, llm, conn, threads, backend);

  free(diffusion);
  free(vae);
  free(audio_vae);
  free(llm);
  free(conn);
  free(backend);

  if (ctx == NULL) {
    js_throw_error(env, NULL, "ltx2_new_ctx failed (check model paths)");
    return NULL;
  }

  js_value_t *handle;
  js_create_external(env, ctx, ltx2_bare__finalize_ctx, NULL, &handle);
  return handle;
}

static js_value_t *
ltx2_bare_generate_t2v(js_env_t *env, js_callback_info_t *info) {
  size_t argc = 8;
  js_value_t *argv[8];
  js_get_callback_info(env, info, &argc, argv, NULL, NULL);

  sd_ctx_t *ctx = NULL;
  js_get_value_external(env, argv[0], (void **) &ctx);
  char *prompt = ltx2_bare__strdup_arg(env, argv[1]);
  char *neg = ltx2_bare__strdup_arg(env, argv[2]);
  int32_t w = 0, h = 0, frames = 0, fps = 0;
  int64_t seed = 0;
  js_get_value_int32(env, argv[3], &w);
  js_get_value_int32(env, argv[4], &h);
  js_get_value_int32(env, argv[5], &frames);
  js_get_value_int32(env, argv[6], &fps);
  js_get_value_int64(env, argv[7], &seed);

  sd_image_t *out = NULL;
  int n = 0;
  sd_audio_t *audio = NULL;
  bool ok = ltx2_generate_t2v(ctx, prompt, neg, w, h, frames, fps, seed, &out, &n, &audio);

  free(prompt);
  free(neg);

  if (!ok || out == NULL || n <= 0) {
    js_throw_error(env, NULL, "ltx2_generate_t2v failed");
    return NULL;
  }
  return ltx2_bare__pack_frames(env, out, n);
}

static js_value_t *
ltx2_bare_generate_i2v(js_env_t *env, js_callback_info_t *info) {
  size_t argc = 12;
  js_value_t *argv[12];
  js_get_callback_info(env, info, &argc, argv, NULL, NULL);

  sd_ctx_t *ctx = NULL;
  js_get_value_external(env, argv[0], (void **) &ctx);
  char *prompt = ltx2_bare__strdup_arg(env, argv[1]);
  char *neg = ltx2_bare__strdup_arg(env, argv[2]);

  void *img_data = NULL;
  size_t img_len = 0;
  js_get_arraybuffer_info(env, argv[3], &img_data, &img_len);
  uint32_t iw = 0, ih = 0, ic = 0;
  js_get_value_uint32(env, argv[4], &iw);
  js_get_value_uint32(env, argv[5], &ih);
  js_get_value_uint32(env, argv[6], &ic);

  int32_t w = 0, h = 0, frames = 0, fps = 0;
  int64_t seed = 0;
  js_get_value_int32(env, argv[7], &w);
  js_get_value_int32(env, argv[8], &h);
  js_get_value_int32(env, argv[9], &frames);
  js_get_value_int32(env, argv[10], &fps);
  js_get_value_int64(env, argv[11], &seed);

  sd_image_t init = {
    .width = iw,
    .height = ih,
    .channel = ic,
    .data = (uint8_t *) img_data,
  };

  sd_image_t *out = NULL;
  int n = 0;
  sd_audio_t *audio = NULL;
  bool ok = ltx2_generate_i2v(ctx, prompt, neg, init, w, h, frames, fps, seed, &out, &n, &audio);

  free(prompt);
  free(neg);

  if (!ok || out == NULL || n <= 0) {
    js_throw_error(env, NULL, "ltx2_generate_i2v failed");
    return NULL;
  }
  return ltx2_bare__pack_frames(env, out, n);
}

// --- module registration ---------------------------------------------------

static js_value_t *
ltx2_bare_exports(js_env_t *env, js_value_t *exports) {
  int err;
#define EXPORT_FN(name, fn) \
  { \
    js_value_t *val; \
    err = js_create_function(env, name, (size_t) -1, fn, NULL, &val); \
    assert(err == 0); \
    err = js_set_named_property(env, exports, name, val); \
    assert(err == 0); \
  }

  EXPORT_FN("createContext", ltx2_bare_create_context)
  EXPORT_FN("generateT2V", ltx2_bare_generate_t2v)
  EXPORT_FN("generateI2V", ltx2_bare_generate_i2v)
#undef EXPORT_FN

  return exports;
}

BARE_MODULE(ltx2_bare, ltx2_bare_exports)
