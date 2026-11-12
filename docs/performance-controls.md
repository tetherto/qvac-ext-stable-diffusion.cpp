# Per-context performance controls

These C API controls preserve the existing defaults. Call them only while the
context is idle, between generation calls. Calls made while generation or another
context operation is running return false, including reentrant calls from callbacks.
Successful settings are retained on the context and restored after runner rebuilds,
ControlNet updates and model-source refreshes.

- `sd_set_h3_local_ops(ctx, enabled)` enables fused SwiGLU and strided QKV views
  in MiniMax-H3. It defaults to enabled, preserving the `2026-11-01` graph and returns false for null or non-H3
  contexts. It reuses existing ggml operations without changing weights or
  sampling steps. Disabling restores materialized QKV slices and separate SiLU/multiply operations. The caller must validate its backend: equivalent operations
  can produce different floating-point results, especially on Metal.
- `sd_set_vae_conv_direct(ctx, enabled)` switches the loaded main, preview and
  audio VAEs between direct convolution and the matrix implementation. It
  returns false for a null context or a context without a VAE. The matrix path
  can use more workspace and can be slower on some backends. Callers can switch
  back before a larger request; the initial setting still follows
  `sd_ctx_params_t.conv_direct_mode` and its legacy `vae_conv_direct` fallback.
  This setter changes 2D convolution only; 3D convolution retains its configured setting.

Device selection, memory checks and automatic profiles belong to the caller.
These controls do not change the ABI layout of `sd_ctx_params_t`.
