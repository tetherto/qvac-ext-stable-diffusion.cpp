# Per-context performance controls

These C API controls preserve the existing defaults. Call them only while the
context is idle, between generation calls; they are not concurrent operations.

- `sd_set_h3_local_ops(ctx, enabled)` enables fused SwiGLU and strided QKV views
  in MiniMax-H3. It defaults to disabled and returns false for null or non-H3
  contexts. It reuses existing ggml operations without changing weights or
  sampling steps. The caller must validate its backend: equivalent operations
  can produce different floating-point results, especially on Metal.
- `sd_set_vae_conv_direct(ctx, enabled)` switches the loaded VAE and optional
  preview VAE between direct convolution and the matrix implementation. It
  returns false for a null context or a context without a VAE. The matrix path
  can use more workspace and can be slower on some backends. Callers can switch
  back before a larger request; the initial setting still comes from
  `sd_ctx_params_t.vae_conv_direct`.

Device selection, memory checks and automatic profiles belong to the caller.
These controls do not change the ABI layout of `sd_ctx_params_t`.
