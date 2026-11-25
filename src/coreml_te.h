#ifndef __SD_COREML_TE_H__
#define __SD_COREML_TE_H__

#include <cstdint>
#include <string>
#include <vector>

// A text encoder exported to CoreML as a folder of chunks run in order (FLUX.2-klein's Qwen3 layers 1-27).
// The first chunk takes input_ids [1, L] int32 and key_mask [1, L] (1 for
// prompt tokens, 0 for padding); every chunk takes key_mask and earlier chunks' outputs by name. Compute units
// come from SD_COREML_COMPUTE_UNITS (cpu_ne by default).
struct CoreMLTextEncoder;

CoreMLTextEncoder* coreml_te_open(const std::string& path, std::string* error);
void coreml_te_close(CoreMLTextEncoder* encoder);

// The fixed number of tokens the model takes.
int64_t coreml_te_length(const CoreMLTextEncoder* encoder);

// Runs the chunks and returns the outputs named in taps, concatenated per token: out is [L][sum of widths] floats,
// which is ggml's [width, L] layout. tokens and mask must hold coreml_te_length() entries.
bool coreml_te_encode(CoreMLTextEncoder* encoder,
                      const std::vector<int32_t>& tokens,
                      const std::vector<float>& mask,
                      const std::vector<std::string>& taps,
                      std::vector<float>& out,
                      int64_t* width,
                      std::string* error);

#endif
