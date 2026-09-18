#include "name_conversion.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <utility>
#include <vector>

int main() {
    const std::pair<const char*, const char*> cases[] = {
        {"first_stage_model.encoder.down_blocks.0.resnets.0.conv1.weight",
         "first_stage_model.encoder.down.0.block.0.conv1.weight"},
        {"first_stage_model.encoder.down_blocks.2.downsamplers.0.conv.weight",
         "first_stage_model.encoder.down.2.downsample.conv.weight"},
        {"first_stage_model.decoder.up_blocks.0.resnets.2.conv_shortcut.weight",
         "first_stage_model.decoder.up.3.block.2.nin_shortcut.weight"},
        {"first_stage_model.decoder.up_blocks.1.upsamplers.0.conv.weight",
         "first_stage_model.decoder.up.2.upsample.conv.weight"},
        {"first_stage_model.encoder.mid_block.resnets.1.conv1.weight",
         "first_stage_model.encoder.mid.block_2.conv1.weight"},
        {"first_stage_model.decoder.mid_block.attentions.0.to_q.weight",
         "first_stage_model.decoder.mid.attn_1.q.weight"},
        {"first_stage_model.encoder.conv1.weight",
         "first_stage_model.encoder.conv1.weight"},
    };
    constexpr int worker_count = 32;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};
    std::vector<std::thread> workers;
    for (int i = 0; i < worker_count; ++i) {
        workers.emplace_back([&] {
            ++ready;
            while (!start.load()) {
                std::this_thread::yield();
            }
            // Model-loader workers can reach VAE conversion together on first use.
            for (int repeat = 0; repeat < 20; ++repeat) {
                for (const auto& entry : cases) {
                    if (convert_tensor_name(entry.first, VERSION_COUNT) != entry.second) {
                        failed = true;
                    }
                }
            }
        });
    }
    while (ready.load() != worker_count) {
        std::this_thread::yield();
    }
    start = true;
    for (auto& worker : workers) {
        worker.join();
    }
    if (failed.load()) {
        std::fprintf(stderr, "Concurrent VAE name conversion produced an incorrect mapping\n");
        return 1;
    }
    return 0;
}
