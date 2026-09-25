#include "debug_trace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "json.hpp"
#include "core/tensor_ggml.hpp"
#include "core/util.h"
#include "stable-diffusion.h"

namespace sd_debug_trace {
namespace {

class Sha256 {
public:
    void update(const uint8_t* data, size_t size) {
        bit_count_ += static_cast<uint64_t>(size) * 8;
        while (size > 0) {
            const size_t take = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, take);
            block_size_ += take;
            data += take;
            size -= take;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::string finish() {
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + block_size_, block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + block_size_, block_.begin() + 56, 0);
        for (int i = 0; i < 8; ++i) {
            block_[63 - i] = static_cast<uint8_t>(bit_count_ >> (8 * i));
        }
        transform(block_.data());
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (uint32_t word : state_) {
            out << std::setw(8) << word;
        }
        return out.str();
    }

private:
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

    void transform(const uint8_t* data) {
        static constexpr uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(data[4 * i]) << 24) | (uint32_t(data[4 * i + 1]) << 16) |
                   (uint32_t(data[4 * i + 2]) << 8) | uint32_t(data[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    size_t block_size_ = 0;
    uint64_t bit_count_ = 0;
};

struct Config {
    std::filesystem::path dir;
    std::string level;
    std::set<std::string> components;
    std::vector<std::string> filters;
    bool full_tensor = false;
    uint64_t sequence = 0;
    std::mutex mutex;

    Config() {
        const char* dir_env = std::getenv("SD_TRACE_DIR");
        if (dir_env == nullptr || *dir_env == '\0') return;
        dir = dir_env;
        level = env("SD_TRACE_LEVEL", "summary");
        full_tensor = env("SD_TRACE_FULL_TENSOR", "0") == "1";
        split(env("SD_TRACE_COMPONENT", "all"), components);
        std::set<std::string> filter_set;
        split(env("SD_TRACE_FILTER", ""), filter_set);
        filters.assign(filter_set.begin(), filter_set.end());
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        nlohmann::ordered_json manifest;
        manifest["schema_version"] = 1;
        manifest["source_state"] = "runtime_unverified";
        manifest["commit_sha"] = sd_commit();
        manifest["trace_level"] = level;
        manifest["trace_component"] = env("SD_TRACE_COMPONENT", "all");
        manifest["trace_filter"] = env("SD_TRACE_FILTER", "");
        manifest["full_tensor"] = full_tensor;
        std::ofstream(dir / "trace-manifest.json") << manifest.dump(2) << '\n';
    }

    static std::string env(const char* name, const char* fallback) {
        const char* value = std::getenv(name);
        return value == nullptr ? fallback : value;
    }

    static void split(const std::string& value, std::set<std::string>& out) {
        size_t start = 0;
        while (start <= value.size()) {
            size_t end = value.find(',', start);
            std::string item = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!item.empty()) out.insert(item);
            if (end == std::string::npos) break;
            start = end + 1;
        }
    }
};

Config& config() {
    static Config value;
    return value;
}

bool selected(const Config& cfg, const char* component, const std::string& name);

bool graph_eval_callback(ggml_tensor* value, bool ask, void*) {
    if (value == nullptr || value->type != GGML_TYPE_F32 || value->name[0] == '\0') return false;
    const std::string name(value->name);
    if (name.find("trace.convrot.") == std::string::npos || name.find(" (") != std::string::npos ||
        !selected(config(), "graph", name)) return false;
    if (!ask) {
        tensor("graph", name, -1, sd::make_sd_tensor_from_ggml<float>(value));
    }
    return true;
}

bool selected(const Config& cfg, const char* component, const std::string& name) {
    if (cfg.dir.empty()) return false;
    if (cfg.components.count("all") == 0 && cfg.components.count(component) == 0) return false;
    if (cfg.filters.empty()) return true;
    for (const std::string& filter : cfg.filters) {
        if (name.find(filter) != std::string::npos) return true;
    }
    return false;
}

std::string safe_name(std::string name) {
    for (char& ch : name) {
        if (!(ch >= 'a' && ch <= 'z') && !(ch >= 'A' && ch <= 'Z') && !(ch >= '0' && ch <= '9') && ch != '-' && ch != '_') ch = '_';
    }
    return name;
}

void add_le_float(Sha256& hash, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint8_t bytes[4] = {static_cast<uint8_t>(bits), static_cast<uint8_t>(bits >> 8),
                        static_cast<uint8_t>(bits >> 16), static_cast<uint8_t>(bits >> 24)};
    hash.update(bytes, sizeof(bytes));
}

template <typename T>
void write_tensor(const char* component, const std::string& name, int step, const sd::Tensor<T>& value) {
    Config& cfg = config();
    if (!selected(cfg, component, name) || value.empty()) return;
    std::lock_guard<std::mutex> lock(cfg.mutex);
    nlohmann::ordered_json record;
    record["schema_version"] = 1;
    record["sequence"] = cfg.sequence;
    record["component"] = component;
    record["name"] = name;
    record["step"] = step;
    record["dtype"] = "f32";
    record["dimensions"] = value.shape();
    record["element_count"] = value.numel();
    record["byte_count"] = value.numel() * sizeof(float);

    size_t nan_count = 0, inf_count = 0;
    double sum = 0.0, sum_sq = 0.0, l2_sq = 0.0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    Sha256 hash;
    std::vector<float> canonical(value.numel());
    for (size_t i = 0; i < canonical.size(); ++i) {
        const float v = static_cast<float>(value.data()[i]);
        canonical[i] = v;
        add_le_float(hash, v);
        if (std::isnan(v)) { ++nan_count; continue; }
        if (std::isinf(v)) { ++inf_count; continue; }
        minimum = std::min(minimum, v);
        maximum = std::max(maximum, v);
        sum += v;
        sum_sq += double(v) * v;
        l2_sq += double(v) * v;
    }
    const size_t finite_count = canonical.size() - nan_count - inf_count;
    const double mean = finite_count == 0 ? 0.0 : sum / finite_count;
    const double variance = finite_count == 0 ? 0.0 : std::max(0.0, sum_sq / finite_count - mean * mean);
    record["nan_count"] = nan_count;
    record["inf_count"] = inf_count;
    record["min"] = finite_count == 0 ? 0.0 : minimum;
    record["max"] = finite_count == 0 ? 0.0 : maximum;
    record["mean"] = mean;
    record["stddev"] = std::sqrt(variance);
    record["l2_norm"] = std::sqrt(l2_sq);
    record["sha256_canonical_le_f32"] = hash.finish();

    std::set<size_t> indices;
    for (size_t i = 0; i < std::min<size_t>(64, canonical.size()); ++i) indices.insert(i);
    const size_t tail = std::min<size_t>(64, canonical.size());
    for (size_t i = canonical.size() - tail; i < canonical.size(); ++i) indices.insert(i);
    uint64_t random = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < 64 && !canonical.empty(); ++i) {
        random = random * 6364136223846793005ULL + 1442695040888963407ULL;
        indices.insert(static_cast<size_t>(random % canonical.size()));
    }
    nlohmann::ordered_json samples = nlohmann::ordered_json::array();
    for (size_t index : indices) samples.push_back({{"index", index}, {"value", canonical[index]}});
    record["samples"] = std::move(samples);

    if (cfg.full_tensor) {
        std::ostringstream filename;
        filename << std::setw(6) << std::setfill('0') << cfg.sequence << '-' << safe_name(name) << ".f32le";
        std::ofstream dump(cfg.dir / filename.str(), std::ios::binary);
        for (float v : canonical) {
            uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            const uint8_t bytes[4] = {static_cast<uint8_t>(bits), static_cast<uint8_t>(bits >> 8),
                                      static_cast<uint8_t>(bits >> 16), static_cast<uint8_t>(bits >> 24)};
            dump.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
        }
        record["full_tensor_file"] = filename.str();
    }
    std::ofstream(cfg.dir / "tensor-summaries.jsonl", std::ios::app) << record.dump() << '\n';
    ++cfg.sequence;
}

}  // namespace

void initialize() {
    Config& cfg = config();
    if (!cfg.dir.empty()) sd_set_backend_eval_callback(graph_eval_callback, nullptr);
}

bool enabled(const char* component, const std::string& name) {
    return selected(config(), component, name);
}

void tensor(const char* component, const std::string& name, int step, const sd::Tensor<float>& value) {
    write_tensor(component, name, step, value);
}

void tensor(const char* component, const std::string& name, int step, const sd::Tensor<int32_t>& value) {
    write_tensor(component, name, step, value);
}

void scalar(const char* component, const std::string& name, int step, double value) {
    Config& cfg = config();
    if (!selected(cfg, component, name)) return;
    std::lock_guard<std::mutex> lock(cfg.mutex);
    nlohmann::ordered_json record = {{"schema_version", 1}, {"sequence", cfg.sequence++},
                                     {"component", component}, {"name", name}, {"step", step}, {"value", value}};
    std::ofstream(cfg.dir / "tensor-summaries.jsonl", std::ios::app) << record.dump() << '\n';
}

}  // namespace sd_debug_trace
