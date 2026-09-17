#ifndef __SD_CORE_DEBUG_TRACE_HPP__
#define __SD_CORE_DEBUG_TRACE_HPP__

#include <string>

#include "core/tensor.hpp"

namespace sd_debug_trace {

void initialize();
bool enabled(const char* component, const std::string& name);
void tensor(const char* component, const std::string& name, int step, const sd::Tensor<float>& value);
void tensor(const char* component, const std::string& name, int step, const sd::Tensor<int32_t>& value);
void scalar(const char* component, const std::string& name, int step, double value);

}  // namespace sd_debug_trace

#endif
