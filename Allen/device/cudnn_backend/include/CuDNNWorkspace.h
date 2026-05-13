#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace Allen::CuDNN {

  struct Workspace {
    void* ptr = nullptr;
    size_t bytes = 0;

    bool empty() const { return ptr == nullptr || bytes == 0; }

    void require(size_t required_bytes, const char* owner) const
    {
      if (required_bytes == 0) return;
      if (ptr == nullptr || bytes < required_bytes) {
        throw std::runtime_error(
          std::string(owner) + " requires " + std::to_string(required_bytes) +
          " workspace bytes, but external workspace has " + std::to_string(bytes));
      }
    }
  };

} // namespace Allen::CuDNN
