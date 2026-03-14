#include "CuDNNWeightRegistry.h"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstring>

#ifdef ALLEN_WITH_CUDNN
#include "CuDNNCheck.h"
#include <cuda_runtime.h>
#define CUDACHECK(stmt) { cudaError_t e=(stmt); \
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
#endif

namespace Allen::CuDNN {

  void WeightRegistry::load(const std::string& key, const std::string& file_path) {
    std::ifstream f(file_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
      throw std::runtime_error("WeightRegistry: cannot open " + file_path);
    }
    const size_t bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> host_buf(bytes);
    f.read(host_buf.data(), static_cast<std::streamsize>(bytes));
    load_from_buffer(key, host_buf.data(), bytes);
  }

  void WeightRegistry::load_from_buffer(const std::string& key,
                                         const void* host_data,
                                         size_t bytes) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_locked) {
      throw std::runtime_error("WeightRegistry: attempted to load weights after initialization phase: " + key);
    }
    
    if (m_registry.count(key)) {
      throw std::runtime_error("WeightRegistry: key already registered: " + key);
    }

#ifdef ALLEN_WITH_CUDNN
    void* dev_ptr = nullptr;
    CUDACHECK(cudaMalloc(&dev_ptr, bytes));
    CUDACHECK(cudaMemcpy(dev_ptr, host_data, bytes, cudaMemcpyHostToDevice));
    m_registry[key] = Entry{dev_ptr, bytes};
#else
    // CPU/HIP stub: store a copy on the host
    void* host_copy = std::malloc(bytes);
    std::memcpy(host_copy, host_data, bytes);
    m_registry[key] = Entry{host_copy, bytes};
#endif
  }

} // namespace Allen::CuDNN
