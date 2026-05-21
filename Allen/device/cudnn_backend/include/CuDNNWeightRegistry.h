#pragma once
#include "CuDNNDeviceWeights.h"
#include <string>
#include <cstddef>

#ifdef ALLEN_WITH_CUDNN
#include "CuDNNCheck.h"
#endif

namespace Allen::CuDNN {

  /**
   * @brief Singleton registry mapping string keys to device-side weight tensors.
   *
   * Legacy-only compatibility facade. New clients should use a namespaced
   * DeviceWeights instance directly so ownership and duplicate-key policy are
   * explicit at the call site.
   *
   * Weights are allocated outside Allen's pool using a direct cudaMalloc.
   * They are permanent for the lifetime of the process.
   */
  class WeightRegistry {
  public:
    static WeightRegistry& instance();

    void load(const std::string& key, const std::string& file_path);
    void load_from_buffer(const std::string& key, const void* host_data, size_t bytes);

    template<typename T>
    const T* get(const std::string& key) const {
      return m_weights.get<T>(key);
    }

    size_t size_bytes(const std::string& key) const {
      return m_weights.size_bytes(key);
    }

    bool contains(const std::string& key) const {
      return m_weights.contains(key);
    }

    WeightRegistry(const WeightRegistry&) = delete;
    WeightRegistry& operator=(const WeightRegistry&) = delete;

  private:
    WeightRegistry();

    DeviceWeights m_weights;
  };

} // namespace Allen::CuDNN
