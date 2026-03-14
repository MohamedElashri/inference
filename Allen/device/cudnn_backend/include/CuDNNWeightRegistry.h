#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <cstddef>

#ifdef ALLEN_WITH_CUDNN
#include "CuDNNCheck.h"
#endif

namespace Allen::CuDNN {

  /**
   * @brief Singleton registry mapping string keys to device-side weight tensors.
   *
   * Weights are allocated outside Allen's pool using a direct cudaMalloc.
   * They are permanent for the lifetime of the process.
   */
  class WeightRegistry {
  public:
    static WeightRegistry& instance() {
      static WeightRegistry s_instance;
      return s_instance;
    }

    void load(const std::string& key, const std::string& file_path);
    void load_from_buffer(const std::string& key, const void* host_data, size_t bytes);

    void lock_allocations() { 
      std::lock_guard<std::mutex> lock(m_mutex);
      m_locked = true; 
    }

    template<typename T>
    const T* get(const std::string& key) const {
      auto it = m_registry.find(key);
      if (it == m_registry.end()) {
        throw std::runtime_error("WeightRegistry: key not found: " + key);
      }
      return static_cast<const T*>(it->second.dev_ptr);
    }

    size_t size_bytes(const std::string& key) const {
      auto it = m_registry.find(key);
      if (it == m_registry.end()) return 0;
      return it->second.bytes;
    }

    bool contains(const std::string& key) const {
      return m_registry.find(key) != m_registry.end();
    }

    WeightRegistry(const WeightRegistry&) = delete;
    WeightRegistry& operator=(const WeightRegistry&) = delete;

  private:
    WeightRegistry() = default;

    struct Entry {
      void*  dev_ptr = nullptr;
      size_t bytes   = 0;
    };

    std::unordered_map<std::string, Entry> m_registry;
    std::mutex m_mutex;
    bool m_locked = false;
  };

} // namespace Allen::CuDNN
