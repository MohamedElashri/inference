#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstddef>

namespace PVFinder {

  /**
   * @brief Singleton registry mapping string keys to device-side weight tensors.
   *
   * Weights are allocated outside Allen's pool using a direct cudaMalloc or hipMalloc
   * and persist for the lifetime of the process.
   */
  class WeightRegistry {
  public:
    static WeightRegistry& instance() {
      static WeightRegistry s_instance;
      return s_instance;
    }

    /**
     * @brief Load weights from a flat binary file into device memory.
     */
    void load(const std::string& key, const std::string& file_path);

    /**
     * @brief Load weights from a host-side buffer.
     */
    void load_from_buffer(const std::string& key, const void* host_data, size_t bytes);

    /**
     * @brief Returns typed const pointer into device memory.
     */
    template<typename T>
    const T* get(const std::string& key) const {
      for (const auto& entry : m_registry) {
        if (entry.key == key) {
          return static_cast<const T*>(entry.dev_ptr);
        }
      }
      throw std::runtime_error("WeightRegistry: key not found: " + key);
    }

    size_t size_bytes(const std::string& key) const {
      for (const auto& entry : m_registry) {
        if (entry.key == key) {
          return entry.bytes;
        }
      }
      return 0;
    }

    bool contains(const std::string& key) const {
      for (const auto& entry : m_registry) {
        if (entry.key == key) {
          return true;
        }
      }
      return false;
    }

    // Prevent accidental copying
    WeightRegistry(const WeightRegistry&) = delete;
    WeightRegistry& operator=(const WeightRegistry&) = delete;

  private:
    WeightRegistry() = default;

    struct Entry {
      std::string key;
      void*  dev_ptr = nullptr;
      size_t bytes   = 0;
    };

    std::vector<Entry> m_registry;
  };

} // namespace PVFinder
