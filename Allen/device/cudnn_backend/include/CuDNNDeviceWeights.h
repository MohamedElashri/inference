#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace Allen::CuDNN {

  enum class DuplicateKeyPolicy {
    Reject,
    ReuseExisting,
    ReplaceExisting
  };

  enum class DeviceWeightOwnership {
    ProcessLifetime,
    ExternalDevicePointer
  };

  class DeviceWeights {
  public:
    explicit DeviceWeights(std::string key_namespace = {});

    void load_file(
      const std::string& key,
      const std::string& file_path,
      size_t expected_bytes = 0,
      DuplicateKeyPolicy duplicate_policy = DuplicateKeyPolicy::Reject);

    void load_from_buffer(
      const std::string& key,
      const void* host_data,
      size_t bytes,
      size_t expected_bytes = 0,
      DuplicateKeyPolicy duplicate_policy = DuplicateKeyPolicy::Reject);

    void register_device_pointer(
      const std::string& key,
      const void* device_data,
      size_t bytes,
      size_t expected_bytes = 0,
      DuplicateKeyPolicy duplicate_policy = DuplicateKeyPolicy::Reject);

    template<typename T>
    const T* get(const std::string& key) const {
      if (size_bytes(key) % sizeof(T) != 0) {
        throw std::runtime_error("DeviceWeights: key size is not aligned to requested type: " + full_key(key));
      }
      return static_cast<const T*>(get_raw(key));
    }

    size_t size_bytes(const std::string& key) const;
    bool contains(const std::string& key) const;
    std::string full_key(const std::string& key) const;
    const std::string& key_namespace() const { return m_namespace; }

    DeviceWeights(const DeviceWeights&) = delete;
    DeviceWeights& operator=(const DeviceWeights&) = delete;

  private:
    const void* get_raw(const std::string& key) const;

    std::string m_namespace;
  };

} // namespace Allen::CuDNN
