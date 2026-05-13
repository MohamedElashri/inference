#include "CuDNNWeightRegistry.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef ALLEN_WITH_CUDNN
#include <cuda_runtime.h>
#define ALLEN_CUDNN_CUDA_CHECK(stmt)                                      \
  do {                                                                    \
    cudaError_t e = (stmt);                                               \
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); \
  } while (0)
#endif

namespace Allen::CuDNN {
  namespace {
    struct WeightEntry {
      std::string key;
      void* device_ptr = nullptr;
      size_t bytes = 0;
      DeviceWeightOwnership ownership = DeviceWeightOwnership::ProcessLifetime;
    };

    class WeightStore {
    public:
      bool contains(const std::string& key) const
      {
        const std::lock_guard<std::mutex> lock {m_mutex};
        return find_entry(key) != m_entries.end();
      }

      size_t size_bytes(const std::string& key) const
      {
        const std::lock_guard<std::mutex> lock {m_mutex};
        auto it = find_entry(key);
        return it == m_entries.end() ? 0 : it->bytes;
      }

      const void* get(const std::string& key) const
      {
        const std::lock_guard<std::mutex> lock {m_mutex};
        auto it = find_entry(key);
        if (it == m_entries.end()) {
          throw std::runtime_error("DeviceWeights: key not found: " + key);
        }
        return it->device_ptr;
      }

      void load_from_host(
        const std::string& key,
        const void* host_data,
        size_t bytes,
        size_t expected_bytes,
        DuplicateKeyPolicy duplicate_policy)
      {
        validate_input(key, host_data, bytes, expected_bytes);

        void* device_ptr = nullptr;
#ifdef ALLEN_WITH_CUDNN
        ALLEN_CUDNN_CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
        ALLEN_CUDNN_CUDA_CHECK(cudaMemcpy(device_ptr, host_data, bytes, cudaMemcpyHostToDevice));
#else
        device_ptr = std::malloc(bytes);
        if (device_ptr == nullptr && bytes != 0) {
          throw std::runtime_error("DeviceWeights: host allocation failed for key: " + key);
        }
        std::memcpy(device_ptr, host_data, bytes);
#endif

        try {
          const bool adopted =
            insert_or_update(key, device_ptr, bytes, DeviceWeightOwnership::ProcessLifetime, duplicate_policy);
          if (!adopted) {
            release_owned(device_ptr);
          }
        }
        catch (...) {
          release_owned(device_ptr);
          throw;
        }
      }

      void register_external(
        const std::string& key,
        const void* device_data,
        size_t bytes,
        size_t expected_bytes,
        DuplicateKeyPolicy duplicate_policy)
      {
        validate_input(key, device_data, bytes, expected_bytes);
        insert_or_update(
          key,
          const_cast<void*>(device_data),
          bytes,
          DeviceWeightOwnership::ExternalDevicePointer,
          duplicate_policy);
      }

    private:
      using EntryIterator = std::vector<WeightEntry>::iterator;
      using ConstEntryIterator = std::vector<WeightEntry>::const_iterator;

      ConstEntryIterator find_entry(const std::string& key) const
      {
        return std::find_if(m_entries.begin(), m_entries.end(), [&](const auto& entry) { return entry.key == key; });
      }

      EntryIterator find_entry(const std::string& key)
      {
        return std::find_if(m_entries.begin(), m_entries.end(), [&](const auto& entry) { return entry.key == key; });
      }

      void validate_input(const std::string& key, const void* data, size_t bytes, size_t expected_bytes) const
      {
        if (key.empty()) {
          throw std::runtime_error("DeviceWeights: empty key");
        }
        if (data == nullptr && bytes != 0) {
          throw std::runtime_error("DeviceWeights: null data for non-empty key: " + key);
        }
        if (expected_bytes != 0 && bytes != expected_bytes) {
          throw std::runtime_error("DeviceWeights: unexpected byte count for key: " + key);
        }
      }

      bool insert_or_update(
        const std::string& key,
        void* device_ptr,
        size_t bytes,
        DeviceWeightOwnership ownership,
        DuplicateKeyPolicy duplicate_policy)
      {
        const std::lock_guard<std::mutex> lock {m_mutex};
        auto it = find_entry(key);
        if (it != m_entries.end()) {
          if (duplicate_policy == DuplicateKeyPolicy::ReuseExisting) return false;
          if (duplicate_policy == DuplicateKeyPolicy::Reject) {
            throw std::runtime_error("DeviceWeights: key already registered: " + key);
          }
          release_if_owned(*it);
          it->device_ptr = device_ptr;
          it->bytes = bytes;
          it->ownership = ownership;
          return true;
        }
        m_entries.push_back(WeightEntry {key, device_ptr, bytes, ownership});
        return true;
      }

      static void release_owned(void* ptr)
      {
        if (ptr == nullptr) return;
#ifdef ALLEN_WITH_CUDNN
        cudaFree(ptr);
#else
        std::free(ptr);
#endif
      }

      static void release_if_owned(const WeightEntry& entry)
      {
        if (entry.ownership == DeviceWeightOwnership::ProcessLifetime) {
          release_owned(entry.device_ptr);
        }
      }

      mutable std::mutex m_mutex;
      std::vector<WeightEntry> m_entries;
    };

    WeightStore& global_weight_store()
    {
      static auto* store = new WeightStore {};
      return *store;
    }
  } // namespace

  DeviceWeights::DeviceWeights(std::string key_namespace) : m_namespace(std::move(key_namespace)) {}

  std::string DeviceWeights::full_key(const std::string& key) const
  {
    return m_namespace.empty() ? key : (m_namespace + "." + key);
  }

  bool DeviceWeights::contains(const std::string& key) const
  {
    return global_weight_store().contains(full_key(key));
  }

  size_t DeviceWeights::size_bytes(const std::string& key) const
  {
    return global_weight_store().size_bytes(full_key(key));
  }

  const void* DeviceWeights::get_raw(const std::string& key) const
  {
    return global_weight_store().get(full_key(key));
  }

  void DeviceWeights::load_file(
    const std::string& key,
    const std::string& file_path,
    size_t expected_bytes,
    DuplicateKeyPolicy duplicate_policy)
  {
    std::ifstream f(file_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
      throw std::runtime_error("DeviceWeights: cannot open " + file_path);
    }
    const size_t bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> host_buf(bytes);
    f.read(host_buf.data(), static_cast<std::streamsize>(bytes));
    load_from_buffer(key, host_buf.data(), bytes, expected_bytes, duplicate_policy);
  }

  void DeviceWeights::load_from_buffer(
    const std::string& key,
    const void* host_data,
    size_t bytes,
    size_t expected_bytes,
    DuplicateKeyPolicy duplicate_policy)
  {
    global_weight_store().load_from_host(full_key(key), host_data, bytes, expected_bytes, duplicate_policy);
  }

  void DeviceWeights::register_device_pointer(
    const std::string& key,
    const void* device_data,
    size_t bytes,
    size_t expected_bytes,
    DuplicateKeyPolicy duplicate_policy)
  {
    global_weight_store().register_external(full_key(key), device_data, bytes, expected_bytes, duplicate_policy);
  }

  WeightRegistry::WeightRegistry() : m_weights("legacy") {}

  WeightRegistry& WeightRegistry::instance()
  {
    static WeightRegistry s_instance;
    return s_instance;
  }

  void WeightRegistry::load(const std::string& key, const std::string& file_path) {
    m_weights.load_file(key, file_path);
  }

  void WeightRegistry::load_from_buffer(const std::string& key, const void* host_data, size_t bytes) {
    m_weights.load_from_buffer(key, host_data, bytes, 0, DuplicateKeyPolicy::Reject);
  }

} // namespace Allen::CuDNN
