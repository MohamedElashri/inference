#include "CuDNNWeightRegistry.h"

#include <cstring>
#include <cstdlib>
#include <fstream>
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

  struct DeviceWeights::State {
    std::string key_namespace;
  };

  namespace {
    struct RegistryEntry {
      char* key = nullptr;
      void* dev_ptr = nullptr;
      size_t bytes = 0;
    };

    struct GlobalRegistry {
      RegistryEntry* entries = nullptr;
      size_t entries_size = 0;
      size_t entries_capacity = 0;
    };

    GlobalRegistry& global_registry()
    {
      static auto* registry = new GlobalRegistry {};
      return *registry;
    }

    char* copy_c_string(const std::string& value)
    {
      auto* copy = static_cast<char*>(std::malloc(value.size() + 1));
      if (copy == nullptr) {
        throw std::runtime_error("DeviceWeights: host key allocation failed");
      }
      std::memcpy(copy, value.c_str(), value.size() + 1);
      return copy;
    }

    void append_entry(const std::string& key, void* dev_ptr, size_t bytes)
    {
      auto& registry = global_registry();
      if (registry.entries_size == registry.entries_capacity) {
        const size_t new_capacity = registry.entries_capacity == 0 ? 32 : registry.entries_capacity * 2;
        auto* new_entries = static_cast<RegistryEntry*>(
          std::realloc(registry.entries, new_capacity * sizeof(RegistryEntry)));
        if (new_entries == nullptr) {
          throw std::runtime_error("DeviceWeights: host registry allocation failed");
        }
        registry.entries = new_entries;
        registry.entries_capacity = new_capacity;
      }
      registry.entries[registry.entries_size++] = RegistryEntry {copy_c_string(key), dev_ptr, bytes};
    }
  } // namespace

  DeviceWeights::DeviceWeights(std::string key_namespace)
  {
    state().key_namespace = std::move(key_namespace);
  }

  DeviceWeights::State& DeviceWeights::state()
  {
    if (m_state == nullptr) {
      m_state = new State {};
    }
    return *m_state;
  }

  const DeviceWeights::State* DeviceWeights::state_if_created() const
  {
    return m_state;
  }

  std::string DeviceWeights::full_key(const std::string& key) const
  {
    const auto* st = state_if_created();
    if (st == nullptr || st->key_namespace.empty()) return key;
    return st->key_namespace + "." + key;
  }

  bool DeviceWeights::contains(const std::string& key) const
  {
    auto& registry = global_registry();
    const auto full = full_key(key);
    for (size_t i = 0; i < registry.entries_size; ++i) {
      if (std::strcmp(registry.entries[i].key, full.c_str()) == 0) return true;
    }
    return false;
  }

  size_t DeviceWeights::size_bytes(const std::string& key) const
  {
    auto& registry = global_registry();
    const auto full = full_key(key);
    for (size_t i = 0; i < registry.entries_size; ++i) {
      if (std::strcmp(registry.entries[i].key, full.c_str()) == 0) return registry.entries[i].bytes;
    }
    return 0;
  }

  const void* DeviceWeights::get_raw(const std::string& key) const
  {
    auto& registry = global_registry();
    const auto full = full_key(key);
    for (size_t i = 0; i < registry.entries_size; ++i) {
      if (std::strcmp(registry.entries[i].key, full.c_str()) == 0) return registry.entries[i].dev_ptr;
    }
    throw std::runtime_error("DeviceWeights: key not found: " + full);
  }

  WeightRegistry::WeightRegistry() : m_weights("legacy") {}

  WeightRegistry& WeightRegistry::instance()
  {
    static WeightRegistry s_instance;
    return s_instance;
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
    if (host_data == nullptr && bytes != 0) {
      throw std::runtime_error("DeviceWeights: null host data for non-empty key: " + full_key(key));
    }
    if (expected_bytes != 0 && bytes != expected_bytes) {
      throw std::runtime_error("DeviceWeights: unexpected byte count for key: " + full_key(key));
    }

    (void) state();
    auto& registry = global_registry();
    const auto full = full_key(key);
    for (size_t i = 0; i < registry.entries_size; ++i) {
      if (std::strcmp(registry.entries[i].key, full.c_str()) == 0) {
        if (duplicate_policy == DuplicateKeyPolicy::ReuseExisting) return;
        throw std::runtime_error("DeviceWeights: key already registered: " + full);
      }
    }

#ifdef ALLEN_WITH_CUDNN
    void* dev_ptr = nullptr;
    ALLEN_CUDNN_CUDA_CHECK(cudaMalloc(&dev_ptr, bytes));
    ALLEN_CUDNN_CUDA_CHECK(cudaMemcpy(dev_ptr, host_data, bytes, cudaMemcpyHostToDevice));
    append_entry(full, dev_ptr, bytes);
#else
    void* host_copy = std::malloc(bytes);
    std::memcpy(host_copy, host_data, bytes);
    append_entry(full, host_copy, bytes);
#endif
  }

  void WeightRegistry::load(const std::string& key, const std::string& file_path) {
    m_weights.load_file(key, file_path);
  }

  void WeightRegistry::load_from_buffer(const std::string& key, const void* host_data, size_t bytes) {
    m_weights.load_from_buffer(key, host_data, bytes, 0, DuplicateKeyPolicy::Reject);
  }

} // namespace Allen::CuDNN
