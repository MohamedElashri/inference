#pragma once
#include "CuDNNCheck.h"
#include "CuDNNBackendShim.h"
#include <mutex>
#include <unordered_map>

namespace Allen::CuDNN {

  /**
   * @brief RAII wrapper around cudnnHandle_t.
   *
   * Legacy per-instance handle — kept for backward compatibility.
   * Prefer get_thread_local_handle() for new code.
   */
  struct Handle {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnHandle_t m_h = nullptr;
    bool m_created = false;

  public:
    Handle() = default;
    // NOTE: ~Handle() intentionally does NOT call cudnnDestroy.
    // Allen calls cudaDeviceReset() at shutdown which implicitly destroys all
    // cuDNN handles. Calling cudnnDestroy after cudaDeviceReset crashes.
    ~Handle() = default;

    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&&) = delete;
    Handle& operator=(Handle&&) = delete;

    void create() {
      ALLEN_CUDNN_CHECK(cudnnCreate(&m_h));
      m_created = true;
    }

    void wrap(cudnnHandle_t h) {
      m_h = h;
      m_created = false;
    }

    void destroy() {
      if (m_created) {
        cudnnDestroy(m_h);
        m_h = nullptr;
        m_created = false;
      }
    }

    void set_stream(cudaStream_t stream) const {
      ALLEN_CUDNN_CHECK(cudnnSetStream(m_h, stream));
    }

    cudnnHandle_t get() const { return m_h; }
    bool created() const { return m_created; }

#else
    void create() {}
    void set_stream(void*) const {}
    void* get() const { return nullptr; }
    bool created() const { return false; }
    void wrap(void*) {}
    void destroy() {}
#endif
  };

#ifdef ALLEN_CUDNN_BACKEND_CUDA
  /**
   * @brief Singleton resource manager for cudnnHandle_t based on cudaStream_t.
   * 
   * Provides 1:1 mapping of cuDNN handles to Allen CUDA streams. This ensures resources 
   * correspond to Allen's thread lifecycle rather than arbitrary OS thread boundaries.
   */
  class CuDNNManager {
  public:
    static CuDNNManager& instance() {
      static CuDNNManager s_instance;
      return s_instance;
    }

    cudnnHandle_t get_handle(cudaStream_t stream) {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto it = m_handles.find(stream);
      if (it == m_handles.end()) {
        cudnnHandle_t h;
        ALLEN_CUDNN_CHECK(cudnnCreate(&h));
        ALLEN_CUDNN_CHECK(cudnnSetStream(h, stream));
        m_handles[stream] = h;
        return h;
      }
      // Ensure the handle continues to recognize this stream
      ALLEN_CUDNN_CHECK(cudnnSetStream(it->second, stream));
      return it->second;
    }

  private:
    CuDNNManager() = default;
    ~CuDNNManager() = default;
    
    CuDNNManager(const CuDNNManager&) = delete;
    CuDNNManager& operator=(const CuDNNManager&) = delete;

    std::mutex m_mutex;
    std::unordered_map<cudaStream_t, cudnnHandle_t> m_handles;
  };

  /**
   * @brief Helper to get the stream-associated handle.
   */
  inline cudnnHandle_t get_thread_local_handle(cudaStream_t stream) {
    return CuDNNManager::instance().get_handle(stream);
  }
#else
  inline void* get_thread_local_handle(void*) { return nullptr; }
#endif

} // namespace Allen::CuDNN
