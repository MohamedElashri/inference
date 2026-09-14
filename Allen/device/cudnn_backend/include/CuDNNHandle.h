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
   * @brief Process-wide map from Allen CUDA stream to its cudnnHandle_t.
   *
   * Handles follow Allen's streams (one per -t slot) rather than whatever OS
   * thread happens to run them, and are shared by every cuDNN-using algorithm
   * on that stream. Created lazily, never destroyed (process lifetime).
   */
  class HandleManager {
  public:
    static HandleManager& instance() {
      static HandleManager s_instance;
      return s_instance;
    }

    cudnnHandle_t handle_for(cudaStream_t stream) {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto [it, inserted] = m_handles.try_emplace(stream, nullptr);
      if (inserted) ALLEN_CUDNN_CHECK(cudnnCreate(&it->second));
      return it->second;
    }

    HandleManager(const HandleManager&) = delete;
    HandleManager& operator=(const HandleManager&) = delete;

  private:
    HandleManager() = default;
    std::mutex m_mutex;
    std::unordered_map<cudaStream_t, cudnnHandle_t> m_handles;
  };

  /**
   * @brief Return the cudnnHandle_t of the given CUDA stream, bound to it.
   *
   * The per-thread cache makes the steady state lock-free: an Allen stream is
   * served by one thread, so the map is only consulted on a thread's first call
   * (or if it is handed a different stream).
   *
   * Usage in operator() const:
   *   cudnnHandle_t h = Allen::CuDNN::get_thread_local_handle(context.stream());
   *   desc.forward(h, ...);
   */
  inline cudnnHandle_t get_thread_local_handle(cudaStream_t stream) {
    thread_local cudaStream_t tl_stream = nullptr;
    thread_local cudnnHandle_t tl_handle = nullptr;
    if (tl_handle == nullptr || tl_stream != stream) {
      tl_handle = HandleManager::instance().handle_for(stream);
      tl_stream = stream;
    }
    ALLEN_CUDNN_CHECK(cudnnSetStream(tl_handle, stream));
    return tl_handle;
  }
#else
  inline void* get_thread_local_handle(void*) { return nullptr; }
#endif

} // namespace Allen::CuDNN
