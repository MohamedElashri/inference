#pragma once
#include "CuDNNCheck.h"
#include "CuDNNBackendShim.h"

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
   * @brief Return a cudnnHandle_t bound to the given CUDA stream.
   *
   * One handle is created lazily per OS thread on first call, then reused.
   * cudnnSetStream is called each time to route work to the correct stream.
   *
   * This replaces the pattern of storing mutable Handle m_handle in each
   * algorithm instance (which caused one handle per Allen thread to be created
   * at startup, spiking GPU memory). With thread_local the handles are created
   * on demand and there is at most one per OS thread.
   *
   * Usage in operator() const:
   *   cudnnHandle_t h = Allen::CuDNN::get_thread_local_handle(context.stream());
   *   desc.forward(h, ...);
   */
  inline cudnnHandle_t get_thread_local_handle(cudaStream_t stream) {
    thread_local cudnnHandle_t tl_handle = nullptr;
    if (tl_handle == nullptr) {
      ALLEN_CUDNN_CHECK(cudnnCreate(&tl_handle));
    }
    ALLEN_CUDNN_CHECK(cudnnSetStream(tl_handle, stream));
    return tl_handle;
  }

  /**
   * @brief Default Allen cuDNN handle provider.
   *
   * Handles are thread-local infrastructure, created lazily, rebound to the
   * caller's stream on every request, and intentionally left alive until process
   * shutdown. Allen may reset the CUDA device during teardown, so destroying
   * thread-local cuDNN handles from C++ static destructors is not reliable.
   */
  struct HandleProvider {
    static cudnnHandle_t get(cudaStream_t stream) { return get_thread_local_handle(stream); }
  };
#else
  inline void* get_thread_local_handle(void*) { return nullptr; }
  struct HandleProvider {
    static void* get(void*) { return nullptr; }
  };
#endif

} // namespace Allen::CuDNN
