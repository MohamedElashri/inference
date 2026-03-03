#pragma once
#include "CuDNNCheck.h"
#include "CuDNNBackendShim.h"

namespace Allen::CuDNN {

  /**
   * @brief RAII wrapper around cudnnHandle_t.
   *
   * Declare as: mutable Allen::CuDNN::Handle m_handle;
   * Initialize in init(): m_handle.create();
   * Use in operator() const: m_handle.set_stream(context.stream());
   *
   * One handle per algorithm instance → one per Allen::Stream → thread-safe.
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

    // Non-copyable, non-movable: tied to algorithm instance
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&&) = delete;
    Handle& operator=(Handle&&) = delete;

    void create() {
      ALLEN_CUDNN_CHECK(cudnnCreate(&m_h));
      m_created = true;
    }

    // Non-owning wrapper: adopt an externally managed cudnnHandle_t.
    // The wrapped handle is NOT destroyed in ~Handle(); the caller owns it.
    void wrap(cudnnHandle_t h) {
      m_h = h;
      m_created = false;  // false → destructor is a no-op
    }

    // Explicit early destroy, call before CUDA context teardown.
    // After this, ~Handle() is a no-op.
    void destroy() {
      if (m_created) {
        cudnnDestroy(m_h);
        m_h = nullptr;
        m_created = false;
      }
    }

    // Called at the start of operator() const — routes cuDNN work to Allen's stream
    void set_stream(cudaStream_t stream) const {
      ALLEN_CUDNN_CHECK(cudnnSetStream(m_h, stream));
    }

    cudnnHandle_t get() const { return m_h; }
    bool created() const { return m_created; }

#else
    // HIP stub / CPU no-op: handle is present in the struct but does nothing.
    // Replace this block with MIOpen equivalents when ready.
    void create() {}
    void set_stream(void*) const {}
    void* get() const { return nullptr; }
    bool created() const { return false; }
#endif
  };

} // namespace Allen::CuDNN
