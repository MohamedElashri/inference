#pragma once
#include "CuDNNCheck.h"
#include "CuDNNHandle.h"
#include <cstddef>
#include <array>

namespace Allen::CuDNN {

  // Shape of a 4-D tensor: {N, C, H, W}
  struct TensorShape {
    int n = 0, c = 0, h = 0, w = 0;
    bool operator==(const TensorShape& o) const {
      return n == o.n && c == o.c && h == o.h && w == o.w;
    }
    bool operator!=(const TensorShape& o) const { return !(*this == o); }
  };

  /**
   * @brief RAII wrapper for a cuDNN convolution's four descriptors.
   *
   * Design constraints (Allen compatibility):
   *  - Algorithm is ALWAYS IMPLICIT_GEMM — zero workspace, no internal cuDNN allocations.
   *  - Tensor descriptors are set once at create() time with the fixed input shape.
   *    Our UNet shapes are compile-time constants so this is safe.
   *  - workspace_bytes() always returns 0 — callers pass nullptr/0 to cuDNN.
   *  - forward() accepts a raw cudnnHandle_t so callers can use thread_local handles
   *    without wrapping them in a Handle object.
   *  - No per-instance mutable workspace cache — eliminates the thread-safety issue
   *    where concurrent calls on the same ConvDescriptors object mutated shared state.
   */
  struct ConvDescriptors {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnTensorDescriptor_t      m_input_desc  = nullptr;
    cudnnFilterDescriptor_t      m_filter_desc = nullptr;
    cudnnConvolutionDescriptor_t m_conv_desc   = nullptr;
    cudnnTensorDescriptor_t      m_output_desc = nullptr;
    bool m_created = false;

  public:
    ConvDescriptors() = default;

    ~ConvDescriptors() {
      if (!m_created) return;
      cudnnDestroyTensorDescriptor(m_input_desc);
      cudnnDestroyFilterDescriptor(m_filter_desc);
      cudnnDestroyConvolutionDescriptor(m_conv_desc);
      cudnnDestroyTensorDescriptor(m_output_desc);
    }

    ConvDescriptors(const ConvDescriptors&) = delete;
    ConvDescriptors& operator=(const ConvDescriptors&) = delete;

    // Create descriptors with fixed input shape.
    // input_shape: {N, C_in, H=1, W} — fixed for the lifetime of this object.
    // filter_shape: {K=C_out, C_in, R=1, S=kernel_w}
    // Pins algorithm to IMPLICIT_GEMM (zero workspace, no internal cuDNN allocation).
    void create(
      std::array<int,4> input_shape,            // {N, C_in, H, W} — fixed
      std::array<int,4> filter_shape,           // {K, C_in, R, S}
      std::array<int,2> pad      = {0, 0},
      std::array<int,2> stride   = {1, 1},
      std::array<int,2> dilation = {1, 1})
    {
      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_input_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_filter_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_conv_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_output_desc));

      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]));

      ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]));

      ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_conv_desc,
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

      // Derive and store output descriptor — fixed for this shape.
      int on, oc, oh, ow;
      ALLEN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        m_conv_desc, m_input_desc, m_filter_desc, &on, &oc, &oh, &ow));
      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));

      m_created = true;
    }

    // Always returns 0: IMPLICIT_GEMM requires no workspace.
    // Kept for API compatibility with any existing callers.
    size_t workspace_bytes(const Handle&, const TensorShape&) const { return 0; }

    // Forward convolution using a thread_local cudnnHandle_t.
    // workspace/ws_bytes should be nullptr/0 — IMPLICIT_GEMM does not use them.
    void forward(
      cudnnHandle_t  handle,
      const float    alpha, const float beta,
      const float*   dev_input,
      const float*   dev_filter,
      float*         dev_output,
      void*          dev_workspace = nullptr,
      size_t         workspace_bytes = 0) const
    {
      ALLEN_CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        m_input_desc,  dev_input,
        m_filter_desc, dev_filter,
        m_conv_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        dev_workspace, workspace_bytes,
        &beta,
        m_output_desc, dev_output));
    }

    // Legacy overload: accept a Handle wrapper (for backward compat).
    void forward(
      const Handle&  handle,
      const float    alpha, const float beta,
      const float*   dev_input,
      const float*   dev_filter,
      float*         dev_output,
      void*          dev_workspace = nullptr,
      size_t         workspace_bytes = 0) const
    {
      forward(handle.get(), alpha, beta, dev_input, dev_filter,
              dev_output, dev_workspace, workspace_bytes);
    }

#else
    void create(std::array<int,4>, std::array<int,4>,
                std::array<int,2> = {0,0}, std::array<int,2> = {1,1},
                std::array<int,2> = {1,1}) {}
    size_t workspace_bytes(const Handle&, const TensorShape&) const { return 0; }
    void forward(cudnnHandle_t, float, float,
                 const float*, const float*, float*,
                 void* = nullptr, size_t = 0) const {}
    void forward(const Handle&, float, float,
                 const float*, const float*, float*,
                 void* = nullptr, size_t = 0) const {}
#endif
  };

} // namespace Allen::CuDNN
