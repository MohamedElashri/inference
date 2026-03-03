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
   * @brief RAII wrapper for a cuDNN convolution's four descriptors plus cached workspace size.
   */
  struct ConvDescriptors {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnTensorDescriptor_t    m_input_desc  = nullptr;
    cudnnFilterDescriptor_t    m_filter_desc = nullptr;
    cudnnConvolutionDescriptor_t m_conv_desc = nullptr;
    cudnnTensorDescriptor_t    m_output_desc = nullptr;
    // IMPLICIT_GEMM requires zero workspace (safe for large batches).
    // IMPLICIT_PRECOMP_GEMM is faster but needs O(N) workspace which overflows Allen's pool.
    cudnnConvolutionFwdAlgo_t  m_algo        = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    bool m_created = false;

    // Filter shape (fixed per algorithm)
    int m_k = 0;  // number of output channels
    int m_c = 0;  // number of input channels
    int m_r = 0;  // filter height
    int m_s = 0;  // filter width

    // Workspace cache: avoid re-querying cuDNN when shape is stable
    mutable TensorShape m_cached_input_shape {};
    mutable size_t      m_cached_workspace_bytes = 0;
    mutable bool        m_cache_valid = false;

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

    void create(
      const Handle& handle,
      std::array<int,4> filter_shape,           // {K, C, R, S}
      std::array<int,2> pad     = {0, 0},
      std::array<int,2> stride  = {1, 1},
      std::array<int,2> dilation= {1, 1},
      cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
    {
      m_k = filter_shape[0]; m_c = filter_shape[1];
      m_r = filter_shape[2]; m_s = filter_shape[3];
      m_algo = algo;

      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_input_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_filter_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_conv_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_output_desc));

      ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        m_k, m_c, m_r, m_s));

      ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_conv_desc,
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

      m_created = true;
    }

    size_t workspace_bytes(const Handle& handle, const TensorShape& input) const {
      if (m_cache_valid && input == m_cached_input_shape) {
        return m_cached_workspace_bytes;
      }

      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        input.n, input.c, input.h, input.w));

      int on, oc, oh, ow;
      ALLEN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        m_conv_desc, m_input_desc, m_filter_desc, &on, &oc, &oh, &ow));
      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));

      size_t ws = 0;
      ALLEN_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle.get(), m_input_desc, m_filter_desc,
        m_conv_desc, m_output_desc, m_algo, &ws));

      m_cached_input_shape      = input;
      m_cached_workspace_bytes  = ws;
      m_cache_valid             = true;

      return ws;
    }

    size_t output_elements() const {
      if (!m_cache_valid) return 0;
      int n, c, h, w;
      cudnnDataType_t dt;
      int ns, cs, hs, ws_stride;
      ALLEN_CUDNN_CHECK(cudnnGetTensor4dDescriptor(
        m_output_desc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws_stride));
      return static_cast<size_t>(n) * c * h * w;
    }

    void forward(
      const Handle& handle,
      const float alpha, const float beta,
      const float* dev_input,
      const float* dev_filter,
      float*       dev_output,
      void*        dev_workspace,
      size_t       workspace_bytes) const
    {
      ALLEN_CUDNN_CHECK(cudnnConvolutionForward(
        handle.get(),
        &alpha,
        m_input_desc,  dev_input,
        m_filter_desc, dev_filter,
        m_conv_desc,
        m_algo,
        dev_workspace, workspace_bytes,
        &beta,
        m_output_desc, dev_output));
    }

#else
    void create(...) {
      static_assert(Allen::CuDNN::backend_available,
        "ConvDescriptors requires CUDA+cuDNN or HIP+MIOpen. "
        "Build with -DWITH_CUDNN=ON and -DTARGET_DEVICE=CUDA.");
    }
    size_t workspace_bytes(const Handle&, const TensorShape&) const { return 0; }
    size_t output_elements() const { return 0; }
    void forward(...) const {}
#endif
  };

} // namespace Allen::CuDNN
