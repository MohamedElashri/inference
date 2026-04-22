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
   *  - Algorithm is selected at create() time via cudnnGetConvolutionForwardAlgorithm_v7
   *    with a 64 MB workspace budget. Falls back to IMPLICIT_GEMM if no result fits.
   *  - Workspace is cudaMalloc'd once at create() and freed in the destructor.
   *    It is shared across all concurrent calls on the same descriptor because it is
   *    read-only during forward() — cuDNN uses it as scratch, not for inter-call state.
   *  - Tensor descriptors are set once at create() time with the fixed input shape.
   *    Our UNet shapes are compile-time constants so this is safe.
   *  - forward() accepts a raw cudnnHandle_t so callers can use thread_local handles.
   */
  struct ConvDescriptors {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnTensorDescriptor_t      m_input_desc   = nullptr;
    cudnnFilterDescriptor_t      m_filter_desc  = nullptr;
    cudnnConvolutionDescriptor_t m_conv_desc    = nullptr;
    cudnnTensorDescriptor_t      m_output_desc  = nullptr;
    cudnnConvolutionFwdAlgo_t    m_algo         = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    void*                        m_workspace    = nullptr;
    size_t                       m_ws_bytes     = 0;
    bool m_created = false;

  public:
    ConvDescriptors() = default;

    ~ConvDescriptors() {
      if (!m_created) return;
      cudnnDestroyTensorDescriptor(m_input_desc);
      cudnnDestroyFilterDescriptor(m_filter_desc);
      cudnnDestroyConvolutionDescriptor(m_conv_desc);
      cudnnDestroyTensorDescriptor(m_output_desc);
      if (m_workspace) cudaFree(m_workspace);
    }

    ConvDescriptors(const ConvDescriptors&) = delete;
    ConvDescriptors& operator=(const ConvDescriptors&) = delete;

    // Create descriptors with fixed input shape. Selects the fastest algorithm
    // within a 64 MB workspace budget via cudnnGetConvolutionForwardAlgorithm_v7.
    // Falls back to IMPLICIT_GEMM (zero workspace) if the query fails or returns
    // no suitable algorithm.
    void create(
      cudnnHandle_t     handle,
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
      // CUDNN_TENSOR_OP_MATH: no-op on SM < 8.0; enables TF32 Tensor Core path on Ampere+.
      ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(m_conv_desc, CUDNN_TENSOR_OP_MATH));

      // Derive and store output descriptor — fixed for this shape.
      int on, oc, oh, ow;
      ALLEN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        m_conv_desc, m_input_desc, m_filter_desc, &on, &oc, &oh, &ow));
      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));

      m_created = true;

      // Select fastest algorithm by benchmarking all candidates on the actual
      // hardware (Find variant). One-time startup cost; returns measured timings
      // rather than heuristic lookup. Allocate kBudget as search workspace, run
      // Find, record the winner, then free the search buffer and reallocate only
      // the winning workspace size.
      static constexpr size_t kBudget  = 64ul * 1024 * 1024;
      static constexpr int    kMaxAlgos = 8;

      size_t in_elems   = (size_t)input_shape[0]  * input_shape[1]  * input_shape[2]  * input_shape[3];
      size_t filt_elems = (size_t)filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3];
      size_t out_elems  = (size_t)on * oc * oh * ow;

      float *tmp_in = nullptr, *tmp_filt = nullptr, *tmp_out = nullptr;
      void  *search_ws = nullptr;
      cudaMalloc(&tmp_in,    in_elems   * sizeof(float));
      cudaMalloc(&tmp_filt,  filt_elems * sizeof(float));
      cudaMalloc(&tmp_out,   out_elems  * sizeof(float));
      cudaMalloc(&search_ws, kBudget);

      int returned = 0;
      cudnnConvolutionFwdAlgoPerf_t perf[kMaxAlgos];
      if (cudnnFindConvolutionForwardAlgorithmEx(
              handle,
              m_input_desc,  tmp_in,
              m_filter_desc, tmp_filt,
              m_conv_desc,
              m_output_desc, tmp_out,
              kMaxAlgos, &returned, perf,
              search_ws, kBudget) == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < returned; ++i) {
          if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= kBudget) {
            m_algo     = perf[i].algo;
            m_ws_bytes = perf[i].memory;
            break;
          }
        }
      }

      cudaFree(tmp_in);
      cudaFree(tmp_filt);
      cudaFree(tmp_out);
      cudaFree(search_ws);
      if (m_ws_bytes > 0) cudaMalloc(&m_workspace, m_ws_bytes);
    }

    size_t workspace_bytes() const { return m_ws_bytes; }

    // Forward convolution using a thread_local cudnnHandle_t.
    void forward(
      cudnnHandle_t  handle,
      const float    alpha, const float beta,
      const float*   dev_input,
      const float*   dev_filter,
      float*         dev_output) const
    {
      ALLEN_CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        m_input_desc,  dev_input,
        m_filter_desc, dev_filter,
        m_conv_desc,
        m_algo,
        m_workspace, m_ws_bytes,
        &beta,
        m_output_desc, dev_output));
    }

    // Legacy overload: accept a Handle wrapper (for backward compat).
    void forward(
      const Handle&  handle,
      const float    alpha, const float beta,
      const float*   dev_input,
      const float*   dev_filter,
      float*         dev_output) const
    {
      forward(handle.get(), alpha, beta, dev_input, dev_filter, dev_output);
    }

#else
    void create(cudnnHandle_t, std::array<int,4>, std::array<int,4>,
                std::array<int,2> = {0,0}, std::array<int,2> = {1,1},
                std::array<int,2> = {1,1}) {}
    size_t workspace_bytes() const { return 0; }
    void forward(cudnnHandle_t, float, float,
                 const float*, const float*, float*) const {}
    void forward(const Handle&, float, float,
                 const float*, const float*, float*) const {}
#endif
  };

} // namespace Allen::CuDNN
