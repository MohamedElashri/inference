#pragma once

#include "CuDNNCheck.h"
#include "CuDNNHandle.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef ALLEN_CUDNN_BACKEND_CUDA
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace Allen::CuDNN {

  enum class AlgorithmSelectionPolicy {
    ZeroWorkspace,
    Heuristic,
    TimedFind
  };

  enum class WorkspacePolicy {
    AllenExternal,
    OwnedInitTime,
    ZeroOnly
  };

  struct TensorShape {
    int n = 0, c = 0, h = 0, w = 0;
    bool operator==(const TensorShape& o) const { return n == o.n && c == o.c && h == o.h && w == o.w; }
    bool operator!=(const TensorShape& o) const { return !(*this == o); }
  };

  struct ConvPlanOptions {
    AlgorithmSelectionPolicy algorithm_policy = AlgorithmSelectionPolicy::TimedFind;
    WorkspacePolicy workspace_policy = WorkspacePolicy::OwnedInitTime;
    size_t workspace_limit_bytes = 64ul * 1024 * 1024;
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;
    cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
#endif
  };

#ifdef ALLEN_CUDNN_BACKEND_CUDA
  namespace detail {
    inline size_t dtype_size(cudnnDataType_t dtype) {
      return dtype == CUDNN_DATA_HALF ? sizeof(__half) : sizeof(float);
    }

    inline void cuda_check(cudaError_t e, const char* what) {
      if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
      }
    }
  } // namespace detail
#endif

  struct TensorDescriptor {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnTensorDescriptor_t m_desc = nullptr;

  public:
    TensorDescriptor() = default;
    ~TensorDescriptor() { reset(); }
    TensorDescriptor(const TensorDescriptor&) = delete;
    TensorDescriptor& operator=(const TensorDescriptor&) = delete;

    void reset() {
      if (m_desc) {
        cudnnDestroyTensorDescriptor(m_desc);
        m_desc = nullptr;
      }
    }

    void create() {
      reset();
      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_desc));
    }

    void set_4d(std::array<int, 4> shape, cudnnDataType_t dtype) {
      if (!m_desc) create();
      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_desc, CUDNN_TENSOR_NCHW, dtype, shape[0], shape[1], shape[2], shape[3]));
    }

    cudnnTensorDescriptor_t get() const { return m_desc; }
#else
    void reset() {}
    void create() {}
    void set_4d(std::array<int, 4>, int) {}
    void* get() const { return nullptr; }
#endif
  };

  struct FilterDescriptor {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnFilterDescriptor_t m_desc = nullptr;

  public:
    FilterDescriptor() = default;
    ~FilterDescriptor() { reset(); }
    FilterDescriptor(const FilterDescriptor&) = delete;
    FilterDescriptor& operator=(const FilterDescriptor&) = delete;

    void reset() {
      if (m_desc) {
        cudnnDestroyFilterDescriptor(m_desc);
        m_desc = nullptr;
      }
    }

    void create() {
      reset();
      ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_desc));
    }

    void set_4d(std::array<int, 4> shape, cudnnDataType_t dtype) {
      if (!m_desc) create();
      ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_desc, dtype, CUDNN_TENSOR_NCHW, shape[0], shape[1], shape[2], shape[3]));
    }

    cudnnFilterDescriptor_t get() const { return m_desc; }
#else
    void reset() {}
    void create() {}
    void set_4d(std::array<int, 4>, int) {}
    void* get() const { return nullptr; }
#endif
  };

  struct ConvolutionDescriptor {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnConvolutionDescriptor_t m_desc = nullptr;

  public:
    ConvolutionDescriptor() = default;
    ~ConvolutionDescriptor() { reset(); }
    ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
    ConvolutionDescriptor& operator=(const ConvolutionDescriptor&) = delete;

    void reset() {
      if (m_desc) {
        cudnnDestroyConvolutionDescriptor(m_desc);
        m_desc = nullptr;
      }
    }

    void create() {
      reset();
      ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_desc));
    }

    void set_2d(
      std::array<int, 2> pad,
      std::array<int, 2> stride,
      std::array<int, 2> dilation,
      cudnnDataType_t compute_type,
      cudnnMathType_t math_type)
    {
      if (!m_desc) create();
      ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_desc,
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        CUDNN_CROSS_CORRELATION, compute_type));
      ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(m_desc, math_type));
    }

    cudnnConvolutionDescriptor_t get() const { return m_desc; }
#else
    void reset() {}
    void create() {}
    void set_2d(std::array<int, 2>, std::array<int, 2>, std::array<int, 2>, int, int) {}
    void* get() const { return nullptr; }
#endif
  };

  struct ForwardConvPlan {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    TensorDescriptor m_input_desc;
    FilterDescriptor m_filter_desc;
    ConvolutionDescriptor m_conv_desc;
    TensorDescriptor m_output_desc;
    cudnnConvolutionFwdAlgo_t m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    ConvPlanOptions m_options {};
    void* m_workspace = nullptr;
    size_t m_ws_bytes = 0;
    bool m_created = false;

    void release_workspace() {
      if (m_workspace) {
        cudaFree(m_workspace);
        m_workspace = nullptr;
      }
      m_ws_bytes = 0;
    }

    void set_workspace(size_t bytes) {
      m_ws_bytes = bytes;
      if (m_options.workspace_policy == WorkspacePolicy::ZeroOnly && bytes != 0) {
        m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        m_ws_bytes = 0;
      }
      if (m_options.workspace_policy == WorkspacePolicy::OwnedInitTime && m_ws_bytes > 0) {
        detail::cuda_check(cudaMalloc(&m_workspace, m_ws_bytes), "ForwardConvPlan workspace allocation failed");
      }
    }

    void select_heuristic(cudnnHandle_t handle) {
      static constexpr int kMaxAlgos = 8;
      int returned = 0;
      cudnnConvolutionFwdAlgoPerf_t perf[kMaxAlgos];
      if (cudnnGetConvolutionForwardAlgorithm_v7(
            handle,
            m_input_desc.get(),
            m_filter_desc.get(),
            m_conv_desc.get(),
            m_output_desc.get(),
            kMaxAlgos,
            &returned,
            perf) == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < returned; ++i) {
          if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= m_options.workspace_limit_bytes) {
            m_algo = perf[i].algo;
            set_workspace(perf[i].memory);
            return;
          }
        }
      }
    }

    void select_timed(
      cudnnHandle_t handle,
      std::array<int, 4> input_shape,
      std::array<int, 4> filter_shape,
      std::array<int, 4> output_shape)
    {
      static constexpr int kMaxAlgos = 8;
      const size_t dtype_bytes = detail::dtype_size(m_options.data_type);
      const size_t in_elems = (size_t) input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
      const size_t filt_elems = (size_t) filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3];
      const size_t out_elems = (size_t) output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];

      void *tmp_in = nullptr, *tmp_filt = nullptr, *tmp_out = nullptr, *search_ws = nullptr;
      detail::cuda_check(cudaMalloc(&tmp_in, in_elems * dtype_bytes), "ForwardConvPlan FindEx input allocation failed");
      detail::cuda_check(cudaMalloc(&tmp_filt, filt_elems * dtype_bytes), "ForwardConvPlan FindEx filter allocation failed");
      detail::cuda_check(cudaMalloc(&tmp_out, out_elems * dtype_bytes), "ForwardConvPlan FindEx output allocation failed");
      detail::cuda_check(cudaMalloc(&search_ws, m_options.workspace_limit_bytes), "ForwardConvPlan FindEx search allocation failed");

      int returned = 0;
      cudnnConvolutionFwdAlgoPerf_t perf[kMaxAlgos];
      if (cudnnFindConvolutionForwardAlgorithmEx(
            handle,
            m_input_desc.get(), tmp_in,
            m_filter_desc.get(), tmp_filt,
            m_conv_desc.get(),
            m_output_desc.get(), tmp_out,
            kMaxAlgos, &returned, perf,
            search_ws, m_options.workspace_limit_bytes) == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < returned; ++i) {
          if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= m_options.workspace_limit_bytes) {
            m_algo = perf[i].algo;
            set_workspace(perf[i].memory);
            break;
          }
        }
      }

      cudaFree(tmp_in);
      cudaFree(tmp_filt);
      cudaFree(tmp_out);
      cudaFree(search_ws);
    }

  public:
    ForwardConvPlan() = default;
    ~ForwardConvPlan() { release_workspace(); }
    ForwardConvPlan(const ForwardConvPlan&) = delete;
    ForwardConvPlan& operator=(const ForwardConvPlan&) = delete;

    void create(
      cudnnHandle_t handle,
      std::array<int, 4> input_shape,
      std::array<int, 4> filter_shape,
      std::array<int, 2> pad = {0, 0},
      std::array<int, 2> stride = {1, 1},
      std::array<int, 2> dilation = {1, 1},
      ConvPlanOptions options = {})
    {
      release_workspace();
      m_options = options;

      m_input_desc.set_4d(input_shape, m_options.data_type);
      m_filter_desc.set_4d(filter_shape, m_options.data_type);
      m_conv_desc.set_2d(pad, stride, dilation, m_options.compute_type, m_options.math_type);

      int on = 0, oc = 0, oh = 0, ow = 0;
      ALLEN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        m_conv_desc.get(), m_input_desc.get(), m_filter_desc.get(), &on, &oc, &oh, &ow));
      const std::array<int, 4> output_shape {on, oc, oh, ow};
      m_output_desc.set_4d(output_shape, m_options.data_type);

      m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      m_ws_bytes = 0;
      if (m_options.algorithm_policy == AlgorithmSelectionPolicy::Heuristic) {
        select_heuristic(handle);
      }
      else if (m_options.algorithm_policy == AlgorithmSelectionPolicy::TimedFind) {
        select_timed(handle, input_shape, filter_shape, output_shape);
      }

      m_created = true;
    }

    void create(
      cudnnHandle_t handle,
      std::array<int, 4> input_shape,
      std::array<int, 4> filter_shape,
      std::array<int, 2> pad,
      std::array<int, 2> stride,
      std::array<int, 2> dilation,
      cudnnDataType_t dtype)
    {
      ConvPlanOptions options {};
      options.data_type = dtype;
      create(handle, input_shape, filter_shape, pad, stride, dilation, options);
    }

    size_t workspace_bytes() const { return m_ws_bytes; }
    WorkspacePolicy workspace_policy() const { return m_options.workspace_policy; }
    AlgorithmSelectionPolicy algorithm_policy() const { return m_options.algorithm_policy; }

    void forward(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const void* dev_input,
      const void* dev_filter,
      void* dev_output,
      void* external_workspace = nullptr) const
    {
      if (m_options.workspace_policy == WorkspacePolicy::AllenExternal && m_ws_bytes > 0 && external_workspace == nullptr) {
        throw std::runtime_error("ForwardConvPlan requires external workspace");
      }
      void* workspace = m_options.workspace_policy == WorkspacePolicy::AllenExternal ? external_workspace : m_workspace;
      ALLEN_CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        m_input_desc.get(), dev_input,
        m_filter_desc.get(), dev_filter,
        m_conv_desc.get(),
        m_algo,
        workspace, m_ws_bytes,
        &beta,
        m_output_desc.get(), dev_output));
    }

    void forward(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const float* dev_input,
      const float* dev_filter,
      float* dev_output) const
    {
      forward(handle, alpha, beta, (const void*) dev_input, (const void*) dev_filter, (void*) dev_output);
    }

    void forward(const Handle& handle, const float alpha, const float beta, const float* dev_input, const float* dev_filter, float* dev_output) const {
      forward(handle.get(), alpha, beta, dev_input, dev_filter, dev_output);
    }

    void forward_half(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const __half* dev_input,
      const __half* dev_filter,
      __half* dev_output) const
    {
      forward(handle, alpha, beta, (const void*) dev_input, (const void*) dev_filter, (void*) dev_output);
    }
#else
    void create(void*, std::array<int, 4>, std::array<int, 4>, std::array<int, 2> = {0, 0}, std::array<int, 2> = {1, 1}, std::array<int, 2> = {1, 1}, ConvPlanOptions = {}) {}
    size_t workspace_bytes() const { return 0; }
    WorkspacePolicy workspace_policy() const { return WorkspacePolicy::ZeroOnly; }
    AlgorithmSelectionPolicy algorithm_policy() const { return AlgorithmSelectionPolicy::ZeroWorkspace; }
    void forward(void*, float, float, const float*, const float*, float*) const {}
    void forward(const Handle&, float, float, const float*, const float*, float*) const {}
    void forward_half(void*, float, float, const void*, const void*, void*) const {}
#endif
  };

  using ConvDescriptors = ForwardConvPlan;

  struct BackwardDataConvPlan {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    FilterDescriptor m_filter_desc;
    ConvolutionDescriptor m_conv_desc;
    TensorDescriptor m_input_desc;
    TensorDescriptor m_output_desc;
    cudnnConvolutionBwdDataAlgo_t m_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    ConvPlanOptions m_options {};
    void* m_workspace = nullptr;
    size_t m_ws_bytes = 0;
    bool m_created = false;

    void release_workspace() {
      if (m_workspace) {
        cudaFree(m_workspace);
        m_workspace = nullptr;
      }
      m_ws_bytes = 0;
    }

    void set_workspace(size_t bytes) {
      m_ws_bytes = bytes;
      if (m_options.workspace_policy == WorkspacePolicy::ZeroOnly && bytes != 0) {
        m_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        m_ws_bytes = 0;
      }
      if (m_options.workspace_policy == WorkspacePolicy::OwnedInitTime && m_ws_bytes > 0) {
        detail::cuda_check(cudaMalloc(&m_workspace, m_ws_bytes), "BackwardDataConvPlan workspace allocation failed");
      }
    }

    void select_heuristic(cudnnHandle_t handle) {
      static constexpr int kMaxAlgos = 8;
      int returned = 0;
      cudnnConvolutionBwdDataAlgoPerf_t perf[kMaxAlgos];
      if (cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle,
            m_filter_desc.get(),
            m_input_desc.get(),
            m_conv_desc.get(),
            m_output_desc.get(),
            kMaxAlgos,
            &returned,
            perf) == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < returned; ++i) {
          if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= m_options.workspace_limit_bytes) {
            m_algo = perf[i].algo;
            set_workspace(perf[i].memory);
            return;
          }
        }
      }
    }

  public:
    BackwardDataConvPlan() = default;
    ~BackwardDataConvPlan() { release_workspace(); }
    BackwardDataConvPlan(const BackwardDataConvPlan&) = delete;
    BackwardDataConvPlan& operator=(const BackwardDataConvPlan&) = delete;

    void create(
      cudnnHandle_t handle,
      std::array<int, 4> filter_shape,
      std::array<int, 4> input_shape,
      std::array<int, 4> output_shape,
      std::array<int, 2> pad = {0, 0},
      std::array<int, 2> stride = {1, 1},
      std::array<int, 2> dilation = {1, 1},
      ConvPlanOptions options = {})
    {
      release_workspace();
      m_options = options;
      m_filter_desc.set_4d(filter_shape, m_options.data_type);
      m_input_desc.set_4d(input_shape, m_options.data_type);
      m_output_desc.set_4d(output_shape, m_options.data_type);
      m_conv_desc.set_2d(pad, stride, dilation, m_options.compute_type, m_options.math_type);

      m_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      m_ws_bytes = 0;
      if (m_options.algorithm_policy == AlgorithmSelectionPolicy::Heuristic ||
          m_options.algorithm_policy == AlgorithmSelectionPolicy::TimedFind) {
        select_heuristic(handle);
      }
      m_created = true;
    }

    size_t workspace_bytes() const { return m_ws_bytes; }

    void backward_data(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const void* dev_filter,
      const void* dev_input,
      void* dev_output,
      void* external_workspace = nullptr) const
    {
      if (m_options.workspace_policy == WorkspacePolicy::AllenExternal && m_ws_bytes > 0 && external_workspace == nullptr) {
        throw std::runtime_error("BackwardDataConvPlan requires external workspace");
      }
      void* workspace = m_options.workspace_policy == WorkspacePolicy::AllenExternal ? external_workspace : m_workspace;
      ALLEN_CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        m_filter_desc.get(), dev_filter,
        m_input_desc.get(), dev_input,
        m_conv_desc.get(),
        m_algo,
        workspace, m_ws_bytes,
        &beta,
        m_output_desc.get(), dev_output));
    }
#else
    void create(void*, std::array<int, 4>, std::array<int, 4>, std::array<int, 4>, std::array<int, 2> = {0, 0}, std::array<int, 2> = {1, 1}, std::array<int, 2> = {1, 1}, ConvPlanOptions = {}) {}
    size_t workspace_bytes() const { return 0; }
    void backward_data(void*, float, float, const void*, const void*, void*, void* = nullptr) const {}
#endif
  };

  // ConvBiasReluGraph: placeholder — implementation removed (cuDNN backend graph
  // API requires NHWC layout and is not supported on all target GPUs).
  struct ConvBiasReluGraph {
    void create(
#ifdef ALLEN_CUDNN_BACKEND_CUDA
      cudnnHandle_t,
#else
      void*,
#endif
      std::array<int, 4>, std::array<int, 4>,
      std::array<int, 2> = {0, 0}, std::array<int, 2> = {1, 1}, std::array<int, 2> = {1, 1})
    {}
    bool is_created() const { return false; }
    void execute(
#ifdef ALLEN_CUDNN_BACKEND_CUDA
      cudnnHandle_t,
#else
      void*,
#endif
      const float*, const float*, const float*, float*) const
    {}
  };

} // namespace Allen::CuDNN
