#pragma once
#include "CuDNNCheck.h"
#include "CuDNNHandle.h"
#include <cstddef>
#include <array>
#ifdef ALLEN_CUDNN_BACKEND_CUDA
#include <cuda_fp16.h>
#endif

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
    // within a 64 MB workspace budget via cudnnFindConvolutionForwardAlgorithmEx.
    // Falls back to IMPLICIT_GEMM (zero workspace) if the query fails or returns
    // no suitable algorithm.
    // dtype: CUDNN_DATA_FLOAT (default) or CUDNN_DATA_HALF for FP16 Tensor Core path.
    // Compute type is always CUDNN_DATA_FLOAT (FP32 accumulation) for both dtypes.
    void create(
      cudnnHandle_t     handle,
      std::array<int,4> input_shape,            // {N, C_in, H, W} — fixed
      std::array<int,4> filter_shape,           // {K, C_in, R, S}
      std::array<int,2> pad      = {0, 0},
      std::array<int,2> stride   = {1, 1},
      std::array<int,2> dilation = {1, 1},
      cudnnDataType_t   dtype    = CUDNN_DATA_FLOAT)
    {
      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_input_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_filter_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_conv_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_output_desc));

      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_input_desc, CUDNN_TENSOR_NCHW, dtype,
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]));

      ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_filter_desc, dtype, CUDNN_TENSOR_NCHW,
        filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]));

      ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_conv_desc,
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
      // CUDNN_TENSOR_OP_MATH: enables TF32 on Ampere+ for FP32, and Tensor Core on all
      // supported GPUs for FP16 (wmma on SM 7.x, HMMA on SM 8.x+).
      ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(m_conv_desc, CUDNN_TENSOR_OP_MATH));

      // Derive and store output descriptor — fixed for this shape.
      int on, oc, oh, ow;
      ALLEN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        m_conv_desc, m_input_desc, m_filter_desc, &on, &oc, &oh, &ow));
      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_output_desc, CUDNN_TENSOR_NCHW, dtype, on, oc, oh, ow));

      m_created = true;

      // Select fastest algorithm by benchmarking all candidates on the actual
      // hardware (Find variant). One-time startup cost; returns measured timings
      // rather than heuristic lookup. Allocate kBudget as search workspace, run
      // Find, record the winner, then free the search buffer and reallocate only
      // the winning workspace size.
      static constexpr size_t kBudget  = 64ul * 1024 * 1024;
      static constexpr int    kMaxAlgos = 8;

      const size_t dtype_bytes = (dtype == CUDNN_DATA_HALF) ? sizeof(__half) : sizeof(float);
      size_t in_elems   = (size_t)input_shape[0]  * input_shape[1]  * input_shape[2]  * input_shape[3];
      size_t filt_elems = (size_t)filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3];
      size_t out_elems  = (size_t)on * oc * oh * ow;

      void *tmp_in = nullptr, *tmp_filt = nullptr, *tmp_out = nullptr;
      void *search_ws = nullptr;
      cudaMalloc(&tmp_in,    in_elems   * dtype_bytes);
      cudaMalloc(&tmp_filt,  filt_elems * dtype_bytes);
      cudaMalloc(&tmp_out,   out_elems  * dtype_bytes);
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

    // FP16 forward: for descriptors created with dtype=CUDNN_DATA_HALF.
    // Alpha/beta are float (cuDNN convention for FP16 tensors with FP32 accumulation).
    void forward_half(
      cudnnHandle_t   handle,
      const float     alpha, const float beta,
      const __half*   dev_input,
      const __half*   dev_filter,
      __half*         dev_output) const
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

#else
    void create(cudnnHandle_t, std::array<int,4>, std::array<int,4>,
                std::array<int,2> = {0,0}, std::array<int,2> = {1,1},
                std::array<int,2> = {1,1}, cudnnDataType_t = CUDNN_DATA_FLOAT) {}
    size_t workspace_bytes() const { return 0; }
    void forward(cudnnHandle_t, float, float,
                 const float*, const float*, float*) const {}
    void forward(const Handle&, float, float,
                 const float*, const float*, float*) const {}
    void forward_half(cudnnHandle_t, float, float,
                      const void*, const void*, void*) const {}
#endif
  };

  // ConvBiasReluGraph: placeholder — implementation removed (cuDNN backend graph
  // API requires NHWC layout and is not supported on all target GPUs).
  // Use ConvDescriptors::forward_fused_bias_relu() instead.
  /**
   * @brief Fused Conv + BiasAdd + ReLU via cuDNN backend graph API.
   *
   * BN parameters are folded into the conv weights/bias by the caller at init
   * time, so no separate batch-norm kernel is needed at runtime. The graph
   * encodes Conv → Pointwise-ADD(bias) → Pointwise-RELU as a single fused op
   * that cuDNN executes without writing the intermediate conv output to global
   * memory. Eliminates bias_bn_relu_kernel and its global-memory round-trip.
   *
   * UIDs for the variant pack (stable per graph instance):
   *   1 = x (input), 2 = w (fused weights), 3 = b (fused bias), 4 = y (output)
   */
  struct ConvBiasReluGraph {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnBackendDescriptor_t m_exec_plan = nullptr;
    void*  m_workspace = nullptr;
    size_t m_ws_bytes  = 0;
    bool   m_created   = false;

    static constexpr int64_t UID_X = 1, UID_W = 2, UID_B = 3, UID_Y = 4;
    static constexpr int64_t UID_ZCONV = 5, UID_ZADD = 6;

  public:
    ConvBiasReluGraph() = default;

    ~ConvBiasReluGraph() {
      if (!m_created) return;
      cudnnBackendDestroyDescriptor(m_exec_plan);
      if (m_workspace) cudaFree(m_workspace);
    }

    ConvBiasReluGraph(const ConvBiasReluGraph&) = delete;
    ConvBiasReluGraph& operator=(const ConvBiasReluGraph&) = delete;

    // Build a Conv+BiasAdd+ReLU graph, compile to an execution plan, and
    // allocate its workspace. input_shape/filter_shape are fixed at this call.
    // filter_shape is {K, C_in, 1, R} where R is the 1-D kernel size.
    // Fused weights (BN folded in) and fused bias are passed at execute() time.
    void create(
      cudnnHandle_t     handle,
      std::array<int,4> input_shape,           // {N, C_in, 1, W}
      std::array<int,4> filter_shape,          // {K, C_in, 1, R}
      std::array<int,2> pad      = {0, 0},
      std::array<int,2> stride   = {1, 1},
      std::array<int,2> dilation = {1, 1})
    {
      int N = input_shape[0], C_in = input_shape[1], W_in = input_shape[3];
      int K = filter_shape[0], R = filter_shape[3];
      int W_out = (W_in + 2*pad[1] - dilation[1]*(R-1) - 1) / stride[1] + 1;

      // Helper: create a backend descriptor of the given type.
      auto mk = [](cudnnBackendDescriptorType_t t) {
        cudnnBackendDescriptor_t d = nullptr;
        ALLEN_CUDNN_CHECK(cudnnBackendCreateDescriptor(t, &d));
        return d;
      };

      // Helper: create, configure, and finalize a tensor descriptor.
      auto mk_tensor = [&](int64_t uid, int64_t n, int64_t c, int64_t h, int64_t w, bool is_virtual) {
        auto d = mk(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        int64_t dims[4]    = {n, c, h, w};
        int64_t strides[4] = {c*h*w, h*w, w, 1};
        int64_t align = 4;
        cudnnDataType_t dt = CUDNN_DATA_FLOAT;
        int8_t  virt = is_virtual ? 1 : 0;
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DATA_TYPE,     CUDNN_TYPE_DATA_TYPE, 1, &dt));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_UNIQUE_ID,     CUDNN_TYPE_INT64,     1, &uid));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DIMENSIONS,    CUDNN_TYPE_INT64,     4, dims));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_STRIDES,       CUDNN_TYPE_INT64,     4, strides));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,     1, &align));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL,    CUDNN_TYPE_BOOLEAN,   1, &virt));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(d));
        return d;
      };

      // Tensor descriptors: real (x, w, b, y) and virtual (zconv, zadd).
      auto x_t     = mk_tensor(UID_X,     N,  C_in, 1, W_in,  false);
      auto w_t     = mk_tensor(UID_W,     K,  C_in, 1, R,     false);
      auto b_t     = mk_tensor(UID_B,     1,  K,    1, 1,     false);
      auto y_t     = mk_tensor(UID_Y,     N,  K,    1, W_out, false);
      auto zconv_t = mk_tensor(UID_ZCONV, N,  K,    1, W_out, true);
      auto zadd_t  = mk_tensor(UID_ZADD,  N,  K,    1, W_out, true);

      // Convolution descriptor (backend variant).
      auto conv_d = mk(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
      {
        cudnnDataType_t       comp  = CUDNN_DATA_FLOAT;
        cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
        int64_t sdims = 2;
        int64_t cpads[2]   = {pad[0],      pad[1]};
        int64_t cstrs[2]   = {stride[0],   stride[1]};
        int64_t cdils[2]   = {dilation[0], dilation[1]};
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_COMP_TYPE,      CUDNN_TYPE_DATA_TYPE,        1, &comp));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_CONV_MODE,      CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,   CUDNN_TYPE_INT64,            1, &sdims));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,   CUDNN_TYPE_INT64,            2, cpads));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,  CUDNN_TYPE_INT64,            2, cpads));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64,            2, cstrs));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_DILATIONS,      CUDNN_TYPE_INT64,            2, cdils));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(conv_d));
      }

      // Pointwise ADD descriptor (for bias add).
      auto add_pw = mk(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
      {
        cudnnPointwiseMode_t m = CUDNN_POINTWISE_ADD;
        cudnnDataType_t      p = CUDNN_DATA_FLOAT;
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_pw, CUDNN_ATTR_POINTWISE_MODE,      CUDNN_TYPE_POINTWISE_MODE, 1, &m));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE,      1, &p));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(add_pw));
      }

      // Pointwise RELU descriptor.
      auto relu_pw = mk(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
      {
        cudnnPointwiseMode_t m = CUDNN_POINTWISE_RELU_FWD;
        cudnnDataType_t      p = CUDNN_DATA_FLOAT;
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_pw, CUDNN_ATTR_POINTWISE_MODE,      CUDNN_TYPE_POINTWISE_MODE, 1, &m));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE,      1, &p));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(relu_pw));
      }

      // Convolution forward operation: x, w → zconv (virtual).
      float falpha = 1.f, fbeta = 0.f;
      auto conv_op = mk(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
      {
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zconv_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_d));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,     CUDNN_TYPE_FLOAT,              1, &falpha));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,      CUDNN_TYPE_FLOAT,              1, &fbeta));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(conv_op));
      }

      // Pointwise ADD operation: zconv + b → zadd (virtual).
      auto add_op = mk(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
      {
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &add_pw));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zconv_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zadd_t));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(add_op));
      }

      // Pointwise RELU operation: zadd → y.
      auto relu_op = mk(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
      {
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &relu_pw));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zadd_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_t));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(relu_op));
      }

      // Operation graph: {conv_op, add_op, relu_op}.
      auto op_graph = mk(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
      {
        cudnnBackendDescriptor_t ops[3] = {conv_op, add_op, relu_op};
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE,             1, &handle));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS,    CUDNN_TYPE_BACKEND_DESCRIPTOR, 3, ops));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(op_graph));
      }

      // Engine heuristics: instant mode (fast lookup, no profiling).
      auto heur = mk(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
      {
        cudnnBackendHeurMode_t hmode = CUDNN_HEUR_MODE_INSTANT;
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_MODE,            CUDNN_TYPE_HEUR_MODE,          1, &hmode));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(heur));
      }

      // Retrieve engine configs; iterate until one compiles to a valid plan.
      static constexpr int kMaxEngines = 10;
      cudnnBackendDescriptor_t eng_cfgs[kMaxEngines] = {};
      for (int i = 0; i < kMaxEngines; ++i)
        ALLEN_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &eng_cfgs[i]));
      int64_t returned = 0;
      ALLEN_CUDNN_CHECK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, kMaxEngines, &returned, eng_cfgs));

      for (int64_t i = 0; i < returned && !m_exec_plan; ++i) {
        cudnnBackendDescriptor_t plan = nullptr;
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan) != CUDNN_STATUS_SUCCESS) continue;
        if (cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE,        CUDNN_TYPE_HANDLE,             1, &handle)        != CUDNN_STATUS_SUCCESS ||
            cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &eng_cfgs[i])   != CUDNN_STATUS_SUCCESS ||
            cudnnBackendFinalize(plan)                                                                                                  != CUDNN_STATUS_SUCCESS) {
          cudnnBackendDestroyDescriptor(plan);
          continue;
        }
        int64_t ws = 0;
        cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, nullptr, &ws);
        m_ws_bytes  = (size_t)ws;
        m_exec_plan = plan;
        if (m_ws_bytes > 0) cudaMalloc(&m_workspace, m_ws_bytes);
      }

      // Free all intermediate descriptors.
      cudnnBackendDestroyDescriptor(x_t);
      cudnnBackendDestroyDescriptor(w_t);
      cudnnBackendDestroyDescriptor(b_t);
      cudnnBackendDestroyDescriptor(y_t);
      cudnnBackendDestroyDescriptor(zconv_t);
      cudnnBackendDestroyDescriptor(zadd_t);
      cudnnBackendDestroyDescriptor(conv_d);
      cudnnBackendDestroyDescriptor(add_pw);
      cudnnBackendDestroyDescriptor(relu_pw);
      cudnnBackendDestroyDescriptor(conv_op);
      cudnnBackendDestroyDescriptor(add_op);
      cudnnBackendDestroyDescriptor(relu_op);
      cudnnBackendDestroyDescriptor(op_graph);
      cudnnBackendDestroyDescriptor(heur);
      for (int i = 0; i < kMaxEngines; ++i)
        if (eng_cfgs[i]) cudnnBackendDestroyDescriptor(eng_cfgs[i]);

      if (!m_exec_plan)
        throw std::invalid_argument("ConvBiasReluGraph::create: no valid engine for Conv+Add+ReLU");
      m_created = true;
    }

    bool is_created() const { return m_created; }

    // Execute the fused graph. x/w/b are device pointers; w and b must be the
    // BN-folded fused weights and biases computed at init time.
    void execute(
      cudnnHandle_t handle,
      const float*  x,
      const float*  w,
      const float*  b,
      float*        y) const
    {
      cudnnBackendDescriptor_t vpack = nullptr;
      ALLEN_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vpack));

      int64_t uids[4] = {UID_X, UID_W, UID_B, UID_Y};
      void*   ptrs[4] = {const_cast<float*>(x), const_cast<float*>(w),
                         const_cast<float*>(b), y};
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,    CUDNN_TYPE_INT64,     4, uids));
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR,  4, ptrs));
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,     CUDNN_TYPE_VOID_PTR,  1, &m_workspace));
      ALLEN_CUDNN_CHECK(cudnnBackendFinalize(vpack));
      ALLEN_CUDNN_CHECK(cudnnBackendExecute(handle, m_exec_plan, vpack));
      cudnnBackendDestroyDescriptor(vpack);
    }

#else
    void create(cudnnHandle_t, std::array<int,4>, std::array<int,4>,
                std::array<int,2> = {0,0}, std::array<int,2> = {1,1},
                std::array<int,2> = {1,1}) {}
    bool is_created() const { return false; }
    void execute(cudnnHandle_t, const float*, const float*, const float*, float*) const {}
#endif
  };

} // namespace Allen::CuDNN
