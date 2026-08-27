#pragma once
#include "CuDNNCheck.h"
#include "CuDNNHandle.h"
#include <cstddef>
#include <array>
#include <unordered_map>
#include <utility>
#ifdef ALLEN_CUDNN_BACKEND_CUDA
#include <cuda_fp16.h>
#include <cuda_bf16.h>
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
   *  - Algorithm is pinned to IMPLICIT_GEMM at create() time by default,
   *    rather than selected via cudnnFindConvolutionForwardAlgorithmEx or
   *    cudnnGetConvolutionForwardAlgorithm_v7's heuristic. For this
   *    network's thin, low-channel shapes under real concurrent multi-thread
   *    load, an algorithm promoted by either search strategy tends to need a
   *    multi-MB per-thread workspace (im2col-style scratch written to and
   *    read from GPU global memory on every call); the resulting
   *    memory-bandwidth contention across many concurrent threads costs more
   *    than any Tensor-Core math it buys back. A one-shot, isolated timing
   *    or cost-model measurement at create() time has no way to see that
   *    concurrent-load cost, so IMPLICIT_GEMM's near-zero workspace
   *    footprint wins for this shape under production load even though it
   *    isn't the fastest single-call candidate in isolation. A bounded,
   *    live-verified heuristic search is available opt-in via
   *    workspace_budget_bytes for callers where that tradeoff differs (see
   *    below).
   *  - Workspace is thread_local, not a single shared buffer: cuDNN writes real
   *    per-call intermediate scratch state into it during forward() (im2col buffers,
   *    partial reductions, etc.) -- it is NOT read-only/inert. A single workspace
   *    buffer shared across concurrent threads on the same descriptor would be a data
   *    race whenever the selected algorithm has nonzero workspace_bytes() (IMPLICIT_GEMM
   *    itself is typically zero-workspace for these shapes, but the size is still queried
   *    via cudnnGetConvolutionForwardWorkspaceSize() rather than assumed, in case that
   *    changes for a future shape). Each descriptor instance still selects ONE
   *    algorithm/size at create() time (single-threaded init); only the workspace
   *    *buffer* backing that fixed size is now lazily allocated per (thread, instance)
   *    via get_thread_local_workspace(), mirroring the thread_local idiom used elsewhere
   *    in Allen::CuDNN (see CuDNNHandle.h's get_thread_local_handle). Never freed (relies
   *    on process teardown, same as that handle) -- sizes here are a few KB at most.
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
    size_t                       m_ws_bytes     = 0;
    bool m_created = false;

    // Thread-local workspace, keyed by this instance within each thread's own map
    // (the map itself is thread_local, so no cross-thread contention/locking is
    // needed -- each thread only ever touches its own map). Lazily grows to
    // needed_bytes on first use per (thread, instance); never shrinks or frees.
    // cudaMalloc's result MUST be checked here: on failure it leaves the pointer
    // null, and if entry.second were still marked as "sized" despite that, every
    // later call would silently hand cudnnConvolutionForward a null workspace
    // with a nonzero requested size -- CUDNN_STATUS_BAD_PARAM, permanently, for
    // the rest of the process's life on that (thread, instance). Only record the
    // new size once the allocation actually succeeds, so a transient failure is
    // retried on the next call instead of being latched in as a fatal state.
    static void* get_thread_local_workspace(const void* instance_key, size_t needed_bytes) {
      if (needed_bytes == 0) return nullptr;
      thread_local std::unordered_map<const void*, std::pair<void*, size_t>> tl_workspaces;
      auto& entry = tl_workspaces[instance_key];
      if (entry.second < needed_bytes) {
        if (entry.first) cudaFree(entry.first);
        entry.first = nullptr;
        cudaCheck(cudaMalloc(&entry.first, needed_bytes));
        entry.second = needed_bytes;
      }
      return entry.first;
    }

  public:
    ConvDescriptors() = default;

    ~ConvDescriptors() {
      if (!m_created) return;
      cudnnDestroyTensorDescriptor(m_input_desc);
      cudnnDestroyFilterDescriptor(m_filter_desc);
      cudnnDestroyConvolutionDescriptor(m_conv_desc);
      cudnnDestroyTensorDescriptor(m_output_desc);
      // Thread-local workspaces (see get_thread_local_workspace) are intentionally
      // not freed here -- they may be owned by threads other than the one running
      // this destructor, and are negligible in size (a few KB per thread).
    }

    // Forces this instance's thread-local workspace to be allocated on the calling
    // thread right now, rather than lazily on the first forward()/forward_half()
    // call. Needed before capturing a CUDA graph that calls forward() on this
    // descriptor: growing the workspace (cudaMalloc/cudaFree) DURING an active
    // stream capture is unsupported, so callers that capture must pre-warm every
    // descriptor they will use first.
    void ensure_thread_local_workspace() const { get_thread_local_workspace(this, m_ws_bytes); }

    ConvDescriptors(const ConvDescriptors&) = delete;
    ConvDescriptors& operator=(const ConvDescriptors&) = delete;

    // Create descriptors with fixed input shape. Algorithm is pinned to
    // IMPLICIT_GEMM by default (see rationale above) rather than selected via
    // cudnnFindConvolutionForwardAlgorithmEx -- unless workspace_budget_bytes
    // is nonzero, in which case a bounded heuristic search is used instead
    // (see the workspace_budget_bytes parameter doc below).
    // dtype: CUDNN_DATA_FLOAT (default) or CUDNN_DATA_HALF for FP16 Tensor Core path.
    // Compute type is always CUDNN_DATA_FLOAT (FP32 accumulation) for both dtypes.
    void create(
      cudnnHandle_t     handle,
      std::array<int,4> input_shape,            // {N, C_in, H, W} — fixed
      std::array<int,4> filter_shape,           // {K, C_in, R, S}
      std::array<int,2> pad      = {0, 0},
      std::array<int,2> stride   = {1, 1},
      std::array<int,2> dilation = {1, 1},
      cudnnDataType_t   dtype    = CUDNN_DATA_FLOAT,
      // 0 (default): keep the pinned-IMPLICIT_GEMM behavior unchanged, bit
      // for bit, from before this parameter existed. Nonzero: run
      // cudnnGetConvolutionForwardAlgorithm_v7 (a static heuristic cost
      // model, not a timed benchmark -- same query style already proven for
      // ConvTranspose's kBwdBudget in PVFinderUNet.cu), then LIVE-VERIFY each
      // within-budget candidate with one real cudnnConvolutionForward() call
      // against scratch buffers before adopting it -- the heuristic can and
      // does report success for algorithms that then fail at real call time
      // for some shape/dtype combinations here (see the loop below). Falls
      // back to pinned IMPLICIT_GEMM if nothing fits the budget and survives
      // verification, or the initial query fails.
      size_t            workspace_budget_bytes = 0)
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

      bool picked_by_search = false;
      if (workspace_budget_bytes > 0) {
        static constexpr int kFwdMaxAlgo = 8;
        cudnnConvolutionFwdAlgoPerf_t perf[kFwdMaxAlgo];
        int returned = 0;
        if (cudnnGetConvolutionForwardAlgorithm_v7(
              handle, m_input_desc, m_filter_desc, m_conv_desc, m_output_desc,
              kFwdMaxAlgo, &returned, perf) == CUDNN_STATUS_SUCCESS) {
          // v7's ranking is a static cost model, not a real execution -- it can
          // (and, empirically, does for some FP16/shape combinations here)
          // report CUDNN_STATUS_SUCCESS for an algorithm that then fails at
          // real cudnnConvolutionForward() call time with CUDNN_STATUS_BAD_PARAM.
          // Guard against that by actually executing each within-budget
          // candidate once, here at init time, against scratch buffers sized
          // to this descriptor's real tensors, and only adopting the first
          // one that genuinely succeeds.
          const size_t elem_size  = (dtype == CUDNN_DATA_HALF) ? sizeof(__half) : sizeof(float);
          const size_t in_elems   = (size_t)input_shape[0]  * input_shape[1]  * input_shape[2]  * input_shape[3];
          const size_t filt_elems = (size_t)filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3];
          const size_t out_elems  = (size_t)on * oc * oh * ow;
          void* dummy_in   = nullptr;
          void* dummy_filt = nullptr;
          void* dummy_out  = nullptr;
          if (cudaMalloc(&dummy_in,   in_elems   * elem_size) == cudaSuccess &&
              cudaMalloc(&dummy_filt, filt_elems * elem_size) == cudaSuccess &&
              cudaMalloc(&dummy_out,  out_elems  * elem_size) == cudaSuccess) {
            cudaMemset(dummy_in,   0, in_elems   * elem_size);
            cudaMemset(dummy_filt, 0, filt_elems * elem_size);
            const float alpha = 1.f, beta = 0.f;
            for (int i = 0; i < returned && !picked_by_search; ++i) {
              if (perf[i].status != CUDNN_STATUS_SUCCESS || perf[i].memory > workspace_budget_bytes) continue;
              void* dummy_ws = nullptr;
              if (perf[i].memory > 0 && cudaMalloc(&dummy_ws, perf[i].memory) != cudaSuccess) continue;
              const cudnnStatus_t trial = cudnnConvolutionForward(
                handle, &alpha, m_input_desc, dummy_in, m_filter_desc, dummy_filt,
                m_conv_desc, perf[i].algo, dummy_ws, perf[i].memory, &beta, m_output_desc, dummy_out);
              if (dummy_ws) cudaFree(dummy_ws);
              if (trial == CUDNN_STATUS_SUCCESS) {
                m_algo        = perf[i].algo;
                m_ws_bytes    = perf[i].memory;
                picked_by_search = true;
              }
            }
          }
          if (dummy_in)   cudaFree(dummy_in);
          if (dummy_filt) cudaFree(dummy_filt);
          if (dummy_out)  cudaFree(dummy_out);
        }
      }
      if (!picked_by_search) {
        m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        ALLEN_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
          handle, m_input_desc, m_filter_desc, m_conv_desc, m_output_desc, m_algo, &m_ws_bytes));
      }
      // Workspace buffer itself is not allocated here -- see
      // get_thread_local_workspace(): it's lazily allocated per (thread,
      // instance) on first forward()/forward_half() call instead.
    }

    size_t workspace_bytes() const { return m_ws_bytes; }
    int algo_id() const { return static_cast<int>(m_algo); }

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
        get_thread_local_workspace(this, m_ws_bytes), m_ws_bytes,
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
        get_thread_local_workspace(this, m_ws_bytes), m_ws_bytes,
        &beta,
        m_output_desc, dev_output));
    }

    // BF16 forward: for descriptors created with dtype=CUDNN_DATA_BFLOAT16.
    // Mirrors forward_half exactly; BF16 shares FP32's exponent range, so it
    // avoids the overflow-to-NaN failure mode FP16 can hit on wide-dynamic-
    // range inputs.
    void forward_bf16(
      cudnnHandle_t         handle,
      const float           alpha, const float beta,
      const __nv_bfloat16*  dev_input,
      const __nv_bfloat16*  dev_filter,
      __nv_bfloat16*        dev_output) const
    {
      ALLEN_CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        m_input_desc,  dev_input,
        m_filter_desc, dev_filter,
        m_conv_desc,
        m_algo,
        get_thread_local_workspace(this, m_ws_bytes), m_ws_bytes,
        &beta,
        m_output_desc, dev_output));
    }

#else
    void create(cudnnHandle_t, std::array<int,4>, std::array<int,4>,
                std::array<int,2> = {0,0}, std::array<int,2> = {1,1},
                std::array<int,2> = {1,1}, cudnnDataType_t = CUDNN_DATA_FLOAT,
                size_t = 0) {}
    size_t workspace_bytes() const { return 0; }
    int algo_id() const { return 0; }
    void ensure_thread_local_workspace() const {}
    void forward(cudnnHandle_t, float, float,
                 const float*, const float*, float*) const {}
    void forward(const Handle&, float, float,
                 const float*, const float*, float*) const {}
    void forward_half(cudnnHandle_t, float, float,
                      const void*, const void*, void*) const {}
    void forward_bf16(cudnnHandle_t, float, float,
                      const void*, const void*, void*) const {}
#endif
  };

  /**
   * @brief Fused Conv + BiasAdd + ReLU via cuDNN backend graph API.
   *
   * BN parameters are folded into the conv weights/bias by the caller at init
   * time, so no separate batch-norm kernel is needed at runtime. The graph
   * encodes Conv → Pointwise-ADD(bias) → Pointwise-RELU as a single fused op
   * that cuDNN executes without writing the intermediate conv output to global
   * memory. Eliminates the conv-output DRAM round trip that a separate
   * bias+ReLU kernel pass otherwise requires.
   *
   * Tensor strides are set to standard NCHW-contiguous (H=1) layout — the
   * cuDNN backend graph API does not require NHWC; it accepts arbitrary
   * strides via CUDNN_ATTR_TENSOR_STRIDES. What IS GPU/shape-dependent is
   * whether any engine advertises support for this op-graph at all: create()
   * throws std::invalid_argument if the engine-heuristic search returns no
   * usable engine config, so callers on unsupported hardware must catch that
   * and fall back to the separate ConvDescriptors + bias/ReLU kernel path.
   *
   * Workspace is thread_local (see get_thread_local_workspace below), mirroring
   * ConvDescriptors: the execution plan is a read-only compiled artifact safe
   * to share across concurrent threads once created, but the workspace buffer
   * cuDNN scribbles into during execute() is not, so each thread gets its own.
   *
   * UIDs for the variant pack (stable per graph instance):
   *   1 = x (input), 2 = w (fused weights), 3 = b (fused bias), 4 = y (output)
   */
  struct ConvBiasReluGraph {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnBackendDescriptor_t m_exec_plan = nullptr;
    size_t m_ws_bytes  = 0;
    bool   m_created   = false;

    static constexpr int64_t UID_X = 1, UID_W = 2, UID_B = 3, UID_Y = 4;
    static constexpr int64_t UID_ZCONV = 5, UID_ZADD = 6;

    // Thread-local workspace, keyed by this instance within each thread's own
    // map — same idiom as ConvDescriptors::get_thread_local_workspace (see
    // that function's comment for the cudaMalloc-failure handling rationale).
    static void* get_thread_local_workspace(const void* instance_key, size_t needed_bytes) {
      if (needed_bytes == 0) return nullptr;
      thread_local std::unordered_map<const void*, std::pair<void*, size_t>> tl_workspaces;
      auto& entry = tl_workspaces[instance_key];
      if (entry.second < needed_bytes) {
        if (entry.first) cudaFree(entry.first);
        entry.first = nullptr;
        cudaCheck(cudaMalloc(&entry.first, needed_bytes));
        entry.second = needed_bytes;
      }
      return entry.first;
    }

  public:
    ConvBiasReluGraph() = default;

    ~ConvBiasReluGraph() {
      if (!m_created) return;
      cudnnBackendDestroyDescriptor(m_exec_plan);
      // Thread-local workspaces (see get_thread_local_workspace) are intentionally
      // not freed here — same rationale as ConvDescriptors's destructor.
    }

    // Pre-allocate this instance's thread-local workspace on the calling thread.
    void ensure_thread_local_workspace() const { get_thread_local_workspace(this, m_ws_bytes); }

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
        // Workspace buffer itself is not allocated here — see
        // get_thread_local_workspace(): it's lazily allocated per (thread,
        // instance) on first execute() call instead, mirroring ConvDescriptors.
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
    size_t workspace_bytes() const { return m_ws_bytes; }

    // Execute the fused graph. x/w/b are device pointers; w and b must be the
    // BN-folded fused weights and biases computed at init time.
    void execute(
      cudnnHandle_t handle,
      const float*  x,
      const float*  w,
      const float*  b,
      float*        y) const
    {
      void* workspace = get_thread_local_workspace(this, m_ws_bytes);

      cudnnBackendDescriptor_t vpack = nullptr;
      ALLEN_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vpack));

      int64_t uids[4] = {UID_X, UID_W, UID_B, UID_Y};
      void*   ptrs[4] = {const_cast<float*>(x), const_cast<float*>(w),
                         const_cast<float*>(b), y};
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,    CUDNN_TYPE_INT64,     4, uids));
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR,  4, ptrs));
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,     CUDNN_TYPE_VOID_PTR,  1, &workspace));
      ALLEN_CUDNN_CHECK(cudnnBackendFinalize(vpack));
      ALLEN_CUDNN_CHECK(cudnnBackendExecute(handle, m_exec_plan, vpack));
      cudnnBackendDestroyDescriptor(vpack);
    }

#else
    void create(cudnnHandle_t, std::array<int,4>, std::array<int,4>,
                std::array<int,2> = {0,0}, std::array<int,2> = {1,1},
                std::array<int,2> = {1,1}) {}
    bool is_created() const { return false; }
    size_t workspace_bytes() const { return 0; }
    void ensure_thread_local_workspace() const {}
    void execute(cudnnHandle_t, const float*, const float*, const float*, float*) const {}
#endif
  };

} // namespace Allen::CuDNN
