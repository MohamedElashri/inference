#include "PVFinderUNet.cuh"
#include "PVFinderUNetKernels.cuh"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_unet::pvfinder_unet_t)

namespace pvfinder_unet {

static constexpr int B_EVENTS_MAX = 20;
static constexpr int N_CHUNK_INTERVALS = B_EVENTS_MAX * N_INTERVALS;

#ifdef ALLEN_CUDNN_BACKEND_CUDA
// ---------------------------------------------------------------------------
// Process-level global descriptor set.
// Created exactly once via s_init_flag — shared read-only across all threads.
// Shapes are compile-time constants so no synchronisation is needed after init.
// ---------------------------------------------------------------------------
struct GlobalDescriptors {
    // CBR layers: cuDNN conv followed by a fused bias+ReLU elementwise pass
    // (BN folded into weights/bias at init time, see fold_bn lambda). The
    // conv output still makes one DRAM round trip between the two — see
    // rcbn1_fused below for the true single-pass alternative (rcbn1 only).
    Allen::CuDNN::ConvDescriptors rcbn1;    // Conv(8→16,  k=25, pad=12)
    Allen::CuDNN::ConvDescriptors rcbn2;    // Conv(16→16, k=7,  pad=3)
    Allen::CuDNN::ConvDescriptors rcbn3;    // Conv(16→16, k=5,  pad=2)
    Allen::CuDNN::ConvDescriptors up1_c;   // Conv(16→16, k=5,  pad=2) after ConvTranspose
    Allen::CuDNN::ConvDescriptors up2_c;   // Conv(16→16, k=5,  pad=2)
    // Non-CBR paths: plain conv (no BN/ReLU fusion).
    Allen::CuDNN::ConvDescriptors oint_half;// Conv(16→16, k=5,  pad=2) — two halves
    Allen::CuDNN::ConvDescriptors outc;     // Conv(16→1,  k=5,  pad=2)

    // Optional true single-pass Conv+Bias+ReLU for rcbn1 (opt-in via
    // use_fused_cbr, FP32 only). Not all GPUs/cuDNN versions expose an engine
    // for this op-graph shape, so creation is attempted and may fail —
    // rcbn1_fused_available records whether it's safe to use.
    Allen::CuDNN::ConvBiasReluGraph rcbn1_fused;
    bool rcbn1_fused_available = false;

    // BN-folded weights and biases for each CBR layer (device pointers, owned here).
    float* rcbn1_w_f = nullptr; float* rcbn1_b_f = nullptr;
    float* rcbn2_w_f = nullptr; float* rcbn2_b_f = nullptr;
    float* rcbn3_w_f = nullptr; float* rcbn3_b_f = nullptr;
    float* up1c_w_f  = nullptr; float* up1c_b_f  = nullptr;
    float* up2c_w_f  = nullptr; float* up2c_b_f  = nullptr;

    // Phase M: FP16 CBR descriptors and weights (CUDNN_DATA_HALF, BN-folded at init).
    Allen::CuDNN::ConvDescriptors rcbn1_h, rcbn2_h, rcbn3_h, up1c_h, up2c_h;
    __half* rcbn1_w_h = nullptr; __half* rcbn1_b_h = nullptr;
    __half* rcbn2_w_h = nullptr; __half* rcbn2_b_h = nullptr;
    __half* rcbn3_w_h = nullptr; __half* rcbn3_b_h = nullptr;
    __half* up1c_w_h  = nullptr; __half* up1c_b_h  = nullptr;
    __half* up2c_w_h  = nullptr; __half* up2c_b_h  = nullptr;

    // FP16 activation pool: contiguous allocation, partitioned per-layer.
    // Offsets: ncw, x1, x2, x3, up1, cat2, up2 (cat2 also reused for up2_c output).
    __half* fp16_pool = nullptr;
    __half* fp16_ncw  = nullptr;  // [N, N_BATCH_CHANNELS, W_IN]
    __half* fp16_x1   = nullptr;  // [N, N_FEAT, W_IN]        — rcbn1 skip preserved
    __half* fp16_x2   = nullptr;  // [N, N_FEAT, W_HALF]
    __half* fp16_x3   = nullptr;  // [N, N_FEAT, W_QTR]
    __half* fp16_up1  = nullptr;  // [N, N_FEAT, W_HALF]
    __half* fp16_cat2 = nullptr;  // [N, 2*N_FEAT, W_HALF]    — also up2_c output
    __half* fp16_up2  = nullptr;  // [N, N_FEAT, W_IN]

    // ConvTranspose descriptors (filter + conv only; tensor descs are local in operator())
    cudnnFilterDescriptor_t       filter_up1_t = nullptr;
    cudnnConvolutionDescriptor_t  conv_up1_t   = nullptr;
    cudnnFilterDescriptor_t       filter_up2_t = nullptr;
    cudnnConvolutionDescriptor_t  conv_up2_t   = nullptr;

    // ConvTranspose algorithm + workspace (selected by cudnnGetConvolutionBackwardDataAlgorithm_v7)
    cudnnConvolutionBwdDataAlgo_t  algo_up1_t   = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdDataAlgo_t  algo_up2_t   = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    void*                          ws_up1_t     = nullptr;
    void*                          ws_up2_t     = nullptr;
    size_t                         ws_up1_bytes = 0;
    size_t                         ws_up2_bytes = 0;

    // Skip-connection ablation ("add"/"none" modes): slimmed ConvTranspose2 with
    // N_FEAT (not 2*N_FEAT) input channels, used when the up1/x2 merge is either
    // an element-wise add or dropped entirely. The first N_FEAT*N_FEAT*2 floats
    // of w_up2t_w (loaded for the full 2*N_FEAT-in filter) are already exactly
    // the [K=N_FEAT, C=N_FEAT, 1, 2] slice we need (row-major [K][C][1][2]
    // layout — the first N_FEAT of 2*N_FEAT "K" slices), so no new weight
    // buffer is allocated; only new descriptors + algorithm/workspace.
    cudnnFilterDescriptor_t       filter_up2_t_slim = nullptr;
    cudnnConvolutionDescriptor_t  conv_up2_t_slim   = nullptr;
    cudnnConvolutionBwdDataAlgo_t algo_up2_t_slim   = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    void*                         ws_up2_t_slim     = nullptr;
    size_t                        ws_up2_bytes_slim = 0;
};

static GlobalDescriptors s_desc;
static std::once_flag    s_init_flag;
static std::once_flag    s_desc_init_flag;

// Serializes the one-time (per-thread) CUDA graph capture sequence in
// get_or_capture_cuda_graph / get_or_capture_cuda_graph_fp16 across ALL
// threads. Each thread's captured graph/exec/scratch-pool is still fully
// independent afterward (thread_local, not shared) -- this mutex only
// prevents multiple threads from being INSIDE cudaStreamBeginCapture /
// cudaStreamEndCapture (plus the workspace pre-warming and scratch-pool
// cudaMalloc calls immediately around it) at the same wall-clock moment.
// Evidence for needing this: a CUDNN_STATUS_BAD_PARAM crash was observed on
// a fresh process's very first repetition, with multiple threads' "scratch
// pool allocated" log lines interleaved right at the crash -- consistent
// with many threads racing to capture for the first time simultaneously at
// startup. Costs a one-time, short serialization at startup only; steady-
// state replay (the actual hot path) is unaffected since captured threads
// never re-enter this block.
static std::mutex s_graph_capture_mutex;

// ---------------------------------------------------------------------------
// Thread-local ConvTranspose tensor descriptors.
// Shapes are compile-time constants (N, N_FEAT, W_QTR/W_HALF/W_IN never
// change), so each OS thread creates its set exactly once — lazily, on first
// use — and reuses it for the thread's lifetime, mirroring the idiom used by
// Allen::CuDNN::get_thread_local_handle (CuDNNHandle.h): null-check-then-create,
// never explicitly destroyed (relies on process teardown, same as that handle).
// A graph captured against these descriptors (see CUDA-graph path) depends on
// them staying alive and unchanged for as long as the graph is replayed, so
// destroying them per-call (as before) is no longer an option once graphs are
// in play — this cache is a hard prerequisite, not just an optimization.
// ---------------------------------------------------------------------------
struct ConvTransposeTensorDescs {
    cudnnTensorDescriptor_t td_up1_in      = nullptr;
    cudnnTensorDescriptor_t td_up1_out     = nullptr;
    cudnnTensorDescriptor_t td_up2_in      = nullptr;
    cudnnTensorDescriptor_t td_up2_out     = nullptr;
    cudnnTensorDescriptor_t td_up2_in_slim = nullptr;
};

static const ConvTransposeTensorDescs& get_thread_local_conv_transpose_descs()
{
    thread_local ConvTransposeTensorDescs descs;
    if (descs.td_up1_in == nullptr) {
        constexpr int N = N_CHUNK_INTERVALS;
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up1_in));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up1_out));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up2_in));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up2_out));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up2_in_slim));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            descs.td_up1_in,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_QTR));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            descs.td_up1_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_HALF));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            descs.td_up2_in,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT*2, 1, W_HALF));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            descs.td_up2_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_IN));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            descs.td_up2_in_slim, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT, 1, W_HALF));
    }
    return descs;
}

// ---------------------------------------------------------------------------
// CUDA graph scratch pool (Part 2: graph capture, FP32/concat only).
//
// Allen's SingleAlloc memory manager runs a full free/reserve cycle for the
// WHOLE sequence's arguments before every repetition (MemoryManager.cuh), so
// no data<ArgumentTag>(arguments) pointer -- not the input, not the output,
// not the internal scratch buffers -- is stable across operator() calls. A
// captured CUDA graph cannot bake in those pointers. Instead, the graph's
// internal nodes operate purely on this fixed, raw-cudaMalloc'd pool (mirrors
// dev_unet_x1_t/x2/x3/up1/cat2 sizes exactly, taken from set_arguments_size()
// below -- note x3 is W_IN-wide, matching its `logits` alias use, NOT the
// W_QTR width its maxpool producer writes). Only the two shuttle-kernel nodes
// (copy real ncw -> pool.ncw_in, copy pool.x1 (=oint) -> real kde) touch
// Allen-managed pointers, and their arguments are patched per replay via
// cudaGraphExecKernelNodeSetParams.
//
// MUST be thread_local, not a shared global like s_desc.fp16_pool: many OS
// threads (one per Allen Stream, per -t N) call operator() on this same
// shared algorithm instance concurrently, each on its own stream. A shared
// pool would let concurrent threads' chunk pipelines corrupt each other's
// activations -- the same class of bug the fp16_pool global risks today.
// Never freed (same lifetime pattern as the thread_local handle/descriptors).
// ---------------------------------------------------------------------------
struct GraphScratchPool {
    float* ncw_in  = nullptr;  // [N, N_BATCH_CHANNELS, W_IN]
    float* x1      = nullptr;  // [N, N_FEAT, W_IN]      -- set_size<dev_unet_x1_t>
    float* x2      = nullptr;  // [N, N_FEAT, W_HALF]    -- set_size<dev_unet_x2_t>
    float* x3      = nullptr;  // [N, N_FEAT, W_IN]      -- set_size<dev_unet_x3_t> (logits alias width)
    float* up1     = nullptr;  // [N, N_FEAT, W_HALF]    -- set_size<dev_unet_up1_t>
    float* cat2    = nullptr;  // [N, N_FEAT, 2*W_HALF]  -- set_size<dev_unet_cat2_t>
    float* kde_out = nullptr;  // [N, W_IN]
};

static const GraphScratchPool& get_thread_local_graph_scratch_pool()
{
    thread_local GraphScratchPool pool;
    if (pool.ncw_in == nullptr) {
        constexpr int N = N_CHUNK_INTERVALS;
        const size_t sz_ncw_in  = (size_t)N * N_BATCH_CHANNELS * W_IN;
        const size_t sz_x1      = (size_t)N * N_FEAT * W_IN;
        const size_t sz_x2      = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_x3      = (size_t)N * N_FEAT * W_IN;
        const size_t sz_up1     = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_cat2    = (size_t)N * N_FEAT * 2 * W_HALF;
        const size_t sz_kde_out = (size_t)N * W_IN;
        cudaMalloc(&pool.ncw_in,  sz_ncw_in  * sizeof(float));
        cudaMalloc(&pool.x1,      sz_x1      * sizeof(float));
        cudaMalloc(&pool.x2,      sz_x2      * sizeof(float));
        cudaMalloc(&pool.x3,      sz_x3      * sizeof(float));
        cudaMalloc(&pool.up1,     sz_up1     * sizeof(float));
        cudaMalloc(&pool.cat2,    sz_cat2    * sizeof(float));
        cudaMalloc(&pool.kde_out, sz_kde_out * sizeof(float));
        const size_t total_bytes =
            (sz_ncw_in + sz_x1 + sz_x2 + sz_x3 + sz_up1 + sz_cat2 + sz_kde_out) * sizeof(float);
        printf("[pvfinder_unet] CUDA-graph scratch pool allocated: %.2f MB (thread_local, "
               "outside Allen's memory manager -- not reflected in -m budget)\n",
               total_bytes / (1024.0 * 1024.0));
    }
    return pool;
}

// ---------------------------------------------------------------------------
// CUDA graph scratch pool -- FP16 counterpart of GraphScratchPool, same
// thread_local/never-freed rules apply (see comment above GraphScratchPool).
// Mirrors s_desc.fp16_pool's layout/sizes exactly, but is NOT that shared
// global -- each thread gets its own copy, avoiding the same class of
// multi-thread race s_desc.fp16_pool has in the eager FP16 path (not fixed
// here; out of scope, flagged separately). FP32-side buffers needed at the
// FP16 path's boundaries (x1, x3/logits, up2/cat2, kde_out shuttle) reuse the
// existing GraphScratchPool rather than duplicating them.
// ---------------------------------------------------------------------------
struct GraphScratchPoolFP16 {
    __half* ncw  = nullptr;  // [N, N_BATCH_CHANNELS, W_IN]
    __half* x1   = nullptr;  // [N, N_FEAT, W_IN]
    __half* x2   = nullptr;  // [N, N_FEAT, W_HALF]
    __half* x3   = nullptr;  // [N, N_FEAT, W_QTR]
    __half* up1  = nullptr;  // [N, N_FEAT, W_HALF]
    __half* cat2 = nullptr;  // [N, N_FEAT, 2*W_HALF] -- also up2_c output
    __half* up2  = nullptr;  // [N, N_FEAT, W_IN]      -- reused as general scratch
};

static const GraphScratchPoolFP16& get_thread_local_graph_scratch_pool_fp16()
{
    thread_local GraphScratchPoolFP16 pool;
    if (pool.ncw == nullptr) {
        constexpr int N = N_CHUNK_INTERVALS;
        const size_t sz_ncw  = (size_t)N * N_BATCH_CHANNELS * W_IN;
        const size_t sz_x1   = (size_t)N * N_FEAT * W_IN;
        const size_t sz_x2   = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_x3   = (size_t)N * N_FEAT * W_QTR;
        const size_t sz_up1  = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_cat2 = (size_t)N * N_FEAT * 2 * W_HALF;
        const size_t sz_up2  = (size_t)N * N_FEAT * W_IN;
        cudaMalloc(&pool.ncw,  sz_ncw  * sizeof(__half));
        cudaMalloc(&pool.x1,   sz_x1   * sizeof(__half));
        cudaMalloc(&pool.x2,   sz_x2   * sizeof(__half));
        cudaMalloc(&pool.x3,   sz_x3   * sizeof(__half));
        cudaMalloc(&pool.up1,  sz_up1  * sizeof(__half));
        cudaMalloc(&pool.cat2, sz_cat2 * sizeof(__half));
        cudaMalloc(&pool.up2,  sz_up2  * sizeof(__half));
        const size_t total_bytes =
            (sz_ncw + sz_x1 + sz_x2 + sz_x3 + sz_up1 + sz_cat2 + sz_up2) * sizeof(__half);
        printf("[pvfinder_unet] CUDA-graph FP16 scratch pool allocated: %.2f MB (thread_local)\n",
               total_bytes / (1024.0 * 1024.0));
    }
    return pool;
}

// Weight blob: device pointers per layer (filled in init(), used in operator()).
struct WeightBlob {
    const float* w_rcbn1_w;  const float* w_rcbn1_b;
    const float* w_rcbn1_gamma; const float* w_rcbn1_beta;
    const float* w_rcbn1_mean;  const float* w_rcbn1_var;
    float rcbn1_eps;

    const float* w_rcbn2_w;  const float* w_rcbn2_b;
    const float* w_rcbn2_gamma; const float* w_rcbn2_beta;
    const float* w_rcbn2_mean;  const float* w_rcbn2_var;
    float rcbn2_eps;

    const float* w_rcbn3_w;  const float* w_rcbn3_b;
    const float* w_rcbn3_gamma; const float* w_rcbn3_beta;
    const float* w_rcbn3_mean;  const float* w_rcbn3_var;
    float rcbn3_eps;

    const float* w_up1t_w;   const float* w_up1t_b;
    const float* w_up1c_w;   const float* w_up1c_b;
    const float* w_up1c_gamma; const float* w_up1c_beta;
    const float* w_up1c_mean;  const float* w_up1c_var;
    float up1c_eps;

    const float* w_up2t_w;   const float* w_up2t_b;
    const float* w_up2c_w;   const float* w_up2c_b;
    const float* w_up2c_gamma; const float* w_up2c_beta;
    const float* w_up2c_mean;  const float* w_up2c_var;
    float up2c_eps;

    const float* w_oint_a_w;
    const float* w_oint_b_w;
    const float* w_oint_b;
    const float* w_outc_w;   const float* w_outc_b;
};
static WeightBlob s_wb {};
static bool s_wb_loaded = false;

static void init_global_descriptors(cudnnHandle_t handle, const WeightBlob& wb, size_t fwd_ws_budget_bytes)
{
    constexpr int N = N_CHUNK_INTERVALS;

    // Helper: allocate device buffer for fused weights/bias and launch the
    // BN-folding kernel.
    // scale[k] = gamma[k]/sqrt(var[k]+eps), w_f[k,...]=scale[k]*w[k,...],
    // b_f[k] = scale[k]*(b[k]-mean[k])+beta[k].
    auto fold_bn = [](const float* w, const float* b,
                      const float* gamma, const float* beta,
                      const float* mean,  const float* var, float eps,
                      int K, int CxHxW,
                      float*& w_f, float*& b_f,
                      cudaStream_t stream)
    {
        cudaMalloc(&w_f, (size_t)K * CxHxW * sizeof(float));
        cudaMalloc(&b_f, (size_t)K * sizeof(float));
        fold_bn_into_conv_kernel<<<K, dim3(256), 0, stream>>>(
            w_f, b_f, w, b, gamma, beta, mean, var, eps, K, CxHxW);
    };

    // Helper: convert FP32 BN-folded weights to FP16 (for Phase M Tensor Core path).
    auto to_half = [](const float* w_f, const float* b_f, int K, int CxHxW,
                      __half*& w_h, __half*& b_h, cudaStream_t stream) {
        size_t wn = (size_t)K * CxHxW;
        cudaMalloc(&w_h, wn * sizeof(__half));
        cudaMalloc(&b_h, (size_t)K * sizeof(__half));
        int threads = 256;
        f32_to_f16_kernel<<<((wn + threads - 1) / threads), threads, 0, stream>>>(w_h, w_f, (int)wn);
        f32_to_f16_kernel<<<((K  + threads - 1) / threads), threads, 0, stream>>>(b_h, b_f, K);
    };

    // Fold BN into conv weights for each CBR layer, then build the fused graph.
    // All fold kernels run on stream 0 (init is single-threaded here).
    fold_bn(wb.w_rcbn1_w, wb.w_rcbn1_b,
            wb.w_rcbn1_gamma, wb.w_rcbn1_beta,
            wb.w_rcbn1_mean,  wb.w_rcbn1_var, wb.rcbn1_eps,
            N_FEAT, N_BATCH_CHANNELS * 25,
            s_desc.rcbn1_w_f, s_desc.rcbn1_b_f, 0);
    cudaDeviceSynchronize();
    s_desc.rcbn1.create(handle, {N, N_BATCH_CHANNELS, 1, W_IN}, {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0,12},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(s_desc.rcbn1_w_f, s_desc.rcbn1_b_f, N_FEAT, N_BATCH_CHANNELS * 25,
            s_desc.rcbn1_w_h, s_desc.rcbn1_b_h, 0);
    cudaDeviceSynchronize();
    s_desc.rcbn1_h.create(handle, {N, N_BATCH_CHANNELS, 1, W_IN}, {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0,12},
                          {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);

    // Optional true single-pass Conv+Bias+ReLU for rcbn1 (Phase 1 of
    // optimization_plan.md). Uses the same BN-folded weights/bias as rcbn1
    // above. Creation can fail if no engine supports this op-graph shape on
    // the current GPU/cuDNN version -- caught here so the process still
    // starts and use_fused_cbr silently falls back to the two-pass path.
    try {
        s_desc.rcbn1_fused.create(handle, {N, N_BATCH_CHANNELS, 1, W_IN}, {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0, 12});
        s_desc.rcbn1_fused_available = true;
    } catch (const std::exception& e) {
        s_desc.rcbn1_fused_available = false;
        fprintf(stderr, "[pvfinder_unet] ConvBiasReluGraph unavailable for rcbn1 (%s); "
                "use_fused_cbr will fall back to the two-pass conv+bias/ReLU path.\n", e.what());
    }

    fold_bn(wb.w_rcbn2_w, wb.w_rcbn2_b,
            wb.w_rcbn2_gamma, wb.w_rcbn2_beta,
            wb.w_rcbn2_mean,  wb.w_rcbn2_var, wb.rcbn2_eps,
            N_FEAT, N_FEAT * 7,
            s_desc.rcbn2_w_f, s_desc.rcbn2_b_f, 0);
    cudaDeviceSynchronize();
    s_desc.rcbn2.create(handle, {N, N_FEAT, 1, W_IN},  {N_FEAT, N_FEAT, 1,  7}, {0, 3},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(s_desc.rcbn2_w_f, s_desc.rcbn2_b_f, N_FEAT, N_FEAT * 7,
            s_desc.rcbn2_w_h, s_desc.rcbn2_b_h, 0);
    cudaDeviceSynchronize();
    s_desc.rcbn2_h.create(handle, {N, N_FEAT, 1, W_IN}, {N_FEAT, N_FEAT, 1, 7}, {0,3},
                          {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);

    fold_bn(wb.w_rcbn3_w, wb.w_rcbn3_b,
            wb.w_rcbn3_gamma, wb.w_rcbn3_beta,
            wb.w_rcbn3_mean,  wb.w_rcbn3_var, wb.rcbn3_eps,
            N_FEAT, N_FEAT * 5,
            s_desc.rcbn3_w_f, s_desc.rcbn3_b_f, 0);
    cudaDeviceSynchronize();
    s_desc.rcbn3.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0, 2},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(s_desc.rcbn3_w_f, s_desc.rcbn3_b_f, N_FEAT, N_FEAT * 5,
            s_desc.rcbn3_w_h, s_desc.rcbn3_b_h, 0);
    cudaDeviceSynchronize();
    s_desc.rcbn3_h.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0,2},
                          {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);

    fold_bn(wb.w_up1c_w, wb.w_up1c_b,
            wb.w_up1c_gamma, wb.w_up1c_beta,
            wb.w_up1c_mean,  wb.w_up1c_var, wb.up1c_eps,
            N_FEAT, N_FEAT * 5,
            s_desc.up1c_w_f, s_desc.up1c_b_f, 0);
    cudaDeviceSynchronize();
    s_desc.up1_c.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0, 2},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(s_desc.up1c_w_f, s_desc.up1c_b_f, N_FEAT, N_FEAT * 5,
            s_desc.up1c_w_h, s_desc.up1c_b_h, 0);
    cudaDeviceSynchronize();
    s_desc.up1c_h.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0,2},
                         {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);

    fold_bn(wb.w_up2c_w, wb.w_up2c_b,
            wb.w_up2c_gamma, wb.w_up2c_beta,
            wb.w_up2c_mean,  wb.w_up2c_var, wb.up2c_eps,
            N_FEAT, N_FEAT * 5,
            s_desc.up2c_w_f, s_desc.up2c_b_f, 0);
    cudaDeviceSynchronize();
    s_desc.up2_c.create(handle, {N, N_FEAT, 1, W_IN},  {N_FEAT, N_FEAT, 1, 5}, {0, 2},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(s_desc.up2c_w_f, s_desc.up2c_b_f, N_FEAT, N_FEAT * 5,
            s_desc.up2c_w_h, s_desc.up2c_b_h, 0);
    cudaDeviceSynchronize();
    s_desc.up2c_h.create(handle, {N, N_FEAT, 1, W_IN}, {N_FEAT, N_FEAT, 1, 5}, {0,2},
                         {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);

    // Allocate dedicated FP16 activation pool for Phase M benchmark.
    // Layout: ncw | x1 | x2 | x3 | up1 | cat2 | up2
    // cat2 slot is also reused as up2_c output (same element count).
    {
        size_t sz_ncw  = (size_t)N * N_BATCH_CHANNELS * W_IN;
        size_t sz_x1   = (size_t)N * N_FEAT * W_IN;
        size_t sz_x2   = (size_t)N * N_FEAT * W_HALF;
        size_t sz_x3   = (size_t)N * N_FEAT * W_QTR;
        size_t sz_up1  = (size_t)N * N_FEAT * W_HALF;
        size_t sz_cat2 = (size_t)N * N_FEAT * 2 * W_HALF;
        size_t sz_up2  = (size_t)N * N_FEAT * W_IN;
        size_t total   = sz_ncw + sz_x1 + sz_x2 + sz_x3 + sz_up1 + sz_cat2 + sz_up2;
        cudaMalloc(&s_desc.fp16_pool, total * sizeof(__half));
        __half* p = s_desc.fp16_pool;
        s_desc.fp16_ncw  = p; p += sz_ncw;
        s_desc.fp16_x1   = p; p += sz_x1;
        s_desc.fp16_x2   = p; p += sz_x2;
        s_desc.fp16_x3   = p; p += sz_x3;
        s_desc.fp16_up1  = p; p += sz_up1;
        s_desc.fp16_cat2 = p; p += sz_cat2;
        s_desc.fp16_up2  = p;
    }

    // Non-CBR paths: plain conv, same pinned-IMPLICIT_GEMM-by-default ConvDescriptors.
    s_desc.oint_half.create(handle, {N, N_FEAT, 1, W_IN},  {N_FEAT, N_FEAT, 1, 5}, {0, 2},
                            {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    s_desc.outc.create(     handle, {N, N_FEAT, 1, W_IN},  {1,      N_FEAT, 1, 5}, {0, 2},
                            {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);

    // One-time diagnostic: confirms the header's design intent -- "IMPLICIT_GEMM
    // pinned everywhere -> zero workspace" -- actually holds when
    // fwd_algo_ws_budget_bytes==0 (the default). ConvDescriptors::create() pins
    // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM unconditionally in that case (no
    // algorithm search); an earlier cudnnFindConvolutionForwardAlgorithmEx-based
    // selection experiment was tried and reverted (see CuDNNDescriptors.h's class
    // comment) because it regressed under real -t16 memory-bandwidth contention
    // despite looking faster in isolation. Phase 2 (optimization_plan.md) reopens
    // this with a workspace-budgeted heuristic search instead of an unrestricted
    // one -- active only when fwd_algo_ws_budget_bytes is set nonzero below. Each
    // ConvDescriptors' workspace is thread_local (not a single shared buffer), so
    // even a nonzero size here would not be a cross-thread race. Logged once so
    // the actual algorithm/workspace state is visible rather than assumed. Routed
    // to stderr (unbuffered) and flushed explicitly: stdout is fully buffered once
    // redirected to a file, so on an abrupt abort() (e.g. the std::terminate path
    // from ALLEN_CUDNN_CHECK) any unflushed printf content -- including this
    // diagnostic -- is silently lost, leaving no evidence of which
    // algorithm/workspace size was actually selected for a crashing run.
    fprintf(stderr, "[pvfinder_unet] fwd_algo_ws_budget_bytes=%zu\n", fwd_ws_budget_bytes);
    fprintf(stderr, "[pvfinder_unet] ConvDescriptors workspace bytes: rcbn1=%zu rcbn2=%zu rcbn3=%zu "
           "up1_c=%zu up2_c=%zu oint_half=%zu outc=%zu | algo ids: rcbn1=%d rcbn2=%d rcbn3=%d "
           "up1_c=%d up2_c=%d oint_half=%d outc=%d\n",
           s_desc.rcbn1.workspace_bytes(), s_desc.rcbn2.workspace_bytes(), s_desc.rcbn3.workspace_bytes(),
           s_desc.up1_c.workspace_bytes(), s_desc.up2_c.workspace_bytes(),
           s_desc.oint_half.workspace_bytes(), s_desc.outc.workspace_bytes(),
           s_desc.rcbn1.algo_id(), s_desc.rcbn2.algo_id(), s_desc.rcbn3.algo_id(),
           s_desc.up1_c.algo_id(), s_desc.up2_c.algo_id(),
           s_desc.oint_half.algo_id(), s_desc.outc.algo_id());
    // Same diagnostic, FP16 descriptors: rcbn1_h in particular picks an algorithm
    // with a ~3.85MB workspace (vs ~1.6KB for every other descriptor here) --
    // large enough that lazily allocating it from inside the hot per-chunk loop,
    // with many threads racing to do so on their first call, caused a real
    // (if rare) CUDNN_STATUS_BAD_PARAM crash under sustained -t16 load. Fixed by
    // pre-warming every descriptor's thread-local workspace once per thread
    // before the chunk loop (see operator()), for both eager and graph paths.
    // Algo ids logged here too, even though algorithm selection is pinned
    // (IMPLICIT_GEMM, no search) -- if that ever changes, this is what would
    // reveal which algorithm/workspace size is actually in effect.
    fprintf(stderr, "[pvfinder_unet] FP16 ConvDescriptors workspace bytes: rcbn1_h=%zu rcbn2_h=%zu "
           "rcbn3_h=%zu up1c_h=%zu up2c_h=%zu | algo ids: rcbn1_h=%d rcbn2_h=%d rcbn3_h=%d "
           "up1c_h=%d up2c_h=%d\n",
           s_desc.rcbn1_h.workspace_bytes(), s_desc.rcbn2_h.workspace_bytes(), s_desc.rcbn3_h.workspace_bytes(),
           s_desc.up1c_h.workspace_bytes(), s_desc.up2c_h.workspace_bytes(),
           s_desc.rcbn1_h.algo_id(), s_desc.rcbn2_h.algo_id(), s_desc.rcbn3_h.algo_id(),
           s_desc.up1c_h.algo_id(), s_desc.up2c_h.algo_id());
    // Phase 1 (optimization_plan.md): whether the fused-graph rcbn1 path is
    // usable on this GPU/cuDNN version at all (see the try/catch around its
    // create() call above).
    fprintf(stderr, "[pvfinder_unet] rcbn1 ConvBiasReluGraph available: %s (workspace bytes: %zu)\n",
           s_desc.rcbn1_fused_available ? "yes" : "no", s_desc.rcbn1_fused.workspace_bytes());
    fflush(stderr);

    // ConvTranspose: filter + conv descriptors only (shared, read-only after init).
    // Tensor descriptors for in/out are thread_local (see
    // get_thread_local_conv_transpose_descs()), not created here.
    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&s_desc.filter_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        s_desc.filter_up1_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&s_desc.conv_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        s_desc.conv_up1_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(s_desc.conv_up1_t, CUDNN_TENSOR_OP_MATH));

    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&s_desc.filter_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        s_desc.filter_up2_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT*2, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&s_desc.conv_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        s_desc.conv_up2_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(s_desc.conv_up2_t, CUDNN_TENSOR_OP_MATH));

    // Algorithm sweep for ConvTranspose (cudnnConvolutionBackwardData)
    // Uses temporary tensor descriptors for the query — not stored, as operator()
    // creates per-call thread-local descriptors.
    static constexpr size_t kBwdBudget  = 64ul * 1024 * 1024;
    static constexpr int    kBwdMaxAlgo = 8;

    // up1_t: dy=[N, N_FEAT, 1, W_QTR], dx=[N, N_FEAT, 1, W_HALF]
    {
        cudnnTensorDescriptor_t dy_desc, dx_desc;
        cudnnCreateTensorDescriptor(&dy_desc);
        cudnnCreateTensorDescriptor(&dx_desc);
        cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_QTR);
        cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_HALF);
        int returned = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perf[kBwdMaxAlgo];
        if (cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle, s_desc.filter_up1_t, dy_desc, s_desc.conv_up1_t, dx_desc,
                kBwdMaxAlgo, &returned, perf) == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= kBwdBudget) {
                    s_desc.algo_up1_t   = perf[i].algo;
                    s_desc.ws_up1_bytes = perf[i].memory;
                    if (s_desc.ws_up1_bytes > 0) cudaMalloc(&s_desc.ws_up1_t, s_desc.ws_up1_bytes);
                    break;
                }
            }
        }
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
    }

    // up2_t: dy=[N, N_FEAT*2, 1, W_HALF], dx=[N, N_FEAT, 1, W_IN]
    {
        cudnnTensorDescriptor_t dy_desc, dx_desc;
        cudnnCreateTensorDescriptor(&dy_desc);
        cudnnCreateTensorDescriptor(&dx_desc);
        cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT*2, 1, W_HALF);
        cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_IN);
        int returned = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perf[kBwdMaxAlgo];
        if (cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle, s_desc.filter_up2_t, dy_desc, s_desc.conv_up2_t, dx_desc,
                kBwdMaxAlgo, &returned, perf) == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= kBwdBudget) {
                    s_desc.algo_up2_t   = perf[i].algo;
                    s_desc.ws_up2_bytes = perf[i].memory;
                    if (s_desc.ws_up2_bytes > 0) cudaMalloc(&s_desc.ws_up2_t, s_desc.ws_up2_bytes);
                    break;
                }
            }
        }
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
    }

    // up2_t_slim: skip-ablation variant ("add"/"none" modes) with N_FEAT (not
    // 2*N_FEAT) input channels — dy=[N, N_FEAT, 1, W_HALF], dx=[N, N_FEAT, 1, W_IN].
    // Reuses w_up2t_w/w_up2t_b as-is (see field comment in GlobalDescriptors).
    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&s_desc.filter_up2_t_slim));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        s_desc.filter_up2_t_slim, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&s_desc.conv_up2_t_slim));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        s_desc.conv_up2_t_slim, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(s_desc.conv_up2_t_slim, CUDNN_TENSOR_OP_MATH));
    {
        cudnnTensorDescriptor_t dy_desc, dx_desc;
        cudnnCreateTensorDescriptor(&dy_desc);
        cudnnCreateTensorDescriptor(&dx_desc);
        cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT, 1, W_HALF);
        cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT, 1, W_IN);
        int returned = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perf[kBwdMaxAlgo];
        if (cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle, s_desc.filter_up2_t_slim, dy_desc, s_desc.conv_up2_t_slim, dx_desc,
                kBwdMaxAlgo, &returned, perf) == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= kBwdBudget) {
                    s_desc.algo_up2_t_slim   = perf[i].algo;
                    s_desc.ws_up2_bytes_slim = perf[i].memory;
                    if (s_desc.ws_up2_bytes_slim > 0) cudaMalloc(&s_desc.ws_up2_t_slim, s_desc.ws_up2_bytes_slim);
                    break;
                }
            }
        }
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
    }
}

// ---------------------------------------------------------------------------
// Binary weight file parser
// Layout (from convert_cnn_weights.py):
//   uint32  magic = 0xCAFE0001
//   conv(8→64,k=25):  int32 in,out,k | float[out*in*k] weights | float[out] bias
//   bn(64):           int32 features | float eps | float[f] gamma,beta,mean,var
//   ... repeated for rcbn2, rcbn3
//   convT(64→64,k=2,s=2): int32 in,out,k,stride | float[in*out*k] | float[out]
//   conv+bn for up1.convbnrelu, up2.convbnrelu
//   conv(128→64,k=5): out_intermediate
//   conv(64→1,k=5):   outc
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Read a block of floats from a host buffer at a given byte offset.
// Returns updated offset.
// ---------------------------------------------------------------------------
static size_t read_float_block(
    const std::vector<char>& buf, size_t offset,
    float* dst, size_t count)
{
    std::memcpy(dst, buf.data() + offset, count * sizeof(float));
    return offset + count * sizeof(float);
}

static size_t read_int32(const std::vector<char>& buf, size_t offset, int& v)
{
    std::memcpy(&v, buf.data() + offset, 4);
    return offset + 4;
}

static size_t read_float32(const std::vector<char>& buf, size_t offset, float& v)
{
    std::memcpy(&v, buf.data() + offset, 4);
    return offset + 4;
}

// ---------------------------------------------------------------------------
// Load weights from binary file into WeightRegistry and fill WeightBlob.
// ---------------------------------------------------------------------------
static WeightBlob load_weights(const std::string& path)
{
    // Read entire file into host buffer
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("PVFinderUNet: cannot open weight file: " + path);
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> buf(file_size);
    fread(buf.data(), 1, file_size, fp);
    fclose(fp);

    size_t off = 0;

    // Magic
    uint32_t magic = 0;
    std::memcpy(&magic, buf.data(), 4);
    off += 4;
    if (magic != 0xCAFE0001u) {
        throw std::runtime_error("PVFinderUNet: bad magic in weight file");
    }

    auto& reg = Allen::CuDNN::WeightRegistry::instance();
    WeightBlob wb {};

    // Helper lambdas
    auto load_conv = [&](const std::string& key_w, const std::string& key_b,
                          const float*& out_w, const float*& out_b) {
        int in_c, out_c, k;
        off = read_int32(buf, off, in_c);
        off = read_int32(buf, off, out_c);
        off = read_int32(buf, off, k);
        size_t wcount = (size_t)out_c * in_c * k;
        // Load weight block
        std::vector<float> w_host(wcount);
        off = read_float_block(buf, off, w_host.data(), wcount);
        if (!reg.contains(key_w)) reg.load_from_buffer(key_w, w_host.data(), wcount * sizeof(float));
        out_w = reg.get<float>(key_w);
        // Load bias block
        std::vector<float> b_host(out_c);
        off = read_float_block(buf, off, b_host.data(), out_c);
        if (!reg.contains(key_b)) reg.load_from_buffer(key_b, b_host.data(), out_c * sizeof(float));
        out_b = reg.get<float>(key_b);
    };

    auto load_bn = [&](const std::string& prefix,
                        const float*& gamma, const float*& beta,
                        const float*& mean,  const float*& var, float& eps) {
        int features;
        off = read_int32(buf, off, features);
        off = read_float32(buf, off, eps);
        std::vector<float> g(features), b(features), m(features), v(features);
        off = read_float_block(buf, off, g.data(), features);
        off = read_float_block(buf, off, b.data(), features);
        off = read_float_block(buf, off, m.data(), features);
        off = read_float_block(buf, off, v.data(), features);
        auto ld = [&](const std::string& k, const std::vector<float>& d, const float*& ptr) {
            if (!reg.contains(k)) reg.load_from_buffer(k, d.data(), d.size() * sizeof(float));
            ptr = reg.get<float>(k);
        };
        ld(prefix + ".gamma", g, gamma);
        ld(prefix + ".beta",  b, beta);
        ld(prefix + ".mean",  m, mean);
        ld(prefix + ".var",   v, var);
    };

    auto load_convt = [&](const std::string& key_w, const std::string& key_b,
                           const float*& out_w, const float*& out_b) {
        int in_c, out_c, k, stride;
        off = read_int32(buf, off, in_c);
        off = read_int32(buf, off, out_c);
        off = read_int32(buf, off, k);
        off = read_int32(buf, off, stride);
        size_t wcount = (size_t)in_c * out_c * k;
        std::vector<float> w_host(wcount);
        off = read_float_block(buf, off, w_host.data(), wcount);
        if (!reg.contains(key_w)) reg.load_from_buffer(key_w, w_host.data(), wcount * sizeof(float));
        out_w = reg.get<float>(key_w);
        std::vector<float> b_host(out_c);
        off = read_float_block(buf, off, b_host.data(), out_c);
        if (!reg.contains(key_b)) reg.load_from_buffer(key_b, b_host.data(), out_c * sizeof(float));
        out_b = reg.get<float>(key_b);
    };

    // rcbn1
    load_conv("rcbn1.w", "rcbn1.b", wb.w_rcbn1_w, wb.w_rcbn1_b);
    load_bn("rcbn1.bn", wb.w_rcbn1_gamma, wb.w_rcbn1_beta, wb.w_rcbn1_mean, wb.w_rcbn1_var, wb.rcbn1_eps);
    // rcbn2
    load_conv("rcbn2.w", "rcbn2.b", wb.w_rcbn2_w, wb.w_rcbn2_b);
    load_bn("rcbn2.bn", wb.w_rcbn2_gamma, wb.w_rcbn2_beta, wb.w_rcbn2_mean, wb.w_rcbn2_var, wb.rcbn2_eps);
    // rcbn3
    load_conv("rcbn3.w", "rcbn3.b", wb.w_rcbn3_w, wb.w_rcbn3_b);
    load_bn("rcbn3.bn", wb.w_rcbn3_gamma, wb.w_rcbn3_beta, wb.w_rcbn3_mean, wb.w_rcbn3_var, wb.rcbn3_eps);
    // up1: ConvTranspose + ConvBNrelu
    load_convt("up1t.w", "up1t.b", wb.w_up1t_w, wb.w_up1t_b);
    load_conv("up1c.w", "up1c.b", wb.w_up1c_w, wb.w_up1c_b);
    load_bn("up1c.bn", wb.w_up1c_gamma, wb.w_up1c_beta, wb.w_up1c_mean, wb.w_up1c_var, wb.up1c_eps);
    // up2: ConvTranspose + ConvBNrelu
    load_convt("up2t.w", "up2t.b", wb.w_up2t_w, wb.w_up2t_b);
    load_conv("up2c.w", "up2c.b", wb.w_up2c_w, wb.w_up2c_b);
    load_bn("up2c.bn", wb.w_up2c_gamma, wb.w_up2c_beta, wb.w_up2c_mean, wb.w_up2c_var, wb.up2c_eps);
    // out_intermediate: Conv(128→64, k=5).  Load full weight [64,128,5] then split into
    // two halves [64,64,5] for channels 0:64 and 64:128 so we can avoid cat_out.
    {
        int in_c, out_c, k;
        off = read_int32(buf, off, in_c);   // 128
        off = read_int32(buf, off, out_c);  // 64
        off = read_int32(buf, off, k);      // 5
        // Full weight: [out_c, in_c, k] = [64, 128, 5]
        size_t full = (size_t)out_c * in_c * k;
        std::vector<float> w_full(full);
        off = read_float_block(buf, off, w_full.data(), full);
        // Split: each output filter has in_c=128 weights per kernel position.
        // Layout (NCHW flattened): [out_c][in_c][k] — split on in_c dimension.
        size_t half_elems = (size_t)out_c * (in_c / 2) * k;  // 64*64*5
        std::vector<float> w_a(half_elems), w_b(half_elems);
        for (int oc = 0; oc < out_c; ++oc) {
            for (int ic = 0; ic < in_c; ++ic) {
                for (int ki = 0; ki < k; ++ki) {
                    float val = w_full[((size_t)oc * in_c + ic) * k + ki];
                    size_t dst_idx = ((size_t)oc * (in_c/2) + (ic % (in_c/2))) * k + ki;
                    if (ic < in_c / 2) w_a[dst_idx] = val;
                    else               w_b[dst_idx] = val;
                }
            }
        }
        if (!reg.contains("oint.a.w")) reg.load_from_buffer("oint.a.w", w_a.data(), half_elems * sizeof(float));
        if (!reg.contains("oint.b.w")) reg.load_from_buffer("oint.b.w", w_b.data(), half_elems * sizeof(float));
        wb.w_oint_a_w = reg.get<float>("oint.a.w");
        wb.w_oint_b_w = reg.get<float>("oint.b.w");
        // Bias [out_c]
        std::vector<float> bias(out_c);
        off = read_float_block(buf, off, bias.data(), out_c);
        if (!reg.contains("oint.b")) reg.load_from_buffer("oint.b", bias.data(), out_c * sizeof(float));
        wb.w_oint_b = reg.get<float>("oint.b");
    }
    // outc
    load_conv("outc.w", "outc.b", wb.w_outc_w, wb.w_outc_b);

    return wb;
}

#endif // ALLEN_CUDNN_BACKEND_CUDA

// ---------------------------------------------------------------------------
// init(): load weights + init global descriptors — both done exactly once.
// ---------------------------------------------------------------------------
void pvfinder_unet_t::init()
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    if (m_init_done) return;
    std::call_once(s_init_flag, [this]() {
        s_wb = load_weights(m_weight_file.value());
        s_wb_loaded = true;
    });
    m_init_done = true;
#endif
}

// ---------------------------------------------------------------------------
// set_arguments_size
// Workspace for cuDNN is owned per ConvDescriptors (cudaMalloc'd at init).
// dev_unet_conv_ws_t is kept at 1 float as Allen requires a non-zero allocation.
// ---------------------------------------------------------------------------
void pvfinder_unet_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    const unsigned padded_events = ((n_events + B_EVENTS_MAX - 1) / B_EVENTS_MAX) * B_EVENTS_MAX;
    constexpr unsigned N_batch = N_CHUNK_INTERVALS;

    set_size<dev_unet_x1_t>   (arguments, N_batch * N_FEAT * W_IN);       
    set_size<dev_unet_x2_t>   (arguments, N_batch * N_FEAT * W_HALF);     
    set_size<dev_unet_x3_t>   (arguments, N_batch * N_FEAT * W_IN);       
    set_size<dev_unet_up1_t>  (arguments, N_batch * N_FEAT * W_HALF);     
    set_size<dev_unet_cat2_t> (arguments, N_batch * N_FEAT * 2 * W_HALF); 
    set_size<dev_unet_conv_ws_t>(arguments, 1u);                    
    set_size<dev_pvfinder_kde_output_t>(arguments, padded_events * N_INTERVALS * W_IN);
}

// ---------------------------------------------------------------------------
// Per-layer helpers
// ---------------------------------------------------------------------------

#ifdef ALLEN_CUDNN_BACKEND_CUDA
// Conv1d + bias + ReLU (BN folded into w_fused/b_fused at init).
// Uses cudnnConvolutionForward with the Phase K timed algorithm, then
// launches bias_relu_kernel (no BN math — BN already in w_fused/b_fused).
void pvfinder_unet_t::run_convbnrelu(
    const Allen::CuDNN::ConvDescriptors& desc,
    const float* input, float* output,
    const float* w_fused, const float* b_fused,
    int K, int W_out, int N,
    cudnnHandle_t handle,
    const dim3& block, const Allen::Context& ctx) const
{
    desc.forward(handle, 1.f, 0.f, input, w_fused, output);
    launch_bias_relu(output, b_fused, K, W_out, N, block, ctx);
}

// FP16 variant: Tensor Core conv (CUDNN_DATA_HALF desc) + FP16 bias+relu kernel.
void pvfinder_unet_t::run_convbnrelu_half(
    const Allen::CuDNN::ConvDescriptors& desc,
    const __half* input, __half* output,
    const __half* w_fused, const __half* b_fused,
    int K, int W_out, int N,
    cudnnHandle_t handle,
    const dim3& block, const Allen::Context& ctx) const
{
    desc.forward_half(handle, 1.f, 0.f, input, w_fused, output);
    launch_bias_relu_half(output, b_fused, K, W_out, N, block, ctx);
}

// Conv1d only (no BN/ReLU).
void pvfinder_unet_t::run_conv(
    const Allen::CuDNN::ConvDescriptors& desc,
    const float* input,  float* output,
    const float* w_ptr,  const float* bias_ptr,
    int N, int C_out, int W,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle,
    float beta_val) const
{
    const float alpha = 1.f;
    desc.forward(handle, alpha, beta_val, input, w_ptr, output);
    if (bias_ptr && beta_val == 0.f)
        launch_bias_add(output, bias_ptr, C_out, W, N, block, ctx);
}

// ConvTranspose1d via cudnnConvolutionBackwardData. Algorithm selected at init time.
void pvfinder_unet_t::run_conv_transpose(
    const float* input, float* output,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t in_desc,
    cudnnTensorDescriptor_t out_desc,
    const float* w_ptr, const float* bias_ptr,
    int N, int C_out, int W_out,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle,
    cudnnConvolutionBwdDataAlgo_t algo,
    void* workspace, size_t ws_bytes) const
{
    const float alpha = 1.f, beta = 0.f;
    ALLEN_CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle, &alpha,
        filter_desc, w_ptr,
        in_desc,     input,
        conv_desc,
        algo,
        workspace, ws_bytes,
        &beta,
        out_desc, output));
    launch_bias_add(output, bias_ptr, C_out, W_out, N, block, ctx);
}

// ---------------------------------------------------------------------------
// CUDA graph capture (thread_local, lazy). See GraphScratchPool comment above
// for why this is needed and why it must be per-thread. Captures the exact
// FP32/concat op sequence from operator() below, operating on pool buffers
// instead of Allen argument pointers, then instantiates a replayable graph
// exec. The two shuttle-kernel nodes' handles are captured live via
// cudaStreamGetCaptureInfo immediately after each is launched during capture
// -- not looked up post-hoc by kernel function pointer, which would be
// ambiguous since both shuttle copies reuse the same squeeze_copy_kernel.
// ---------------------------------------------------------------------------
void pvfinder_unet_t::get_or_capture_cuda_graph(
    cudnnHandle_t handle,
    const dim3& block,
    const Allen::Context& ctx,
    const float* seed_ncw,
    float* seed_kde,
    cudaGraphExec_t& out_exec,
    cudaGraphNode_t& out_copy_in_node,
    cudaGraphNode_t& out_copy_out_node) const
{
    thread_local cudaGraphExec_t tl_exec     = nullptr;
    thread_local cudaGraphNode_t tl_copy_in  = nullptr;
    thread_local cudaGraphNode_t tl_copy_out = nullptr;
    // The template cudaGraph_t is intentionally kept alive (never destroyed) for
    // the thread's lifetime: cudaGraphExecKernelNodeSetParams patches tl_exec by
    // referencing node handles owned by THIS template graph, and destroying it
    // would invalidate those handles even though tl_exec itself would keep working
    // for plain (unpatched) relaunches. Must stay alive as long as tl_exec is used.
    thread_local cudaGraph_t tl_template_graph = nullptr;

    if (tl_exec == nullptr) {
        // See s_graph_capture_mutex's declaration comment: serializes first-time
        // capture across threads. Only this thread's own tl_exec is set inside;
        // once set, this thread never re-enters this block or takes the lock again.
        std::lock_guard<std::mutex> capture_lock(s_graph_capture_mutex);
        constexpr int N = N_CHUNK_INTERVALS;
        const GraphScratchPool& pool = get_thread_local_graph_scratch_pool();
        const ConvTransposeTensorDescs& td = get_thread_local_conv_transpose_descs();
        cudaStream_t stream = ctx.stream();

        // Aliases within the pool, mirroring the eager path's proven-safe
        // x1/x2/x3/up1/cat2 aliasing scheme (up2=cat2, oint=x1, logits=x3).
        float* g_x1 = pool.x1; float* g_x2 = pool.x2; float* g_x3 = pool.x3;
        float* g_up1 = pool.up1; float* g_cat2 = pool.cat2;
        float* g_up2 = g_cat2; float* g_oint = g_x1; float* g_logits = g_x3;

        const unsigned total_in  = (unsigned)N * N_BATCH_CHANNELS * W_IN;
        const unsigned total_out = (unsigned)N * W_IN;
        const dim3 grid_in ((total_in  + block.x - 1) / block.x);
        const dim3 grid_out((total_out + block.x - 1) / block.x);

        // Pre-warm every ConvDescriptors' thread-local workspace on this thread
        // BEFORE capture begins: growing a thread-local workspace (cudaMalloc) is
        // not something that can happen mid-capture, so every descriptor forward()
        // will touch during the captured sequence below must already be sized.
        s_desc.rcbn1.ensure_thread_local_workspace();
        s_desc.rcbn2.ensure_thread_local_workspace();
        s_desc.rcbn3.ensure_thread_local_workspace();
        s_desc.up1_c.ensure_thread_local_workspace();
        s_desc.up2_c.ensure_thread_local_workspace();
        s_desc.oint_half.ensure_thread_local_workspace();
        s_desc.outc.ensure_thread_local_workspace();

        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

        // Copy-in shuttle: seed_ncw is only valid/meaningful at this exact
        // capture-time call; every subsequent replay patches this node's
        // source pointer to that call's real ncw slice.
        squeeze_copy_kernel<<<grid_in, block, 0, stream>>>(seed_ncw, pool.ncw_in, (int)total_in);
        {
            cudaStreamCaptureStatus status;
            const cudaGraphNode_t* deps = nullptr;
            size_t num_deps = 0;
            cudaCheck(cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr, &deps, &num_deps));
            tl_copy_in = deps[num_deps - 1];
        }

        run_convbnrelu(s_desc.rcbn1, pool.ncw_in, g_x1,  s_desc.rcbn1_w_f, s_desc.rcbn1_b_f, N_FEAT, W_IN,   N, handle, block, ctx);
        run_convbnrelu(s_desc.rcbn2, g_x1,  g_up2, s_desc.rcbn2_w_f, s_desc.rcbn2_b_f, N_FEAT, W_IN,   N, handle, block, ctx);
        launch_maxpool(g_up2, g_x2, N, N_FEAT, W_IN, block, ctx);

        run_convbnrelu(s_desc.rcbn3, g_x2, g_up2, s_desc.rcbn3_w_f, s_desc.rcbn3_b_f, N_FEAT, W_HALF, N, handle, block, ctx);
        launch_maxpool(g_up2, g_x3, N, N_FEAT, W_HALF, block, ctx);

        run_conv_transpose(g_x3, g_up2,
            s_desc.filter_up1_t, s_desc.conv_up1_t, td.td_up1_in, td.td_up1_out,
            s_wb.w_up1t_w, s_wb.w_up1t_b,
            N, N_FEAT, W_HALF, block, ctx, handle,
            s_desc.algo_up1_t, s_desc.ws_up1_t, s_desc.ws_up1_bytes);
        run_convbnrelu(s_desc.up1_c, g_up2, g_up1, s_desc.up1c_w_f, s_desc.up1c_b_f, N_FEAT, W_HALF, N, handle, block, ctx);

        launch_concat(g_up1, g_x2, g_cat2, N, N_FEAT, N_FEAT, W_HALF, block, ctx);
        run_conv_transpose(g_cat2, g_logits,
            s_desc.filter_up2_t, s_desc.conv_up2_t, td.td_up2_in, td.td_up2_out,
            s_wb.w_up2t_w, s_wb.w_up2t_b,
            N, N_FEAT, W_IN, block, ctx, handle,
            s_desc.algo_up2_t, s_desc.ws_up2_t, s_desc.ws_up2_bytes);
        run_convbnrelu(s_desc.up2_c, g_logits, g_up2, s_desc.up2c_w_f, s_desc.up2c_b_f, N_FEAT, W_IN, N, handle, block, ctx);

        run_conv(s_desc.oint_half, g_x1, g_logits,
            s_wb.w_oint_b_w, nullptr,
            N, N_FEAT, W_IN, block, ctx, handle, 0.f);
        run_conv(s_desc.oint_half, g_up2, g_logits,
            s_wb.w_oint_a_w, nullptr,
            N, N_FEAT, W_IN, block, ctx, handle, 1.f);
        launch_bias_add(g_logits, s_wb.w_oint_b, N_FEAT, W_IN, N, block, ctx);
        run_conv(s_desc.outc, g_logits, g_oint,
            s_wb.w_outc_w, s_wb.w_outc_b,
            N, 1, W_IN, block, ctx, handle, 0.f);

        launch_softplus_scale(g_oint, KDE_SCALE, N * W_IN, block, ctx);

        // Copy-out shuttle: seed_kde is only valid/meaningful at this exact
        // capture-time call; every subsequent replay patches this node's
        // destination pointer to that call's real kde slice.
        squeeze_copy_kernel<<<grid_out, block, 0, stream>>>(g_oint, seed_kde, (int)total_out);
        {
            cudaStreamCaptureStatus status;
            const cudaGraphNode_t* deps = nullptr;
            size_t num_deps = 0;
            cudaCheck(cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr, &deps, &num_deps));
            tl_copy_out = deps[num_deps - 1];
        }

        cudaCheck(cudaStreamEndCapture(stream, &tl_template_graph));
        cudaCheck(cudaGraphInstantiate(&tl_exec, tl_template_graph, 0));
        // tl_template_graph is deliberately NOT destroyed -- see declaration comment.

        printf("[pvfinder_unet] CUDA graph captured (thread_local, FP32+concat pipeline)\n");
    }

    out_exec          = tl_exec;
    out_copy_in_node  = tl_copy_in;
    out_copy_out_node = tl_copy_out;
}

// ---------------------------------------------------------------------------
// CUDA graph capture, FP16 counterpart of get_or_capture_cuda_graph. Same
// idiom (thread_local exec/nodes/template graph, capture-once, live node
// handles via cudaStreamGetCaptureInfo). Reuses the existing FP32
// GraphScratchPool for the FP32-side buffers this sequence needs (x1, x3,
// cat2/up2 -- same proven-safe aliasing as the eager FP32/FP16 paths) and a
// new GraphScratchPoolFP16 for the FP16-side ones. The leading f32_to_f16
// conversion doubles as the input shuttle -- no separate copy kernel needed.
// ---------------------------------------------------------------------------
void pvfinder_unet_t::get_or_capture_cuda_graph_fp16(
    cudnnHandle_t handle,
    const dim3& block,
    const Allen::Context& ctx,
    const float* seed_ncw,
    float* seed_kde,
    cudaGraphExec_t& out_exec,
    cudaGraphNode_t& out_copy_in_node,
    cudaGraphNode_t& out_copy_out_node) const
{
    thread_local cudaGraphExec_t tl_exec     = nullptr;
    thread_local cudaGraphNode_t tl_copy_in  = nullptr;
    thread_local cudaGraphNode_t tl_copy_out = nullptr;
    thread_local cudaGraph_t tl_template_graph = nullptr;

    if (tl_exec == nullptr) {
        // See s_graph_capture_mutex's declaration comment (shared with the FP32
        // capture function above -- one global mutex, both capture paths).
        std::lock_guard<std::mutex> capture_lock(s_graph_capture_mutex);
        constexpr int N = N_CHUNK_INTERVALS;
        const GraphScratchPool& pool32 = get_thread_local_graph_scratch_pool();
        const GraphScratchPoolFP16& pool16 = get_thread_local_graph_scratch_pool_fp16();
        const ConvTransposeTensorDescs& td = get_thread_local_conv_transpose_descs();
        cudaStream_t stream = ctx.stream();

        // FP32-side aliases (reusing the existing FP32 pool -- same proven-safe
        // aliasing scheme as the eager path: up2=cat2, oint=x1, logits=x3).
        float* g_x1 = pool32.x1; float* g_x3 = pool32.x3; float* g_cat2 = pool32.cat2;
        float* g_up2 = g_cat2; float* g_oint = g_x1; float* g_logits = g_x3;

        const unsigned total_in  = (unsigned)N * N_BATCH_CHANNELS * W_IN;
        const unsigned total_out = (unsigned)N * W_IN;
        const dim3 grid_out((total_out + block.x - 1) / block.x);

        // Pre-warm every ConvDescriptors' thread-local workspace this sequence
        // touches, BEFORE capture begins (see get_or_capture_cuda_graph's
        // comment for why). Cheap no-op if already warmed (e.g. by the FP32
        // graph on this thread).
        s_desc.rcbn1_h.ensure_thread_local_workspace();
        s_desc.rcbn2_h.ensure_thread_local_workspace();
        s_desc.rcbn3_h.ensure_thread_local_workspace();
        s_desc.up1c_h.ensure_thread_local_workspace();
        s_desc.up2c_h.ensure_thread_local_workspace();
        s_desc.oint_half.ensure_thread_local_workspace();
        s_desc.outc.ensure_thread_local_workspace();

        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

        // Copy-in shuttle IS the leading f32->f16 conversion: seed_ncw is only
        // valid/meaningful at this exact capture-time call; every subsequent
        // replay patches this node's src argument to that call's real ncw slice.
        launch_f32_to_f16(pool16.ncw, seed_ncw, (int)total_in, block, ctx);
        {
            cudaStreamCaptureStatus status;
            const cudaGraphNode_t* deps = nullptr;
            size_t num_deps = 0;
            cudaCheck(cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr, &deps, &num_deps));
            tl_copy_in = deps[num_deps - 1];
        }

        run_convbnrelu_half(s_desc.rcbn1_h, pool16.ncw, pool16.x1,
            s_desc.rcbn1_w_h, s_desc.rcbn1_b_h, N_FEAT, W_IN, N, handle, block, ctx);
        run_convbnrelu_half(s_desc.rcbn2_h, pool16.x1, pool16.up2,
            s_desc.rcbn2_w_h, s_desc.rcbn2_b_h, N_FEAT, W_IN, N, handle, block, ctx);
        launch_maxpool_half(pool16.up2, pool16.x2, N, N_FEAT, W_IN, block, ctx);

        run_convbnrelu_half(s_desc.rcbn3_h, pool16.x2, pool16.up2,
            s_desc.rcbn3_w_h, s_desc.rcbn3_b_h, N_FEAT, W_HALF, N, handle, block, ctx);
        launch_maxpool_half(pool16.up2, pool16.x3, N, N_FEAT, W_HALF, block, ctx);

        // ConvTranspose1: needs FP32. Convert fp16 x3 -> g_x3.
        launch_f16_to_f32(g_x3, pool16.x3, N * N_FEAT * W_QTR, block, ctx);
        run_conv_transpose(g_x3, g_up2,
            s_desc.filter_up1_t, s_desc.conv_up1_t, td.td_up1_in, td.td_up1_out,
            s_wb.w_up1t_w, s_wb.w_up1t_b,
            N, N_FEAT, W_HALF, block, ctx, handle,
            s_desc.algo_up1_t, s_desc.ws_up1_t, s_desc.ws_up1_bytes);

        // up1_c FP16: convert FP32 g_up2 -> fp16 pool16.up2, then conv.
        launch_f32_to_f16(pool16.up2, g_up2, N * N_FEAT * W_HALF, block, ctx);
        run_convbnrelu_half(s_desc.up1c_h, pool16.up2, pool16.up1,
            s_desc.up1c_w_h, s_desc.up1c_b_h, N_FEAT, W_HALF, N, handle, block, ctx);

        launch_concat_half(pool16.up1, pool16.x2, pool16.cat2, N, N_FEAT, N_FEAT, W_HALF, block, ctx);

        // ConvTranspose2: needs FP32. Convert fp16 cat2 -> g_cat2.
        launch_f16_to_f32(g_cat2, pool16.cat2, N * N_FEAT * 2 * W_HALF, block, ctx);
        run_conv_transpose(g_cat2, g_logits,
            s_desc.filter_up2_t, s_desc.conv_up2_t, td.td_up2_in, td.td_up2_out,
            s_wb.w_up2t_w, s_wb.w_up2t_b,
            N, N_FEAT, W_IN, block, ctx, handle,
            s_desc.algo_up2_t, s_desc.ws_up2_t, s_desc.ws_up2_bytes);

        // up2_c FP16: convert FP32 g_logits -> fp16 pool16.up2, conv -> pool16.cat2 (reused).
        launch_f32_to_f16(pool16.up2, g_logits, N * N_FEAT * W_IN, block, ctx);
        run_convbnrelu_half(s_desc.up2c_h, pool16.up2, pool16.cat2,
            s_desc.up2c_w_h, s_desc.up2c_b_h, N_FEAT, W_IN, N, handle, block, ctx);

        // Output stage: FP32. Convert fp16 x1 (rcbn1 skip) -> g_x1, fp16 cat2 -> g_up2.
        launch_f16_to_f32(g_x1, pool16.x1, N * N_FEAT * W_IN, block, ctx);
        launch_f16_to_f32(g_up2, pool16.cat2, N * N_FEAT * W_IN, block, ctx);

        run_conv(s_desc.oint_half, g_x1, g_logits,
            s_wb.w_oint_b_w, nullptr,
            N, N_FEAT, W_IN, block, ctx, handle, 0.f);
        run_conv(s_desc.oint_half, g_up2, g_logits,
            s_wb.w_oint_a_w, nullptr,
            N, N_FEAT, W_IN, block, ctx, handle, 1.f);
        launch_bias_add(g_logits, s_wb.w_oint_b, N_FEAT, W_IN, N, block, ctx);
        run_conv(s_desc.outc, g_logits, g_oint,
            s_wb.w_outc_w, s_wb.w_outc_b,
            N, 1, W_IN, block, ctx, handle, 0.f);

        launch_softplus_scale(g_oint, KDE_SCALE, N * W_IN, block, ctx);

        // Copy-out shuttle: same squeeze_copy_kernel pattern as the FP32 graph
        // (output stage is FP32-only here too).
        squeeze_copy_kernel<<<grid_out, block, 0, stream>>>(g_oint, seed_kde, (int)total_out);
        {
            cudaStreamCaptureStatus status;
            const cudaGraphNode_t* deps = nullptr;
            size_t num_deps = 0;
            cudaCheck(cudaStreamGetCaptureInfo(stream, &status, nullptr, nullptr, &deps, &num_deps));
            tl_copy_out = deps[num_deps - 1];
        }

        cudaCheck(cudaStreamEndCapture(stream, &tl_template_graph));
        cudaCheck(cudaGraphInstantiate(&tl_exec, tl_template_graph, 0));
        // tl_template_graph is deliberately NOT destroyed -- see the FP32
        // get_or_capture_cuda_graph's declaration comment for why.

        printf("[pvfinder_unet] CUDA graph captured (thread_local, FP16+concat pipeline)\n");
    }

    out_exec          = tl_exec;
    out_copy_in_node  = tl_copy_in;
    out_copy_out_node = tl_copy_out;
}
#endif // ALLEN_CUDNN_BACKEND_CUDA

// ---------------------------------------------------------------------------
// operator(): full UNet forward pass
// ---------------------------------------------------------------------------
void pvfinder_unet_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    if (!m_init_done || !s_wb_loaded) return;

    const unsigned n_events = first<host_number_of_events_t>(arguments);

    // One thread_local handle per OS thread — created lazily, routed to this stream.
    cudnnHandle_t handle = Allen::CuDNN::get_thread_local_handle(context.stream());

    // Descriptor creation needs a live handle (for algorithm selection), so it runs
    // here on first operator() call rather than in init(). fwd_ws_budget_bytes is
    // read from whichever call happens to win the call_once race -- fine here since
    // it's a benchmark-only property set once at process/config level, not expected
    // to vary between concurrent operator() calls.
    const size_t fwd_ws_budget_bytes = m_fwd_algo_ws_budget_bytes.value();
    std::call_once(s_desc_init_flag, [handle, fwd_ws_budget_bytes]() {
        init_global_descriptors(handle, s_wb, fwd_ws_budget_bytes);
    });

    const dim3 block = m_block_dim;
    constexpr int N = N_CHUNK_INTERVALS;  // batch size = 20 * 40 = 800

    // Scratch buffers (fixed size, reused each event iteration)
    float* x1   = data<dev_unet_x1_t>(arguments);
    float* x2   = data<dev_unet_x2_t>(arguments);
    float* x3   = data<dev_unet_x3_t>(arguments);
    float* up1  = data<dev_unet_up1_t>(arguments);
    float* cat2 = data<dev_unet_cat2_t>(arguments);

    // Buffer aliases (liveness-proven safe)
    float* up2   = cat2;  // cat2[N,128,50] and up2[N,64,100] have same element count
    float* oint  = x1;    // x1 skip consumed before oint written
    float* logits = x3;   // x3 consumed after maxpool; reused as logits

    constexpr unsigned ncw_stride = N_INTERVALS * N_BATCH_CHANNELS * W_IN;
    constexpr unsigned kde_stride = N_INTERVALS * W_IN;

    // ConvTranspose tensor descriptors — thread_local, created once per OS thread
    // and reused for its lifetime (shapes are compile-time constants). See
    // get_thread_local_conv_transpose_descs() for the lazy-init idiom.
    const ConvTransposeTensorDescs& td = get_thread_local_conv_transpose_descs();
    cudnnTensorDescriptor_t td_up1_in      = td.td_up1_in;
    cudnnTensorDescriptor_t td_up1_out     = td.td_up1_out;
    cudnnTensorDescriptor_t td_up2_in      = td.td_up2_in;
    cudnnTensorDescriptor_t td_up2_out     = td.td_up2_out;
    cudnnTensorDescriptor_t td_up2_in_slim = td.td_up2_in_slim;

    const float* ncw_base = data<dev_pvfinder_interval_features_t>(arguments);
    float*       kde_base = data<dev_pvfinder_kde_output_t>(arguments);

    const unsigned padded_events = ((n_events + B_EVENTS_MAX - 1) / B_EVENTS_MAX) * B_EVENTS_MAX;

    const bool use_fp16 = m_use_fp16.value();
    // Skip-connection ablation ("concat" | "add" | "none") — applies to the FP32
    // path only; the FP16 Tensor Core path below always uses "concat".
    const std::string& skip_mode = m_skip_mode.value();
    // CUDA graph path: re-evaluated fresh every call (not just cached from first
    // capture) — each captured graph only ever represents one fixed topology
    // (FP32+concat or FP16+concat), so any call outside skip_mode=="concat" simply
    // takes the eager branch below instead, by construction, with no risk of
    // replaying a stale/mismatched graph.
    const bool graph_eligible = skip_mode == "concat" && m_use_cuda_graph.value();
    // Phase 1 (optimization_plan.md): true single-pass Conv+Bias+ReLU for rcbn1,
    // eager FP32 path only. FP16 has no fused-graph variant (see m_use_fused_cbr's
    // doc comment), so this is simply ignored whenever use_fp16=true.
    const bool use_fused_cbr = m_use_fused_cbr.value() && s_desc.rcbn1_fused_available;
    // Phase 3 (optimization_plan.md): hand-written fused rcbn3, eager FP32 path only.
    const bool use_fused_rcbn3 = m_use_fused_rcbn3.value();
    const bool use_graph_fp32 = graph_eligible && !use_fp16;
    const bool use_graph_fp16 = graph_eligible && use_fp16;

    // Pre-warm every ConvDescriptors' thread-local workspace once per thread,
    // for BOTH the eager and graph paths (previously only done before graph
    // capture). Without this, each thread's first-ever eager call lazily
    // cudaMalloc's its workspace from inside the hot per-chunk loop -- for
    // rcbn1_h that's ~3.85 MB (vs. ~1.6 KB for every other descriptor), and
    // with many threads starting their first call at close to the same time,
    // that turned into a real, if rare, CUDNN_STATUS_BAD_PARAM crash under
    // sustained -t16 load. Warming here decouples allocation from the hot
    // path and from other threads' concurrent first-touch timing entirely.
    {
        thread_local bool tl_warmed = false;
        if (!tl_warmed) {
            s_desc.rcbn1.ensure_thread_local_workspace();
            s_desc.rcbn2.ensure_thread_local_workspace();
            s_desc.rcbn3.ensure_thread_local_workspace();
            s_desc.up1_c.ensure_thread_local_workspace();
            s_desc.up2_c.ensure_thread_local_workspace();
            s_desc.oint_half.ensure_thread_local_workspace();
            s_desc.outc.ensure_thread_local_workspace();
            s_desc.rcbn1_h.ensure_thread_local_workspace();
            s_desc.rcbn2_h.ensure_thread_local_workspace();
            s_desc.rcbn3_h.ensure_thread_local_workspace();
            s_desc.up1c_h.ensure_thread_local_workspace();
            s_desc.up2c_h.ensure_thread_local_workspace();
            if (s_desc.rcbn1_fused_available) s_desc.rcbn1_fused.ensure_thread_local_workspace();
            tl_warmed = true;
        }
    }

    // FP16 pool pointers (only used when use_fp16 is true).
    //
    // Deliberately NOT s_desc.fp16_* here: those are a single process-wide
    // shared allocation (see GlobalDescriptors::fp16_pool) -- with many OS
    // threads (one per Allen Stream, per -t N) all running the eager FP16
    // path concurrently, every thread would read/write the EXACT SAME
    // fp16_ncw/x1/x2/x3/up1/cat2/up2 addresses simultaneously with zero
    // synchronization. This is very likely the root cause of the intermittent
    // CUDNN_STATUS_BAD_PARAM crashes seen under sustained -t16 load: this bug
    // predates today's session (flagged earlier as a known-but-unfixed issue
    // when the FP16 CUDA graph path was built, since that new code correctly
    // used its own thread_local pool instead and has run crash-free). Reusing
    // that same thread_local GraphScratchPoolFP16 here for the eager path
    // fixes it the same way, using infrastructure already built and validated.
    const GraphScratchPoolFP16* fp16_pool_tl = use_fp16 ? &get_thread_local_graph_scratch_pool_fp16() : nullptr;
    __half* fp16_ncw  = use_fp16 ? fp16_pool_tl->ncw  : nullptr;
    __half* fp16_x1   = use_fp16 ? fp16_pool_tl->x1   : nullptr;
    __half* fp16_x2   = use_fp16 ? fp16_pool_tl->x2   : nullptr;
    __half* fp16_x3   = use_fp16 ? fp16_pool_tl->x3   : nullptr;
    __half* fp16_up1  = use_fp16 ? fp16_pool_tl->up1  : nullptr;
    __half* fp16_cat2 = use_fp16 ? fp16_pool_tl->cat2 : nullptr;
    __half* fp16_up2  = use_fp16 ? fp16_pool_tl->up2  : nullptr;

    for (unsigned chunk_start = 0; chunk_start < padded_events; chunk_start += B_EVENTS_MAX) {
        const float* ncw = ncw_base + chunk_start * ncw_stride;
        float*       kde = kde_base + chunk_start * kde_stride;

        if (use_graph_fp32) {
            // ---- CUDA graph path (FP32 + concat only) ----
            // Captures once (thread_local, lazy); every call after the first just
            // patches the two shuttle-kernel nodes' pointers and replays.
            cudaGraphExec_t graphExec  = nullptr;
            cudaGraphNode_t copyInNode = nullptr, copyOutNode = nullptr;
            get_or_capture_cuda_graph(handle, block, context, ncw, kde, graphExec, copyInNode, copyOutNode);
            const GraphScratchPool& pool = get_thread_local_graph_scratch_pool();

            const int total_in  = (int)(N_CHUNK_INTERVALS * N_BATCH_CHANNELS * W_IN);
            const int total_out = (int)(N_CHUNK_INTERVALS * W_IN);
            const dim3 grid_in ((unsigned(total_in)  + block.x - 1) / block.x);
            const dim3 grid_out((unsigned(total_out) + block.x - 1) / block.x);

            const float* copy_in_src = ncw;
            float*       copy_in_dst = pool.ncw_in;
            void* copy_in_args[3] = {(void*)&copy_in_src, (void*)&copy_in_dst, (void*)&total_in};
            cudaKernelNodeParams copy_in_params{};
            copy_in_params.func          = (void*)squeeze_copy_kernel;
            copy_in_params.gridDim       = grid_in;
            copy_in_params.blockDim      = block;
            copy_in_params.sharedMemBytes = 0;
            copy_in_params.kernelParams  = copy_in_args;
            copy_in_params.extra         = nullptr;
            cudaCheck(cudaGraphExecKernelNodeSetParams(graphExec, copyInNode, &copy_in_params));

            const float* copy_out_src = pool.x1;  // g_oint alias inside the captured graph
            float*       copy_out_dst = kde;
            void* copy_out_args[3] = {(void*)&copy_out_src, (void*)&copy_out_dst, (void*)&total_out};
            cudaKernelNodeParams copy_out_params{};
            copy_out_params.func          = (void*)squeeze_copy_kernel;
            copy_out_params.gridDim       = grid_out;
            copy_out_params.blockDim      = block;
            copy_out_params.sharedMemBytes = 0;
            copy_out_params.kernelParams  = copy_out_args;
            copy_out_params.extra         = nullptr;
            cudaCheck(cudaGraphExecKernelNodeSetParams(graphExec, copyOutNode, &copy_out_params));

            cudaCheck(cudaGraphLaunch(graphExec, context.stream()));
            // Graph already applies softplus_scale + copy-out into the real kde
            // buffer internally — skip the shared eager-path tail below.
            continue;
        } else if (use_graph_fp16) {
            // ---- CUDA graph path (FP16 + concat only) ----
            // Same replay pattern as the FP32 graph, but the copy-in node is the
            // leading f32_to_f16 conversion (dst fixed, src patched), not a plain
            // squeeze_copy_kernel — different kernel, different argument order.
            cudaGraphExec_t graphExec  = nullptr;
            cudaGraphNode_t copyInNode = nullptr, copyOutNode = nullptr;
            get_or_capture_cuda_graph_fp16(handle, block, context, ncw, kde, graphExec, copyInNode, copyOutNode);
            const GraphScratchPool& pool = get_thread_local_graph_scratch_pool();
            const GraphScratchPoolFP16& pool16 = get_thread_local_graph_scratch_pool_fp16();

            const int total_in  = (int)(N_CHUNK_INTERVALS * N_BATCH_CHANNELS * W_IN);
            const int total_out = (int)(N_CHUNK_INTERVALS * W_IN);
            const dim3 grid_out((unsigned(total_out) + block.x - 1) / block.x);

            // f32_to_f16_kernel(__half* dst, const float* src, int n) — dst fixed
            // (pool16.ncw), src patched to this chunk's real ncw, n fixed.
            __half*      copy_in_dst = pool16.ncw;
            const float* copy_in_src = ncw;
            void* copy_in_args[3] = {(void*)&copy_in_dst, (void*)&copy_in_src, (void*)&total_in};
            cudaKernelNodeParams copy_in_params{};
            copy_in_params.func           = (void*)f32_to_f16_kernel;
            copy_in_params.gridDim        = dim3((unsigned(total_in) + block.x - 1) / block.x);
            copy_in_params.blockDim       = block;
            copy_in_params.sharedMemBytes = 0;
            copy_in_params.kernelParams   = copy_in_args;
            copy_in_params.extra          = nullptr;
            cudaCheck(cudaGraphExecKernelNodeSetParams(graphExec, copyInNode, &copy_in_params));

            const float* copy_out_src = pool.x1;  // g_oint alias inside the captured graph
            float*       copy_out_dst = kde;
            void* copy_out_args[3] = {(void*)&copy_out_src, (void*)&copy_out_dst, (void*)&total_out};
            cudaKernelNodeParams copy_out_params{};
            copy_out_params.func          = (void*)squeeze_copy_kernel;
            copy_out_params.gridDim       = grid_out;
            copy_out_params.blockDim      = block;
            copy_out_params.sharedMemBytes = 0;
            copy_out_params.kernelParams  = copy_out_args;
            copy_out_params.extra         = nullptr;
            cudaCheck(cudaGraphExecKernelNodeSetParams(graphExec, copyOutNode, &copy_out_params));

            cudaCheck(cudaGraphLaunch(graphExec, context.stream()));
            continue;
        } else if (!use_fp16) {
            // ---- FP32 path (Phase L baseline) ----
            if (use_fused_cbr) {
                // Single-pass Conv+Bias+ReLU: no separate bias_relu_kernel launch,
                // no extra DRAM round trip on the conv output.
                s_desc.rcbn1_fused.execute(handle, ncw, s_desc.rcbn1_w_f, s_desc.rcbn1_b_f, x1);
            } else {
                run_convbnrelu(s_desc.rcbn1, ncw, x1,  s_desc.rcbn1_w_f, s_desc.rcbn1_b_f, N_FEAT, W_IN,   N, handle, block, context);
            }
            run_convbnrelu(s_desc.rcbn2, x1,  up2, s_desc.rcbn2_w_f, s_desc.rcbn2_b_f, N_FEAT, W_IN,   N, handle, block, context);
            launch_maxpool(up2, x2, N, N_FEAT, W_IN, block, context);

            if (use_fused_rcbn3) {
                // Single kernel: conv + bias + ReLU with the activation slice
                // kept in shared memory, no DRAM round trip on the raw conv output.
                launch_fused_rcbn3(x2, up2, s_desc.rcbn3_w_f, s_desc.rcbn3_b_f, N, block, context);
            } else {
                run_convbnrelu(s_desc.rcbn3, x2, up2, s_desc.rcbn3_w_f, s_desc.rcbn3_b_f, N_FEAT, W_HALF, N, handle, block, context);
            }
            launch_maxpool(up2, x3, N, N_FEAT, W_HALF, block, context);

            run_conv_transpose(x3, up2,
                s_desc.filter_up1_t, s_desc.conv_up1_t, td_up1_in, td_up1_out,
                s_wb.w_up1t_w, s_wb.w_up1t_b,
                N, N_FEAT, W_HALF, block, context, handle,
                s_desc.algo_up1_t, s_desc.ws_up1_t, s_desc.ws_up1_bytes);
            run_convbnrelu(s_desc.up1_c, up2, up1, s_desc.up1c_w_f, s_desc.up1c_b_f, N_FEAT, W_HALF, N, handle, block, context);

            // Skip 1: merge up1 (decoder) with x2 (encoder) ahead of ConvTranspose2.
            if (skip_mode == "concat") {
                launch_concat(up1, x2, cat2, N, N_FEAT, N_FEAT, W_HALF, block, context);
                run_conv_transpose(cat2, logits,
                    s_desc.filter_up2_t, s_desc.conv_up2_t, td_up2_in, td_up2_out,
                    s_wb.w_up2t_w, s_wb.w_up2t_b,
                    N, N_FEAT, W_IN, block, context, handle,
                    s_desc.algo_up2_t, s_desc.ws_up2_t, s_desc.ws_up2_bytes);
            } else {
                // "add": up1 += x2 in place, then a slim N_FEAT-in ConvTranspose.
                // "none": same slim ConvTranspose, x2 never touches up1.
                if (skip_mode == "add") {
                    launch_accumulate(up1, x2, N * N_FEAT * W_HALF, block, context);
                }
                run_conv_transpose(up1, logits,
                    s_desc.filter_up2_t_slim, s_desc.conv_up2_t_slim, td_up2_in_slim, td_up2_out,
                    s_wb.w_up2t_w, s_wb.w_up2t_b,
                    N, N_FEAT, W_IN, block, context, handle,
                    s_desc.algo_up2_t_slim, s_desc.ws_up2_t_slim, s_desc.ws_up2_bytes_slim);
            }
            run_convbnrelu(s_desc.up2_c, logits, up2, s_desc.up2c_w_f, s_desc.up2c_b_f, N_FEAT, W_IN, N, handle, block, context);

            // Skip 2: merge up2 (decoder) with x1 (encoder) ahead of the output conv.
            if (skip_mode == "concat") {
                run_conv(s_desc.oint_half, x1, logits,
                    s_wb.w_oint_b_w, nullptr,
                    N, N_FEAT, W_IN, block, context, handle, 0.f);
                run_conv(s_desc.oint_half, up2, logits,
                    s_wb.w_oint_a_w, nullptr,
                    N, N_FEAT, W_IN, block, context, handle, 1.f);
            } else {
                if (skip_mode == "add") {
                    launch_accumulate(up2, x1, N * N_FEAT * W_IN, block, context);
                }
                run_conv(s_desc.oint_half, up2, logits,
                    s_wb.w_oint_a_w, nullptr,
                    N, N_FEAT, W_IN, block, context, handle, 0.f);
            }
            launch_bias_add(logits, s_wb.w_oint_b, N_FEAT, W_IN, N, block, context);
            run_conv(s_desc.outc, logits, oint,
                s_wb.w_outc_w, s_wb.w_outc_b,
                N, 1, W_IN, block, context, handle, 0.f);
        } else {
            // ---- FP16 path (Phase M benchmark) ----
            // CBR layers run as Tensor Core FP16 convs; ConvTranspose and output
            // layers stay FP32. Explicit F32↔F16 conversions at the boundaries.

            // Encoder
            launch_f32_to_f16(fp16_ncw, ncw, N * N_BATCH_CHANNELS * W_IN, block, context);
            run_convbnrelu_half(s_desc.rcbn1_h, fp16_ncw, fp16_x1,
                s_desc.rcbn1_w_h, s_desc.rcbn1_b_h, N_FEAT, W_IN, N, handle, block, context);
            run_convbnrelu_half(s_desc.rcbn2_h, fp16_x1, fp16_up2,
                s_desc.rcbn2_w_h, s_desc.rcbn2_b_h, N_FEAT, W_IN, N, handle, block, context);
            launch_maxpool_half(fp16_up2, fp16_x2, N, N_FEAT, W_IN, block, context);

            run_convbnrelu_half(s_desc.rcbn3_h, fp16_x2, fp16_up2,
                s_desc.rcbn3_w_h, s_desc.rcbn3_b_h, N_FEAT, W_HALF, N, handle, block, context);
            launch_maxpool_half(fp16_up2, fp16_x3, N, N_FEAT, W_HALF, block, context);

            // ConvTranspose1: needs FP32. Convert fp16_x3 → x3.
            launch_f16_to_f32(x3, fp16_x3, N * N_FEAT * W_QTR, block, context);
            run_conv_transpose(x3, up2,
                s_desc.filter_up1_t, s_desc.conv_up1_t, td_up1_in, td_up1_out,
                s_wb.w_up1t_w, s_wb.w_up1t_b,
                N, N_FEAT, W_HALF, block, context, handle,
                s_desc.algo_up1_t, s_desc.ws_up1_t, s_desc.ws_up1_bytes);

            // up1_c FP16: convert FP32 up2 → fp16_up2, then conv.
            launch_f32_to_f16(fp16_up2, up2, N * N_FEAT * W_HALF, block, context);
            run_convbnrelu_half(s_desc.up1c_h, fp16_up2, fp16_up1,
                s_desc.up1c_w_h, s_desc.up1c_b_h, N_FEAT, W_HALF, N, handle, block, context);

            // Concat FP16: fp16_up1 + fp16_x2 → fp16_cat2.
            launch_concat_half(fp16_up1, fp16_x2, fp16_cat2, N, N_FEAT, N_FEAT, W_HALF, block, context);

            // ConvTranspose2: needs FP32. Convert fp16_cat2 → cat2.
            launch_f16_to_f32(cat2, fp16_cat2, N * N_FEAT * 2 * W_HALF, block, context);
            run_conv_transpose(cat2, logits,
                s_desc.filter_up2_t, s_desc.conv_up2_t, td_up2_in, td_up2_out,
                s_wb.w_up2t_w, s_wb.w_up2t_b,
                N, N_FEAT, W_IN, block, context, handle,
                s_desc.algo_up2_t, s_desc.ws_up2_t, s_desc.ws_up2_bytes);

            // up2_c FP16: convert FP32 logits → fp16_up2, conv → fp16_cat2 (reused).
            launch_f32_to_f16(fp16_up2, logits, N * N_FEAT * W_IN, block, context);
            run_convbnrelu_half(s_desc.up2c_h, fp16_up2, fp16_cat2,
                s_desc.up2c_w_h, s_desc.up2c_b_h, N_FEAT, W_IN, N, handle, block, context);

            // Output stage: FP32. Convert fp16_x1 (rcbn1 skip) → x1, fp16_cat2 → up2.
            launch_f16_to_f32(x1, fp16_x1, N * N_FEAT * W_IN, block, context);
            launch_f16_to_f32(up2, fp16_cat2, N * N_FEAT * W_IN, block, context);

            run_conv(s_desc.oint_half, x1, logits,
                s_wb.w_oint_b_w, nullptr,
                N, N_FEAT, W_IN, block, context, handle, 0.f);
            run_conv(s_desc.oint_half, up2, logits,
                s_wb.w_oint_a_w, nullptr,
                N, N_FEAT, W_IN, block, context, handle, 1.f);
            launch_bias_add(logits, s_wb.w_oint_b, N_FEAT, W_IN, N, block, context);
            run_conv(s_desc.outc, logits, oint,
                s_wb.w_outc_w, s_wb.w_outc_b,
                N, 1, W_IN, block, context, handle, 0.f);
        }

        launch_softplus_scale(oint, KDE_SCALE, N * W_IN, block, context);
        squeeze_copy_kernel<<<
            ((unsigned)(N * W_IN) + block.x - 1) / block.x, block,
            0, context.stream()>>>(oint, kde, N * W_IN);
    }

    // ConvTranspose tensor descriptors are thread_local (see
    // get_thread_local_conv_transpose_descs()) and intentionally never destroyed here.

    // Validation dump (on the m_dump_repetition-th call, when dump_dir property is
    // set; default 0 dumps the first call, matching prior behaviour). A later index
    // is needed to validate the CUDA-graph path against pointer drift -- see
    // m_dump_repetition's doc comment in PVFinderUNet.cuh.
    const unsigned this_call = m_call_count++;
    const std::string& dump_dir = m_dump_dir.value();
    if (!dump_dir.empty() && !m_dump_done && this_call == m_dump_repetition.value()) {
        cudaStreamSynchronize(context.stream());
        const unsigned ncw_elems = n_events * N_INTERVALS * N_BATCH_CHANNELS * W_IN;
        const unsigned kde_elems = n_events * N_INTERVALS * W_IN;
        std::vector<float> h_ncw(ncw_elems), h_kde(kde_elems);
        cudaMemcpy(h_ncw.data(), data<dev_pvfinder_interval_features_t>(arguments),
                   ncw_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_kde.data(), data<dev_pvfinder_kde_output_t>(arguments),
                   kde_elems * sizeof(float), cudaMemcpyDeviceToHost);
        const uint32_t magic = 0xAB1EU;
        auto write_bin = [&](const std::string& path, const float* d, unsigned n) {
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(&magic),    sizeof(magic));
            f.write(reinterpret_cast<const char*>(&n_events), sizeof(n_events));
            f.write(reinterpret_cast<const char*>(d),         n * sizeof(float));
        };
        write_bin(dump_dir + "/allen_ncw_input.bin",  h_ncw.data(), ncw_elems);
        write_bin(dump_dir + "/allen_kde_output.bin", h_kde.data(), kde_elems);
        printf("[pvfinder_unet] Validation dump written to %s (%u events)\n",
               dump_dir.c_str(), n_events);
        m_dump_done = true;
    }
#endif
}

} // namespace pvfinder_unet
