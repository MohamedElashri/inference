#include "PVFinderUNet.cuh"
#include "PVFinderUNetKernels.cuh"
#include "PVFinderUNetLowPrecision.cuh"
#include "PVFinderConvGraph.cuh"
#include "PVFinderUNetFused.cuh"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_unet::pvfinder_unet_t)

namespace pvfinder_unet {

// Events per cuDNN batch come from the unet_batch_events property. Every
// shape-dependent resource below (descriptors, scratch pools, CUDA graphs)
// takes N = unet_batch_events * N_INTERVALS, fixed for the process lifetime.

#ifdef ALLEN_CUDNN_BACKEND_CUDA
// ---------------------------------------------------------------------------
// Descriptor set of one pvfinder_unet instance (held in UNetState below).
// Created once per instance on its first operator() call, then shared
// read-only by all threads. Shapes are fixed for the process lifetime.
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
    Allen::CuDNN::ConvDescriptors oint;     // Conv(16→16, k=5,  pad=2) out_intermediate
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
    // Offsets: ncw, x1, x2, x3, up1, up2 (x1 also reused for up2_c output).
    __half* fp16_pool = nullptr;
    __half* fp16_ncw  = nullptr;  // [N, N_BATCH_CHANNELS, W_IN]
    __half* fp16_x1   = nullptr;  // [N, N_FEAT, W_IN]        — also up2_c output
    __half* fp16_x2   = nullptr;  // [N, N_FEAT, W_HALF]
    __half* fp16_x3   = nullptr;  // [N, N_FEAT, W_QTR]
    __half* fp16_up1  = nullptr;  // [N, N_FEAT, W_HALF]
    __half* fp16_up2  = nullptr;  // [N, N_FEAT, W_IN]

    // BF16 CBR descriptors and
    // weights (CUDNN_DATA_BFLOAT16, BN-folded at init) -- exact structural
    // mirror of the FP16 fields above, added instead of reusing them so the
    // already-validated FP16 path is untouched by this addition. No shared
    // bf16_pool activation buffer here, deliberately -- this phase's BF16
    // support is eager-path-only (no CUDA graph capture), and the eager
    // FP16 path itself does not use the (unused) fp16_pool either; it uses a
    // thread_local pool instead (GraphScratchPoolBF16 below), for the same
    // reason the FP16 one does (see that struct's comment).
    Allen::CuDNN::ConvDescriptors rcbn1_bf, rcbn2_bf, rcbn3_bf, up1c_bf, up2c_bf;
    // Reduced-precision output stage: out_intermediate and outc composed
    // into one convolution (OutputStage<N_FEAT> layout, see
    // PVFinderUNetLowPrecision.cuh).
    float* output_stage_params = nullptr;
    __nv_bfloat16* rcbn1_w_bf = nullptr; __nv_bfloat16* rcbn1_b_bf = nullptr;
    __nv_bfloat16* rcbn2_w_bf = nullptr; __nv_bfloat16* rcbn2_b_bf = nullptr;
    __nv_bfloat16* rcbn3_w_bf = nullptr; __nv_bfloat16* rcbn3_b_bf = nullptr;
    __nv_bfloat16* up1c_w_bf  = nullptr; __nv_bfloat16* up1c_b_bf  = nullptr;
    __nv_bfloat16* up2c_w_bf  = nullptr; __nv_bfloat16* up2c_b_bf  = nullptr;

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

    // Merged up1 ConvTranspose+Conv
    // phase-dependent taps ([N_FEAT,N_FEAT,3] each) and scalar-per-channel
    // bias ([N_FEAT]), folded once at init -- see fold_up1_merge_kernel.
    // Eager FP32 path only; opt-in via use_merged_up1.
    float* up1_merge_K_even = nullptr;
    float* up1_merge_K_odd  = nullptr;
    float* up1_merge_bias   = nullptr;
};

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
};

static const ConvTransposeTensorDescs& get_thread_local_conv_transpose_descs(const void* owner, int N)
{
    // One per (thread, algorithm instance): shapes and contents belong to one instance.
    thread_local std::unordered_map<const void*, ConvTransposeTensorDescs> cache;
    ConvTransposeTensorDescs& descs = cache[owner];
    if (descs.td_up1_in == nullptr) {
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up1_in));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up1_out));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up2_in));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up2_out));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            descs.td_up1_in,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_QTR));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            descs.td_up1_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_HALF));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            descs.td_up2_in,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_HALF));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            descs.td_up2_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_IN));
    }
    return descs;
}

// ---------------------------------------------------------------------------
// CUDA graph scratch pool (Part 2: graph capture, FP32).
//
// Allen's SingleAlloc memory manager runs a full free/reserve cycle for the
// WHOLE sequence's arguments before every repetition (MemoryManager.cuh), so
// no data<ArgumentTag>(arguments) pointer -- not the input, not the output,
// not the internal scratch buffers -- is stable across operator() calls. A
// captured CUDA graph cannot bake in those pointers. Instead, the graph's
// internal nodes operate purely on this fixed, raw-cudaMalloc'd pool (mirrors
// dev_unet_x1_t/x2/x3/up1/up2 sizes exactly, taken from set_arguments_size()
// below -- note x3 is W_IN-wide, matching its `logits` alias use, NOT the
// W_QTR width its maxpool producer writes). Only the two shuttle-kernel nodes
// (copy real ncw -> pool.ncw_in, copy pool.x1 (=oint) -> real kde) touch
// Allen-managed pointers, and their arguments are patched per replay via
// cudaGraphExecKernelNodeSetParams.
//
// MUST be thread_local, not one shared pool like the unused fp16_pool field: many OS
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
    float* up2     = nullptr;  // [N, N_FEAT, W_IN]      -- set_size<dev_unet_up2_t>
    float* kde_out = nullptr;  // [N, W_IN]
};

static const GraphScratchPool& get_thread_local_graph_scratch_pool(const void* owner, int N)
{
    // One per (thread, algorithm instance): shapes and contents belong to one instance.
    thread_local std::unordered_map<const void*, GraphScratchPool> cache;
    GraphScratchPool& pool = cache[owner];
    if (pool.ncw_in == nullptr) {
        const size_t sz_ncw_in  = (size_t)N * N_BATCH_CHANNELS * W_IN;
        const size_t sz_x1      = (size_t)N * N_FEAT * W_IN;
        const size_t sz_x2      = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_x3      = (size_t)N * N_FEAT * W_IN;
        const size_t sz_up1     = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_up2     = (size_t)N * N_FEAT * W_IN;
        const size_t sz_kde_out = (size_t)N * W_IN;
        cudaMalloc(&pool.ncw_in,  sz_ncw_in  * sizeof(float));
        cudaMalloc(&pool.x1,      sz_x1      * sizeof(float));
        cudaMalloc(&pool.x2,      sz_x2      * sizeof(float));
        cudaMalloc(&pool.x3,      sz_x3      * sizeof(float));
        cudaMalloc(&pool.up1,     sz_up1     * sizeof(float));
        cudaMalloc(&pool.up2,     sz_up2     * sizeof(float));
        cudaMalloc(&pool.kde_out, sz_kde_out * sizeof(float));
        const size_t total_bytes =
            (sz_ncw_in + sz_x1 + sz_x2 + sz_x3 + sz_up1 + sz_up2 + sz_kde_out) * sizeof(float);
        printf("[pvfinder_unet] CUDA-graph scratch pool allocated: %.2f MB (thread_local, "
               "outside Allen's memory manager -- not reflected in -m budget)\n",
               total_bytes / (1024.0 * 1024.0));
    }
    return pool;
}

// ---------------------------------------------------------------------------
// CUDA graph scratch pool -- FP16 counterpart of GraphScratchPool, same
// thread_local/never-freed rules apply (see comment above GraphScratchPool).
// Mirrors the fp16_pool field's layout/sizes exactly, but is NOT that shared
// global -- each thread gets its own copy, avoiding the same class of
// multi-thread race a shared fp16_pool has in the eager FP16 path (not fixed
// here; out of scope, flagged separately). FP32-side buffers needed at the
// FP16 path's boundaries (x1/oint, x3/logits, up1, up2) reuse the
// existing GraphScratchPool rather than duplicating them.
// ---------------------------------------------------------------------------
struct GraphScratchPoolFP16 {
    __half* ncw  = nullptr;  // [N, N_BATCH_CHANNELS, W_IN]
    __half* x1   = nullptr;  // [N, N_FEAT, W_IN]      -- also up2_c output
    __half* x2   = nullptr;  // [N, N_FEAT, W_HALF]
    __half* x3   = nullptr;  // [N, N_FEAT, W_QTR]
    __half* up1  = nullptr;  // [N, N_FEAT, W_HALF]
    __half* up2  = nullptr;  // [N, N_FEAT, W_IN]      -- reused as general scratch
};

static const GraphScratchPoolFP16& get_thread_local_graph_scratch_pool_fp16(const void* owner, int N)
{
    // One per (thread, algorithm instance): shapes and contents belong to one instance.
    thread_local std::unordered_map<const void*, GraphScratchPoolFP16> cache;
    GraphScratchPoolFP16& pool = cache[owner];
    if (pool.ncw == nullptr) {
        const size_t sz_ncw  = (size_t)N * N_BATCH_CHANNELS * W_IN;
        const size_t sz_x1   = (size_t)N * N_FEAT * W_IN;
        const size_t sz_x2   = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_x3   = (size_t)N * N_FEAT * W_QTR;
        const size_t sz_up1  = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_up2  = (size_t)N * N_FEAT * W_IN;
        cudaMalloc(&pool.ncw,  sz_ncw  * sizeof(__half));
        cudaMalloc(&pool.x1,   sz_x1   * sizeof(__half));
        cudaMalloc(&pool.x2,   sz_x2   * sizeof(__half));
        cudaMalloc(&pool.x3,   sz_x3   * sizeof(__half));
        cudaMalloc(&pool.up1,  sz_up1  * sizeof(__half));
        cudaMalloc(&pool.up2,  sz_up2  * sizeof(__half));
        const size_t total_bytes =
            (sz_ncw + sz_x1 + sz_x2 + sz_x3 + sz_up1 + sz_up2) * sizeof(__half);
        printf("[pvfinder_unet] CUDA-graph FP16 scratch pool allocated: %.2f MB (thread_local)\n",
               total_bytes / (1024.0 * 1024.0));
    }
    return pool;
}

// ---------------------------------------------------------------------------
// BF16 counterpart of GraphScratchPoolFP16. Same thread_local/never-freed
// rules, same layout/sizes, same
// reuse of the existing FP32 GraphScratchPool at the path's boundaries
// (x3/logits, up1, up2). Eager-path-only for now -- not wired into
// either CUDA graph capture function, unlike the FP16 pool (which serves
// both) -- so this is simpler than its FP16 counterpart in that respect,
// not because the underlying risk differs.
// ---------------------------------------------------------------------------
struct GraphScratchPoolBF16 {
    __nv_bfloat16* ncw  = nullptr;  // [N, N_BATCH_CHANNELS, W_IN]
    __nv_bfloat16* x1   = nullptr;  // [N, N_FEAT, W_IN]      -- also up2_c output
    __nv_bfloat16* x2   = nullptr;  // [N, N_FEAT, W_HALF]
    __nv_bfloat16* x3   = nullptr;  // [N, N_FEAT, W_QTR]
    __nv_bfloat16* up1  = nullptr;  // [N, N_FEAT, W_HALF]
    __nv_bfloat16* up2  = nullptr;  // [N, N_FEAT, W_IN]      -- reused as general scratch
};

static const GraphScratchPoolBF16& get_thread_local_graph_scratch_pool_bf16(const void* owner, int N)
{
    // One per (thread, algorithm instance): shapes and contents belong to one instance.
    thread_local std::unordered_map<const void*, GraphScratchPoolBF16> cache;
    GraphScratchPoolBF16& pool = cache[owner];
    if (pool.ncw == nullptr) {
        const size_t sz_ncw  = (size_t)N * N_BATCH_CHANNELS * W_IN;
        const size_t sz_x1   = (size_t)N * N_FEAT * W_IN;
        const size_t sz_x2   = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_x3   = (size_t)N * N_FEAT * W_QTR;
        const size_t sz_up1  = (size_t)N * N_FEAT * W_HALF;
        const size_t sz_up2  = (size_t)N * N_FEAT * W_IN;
        cudaMalloc(&pool.ncw,  sz_ncw  * sizeof(__nv_bfloat16));
        cudaMalloc(&pool.x1,   sz_x1   * sizeof(__nv_bfloat16));
        cudaMalloc(&pool.x2,   sz_x2   * sizeof(__nv_bfloat16));
        cudaMalloc(&pool.x3,   sz_x3   * sizeof(__nv_bfloat16));
        cudaMalloc(&pool.up1,  sz_up1  * sizeof(__nv_bfloat16));
        cudaMalloc(&pool.up2,  sz_up2  * sizeof(__nv_bfloat16));
        const size_t total_bytes =
            (sz_ncw + sz_x1 + sz_x2 + sz_x3 + sz_up1 + sz_up2) * sizeof(__nv_bfloat16);
        printf("[pvfinder_unet] eager BF16 scratch pool allocated: %.2f MB (thread_local)\n",
               total_bytes / (1024.0 * 1024.0));
    }
    return pool;
}

// A captured CUDA graph plus its patchable shuttle nodes; the template graph
// must outlive the exec (see get_or_capture_cuda_graph). One per (thread,
// algorithm instance): a graph bakes in that instance's descriptors, weights
// and scratch pool, so replaying another instance's graph would be wrong.
struct CapturedGraph {
    cudaGraphExec_t exec           = nullptr;
    cudaGraphNode_t copy_in        = nullptr;
    cudaGraphNode_t copy_out       = nullptr;
    cudaGraph_t     template_graph = nullptr;
};

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

    const float* w_oint_w;   const float* w_oint_b;
    const float* w_outc_w;   const float* w_outc_b;
};

// Everything one pvfinder_unet instance owns: its weights (loaded in init()),
// its descriptors (created on its first operator() call, since algorithm
// selection needs a live cuDNN handle) and the once-flag guarding them.
struct pvfinder_unet_t::UNetState {
    WeightBlob        wb {};
    GlobalDescriptors desc;
    std::once_flag    desc_init_flag;
    // The UNet's output for an all-zero interval ([W_IN] floats, device),
    // written to every interval it skips when the features come in compact
    // rows. Computed once, by the configured path, on the first compact call.
    float*            empty_response = nullptr;
    std::once_flag    empty_response_flag;
    // BF16 path, channels last (bf16_layout = "nwc"): one fused
    // Conv+Bias+ReLU graph per CBR layer (rcbn1, rcbn2, rcbn3, up1c, up2c)
    // and its BN-folded weights as BF16 [K][R][C]. Built on the first such call.
    ConvBiasReluGraphNWC nwc_conv[5];
    __nv_bfloat16*    nwc_w[5] = {};
    std::once_flag    nwc_flag;
    // fused_kernel: the whole channels-last BF16 UNet in one kernel
    // (PVFinderUNetFused.cuh), from this shared-memory image of its weights.
    unsigned char*    fused_blob = nullptr;
    int               fused_grid = 0;
};

static void init_descriptors(GlobalDescriptors& desc, cudnnHandle_t handle, const WeightBlob& wb, size_t fwd_ws_budget_bytes, int N)
{

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

    // Helper: convert FP32 BN-folded weights to BF16. Structurally identical
    // to to_half above.
    auto to_bf16 = [](const float* w_f, const float* b_f, int K, int CxHxW,
                      __nv_bfloat16*& w_bf, __nv_bfloat16*& b_bf, cudaStream_t stream) {
        size_t wn = (size_t)K * CxHxW;
        cudaMalloc(&w_bf, wn * sizeof(__nv_bfloat16));
        cudaMalloc(&b_bf, (size_t)K * sizeof(__nv_bfloat16));
        int threads = 256;
        f32_to_bf16_kernel<<<((wn + threads - 1) / threads), threads, 0, stream>>>(w_bf, w_f, (int)wn);
        f32_to_bf16_kernel<<<((K  + threads - 1) / threads), threads, 0, stream>>>(b_bf, b_f, K);
    };

    // Fold BN into conv weights for each CBR layer, then build the fused graph.
    // All fold kernels run on stream 0 (init is single-threaded here).
    fold_bn(wb.w_rcbn1_w, wb.w_rcbn1_b,
            wb.w_rcbn1_gamma, wb.w_rcbn1_beta,
            wb.w_rcbn1_mean,  wb.w_rcbn1_var, wb.rcbn1_eps,
            N_FEAT, N_BATCH_CHANNELS * 25,
            desc.rcbn1_w_f, desc.rcbn1_b_f, 0);
    cudaDeviceSynchronize();
    desc.rcbn1.create(handle, {N, N_BATCH_CHANNELS, 1, W_IN}, {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0,12},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(desc.rcbn1_w_f, desc.rcbn1_b_f, N_FEAT, N_BATCH_CHANNELS * 25,
            desc.rcbn1_w_h, desc.rcbn1_b_h, 0);
    cudaDeviceSynchronize();
    desc.rcbn1_h.create(handle, {N, N_BATCH_CHANNELS, 1, W_IN}, {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0,12},
                          {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);
    to_bf16(desc.rcbn1_w_f, desc.rcbn1_b_f, N_FEAT, N_BATCH_CHANNELS * 25,
            desc.rcbn1_w_bf, desc.rcbn1_b_bf, 0);
    cudaDeviceSynchronize();
    desc.rcbn1_bf.create(handle, {N, N_BATCH_CHANNELS, 1, W_IN}, {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0,12},
                          {1,1}, {1,1}, CUDNN_DATA_BFLOAT16, fwd_ws_budget_bytes);

    // Optional true single-pass Conv+Bias+ReLU for rcbn1. Uses the same
    // BN-folded weights/bias as rcbn1
    // above. Creation can fail if no engine supports this op-graph shape on
    // the current GPU/cuDNN version -- caught here so the process still
    // starts and use_fused_cbr silently falls back to the two-pass path.
    try {
        desc.rcbn1_fused.create(handle, {N, N_BATCH_CHANNELS, 1, W_IN}, {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0, 12});
        desc.rcbn1_fused_available = true;
    } catch (const std::exception& e) {
        desc.rcbn1_fused_available = false;
        fprintf(stderr, "[pvfinder_unet] ConvBiasReluGraph unavailable for rcbn1 (%s); "
                "use_fused_cbr will fall back to the two-pass conv+bias/ReLU path.\n", e.what());
    }

    fold_bn(wb.w_rcbn2_w, wb.w_rcbn2_b,
            wb.w_rcbn2_gamma, wb.w_rcbn2_beta,
            wb.w_rcbn2_mean,  wb.w_rcbn2_var, wb.rcbn2_eps,
            N_FEAT, N_FEAT * 7,
            desc.rcbn2_w_f, desc.rcbn2_b_f, 0);
    cudaDeviceSynchronize();
    desc.rcbn2.create(handle, {N, N_FEAT, 1, W_IN},  {N_FEAT, N_FEAT, 1,  7}, {0, 3},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(desc.rcbn2_w_f, desc.rcbn2_b_f, N_FEAT, N_FEAT * 7,
            desc.rcbn2_w_h, desc.rcbn2_b_h, 0);
    cudaDeviceSynchronize();
    desc.rcbn2_h.create(handle, {N, N_FEAT, 1, W_IN}, {N_FEAT, N_FEAT, 1, 7}, {0,3},
                          {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);
    to_bf16(desc.rcbn2_w_f, desc.rcbn2_b_f, N_FEAT, N_FEAT * 7,
            desc.rcbn2_w_bf, desc.rcbn2_b_bf, 0);
    cudaDeviceSynchronize();
    desc.rcbn2_bf.create(handle, {N, N_FEAT, 1, W_IN}, {N_FEAT, N_FEAT, 1, 7}, {0,3},
                          {1,1}, {1,1}, CUDNN_DATA_BFLOAT16, fwd_ws_budget_bytes);

    fold_bn(wb.w_rcbn3_w, wb.w_rcbn3_b,
            wb.w_rcbn3_gamma, wb.w_rcbn3_beta,
            wb.w_rcbn3_mean,  wb.w_rcbn3_var, wb.rcbn3_eps,
            N_FEAT, N_FEAT * 5,
            desc.rcbn3_w_f, desc.rcbn3_b_f, 0);
    cudaDeviceSynchronize();
    desc.rcbn3.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0, 2},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(desc.rcbn3_w_f, desc.rcbn3_b_f, N_FEAT, N_FEAT * 5,
            desc.rcbn3_w_h, desc.rcbn3_b_h, 0);
    cudaDeviceSynchronize();
    desc.rcbn3_h.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0,2},
                          {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);
    to_bf16(desc.rcbn3_w_f, desc.rcbn3_b_f, N_FEAT, N_FEAT * 5,
            desc.rcbn3_w_bf, desc.rcbn3_b_bf, 0);
    cudaDeviceSynchronize();
    desc.rcbn3_bf.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0,2},
                          {1,1}, {1,1}, CUDNN_DATA_BFLOAT16, fwd_ws_budget_bytes);

    fold_bn(wb.w_up1c_w, wb.w_up1c_b,
            wb.w_up1c_gamma, wb.w_up1c_beta,
            wb.w_up1c_mean,  wb.w_up1c_var, wb.up1c_eps,
            N_FEAT, N_FEAT * 5,
            desc.up1c_w_f, desc.up1c_b_f, 0);
    cudaDeviceSynchronize();
    desc.up1_c.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0, 2},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(desc.up1c_w_f, desc.up1c_b_f, N_FEAT, N_FEAT * 5,
            desc.up1c_w_h, desc.up1c_b_h, 0);
    cudaDeviceSynchronize();
    desc.up1c_h.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0,2},
                         {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);
    to_bf16(desc.up1c_w_f, desc.up1c_b_f, N_FEAT, N_FEAT * 5,
            desc.up1c_w_bf, desc.up1c_b_bf, 0);
    cudaDeviceSynchronize();
    desc.up1c_bf.create(handle, {N, N_FEAT, 1, W_HALF}, {N_FEAT, N_FEAT, 1, 5}, {0,2},
                         {1,1}, {1,1}, CUDNN_DATA_BFLOAT16, fwd_ws_budget_bytes);

    // Fold up1's ConvTranspose+Conv into phase-dependent merged taps, from
    // the raw (unfused) ConvTranspose
    // weight/bias and the already-BN-folded up1c_w_f/b_f above.
    {
        cudaMalloc(&desc.up1_merge_K_even, (size_t)N_FEAT * N_FEAT * 3 * sizeof(float));
        cudaMalloc(&desc.up1_merge_K_odd,  (size_t)N_FEAT * N_FEAT * 3 * sizeof(float));
        cudaMalloc(&desc.up1_merge_bias,   (size_t)N_FEAT * sizeof(float));
        launch_fold_up1_merge(
            desc.up1_merge_K_even, desc.up1_merge_K_odd, desc.up1_merge_bias,
            wb.w_up1t_w, wb.w_up1t_b, desc.up1c_w_f, desc.up1c_b_f,
            N_FEAT, /*stream=*/0);
        cudaDeviceSynchronize();
    }

    fold_bn(wb.w_up2c_w, wb.w_up2c_b,
            wb.w_up2c_gamma, wb.w_up2c_beta,
            wb.w_up2c_mean,  wb.w_up2c_var, wb.up2c_eps,
            N_FEAT, N_FEAT * 5,
            desc.up2c_w_f, desc.up2c_b_f, 0);
    cudaDeviceSynchronize();
    desc.up2_c.create(handle, {N, N_FEAT, 1, W_IN},  {N_FEAT, N_FEAT, 1, 5}, {0, 2},
                        {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    to_half(desc.up2c_w_f, desc.up2c_b_f, N_FEAT, N_FEAT * 5,
            desc.up2c_w_h, desc.up2c_b_h, 0);
    cudaDeviceSynchronize();
    desc.up2c_h.create(handle, {N, N_FEAT, 1, W_IN}, {N_FEAT, N_FEAT, 1, 5}, {0,2},
                         {1,1}, {1,1}, CUDNN_DATA_HALF, fwd_ws_budget_bytes);
    to_bf16(desc.up2c_w_f, desc.up2c_b_f, N_FEAT, N_FEAT * 5,
            desc.up2c_w_bf, desc.up2c_b_bf, 0);
    cudaDeviceSynchronize();
    desc.up2c_bf.create(handle, {N, N_FEAT, 1, W_IN}, {N_FEAT, N_FEAT, 1, 5}, {0,2},
                         {1,1}, {1,1}, CUDNN_DATA_BFLOAT16, fwd_ws_budget_bytes);

    // No global FP16 activation pool: the eager and graph FP16 paths both use
    // thread_local pools (GraphScratchPoolFP16), and a global one sized by N
    // would only waste memory outside Allen's -m budget at large batch sizes.

    // Non-CBR paths: plain conv, same pinned-IMPLICIT_GEMM-by-default ConvDescriptors.
    desc.oint.create(     handle, {N, N_FEAT, 1, W_IN},  {N_FEAT, N_FEAT, 1, 5}, {0, 2},
                            {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);
    desc.outc.create(     handle, {N, N_FEAT, 1, W_IN},  {1,      N_FEAT, 1, 5}, {0, 2},
                            {1,1}, {1,1}, CUDNN_DATA_FLOAT, fwd_ws_budget_bytes);

    // Reduced-precision output stage parameters, composed once in double
    // precision from the same FP32 weights (only the BF16 path reads them).
    {
        std::vector<float> w_oint(N_FEAT * N_FEAT * 5), b_oint(N_FEAT), w_outc(N_FEAT * 5);
        float b_outc = 0.f;
        cudaMemcpy(w_oint.data(), wb.w_oint_w, w_oint.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(b_oint.data(), wb.w_oint_b, b_oint.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(w_outc.data(), wb.w_outc_w, w_outc.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b_outc, wb.w_outc_b, sizeof(float), cudaMemcpyDeviceToHost);
        const std::vector<float> params = make_output_stage_params<N_FEAT>(w_oint, b_oint, w_outc, b_outc);
        cudaCheck(cudaMalloc(&desc.output_stage_params, params.size() * sizeof(float)));
        cudaCheck(cudaMemcpy(desc.output_stage_params, params.data(), params.size() * sizeof(float),
                             cudaMemcpyHostToDevice));
    }

    // One-time diagnostic: confirms the header's design intent -- "IMPLICIT_GEMM
    // pinned everywhere -> zero workspace" -- actually holds when
    // fwd_algo_ws_budget_bytes==0 (the default). ConvDescriptors::create() pins
    // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM unconditionally in that case (no
    // algorithm search); an earlier cudnnFindConvolutionForwardAlgorithmEx-based
    // selection experiment was tried and reverted (see CuDNNDescriptors.h's class
    // comment) because it regressed under real -t16 memory-bandwidth contention
    // despite looking faster in isolation. fwd_algo_ws_budget_bytes below reopens
    // this with a workspace-budgeted heuristic search instead of an unrestricted
    // one -- active only when it is set nonzero. Each
    // ConvDescriptors' workspace is thread_local (not a single shared buffer), so
    // even a nonzero size here would not be a cross-thread race. Logged once so
    // the actual algorithm/workspace state is visible rather than assumed. Routed
    // to stderr (unbuffered) and flushed explicitly: stdout is fully buffered once
    // redirected to a file, so on an abrupt abort() (e.g. the std::terminate path
    // from ALLEN_CUDNN_CHECK) any unflushed printf content -- including this
    // diagnostic -- is silently lost, leaving no evidence of which
    // algorithm/workspace size was actually selected for a crashing run.
    fprintf(stderr, "[pvfinder_unet] cuDNN batch N=%d samples (%d events)\n", N, N / N_INTERVALS);
    fprintf(stderr, "[pvfinder_unet] fwd_algo_ws_budget_bytes=%zu\n", fwd_ws_budget_bytes);
    fprintf(stderr, "[pvfinder_unet] ConvDescriptors workspace bytes: rcbn1=%zu rcbn2=%zu rcbn3=%zu "
           "up1_c=%zu up2_c=%zu oint=%zu outc=%zu | algo ids: rcbn1=%d rcbn2=%d rcbn3=%d "
           "up1_c=%d up2_c=%d oint=%d outc=%d\n",
           desc.rcbn1.workspace_bytes(), desc.rcbn2.workspace_bytes(), desc.rcbn3.workspace_bytes(),
           desc.up1_c.workspace_bytes(), desc.up2_c.workspace_bytes(),
           desc.oint.workspace_bytes(), desc.outc.workspace_bytes(),
           desc.rcbn1.algo_id(), desc.rcbn2.algo_id(), desc.rcbn3.algo_id(),
           desc.up1_c.algo_id(), desc.up2_c.algo_id(),
           desc.oint.algo_id(), desc.outc.algo_id());
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
           desc.rcbn1_h.workspace_bytes(), desc.rcbn2_h.workspace_bytes(), desc.rcbn3_h.workspace_bytes(),
           desc.up1c_h.workspace_bytes(), desc.up2c_h.workspace_bytes(),
           desc.rcbn1_h.algo_id(), desc.rcbn2_h.algo_id(), desc.rcbn3_h.algo_id(),
           desc.up1c_h.algo_id(), desc.up2c_h.algo_id());
    // Whether the fused-graph rcbn1 path is
    // usable on this GPU/cuDNN version at all (see the try/catch around its
    // create() call above).
    fprintf(stderr, "[pvfinder_unet] rcbn1 ConvBiasReluGraph available: %s (workspace bytes: %zu)\n",
           desc.rcbn1_fused_available ? "yes" : "no", desc.rcbn1_fused.workspace_bytes());
    fflush(stderr);

    // ConvTranspose: filter + conv descriptors only (shared, read-only after init).
    // Tensor descriptors for in/out are thread_local (see
    // get_thread_local_conv_transpose_descs()), not created here.
    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc.filter_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        desc.filter_up1_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc.conv_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        desc.conv_up1_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(desc.conv_up1_t, CUDNN_TENSOR_OP_MATH));

    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc.filter_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        desc.filter_up2_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc.conv_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        desc.conv_up2_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(desc.conv_up2_t, CUDNN_TENSOR_OP_MATH));

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
                handle, desc.filter_up1_t, dy_desc, desc.conv_up1_t, dx_desc,
                kBwdMaxAlgo, &returned, perf) == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= kBwdBudget) {
                    desc.algo_up1_t   = perf[i].algo;
                    desc.ws_up1_bytes = perf[i].memory;
                    if (desc.ws_up1_bytes > 0) cudaMalloc(&desc.ws_up1_t, desc.ws_up1_bytes);
                    break;
                }
            }
        }
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
    }

    // up2_t: dy=[N, N_FEAT, 1, W_HALF], dx=[N, N_FEAT, 1, W_IN]
    {
        cudnnTensorDescriptor_t dy_desc, dx_desc;
        cudnnCreateTensorDescriptor(&dy_desc);
        cudnnCreateTensorDescriptor(&dx_desc);
        cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_HALF);
        cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_IN);
        int returned = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perf[kBwdMaxAlgo];
        if (cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle, desc.filter_up2_t, dy_desc, desc.conv_up2_t, dx_desc,
                kBwdMaxAlgo, &returned, perf) == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= kBwdBudget) {
                    desc.algo_up2_t   = perf[i].algo;
                    desc.ws_up2_bytes = perf[i].memory;
                    if (desc.ws_up2_bytes > 0) cudaMalloc(&desc.ws_up2_t, desc.ws_up2_bytes);
                    break;
                }
            }
        }
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
    }

    // The ConvTranspose workspaces are shared by all threads, so a nonzero
    // size here would be a cross-thread race; log it so it is never silent.
    fprintf(stderr, "[pvfinder_unet] ConvTranspose workspace bytes: up1=%zu up2=%zu | algo ids: "
            "up1=%d up2=%d\n",
            desc.ws_up1_bytes, desc.ws_up2_bytes,
            (int)desc.algo_up1_t, (int)desc.algo_up2_t);
    fflush(stderr);
}

// ---------------------------------------------------------------------------
// Binary weight file parser
// Layout (written by write_cnn_weights in weights/scripts/convert.py):
//   uint32  magic = 0xCAFE0001
//   conv(C→F,k=25):  int32 in,out,k | float[out*in*k] weights | float[out] bias
//   bn(F):           int32 features | float eps | float[f] gamma,beta,mean,var
//   ... repeated for rcbn2, rcbn3
//   convT(F→F,k=2,s=2): int32 in,out,k,stride | float[in*out*k] | float[out]
//   conv+bn for up1.convbnrelu, up2.convbnrelu
//   conv(F→F,k=5):   out_intermediate
//   conv(F→1,k=5):   outc
// with C = N_BATCH_CHANNELS and F = N_FEAT. No skip connections, so up2's
// ConvTranspose and out_intermediate take F channels, not 2F.
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
    // Keys are namespaced by weight file: instances loading the same file share
    // one device copy, instances with different files never collide.
    const std::string ns = "pvfinder_unet:" + path + ":";
    WeightBlob wb {};

    // Every layer's shape is fixed by this build (N_BATCH_CHANNELS, N_FEAT) and
    // by the no-skip architecture, so check it instead of trusting the file: a
    // checkpoint trained with skip connections has 2*N_FEAT inputs at up2's
    // ConvTranspose and at out_intermediate.
    auto expect_shape = [&](const std::string& key, int in_c, int out_c, int k,
                            int want_in, int want_out, int want_k) {
        if (in_c != want_in || out_c != want_out || k != want_k) {
            throw std::runtime_error(
                "PVFinderUNet: " + key + " in " + path + " has (in=" + std::to_string(in_c) +
                ", out=" + std::to_string(out_c) + ", k=" + std::to_string(k) +
                "), this build expects (in=" + std::to_string(want_in) + ", out=" +
                std::to_string(want_out) + ", k=" + std::to_string(want_k) +
                "); check --unet-feat/--unet-batch-channels and that the model has no skip connections");
        }
    };

    // Helper lambdas
    auto load_conv = [&](const std::string& key_w, const std::string& key_b,
                          const float*& out_w, const float*& out_b,
                          int want_in, int want_out, int want_k) {
        int in_c, out_c, k;
        off = read_int32(buf, off, in_c);
        off = read_int32(buf, off, out_c);
        off = read_int32(buf, off, k);
        expect_shape(key_w, in_c, out_c, k, want_in, want_out, want_k);
        size_t wcount = (size_t)out_c * in_c * k;
        // Load weight block
        std::vector<float> w_host(wcount);
        off = read_float_block(buf, off, w_host.data(), wcount);
        if (!reg.contains(ns + key_w)) reg.load_from_buffer(ns + key_w, w_host.data(), wcount * sizeof(float));
        out_w = reg.get<float>(ns + key_w);
        // Load bias block
        std::vector<float> b_host(out_c);
        off = read_float_block(buf, off, b_host.data(), out_c);
        if (!reg.contains(ns + key_b)) reg.load_from_buffer(ns + key_b, b_host.data(), out_c * sizeof(float));
        out_b = reg.get<float>(ns + key_b);
    };

    auto load_bn = [&](const std::string& prefix,
                        const float*& gamma, const float*& beta,
                        const float*& mean,  const float*& var, float& eps) {
        int features;
        off = read_int32(buf, off, features);
        if (features != N_FEAT) {
            throw std::runtime_error("PVFinderUNet: " + prefix + " in " + path + " has " +
                                     std::to_string(features) + " features, this build expects N_FEAT=" +
                                     std::to_string(N_FEAT));
        }
        off = read_float32(buf, off, eps);
        std::vector<float> g(features), b(features), m(features), v(features);
        off = read_float_block(buf, off, g.data(), features);
        off = read_float_block(buf, off, b.data(), features);
        off = read_float_block(buf, off, m.data(), features);
        off = read_float_block(buf, off, v.data(), features);
        auto ld = [&](const std::string& k, const std::vector<float>& d, const float*& ptr) {
            if (!reg.contains(ns + k)) reg.load_from_buffer(ns + k, d.data(), d.size() * sizeof(float));
            ptr = reg.get<float>(ns + k);
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
        expect_shape(key_w, in_c, out_c, k, N_FEAT, N_FEAT, 2);
        if (stride != 2) {
            throw std::runtime_error("PVFinderUNet: " + key_w + " in " + path + " has stride " +
                                     std::to_string(stride) + ", expected 2");
        }
        size_t wcount = (size_t)in_c * out_c * k;
        std::vector<float> w_host(wcount);
        off = read_float_block(buf, off, w_host.data(), wcount);
        if (!reg.contains(ns + key_w)) reg.load_from_buffer(ns + key_w, w_host.data(), wcount * sizeof(float));
        out_w = reg.get<float>(ns + key_w);
        std::vector<float> b_host(out_c);
        off = read_float_block(buf, off, b_host.data(), out_c);
        if (!reg.contains(ns + key_b)) reg.load_from_buffer(ns + key_b, b_host.data(), out_c * sizeof(float));
        out_b = reg.get<float>(ns + key_b);
    };

    // rcbn1
    load_conv("rcbn1.w", "rcbn1.b", wb.w_rcbn1_w, wb.w_rcbn1_b, N_BATCH_CHANNELS, N_FEAT, 25);
    load_bn("rcbn1.bn", wb.w_rcbn1_gamma, wb.w_rcbn1_beta, wb.w_rcbn1_mean, wb.w_rcbn1_var, wb.rcbn1_eps);
    // rcbn2
    load_conv("rcbn2.w", "rcbn2.b", wb.w_rcbn2_w, wb.w_rcbn2_b, N_FEAT, N_FEAT, 7);
    load_bn("rcbn2.bn", wb.w_rcbn2_gamma, wb.w_rcbn2_beta, wb.w_rcbn2_mean, wb.w_rcbn2_var, wb.rcbn2_eps);
    // rcbn3
    load_conv("rcbn3.w", "rcbn3.b", wb.w_rcbn3_w, wb.w_rcbn3_b, N_FEAT, N_FEAT, 5);
    load_bn("rcbn3.bn", wb.w_rcbn3_gamma, wb.w_rcbn3_beta, wb.w_rcbn3_mean, wb.w_rcbn3_var, wb.rcbn3_eps);
    // up1: ConvTranspose + ConvBNrelu
    load_convt("up1t.w", "up1t.b", wb.w_up1t_w, wb.w_up1t_b);
    load_conv("up1c.w", "up1c.b", wb.w_up1c_w, wb.w_up1c_b, N_FEAT, N_FEAT, 5);
    load_bn("up1c.bn", wb.w_up1c_gamma, wb.w_up1c_beta, wb.w_up1c_mean, wb.w_up1c_var, wb.up1c_eps);
    // up2: ConvTranspose + ConvBNrelu
    load_convt("up2t.w", "up2t.b", wb.w_up2t_w, wb.w_up2t_b);
    load_conv("up2c.w", "up2c.b", wb.w_up2c_w, wb.w_up2c_b, N_FEAT, N_FEAT, 5);
    load_bn("up2c.bn", wb.w_up2c_gamma, wb.w_up2c_beta, wb.w_up2c_mean, wb.w_up2c_var, wb.up2c_eps);
    // out_intermediate: Conv(N_FEAT→N_FEAT, k=5)
    load_conv("oint.w", "oint.b", wb.w_oint_w, wb.w_oint_b, N_FEAT, N_FEAT, 5);
    // outc: Conv(N_FEAT→1, k=5)
    load_conv("outc.w", "outc.b", wb.w_outc_w, wb.w_outc_b, N_FEAT, 1, 5);

    if (off != buf.size()) {
        throw std::runtime_error("PVFinderUNet: " + std::to_string(buf.size() - off) +
                                 " trailing bytes in weight file " + path);
    }

    return wb;
}

#endif // ALLEN_CUDNN_BACKEND_CUDA

// ---------------------------------------------------------------------------
// init(): load this instance's weights (descriptors follow on its first operator() call).
// ---------------------------------------------------------------------------
void pvfinder_unet_t::init()
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    if (m_state) return;
    if (m_weight_file.value().empty()) {
        throw std::runtime_error(
            "pvfinder_unet: weight_file is not set. Produce weights with the repository's weights/ "
            "pipeline (make -C weights verify MODEL=<name>) and generate the sequence configuration "
            "with PVFINDER_WEIGHTS_DIR pointing at them (make -C weights env MODEL=<name>).");
    }
    auto state = std::make_shared<UNetState>();
    state->wb = load_weights(m_weight_file.value());
    // BF16 channels-last path: build the fused convolution plans here, at
    // start-up, since their runtime-compiled engines take about a second
    // each to compile. The BF16 weights they run on are prepared on the
    // first call (they need the BN folding done with the descriptors).
    if (m_use_bf16.value() && m_bf16_layout.value() == "nwc") {
        cudnnHandle_t handle = Allen::CuDNN::get_thread_local_handle(nullptr);
        const int N = (int) m_unet_batch_events.value() * N_INTERVALS;
        struct Layer { int C_in, R, pad, W; const char* name; };
        const Layer layers[5] = {{N_BATCH_CHANNELS, 25, 12, W_IN, "rcbn1"}, {N_FEAT, 7, 3, W_IN, "rcbn2"},
                                 {N_FEAT, 5, 2, W_HALF, "rcbn3"}, {N_FEAT, 5, 2, W_HALF, "up1c"},
                                 {N_FEAT, 5, 2, W_IN, "up2c"}};
        for (int l = 0; l < 5; ++l) {
            const Layer& L = layers[l];
            const std::string chosen = state->nwc_conv[l].create(handle, nullptr, CUDNN_DATA_BFLOAT16, N, L.C_in, N_FEAT,
                                                                 L.W, L.R, L.pad, (int) m_bf16_nwc_candidates.value());
            printf("[pvfinder_unet] BF16 NWC %s: fused Conv+Bias+ReLU graph plan, %s\n", L.name, chosen.c_str());
        }
    }
    m_state = std::move(state);
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
    const unsigned batch_events = m_unet_batch_events.value();
    if (batch_events == 0) {
        throw std::runtime_error("pvfinder_unet: unet_batch_events must be >= 1");
    }
    const unsigned padded_events = ((n_events + batch_events - 1) / batch_events) * batch_events;
    const unsigned N_batch = batch_events * N_INTERVALS;
    const unsigned* unet_rows = data<host_pvfinder_unet_rows_t>(arguments);
    const bool compact = unet_rows[0] == 1u;
    const unsigned padded_rows = compact ? (unet_rows[1] + N_batch - 1) / N_batch * N_batch : 0u;

    set_size<dev_unet_x1_t>   (arguments, N_batch * N_FEAT * W_IN);       
    set_size<dev_unet_x2_t>   (arguments, N_batch * N_FEAT * W_HALF);     
    set_size<dev_unet_x3_t>   (arguments, N_batch * N_FEAT * W_IN);       
    set_size<dev_unet_up1_t>  (arguments, N_batch * N_FEAT * W_HALF);     
    set_size<dev_unet_up2_t>  (arguments, N_batch * N_FEAT * W_IN); 
    set_size<dev_unet_conv_ws_t>(arguments, 1u);                    
    set_size<dev_unet_kde_rows_t>(arguments, padded_rows > 0 ? padded_rows * W_IN : 1u);
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

// BF16 counterpart of run_convbnrelu_half.
void pvfinder_unet_t::run_convbnrelu_bf16(
    const Allen::CuDNN::ConvDescriptors& desc,
    const __nv_bfloat16* input, __nv_bfloat16* output,
    const __nv_bfloat16* w_fused, const __nv_bfloat16* b_fused,
    int K, int W_out, int N,
    cudnnHandle_t handle,
    const dim3& block, const Allen::Context& ctx) const
{
    desc.forward_bf16(handle, 1.f, 0.f, input, w_fused, output);
    launch_bias_relu_bf16(output, b_fused, K, W_out, N, block, ctx);
}

// Conv1d only (no BN/ReLU). bias_ptr may be null to keep the raw conv output.
void pvfinder_unet_t::run_conv(
    const Allen::CuDNN::ConvDescriptors& desc,
    const float* input,  float* output,
    const float* w_ptr,  const float* bias_ptr,
    int N, int C_out, int W,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle) const
{
    desc.forward(handle, 1.f, 0.f, input, w_ptr, output);
    if (bias_ptr)
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
// FP32 op sequence from operator() below, operating on pool buffers
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
    // This instance's graph on this thread (see CapturedGraph).
    thread_local std::unordered_map<const void*, CapturedGraph> tl_graphs;
    CapturedGraph& tl_graph = tl_graphs[m_state.get()];
    cudaGraphExec_t& tl_exec     = tl_graph.exec;
    cudaGraphNode_t& tl_copy_in  = tl_graph.copy_in;
    cudaGraphNode_t& tl_copy_out = tl_graph.copy_out;
    const GlobalDescriptors& desc = m_state->desc;
    const WeightBlob& wb = m_state->wb;
    // The template cudaGraph_t is intentionally kept alive (never destroyed) for
    // the thread's lifetime: cudaGraphExecKernelNodeSetParams patches tl_exec by
    // referencing node handles owned by THIS template graph, and destroying it
    // would invalidate those handles even though tl_exec itself would keep working
    // for plain (unpatched) relaunches. Must stay alive as long as tl_exec is used.
    cudaGraph_t& tl_template_graph = tl_graph.template_graph;

    if (tl_exec == nullptr) {
        // See s_graph_capture_mutex's declaration comment: serializes first-time
        // capture across threads. Only this thread's own tl_exec is set inside;
        // once set, this thread never re-enters this block or takes the lock again.
        std::lock_guard<std::mutex> capture_lock(s_graph_capture_mutex);
        const int N = (int)m_unet_batch_events.value() * N_INTERVALS;
        const GraphScratchPool& pool = get_thread_local_graph_scratch_pool(m_state.get(), N);
        const ConvTransposeTensorDescs& td = get_thread_local_conv_transpose_descs(m_state.get(), N);
        cudaStream_t stream = ctx.stream();

        // Aliases within the pool, mirroring the eager path's proven-safe
        // aliasing scheme (oint=x1, logits=x3).
        float* g_x1 = pool.x1; float* g_x2 = pool.x2; float* g_x3 = pool.x3;
        float* g_up1 = pool.up1; float* g_up2 = pool.up2;
        float* g_oint = g_x1; float* g_logits = g_x3;

        const unsigned total_in  = (unsigned)N * N_BATCH_CHANNELS * W_IN;
        const unsigned total_out = (unsigned)N * W_IN;
        const dim3 grid_in ((total_in  + block.x - 1) / block.x);
        const dim3 grid_out((total_out + block.x - 1) / block.x);

        // Pre-warm every ConvDescriptors' thread-local workspace on this thread
        // BEFORE capture begins: growing a thread-local workspace (cudaMalloc) is
        // not something that can happen mid-capture, so every descriptor forward()
        // will touch during the captured sequence below must already be sized.
        desc.rcbn1.ensure_thread_local_workspace();
        desc.rcbn2.ensure_thread_local_workspace();
        desc.rcbn3.ensure_thread_local_workspace();
        desc.up1_c.ensure_thread_local_workspace();
        desc.up2_c.ensure_thread_local_workspace();
        desc.oint.ensure_thread_local_workspace();
        desc.outc.ensure_thread_local_workspace();

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

        run_convbnrelu(desc.rcbn1, pool.ncw_in, g_x1,  desc.rcbn1_w_f, desc.rcbn1_b_f, N_FEAT, W_IN,   N, handle, block, ctx);
        run_convbnrelu(desc.rcbn2, g_x1,  g_up2, desc.rcbn2_w_f, desc.rcbn2_b_f, N_FEAT, W_IN,   N, handle, block, ctx);
        launch_maxpool(g_up2, g_x2, N, N_FEAT, W_IN, block, ctx);

        run_convbnrelu(desc.rcbn3, g_x2, g_up2, desc.rcbn3_w_f, desc.rcbn3_b_f, N_FEAT, W_HALF, N, handle, block, ctx);
        launch_maxpool(g_up2, g_x3, N, N_FEAT, W_HALF, block, ctx);

        run_conv_transpose(g_x3, g_up2,
            desc.filter_up1_t, desc.conv_up1_t, td.td_up1_in, td.td_up1_out,
            wb.w_up1t_w, wb.w_up1t_b,
            N, N_FEAT, W_HALF, block, ctx, handle,
            desc.algo_up1_t, desc.ws_up1_t, desc.ws_up1_bytes);
        run_convbnrelu(desc.up1_c, g_up2, g_up1, desc.up1c_w_f, desc.up1c_b_f, N_FEAT, W_HALF, N, handle, block, ctx);

        run_conv_transpose(g_up1, g_logits,
            desc.filter_up2_t, desc.conv_up2_t, td.td_up2_in, td.td_up2_out,
            wb.w_up2t_w, wb.w_up2t_b,
            N, N_FEAT, W_IN, block, ctx, handle,
            desc.algo_up2_t, desc.ws_up2_t, desc.ws_up2_bytes);
        run_convbnrelu(desc.up2_c, g_logits, g_up2, desc.up2c_w_f, desc.up2c_b_f, N_FEAT, W_IN, N, handle, block, ctx);

        run_conv(desc.oint, g_up2, g_logits, wb.w_oint_w, wb.w_oint_b, N, N_FEAT, W_IN, block, ctx, handle);
        run_conv(desc.outc, g_logits, g_oint, wb.w_outc_w, wb.w_outc_b, N, 1, W_IN, block, ctx, handle);

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

        printf("[pvfinder_unet] CUDA graph captured (thread_local, FP32 pipeline)\n");
    }

    out_exec          = tl_exec;
    out_copy_in_node  = tl_copy_in;
    out_copy_out_node = tl_copy_out;
}

// ---------------------------------------------------------------------------
// CUDA graph capture, FP16 counterpart of get_or_capture_cuda_graph. Same
// idiom (thread_local exec/nodes/template graph, capture-once, live node
// handles via cudaStreamGetCaptureInfo). Reuses the existing FP32
// GraphScratchPool for the FP32-side buffers this sequence needs (x1/oint,
// x3/logits, up1, up2 -- same aliasing as the eager FP32/FP16 paths) and a
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
    // This instance's graph on this thread (see CapturedGraph).
    thread_local std::unordered_map<const void*, CapturedGraph> tl_graphs;
    CapturedGraph& tl_graph = tl_graphs[m_state.get()];
    cudaGraphExec_t& tl_exec     = tl_graph.exec;
    cudaGraphNode_t& tl_copy_in  = tl_graph.copy_in;
    cudaGraphNode_t& tl_copy_out = tl_graph.copy_out;
    const GlobalDescriptors& desc = m_state->desc;
    const WeightBlob& wb = m_state->wb;
    cudaGraph_t& tl_template_graph = tl_graph.template_graph;

    if (tl_exec == nullptr) {
        // See s_graph_capture_mutex's declaration comment (shared with the FP32
        // capture function above -- one global mutex, both capture paths).
        std::lock_guard<std::mutex> capture_lock(s_graph_capture_mutex);
        const int N = (int)m_unet_batch_events.value() * N_INTERVALS;
        const GraphScratchPool& pool32 = get_thread_local_graph_scratch_pool(m_state.get(), N);
        const GraphScratchPoolFP16& pool16 = get_thread_local_graph_scratch_pool_fp16(m_state.get(), N);
        const ConvTransposeTensorDescs& td = get_thread_local_conv_transpose_descs(m_state.get(), N);
        cudaStream_t stream = ctx.stream();

        // FP32-side buffers (reusing the existing FP32 pool -- same proven-safe
        // aliasing scheme as the eager path: oint=x1, logits=x3).
        float* g_x1 = pool32.x1; float* g_x3 = pool32.x3;
        float* g_up1 = pool32.up1; float* g_up2 = pool32.up2;
        float* g_oint = g_x1; float* g_logits = g_x3;

        const unsigned total_in  = (unsigned)N * N_BATCH_CHANNELS * W_IN;
        const unsigned total_out = (unsigned)N * W_IN;
        const dim3 grid_out((total_out + block.x - 1) / block.x);

        // Pre-warm every ConvDescriptors' thread-local workspace this sequence
        // touches, BEFORE capture begins (see get_or_capture_cuda_graph's
        // comment for why). Cheap no-op if already warmed (e.g. by the FP32
        // graph on this thread).
        desc.rcbn1_h.ensure_thread_local_workspace();
        desc.rcbn2_h.ensure_thread_local_workspace();
        desc.rcbn3_h.ensure_thread_local_workspace();
        desc.up1c_h.ensure_thread_local_workspace();
        desc.up2c_h.ensure_thread_local_workspace();
        desc.oint.ensure_thread_local_workspace();
        desc.outc.ensure_thread_local_workspace();

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

        run_convbnrelu_half(desc.rcbn1_h, pool16.ncw, pool16.x1,
            desc.rcbn1_w_h, desc.rcbn1_b_h, N_FEAT, W_IN, N, handle, block, ctx);
        run_convbnrelu_half(desc.rcbn2_h, pool16.x1, pool16.up2,
            desc.rcbn2_w_h, desc.rcbn2_b_h, N_FEAT, W_IN, N, handle, block, ctx);
        launch_maxpool_half(pool16.up2, pool16.x2, N, N_FEAT, W_IN, block, ctx);

        run_convbnrelu_half(desc.rcbn3_h, pool16.x2, pool16.up2,
            desc.rcbn3_w_h, desc.rcbn3_b_h, N_FEAT, W_HALF, N, handle, block, ctx);
        launch_maxpool_half(pool16.up2, pool16.x3, N, N_FEAT, W_HALF, block, ctx);

        // ConvTranspose1: needs FP32. Convert fp16 x3 -> g_x3.
        launch_f16_to_f32(g_x3, pool16.x3, N * N_FEAT * W_QTR, block, ctx);
        run_conv_transpose(g_x3, g_up2,
            desc.filter_up1_t, desc.conv_up1_t, td.td_up1_in, td.td_up1_out,
            wb.w_up1t_w, wb.w_up1t_b,
            N, N_FEAT, W_HALF, block, ctx, handle,
            desc.algo_up1_t, desc.ws_up1_t, desc.ws_up1_bytes);

        // up1_c FP16: convert FP32 g_up2 -> fp16 pool16.up2, then conv.
        launch_f32_to_f16(pool16.up2, g_up2, N * N_FEAT * W_HALF, block, ctx);
        run_convbnrelu_half(desc.up1c_h, pool16.up2, pool16.up1,
            desc.up1c_w_h, desc.up1c_b_h, N_FEAT, W_HALF, N, handle, block, ctx);

        // ConvTranspose2: needs FP32. Convert fp16 up1 -> g_up1.
        launch_f16_to_f32(g_up1, pool16.up1, N * N_FEAT * W_HALF, block, ctx);
        run_conv_transpose(g_up1, g_logits,
            desc.filter_up2_t, desc.conv_up2_t, td.td_up2_in, td.td_up2_out,
            wb.w_up2t_w, wb.w_up2t_b,
            N, N_FEAT, W_IN, block, ctx, handle,
            desc.algo_up2_t, desc.ws_up2_t, desc.ws_up2_bytes);

        // up2_c FP16: convert FP32 g_logits -> fp16 pool16.up2, conv -> pool16.x1
        // (free once rcbn2 has read it), then back to FP32 for the output stage.
        launch_f32_to_f16(pool16.up2, g_logits, N * N_FEAT * W_IN, block, ctx);
        run_convbnrelu_half(desc.up2c_h, pool16.up2, pool16.x1,
            desc.up2c_w_h, desc.up2c_b_h, N_FEAT, W_IN, N, handle, block, ctx);
        launch_f16_to_f32(g_up2, pool16.x1, N * N_FEAT * W_IN, block, ctx);

        // Output stage: FP32.
        run_conv(desc.oint, g_up2, g_logits, wb.w_oint_w, wb.w_oint_b, N, N_FEAT, W_IN, block, ctx, handle);
        run_conv(desc.outc, g_logits, g_oint, wb.w_outc_w, wb.w_outc_b, N, 1, W_IN, block, ctx, handle);

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

        printf("[pvfinder_unet] CUDA graph captured (thread_local, FP16 pipeline)\n");
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
    if (!m_state) return;
    UNetState& state = *m_state;
    GlobalDescriptors& desc = state.desc;
    const WeightBlob& wb = state.wb;

    const unsigned n_events = first<host_number_of_events_t>(arguments);

    // One thread_local handle per OS thread — created lazily, routed to this stream.
    cudnnHandle_t handle = Allen::CuDNN::get_thread_local_handle(context.stream());

    // Descriptor creation needs a live handle (for algorithm selection), so it runs
    // here on first operator() call rather than in init(). fwd_ws_budget_bytes is
    // read from whichever call happens to win the call_once race -- fine here since
    // it's a benchmark-only property set once at process/config level, not expected
    // to vary between concurrent operator() calls.
    const unsigned batch_events = m_unet_batch_events.value();
    const int N = (int)batch_events * N_INTERVALS;  // samples per cuDNN batch
    const size_t fwd_ws_budget_bytes = m_fwd_algo_ws_budget_bytes.value();
    std::call_once(state.desc_init_flag, [&state, handle, fwd_ws_budget_bytes, N]() {
        init_descriptors(state.desc, handle, state.wb, fwd_ws_budget_bytes, N);
        // Event processing has started, so every instance's init() (and its
        // weight upload) is done: refuse any later registry allocation.
        Allen::CuDNN::WeightRegistry::instance().lock_allocations();
    });

    const dim3 block = m_block_dim;

    // Scratch buffers (fixed size, reused each event iteration)
    float* x1   = data<dev_unet_x1_t>(arguments);
    float* x2   = data<dev_unet_x2_t>(arguments);
    float* x3   = data<dev_unet_x3_t>(arguments);
    float* up1  = data<dev_unet_up1_t>(arguments);
    float* up2  = data<dev_unet_up2_t>(arguments);

    // Buffer aliases (liveness-proven safe)
    float* oint   = x1;   // x1 is last read by rcbn2, long before oint is written
    float* logits = x3;   // x3 consumed after maxpool; reused as logits

    constexpr unsigned ncw_stride = N_INTERVALS * N_BATCH_CHANNELS * W_IN;
    constexpr unsigned kde_stride = N_INTERVALS * W_IN;

    // ConvTranspose tensor descriptors — thread_local, created once per OS thread
    // and reused for its lifetime (shapes are compile-time constants). See
    // get_thread_local_conv_transpose_descs() for the lazy-init idiom.
    const ConvTransposeTensorDescs& td = get_thread_local_conv_transpose_descs(m_state.get(), N);
    cudnnTensorDescriptor_t td_up1_in      = td.td_up1_in;
    cudnnTensorDescriptor_t td_up1_out     = td.td_up1_out;
    cudnnTensorDescriptor_t td_up2_in      = td.td_up2_in;
    cudnnTensorDescriptor_t td_up2_out     = td.td_up2_out;

    const float* ncw_base = data<dev_pvfinder_interval_features_t>(arguments);
    const char*  ncw_bytes = reinterpret_cast<const char*>(ncw_base);
    float*       kde_base = data<dev_pvfinder_kde_output_t>(arguments);

    const unsigned padded_events = ((n_events + batch_events - 1) / batch_events) * batch_events;
    const unsigned* unet_rows = data<host_pvfinder_unet_rows_t>(arguments);
    const bool compact = unet_rows[0] == 1u;
    const unsigned n_rows = compact ? unet_rows[1] : padded_events * N_INTERVALS;
    const unsigned padded_rows = (n_rows + N - 1) / N * N;
    // Storage type of the interval features (pvfinder_fc_aggregation's
    // unet_input_dtype): float32, or bfloat16 for the BF16 path to read directly.
    const bool input_bf16 = unet_rows[2] == 1u;
    const bool input_nwc = unet_rows[3] == 1u;   // bfloat16 rows stored [bin][channel]
    const size_t input_elem_bytes = input_bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
    // FC pads the interval features to its own unet_batch_events; if the two
    // disagree the last batch would read past that buffer.
    if (size<dev_pvfinder_interval_features_t>(arguments) < (size_t)padded_rows * N_BATCH_CHANNELS * W_IN) {
        throw std::runtime_error(
            "pvfinder_unet: interval features are not padded to unet_batch_events; "
            "set pvfinder_fc_aggregation.unet_batch_events to the same value");
    }

    // BF16 takes precedence over
    // FP16 if both are somehow set (not a supported configuration, just a
    // deterministic tie-break) -- forcing use_fp16 false here means every
    // existing use_fp16-gated branch below (including the CUDA graph FP16
    // path, which BF16 does not support) is correctly bypassed without
    // needing to touch that logic.
    const bool use_bf16 = m_use_bf16.value();
    const bool use_fp16 = m_use_fp16.value() && !use_bf16;
    const std::string& bf16_layout = m_bf16_layout.value();
    if (bf16_layout != "ncw" && bf16_layout != "nwc") {
        throw std::runtime_error("pvfinder_unet: bf16_layout must be ncw or nwc, got '" + bf16_layout + "'");
    }
    const bool bf16_nwc = use_bf16 && bf16_layout == "nwc";
    if (input_nwc && !bf16_nwc) {
        throw std::runtime_error(
            "pvfinder_unet: pvfinder_fc_aggregation.unet_input_layout is nwc, which only the channels-last "
            "BF16 path reads; set pvfinder_unet.use_bf16 = true and bf16_layout = nwc, or unet_input_layout = ncw");
    }
    // The fused kernel is written for this build's shapes (16 feature maps,
    // 4 input channels) and BF16 tensor cores.
    // Otherwise the channels-last BF16 path runs on the cuDNN plans.
    thread_local int tl_cc_major = -1;
    if (tl_cc_major < 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&tl_cc_major, cudaDevAttrComputeCapabilityMajor, device_id);
    }
    const bool use_fused_unet = m_fused_kernel.value() && bf16_nwc && N_FEAT == fused::C &&
                                N_BATCH_CHANNELS == fused::CIN && tl_cc_major >= 8;
    if (bf16_nwc) {
        // Channels-last fused convolutions (plans built in init()): the
        // BN-folded weights transposed to [K][R][C] and rounded to BF16.
        std::call_once(state.nwc_flag, [&]() {
            struct Layer { const float* w; int C_in, R; };
            const Layer layers[5] = {{desc.rcbn1_w_f, N_BATCH_CHANNELS, 25}, {desc.rcbn2_w_f, N_FEAT, 7},
                                     {desc.rcbn3_w_f, N_FEAT, 5}, {desc.up1c_w_f, N_FEAT, 5},
                                     {desc.up2c_w_f, N_FEAT, 5}};
            for (int l = 0; l < 5; ++l) {
                const Layer& L = layers[l];
                std::vector<float> w((size_t)N_FEAT * L.C_in * L.R);
                cudaCheck(cudaMemcpy(w.data(), L.w, w.size() * sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<__nv_bfloat16> t(w.size());
                for (int k = 0; k < N_FEAT; ++k)
                    for (int c = 0; c < L.C_in; ++c)
                        for (int r = 0; r < L.R; ++r)
                            t[((size_t)k * L.R + r) * L.C_in + c] = __float2bfloat16(w[((size_t)k * L.C_in + c) * L.R + r]);
                cudaCheck(cudaMalloc(&state.nwc_w[l], t.size() * sizeof(__nv_bfloat16)));
                cudaCheck(cudaMemcpy(state.nwc_w[l], t.data(), t.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            }
            if (use_fused_unet) {
                // The fused kernel's weight image: the same BF16 [K][R][C]
                // convolution weights and BF16 biases the cuDNN plans use.
                std::vector<float> conv_w[5], conv_b[5], ct_w[2], ct_b[2];
                const __nv_bfloat16* biases[5] = {desc.rcbn1_b_bf, desc.rcbn2_b_bf, desc.rcbn3_b_bf, desc.up1c_b_bf,
                                                  desc.up2c_b_bf};
                for (int l = 0; l < 5; ++l) {
                    const size_t n = (size_t) N_FEAT * layers[l].C_in * layers[l].R;
                    std::vector<__nv_bfloat16> wb16(n), bb16(N_FEAT);
                    cudaCheck(cudaMemcpy(wb16.data(), state.nwc_w[l], n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                    cudaCheck(cudaMemcpy(bb16.data(), biases[l], N_FEAT * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                    conv_w[l].resize(n);
                    for (size_t i = 0; i < n; ++i) conv_w[l][i] = __bfloat162float(wb16[i]);
                    conv_b[l].resize(N_FEAT);
                    for (int i = 0; i < N_FEAT; ++i) conv_b[l][i] = __bfloat162float(bb16[i]);
                }
                const float* ctw[2] = {wb.w_up1t_w, wb.w_up2t_w};
                const float* ctb[2] = {wb.w_up1t_b, wb.w_up2t_b};
                for (int t = 0; t < 2; ++t) {
                    ct_w[t].resize((size_t) N_FEAT * N_FEAT * 2);
                    ct_b[t].resize(N_FEAT);
                    cudaCheck(cudaMemcpy(ct_w[t].data(), ctw[t], ct_w[t].size() * sizeof(float), cudaMemcpyDeviceToHost));
                    cudaCheck(cudaMemcpy(ct_b[t].data(), ctb[t], N_FEAT * sizeof(float), cudaMemcpyDeviceToHost));
                }
                std::vector<float> out_params(OutputStage<N_FEAT>::n_params);
                cudaCheck(cudaMemcpy(out_params.data(), desc.output_stage_params, out_params.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost));
                const std::vector<unsigned char> blob = fused::make_fused_unet_blob(conv_w, conv_b, ct_w, ct_b, out_params);
                cudaCheck(cudaMalloc(&state.fused_blob, blob.size()));
                cudaCheck(cudaMemcpy(state.fused_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice));
                cudaCheck(cudaFuncSetAttribute(fused::fused_unet_bf16_kernel,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize, fused::SMEM_BYTES));
                int device_id = 0, sm_count = 0, per_sm = 0;
                cudaCheck(cudaGetDevice(&device_id));
                cudaCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
                cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fused::fused_unet_bf16_kernel,
                                                                        fused::THREADS, fused::SMEM_BYTES));
                state.fused_grid = sm_count * std::max(per_sm, 1);
            }
        });
    }
    if (input_bf16 && !use_bf16) {
        throw std::runtime_error(
            "pvfinder_unet: pvfinder_fc_aggregation.unet_input_dtype is bfloat16, which only the BF16 "
            "path reads; set pvfinder_unet.use_bf16 = true or unet_input_dtype = float32");
    }
    // CUDA graph path. BF16 has no CUDA-graph-capture variant (eager-path-only) --
    // excluded here so a use_bf16=true call always takes the eager branch
    // below, never the FP32 graph path (which use_fp16=false alone, forced
    // above when use_bf16 is set, would otherwise incorrectly make eligible).
    const bool graph_eligible = m_use_cuda_graph.value() && !use_bf16;
    // True single-pass Conv+Bias+ReLU for rcbn1, eager FP32 path only. FP16
    // has no fused-graph variant (see m_use_fused_cbr's doc comment), so
    // this is simply ignored whenever use_fp16=true.
    const bool use_fused_cbr = m_use_fused_cbr.value() && desc.rcbn1_fused_available;
    // Hand-written fused rcbn3, eager FP32 path only.
    const bool use_fused_rcbn3 = m_use_fused_rcbn3.value();
    const bool fuse_pool       = m_use_fused_bias_relu_pool.value();
    // Merged up1, eager FP32 path only.
    const bool use_merged_up1 = m_use_merged_up1.value();
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
        thread_local std::unordered_map<const void*, bool> tl_warmed_by_instance;
        bool& tl_warmed = tl_warmed_by_instance[m_state.get()];
        if (!tl_warmed) {
            // Only the active precision's descriptors: workspace sizes grow
            // with N and these allocations sit outside Allen's -m pool.
            desc.rcbn1.ensure_thread_local_workspace();
            desc.rcbn2.ensure_thread_local_workspace();
            desc.rcbn3.ensure_thread_local_workspace();
            desc.up1_c.ensure_thread_local_workspace();
            desc.up2_c.ensure_thread_local_workspace();
            desc.oint.ensure_thread_local_workspace();
            desc.outc.ensure_thread_local_workspace();
            if (use_fp16) {
                desc.rcbn1_h.ensure_thread_local_workspace();
                desc.rcbn2_h.ensure_thread_local_workspace();
                desc.rcbn3_h.ensure_thread_local_workspace();
                desc.up1c_h.ensure_thread_local_workspace();
                desc.up2c_h.ensure_thread_local_workspace();
            }
            if (use_bf16) {
                desc.rcbn1_bf.ensure_thread_local_workspace();
                desc.rcbn2_bf.ensure_thread_local_workspace();
                desc.rcbn3_bf.ensure_thread_local_workspace();
                desc.up1c_bf.ensure_thread_local_workspace();
                desc.up2c_bf.ensure_thread_local_workspace();
            }
            if (use_fused_cbr) desc.rcbn1_fused.ensure_thread_local_workspace();
            tl_warmed = true;
        }
    }

    // FP16 pool pointers (only used when use_fp16 is true).
    //
    // Deliberately NOT desc.fp16_* here: those are a single process-wide
    // shared allocation (see GlobalDescriptors::fp16_pool) -- with many OS
    // threads (one per Allen Stream, per -t N) all running the eager FP16
    // path concurrently, every thread would read/write the EXACT SAME
    // fp16_ncw/x1/x2/x3/up1/up2 addresses simultaneously with zero
    // synchronization. This is very likely the root cause of the intermittent
    // CUDNN_STATUS_BAD_PARAM crashes seen under sustained -t16 load: this bug
    // predates today's session (flagged earlier as a known-but-unfixed issue
    // when the FP16 CUDA graph path was built, since that new code correctly
    // used its own thread_local pool instead and has run crash-free). Reusing
    // that same thread_local GraphScratchPoolFP16 here for the eager path
    // fixes it the same way, using infrastructure already built and validated.
    const GraphScratchPoolFP16* fp16_pool_tl = use_fp16 ? &get_thread_local_graph_scratch_pool_fp16(m_state.get(), N) : nullptr;
    __half* fp16_ncw  = use_fp16 ? fp16_pool_tl->ncw  : nullptr;
    __half* fp16_x1   = use_fp16 ? fp16_pool_tl->x1   : nullptr;
    __half* fp16_x2   = use_fp16 ? fp16_pool_tl->x2   : nullptr;
    __half* fp16_x3   = use_fp16 ? fp16_pool_tl->x3   : nullptr;
    __half* fp16_up1  = use_fp16 ? fp16_pool_tl->up1  : nullptr;
    __half* fp16_up2  = use_fp16 ? fp16_pool_tl->up2  : nullptr;

    // BF16 pool pointers (only used when use_bf16 is true) -- same
    // thread_local-per-OS-thread rationale as the FP16 pool above.
    const GraphScratchPoolBF16* bf16_pool_tl = use_bf16 ? &get_thread_local_graph_scratch_pool_bf16(m_state.get(), N) : nullptr;
    __nv_bfloat16* bf16_ncw  = use_bf16 ? bf16_pool_tl->ncw  : nullptr;
    __nv_bfloat16* bf16_x1   = use_bf16 ? bf16_pool_tl->x1   : nullptr;
    __nv_bfloat16* bf16_x2   = use_bf16 ? bf16_pool_tl->x2   : nullptr;
    __nv_bfloat16* bf16_x3   = use_bf16 ? bf16_pool_tl->x3   : nullptr;
    __nv_bfloat16* bf16_up1  = use_bf16 ? bf16_pool_tl->up1  : nullptr;
    __nv_bfloat16* bf16_up2  = use_bf16 ? bf16_pool_tl->up2  : nullptr;

    // One cuDNN batch: N rows of features at ncw -> N rows of KDE at kde.
    // ncw_in is float, or __nv_bfloat16 when input_bf16 (only the BF16 branch reads it then).
    auto run_batch = [&](const void* ncw_in, float* kde) {
        const float* ncw = static_cast<const float*>(ncw_in);

        if (use_graph_fp32) {
            // ---- CUDA graph path (FP32) ----
            // Captures once (thread_local, lazy); every call after the first just
            // patches the two shuttle-kernel nodes' pointers and replays.
            cudaGraphExec_t graphExec  = nullptr;
            cudaGraphNode_t copyInNode = nullptr, copyOutNode = nullptr;
            get_or_capture_cuda_graph(handle, block, context, ncw, kde, graphExec, copyInNode, copyOutNode);
            const GraphScratchPool& pool = get_thread_local_graph_scratch_pool(m_state.get(), N);

            const int total_in  = (int)(N * N_BATCH_CHANNELS * W_IN);
            const int total_out = (int)(N * W_IN);
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
            return;
        } else if (use_graph_fp16) {
            // ---- CUDA graph path (FP16) ----
            // Same replay pattern as the FP32 graph, but the copy-in node is the
            // leading f32_to_f16 conversion (dst fixed, src patched), not a plain
            // squeeze_copy_kernel — different kernel, different argument order.
            cudaGraphExec_t graphExec  = nullptr;
            cudaGraphNode_t copyInNode = nullptr, copyOutNode = nullptr;
            get_or_capture_cuda_graph_fp16(handle, block, context, ncw, kde, graphExec, copyInNode, copyOutNode);
            const GraphScratchPool& pool = get_thread_local_graph_scratch_pool(m_state.get(), N);
            const GraphScratchPoolFP16& pool16 = get_thread_local_graph_scratch_pool_fp16(m_state.get(), N);

            const int total_in  = (int)(N * N_BATCH_CHANNELS * W_IN);
            const int total_out = (int)(N * W_IN);
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
            return;
        } else if (use_bf16) {
            // ---- BF16 path, no FP32 <-> BF16 conversion passes ----
            // Every activation stays BF16 and all arithmetic is FP32: cuDNN's
            // BF16 convolutions accumulate in FP32, and so do the two
            // hand-written stages (PVFinderUNetLowPrecision.cuh). The input
            // arrives in BF16 when pvfinder_fc_aggregation.unet_input_dtype
            // is "bfloat16"; otherwise it is converted once here. Both
            // ConvTransposes run in conv_transpose_k2s2_kernel, and
            // output_stage_kernel does out_intermediate, outc, softplus and
            // the FP32 KDE write in one pass, so the shared FP32 tail below
            // is skipped (out_intermediate and outc are applied as their exact
            // composition, one 9-tap convolution). Eager path only (no CUDA
            // graph variant).
            if (bf16_nwc) {
                // Channels last: every CBR layer is one fused cuDNN graph plan
                // (bias + ReLU included); max-pools, ConvTransposes and the
                // output stage run in NWC. The FC stage writes the input
                // channels last (unet_input_layout = nwc); otherwise it is
                // transposed (and converted) once here.
                const __nv_bfloat16* const* w = state.nwc_w;
                const __nv_bfloat16* in_nwc = bf16_ncw;
                if (input_nwc) {
                    in_nwc = static_cast<const __nv_bfloat16*>(ncw_in);
                }
                else if (input_bf16) {
                    launch_ncw_to_nwc(static_cast<const __nv_bfloat16*>(ncw_in), bf16_ncw, N, N_BATCH_CHANNELS, W_IN, context);
                }
                else {
                    launch_ncw_to_nwc(ncw, bf16_ncw, N, N_BATCH_CHANNELS, W_IN, context);
                }
                if (use_fused_unet) {
                    fused::fused_unet_bf16_kernel<<<std::min(state.fused_grid, (N + fused::WARPS - 1) / fused::WARPS),
                                                    fused::THREADS, fused::SMEM_BYTES, context.stream()>>>(
                        in_nwc, state.fused_blob, kde, KDE_SCALE, N, nullptr, nullptr, nullptr, 0);
                    return;
                }
                state.nwc_conv[0].execute(handle, in_nwc, w[0], desc.rcbn1_b_bf, bf16_x1);
                state.nwc_conv[1].execute(handle, bf16_x1, w[1], desc.rcbn2_b_bf, bf16_up2);
                launch_maxpool2_nwc(bf16_up2, bf16_x2, N, N_FEAT, W_IN, context);
                state.nwc_conv[2].execute(handle, bf16_x2, w[2], desc.rcbn3_b_bf, bf16_up2);
                launch_maxpool2_nwc(bf16_up2, bf16_x3, N, N_FEAT, W_HALF, context);
                launch_conv_transpose_k2s2_nwc<__nv_bfloat16, N_FEAT>(bf16_x3, bf16_up2, wb.w_up1t_w, wb.w_up1t_b,
                    N, W_QTR, context);
                state.nwc_conv[3].execute(handle, bf16_up2, w[3], desc.up1c_b_bf, bf16_up1);
                launch_conv_transpose_k2s2_nwc<__nv_bfloat16, N_FEAT>(bf16_up1, bf16_up2, wb.w_up2t_w, wb.w_up2t_b,
                    N, W_HALF, context);
                state.nwc_conv[4].execute(handle, bf16_up2, w[4], desc.up2c_b_bf, bf16_x1);
                launch_output_stage_nwc<__nv_bfloat16, N_FEAT, W_IN>(bf16_x1, desc.output_stage_params,
                    kde, KDE_SCALE, N, context);
                return;
            }
            const __nv_bfloat16* in_bf = static_cast<const __nv_bfloat16*>(ncw_in);
            if (!input_bf16) {
                launch_f32_to_bf16(bf16_ncw, ncw, N * N_BATCH_CHANNELS * W_IN, block, context);
                in_bf = bf16_ncw;
            }

            // Only rcbn1 keeps a separate bias + ReLU pass (its consumer is a
            // cuDNN convolution). rcbn2 and rcbn3 fold theirs into the
            // max-pool, up1c into the second ConvTranspose's input load and
            // up2c into the output stage's, so those convolutions write
            // their raw output.
            // Encoder
            run_convbnrelu_bf16(desc.rcbn1_bf, in_bf, bf16_x1,
                desc.rcbn1_w_bf, desc.rcbn1_b_bf, N_FEAT, W_IN, N, handle, block, context);
            desc.rcbn2_bf.forward_bf16(handle, 1.f, 0.f, bf16_x1, desc.rcbn2_w_bf, bf16_up2);
            launch_bias_relu_maxpool2(bf16_up2, bf16_x2, desc.rcbn2_b_bf, N, N_FEAT, W_IN, context);
            desc.rcbn3_bf.forward_bf16(handle, 1.f, 0.f, bf16_x2, desc.rcbn3_w_bf, bf16_up2);
            launch_bias_relu_maxpool2(bf16_up2, bf16_x3, desc.rcbn3_b_bf, N, N_FEAT, W_HALF, context);

            // Decoder: ConvTranspose (W_QTR -> W_HALF), conv, ConvTranspose (W_HALF -> W_IN), conv
            launch_conv_transpose_k2s2<__nv_bfloat16, N_FEAT>(bf16_x3, bf16_up2, wb.w_up1t_w, wb.w_up1t_b, nullptr,
                N, W_QTR, context);
            desc.up1c_bf.forward_bf16(handle, 1.f, 0.f, bf16_up2, desc.up1c_w_bf, bf16_up1);
            launch_conv_transpose_k2s2<__nv_bfloat16, N_FEAT>(bf16_up1, bf16_up2, wb.w_up2t_w, wb.w_up2t_b, desc.up1c_b_bf,
                N, W_HALF, context);
            desc.up2c_bf.forward_bf16(handle, 1.f, 0.f, bf16_up2, desc.up2c_w_bf, bf16_x1);

            // Output stage (with up2c's bias + ReLU), straight to the FP32 KDE rows
            launch_output_stage<__nv_bfloat16, N_FEAT, W_IN>(bf16_x1, desc.up2c_b_bf, desc.output_stage_params,
                kde, KDE_SCALE, N, context);
            return;
        } else if (!use_fp16) {
            // ---- FP32 path (Phase L baseline) ----
            if (use_fused_cbr) {
                // Single-pass Conv+Bias+ReLU: no separate bias_relu_kernel launch,
                // no extra DRAM round trip on the conv output.
                desc.rcbn1_fused.execute(handle, ncw, desc.rcbn1_w_f, desc.rcbn1_b_f, x1);
            } else {
                run_convbnrelu(desc.rcbn1, ncw, x1,  desc.rcbn1_w_f, desc.rcbn1_b_f, N_FEAT, W_IN,   N, handle, block, context);
            }
            if (fuse_pool) {
                // conv writes raw output; bias, ReLU and pooling happen in one
                // pass, so the full-resolution activation is read once and
                // never rewritten.
                run_conv(desc.rcbn2, x1, up2, desc.rcbn2_w_f, nullptr,
                         N, N_FEAT, W_IN, block, context, handle);
                launch_bias_relu_maxpool(up2, x2, desc.rcbn2_b_f, N, N_FEAT, W_IN, block, context);
            } else {
                run_convbnrelu(desc.rcbn2, x1,  up2, desc.rcbn2_w_f, desc.rcbn2_b_f, N_FEAT, W_IN,   N, handle, block, context);
                launch_maxpool(up2, x2, N, N_FEAT, W_IN, block, context);
            }

            if (use_fused_rcbn3) {
                // Single kernel: conv + bias + ReLU with the activation slice
                // kept in shared memory, no DRAM round trip on the raw conv output.
                launch_fused_rcbn3(x2, up2, desc.rcbn3_w_f, desc.rcbn3_b_f, N, block, context);
            } else if (fuse_pool) {
                run_conv(desc.rcbn3, x2, up2, desc.rcbn3_w_f, nullptr,
                         N, N_FEAT, W_HALF, block, context, handle);
                launch_bias_relu_maxpool(up2, x3, desc.rcbn3_b_f, N, N_FEAT, W_HALF, block, context);
            } else {
                run_convbnrelu(desc.rcbn3, x2, up2, desc.rcbn3_w_f, desc.rcbn3_b_f, N_FEAT, W_HALF, N, handle, block, context);
            }
            if (!fuse_pool || use_fused_rcbn3) {
                launch_maxpool(up2, x3, N, N_FEAT, W_HALF, block, context);
            }

            // Merged up1 ConvTranspose+Conv+BiasReLU -- writes x3 -> up1 directly,
            // skipping the intermediate `up2` scratch write entirely
            // (safe: up2 is unconditionally overwritten again below
            // before anything reads it).
            if (use_merged_up1) {
                launch_up1_merge(x3, up1,
                    desc.up1_merge_K_even, desc.up1_merge_K_odd, desc.up1_merge_bias,
                    wb.w_up1t_w, wb.w_up1t_b, desc.up1c_w_f, desc.up1c_b_f,
                    N_FEAT, W_QTR, N, block, context);
            } else {
                run_conv_transpose(x3, up2,
                    desc.filter_up1_t, desc.conv_up1_t, td_up1_in, td_up1_out,
                    wb.w_up1t_w, wb.w_up1t_b,
                    N, N_FEAT, W_HALF, block, context, handle,
                    desc.algo_up1_t, desc.ws_up1_t, desc.ws_up1_bytes);
                run_convbnrelu(desc.up1_c, up2, up1, desc.up1c_w_f, desc.up1c_b_f, N_FEAT, W_HALF, N, handle, block, context);
            }

            // up2: ConvTranspose (W_HALF -> W_IN) + ConvBNReLU.
            run_conv_transpose(up1, logits,
                desc.filter_up2_t, desc.conv_up2_t, td_up2_in, td_up2_out,
                wb.w_up2t_w, wb.w_up2t_b,
                N, N_FEAT, W_IN, block, context, handle,
                desc.algo_up2_t, desc.ws_up2_t, desc.ws_up2_bytes);
            run_convbnrelu(desc.up2_c, logits, up2, desc.up2c_w_f, desc.up2c_b_f, N_FEAT, W_IN, N, handle, block, context);
        } else {
            // ---- FP16 path (Phase M benchmark) ----
            // CBR layers run as Tensor Core FP16 convs; ConvTranspose and output
            // layers stay FP32. Explicit F32↔F16 conversions at the boundaries.

            // Encoder
            launch_f32_to_f16(fp16_ncw, ncw, N * N_BATCH_CHANNELS * W_IN, block, context);
            run_convbnrelu_half(desc.rcbn1_h, fp16_ncw, fp16_x1,
                desc.rcbn1_w_h, desc.rcbn1_b_h, N_FEAT, W_IN, N, handle, block, context);
            run_convbnrelu_half(desc.rcbn2_h, fp16_x1, fp16_up2,
                desc.rcbn2_w_h, desc.rcbn2_b_h, N_FEAT, W_IN, N, handle, block, context);
            launch_maxpool_half(fp16_up2, fp16_x2, N, N_FEAT, W_IN, block, context);

            run_convbnrelu_half(desc.rcbn3_h, fp16_x2, fp16_up2,
                desc.rcbn3_w_h, desc.rcbn3_b_h, N_FEAT, W_HALF, N, handle, block, context);
            launch_maxpool_half(fp16_up2, fp16_x3, N, N_FEAT, W_HALF, block, context);

            // ConvTranspose1: needs FP32. Convert fp16_x3 → x3.
            launch_f16_to_f32(x3, fp16_x3, N * N_FEAT * W_QTR, block, context);
            run_conv_transpose(x3, up2,
                desc.filter_up1_t, desc.conv_up1_t, td_up1_in, td_up1_out,
                wb.w_up1t_w, wb.w_up1t_b,
                N, N_FEAT, W_HALF, block, context, handle,
                desc.algo_up1_t, desc.ws_up1_t, desc.ws_up1_bytes);

            // up1_c FP16: convert FP32 up2 → fp16_up2, then conv.
            launch_f32_to_f16(fp16_up2, up2, N * N_FEAT * W_HALF, block, context);
            run_convbnrelu_half(desc.up1c_h, fp16_up2, fp16_up1,
                desc.up1c_w_h, desc.up1c_b_h, N_FEAT, W_HALF, N, handle, block, context);

            // ConvTranspose2: needs FP32. Convert fp16_up1 → up1.
            launch_f16_to_f32(up1, fp16_up1, N * N_FEAT * W_HALF, block, context);
            run_conv_transpose(up1, logits,
                desc.filter_up2_t, desc.conv_up2_t, td_up2_in, td_up2_out,
                wb.w_up2t_w, wb.w_up2t_b,
                N, N_FEAT, W_IN, block, context, handle,
                desc.algo_up2_t, desc.ws_up2_t, desc.ws_up2_bytes);

            // up2_c FP16: convert FP32 logits → fp16_up2, conv → fp16_x1 (free
            // once rcbn2 has read it), then back to FP32 up2 for the output stage.
            launch_f32_to_f16(fp16_up2, logits, N * N_FEAT * W_IN, block, context);
            run_convbnrelu_half(desc.up2c_h, fp16_up2, fp16_x1,
                desc.up2c_w_h, desc.up2c_b_h, N_FEAT, W_IN, N, handle, block, context);
            launch_f16_to_f32(up2, fp16_x1, N * N_FEAT * W_IN, block, context);
        }

        // Output stage, FP32 in every eager path: out_intermediate, outc, softplus.
        run_conv(desc.oint, up2, logits, wb.w_oint_w, wb.w_oint_b, N, N_FEAT, W_IN, block, context, handle);
        run_conv(desc.outc, logits, oint, wb.w_outc_w, wb.w_outc_b, N, 1, W_IN, block, context, handle);
        launch_softplus_scale(oint, KDE_SCALE, N * W_IN, block, context);
        squeeze_copy_kernel<<<
            ((unsigned)(N * W_IN) + block.x - 1) / block.x, block,
            0, context.stream()>>>(oint, kde, N * W_IN);
    };

    if (!compact) {
        for (unsigned chunk_start = 0; chunk_start < padded_events; chunk_start += batch_events) {
            run_batch(ncw_bytes + (size_t)chunk_start * ncw_stride * input_elem_bytes,
                      kde_base + chunk_start * kde_stride);
        }
    } else {
        // The UNet's response to an all-zero interval, through exactly the
        // path configured for the real batches, so skipped intervals get
        // bit-for-bit what running them would have produced. Once per
        // instance; other threads wait in call_once until it is on the device.
        std::call_once(state.empty_response_flag, [&]() {
            float* zeros = nullptr;
            float* out = nullptr;
            cudaCheck(cudaMalloc(&zeros, (size_t)N * N_BATCH_CHANNELS * W_IN * sizeof(float)));
            cudaCheck(cudaMalloc(&out, (size_t)N * W_IN * sizeof(float)));
            cudaCheck(cudaMemsetAsync(zeros, 0, (size_t)N * N_BATCH_CHANNELS * W_IN * sizeof(float), context.stream()));
            run_batch(zeros, out);
            cudaCheck(cudaStreamSynchronize(context.stream()));
            cudaCheck(cudaFree(zeros));
            state.empty_response = out;   // row 0; never freed, like the scratch pools
        });

        float* kde_rows = data<dev_unet_kde_rows_t>(arguments);
        const unsigned row_stride = N_BATCH_CHANNELS * W_IN;
        const int n_slots = (int)(n_events * N_INTERVALS);
        if (use_fused_unet && input_nwc) {
            // Every row in one launch (the kernel has no batch size), each
            // row's KDE straight to its slot, and the empty-interval response
            // to the slots without a row: no expansion pass.
            if (n_slots > 0) {
                // See m_fused_grid_fraction: leave room for other streams' kernels.
                const int grid = std::max(1, (int) (state.fused_grid * m_fused_grid_fraction.value()));
                const int blocks = std::max(1, std::min(grid, (int) (n_rows + fused::WARPS - 1) / fused::WARPS));
                fused::fused_unet_bf16_kernel<<<std::max(blocks, std::min(grid, n_slots / fused::THREADS + 1)),
                                                fused::THREADS, fused::SMEM_BYTES, context.stream()>>>(
                    static_cast<const __nv_bfloat16*>(static_cast<const void*>(ncw_base)), state.fused_blob, kde_base,
                    KDE_SCALE, (int) n_rows, data<dev_pvfinder_row_slot_t>(arguments),
                    data<dev_pvfinder_slot_row_t>(arguments), state.empty_response, n_slots);
            }
        }
        else {
            for (unsigned row = 0; row < n_rows; row += N) {
                run_batch(ncw_bytes + (size_t)row * row_stride * input_elem_bytes, kde_rows + (size_t)row * W_IN);
            }
            const unsigned threads = n_slots * (W_IN / 4);
            if (threads > 0) {
                expand_kde_rows_kernel<<<(threads + 255) / 256, 256, 0, context.stream()>>>(
                    kde_rows, data<dev_pvfinder_slot_row_t>(arguments), state.empty_response, kde_base, n_slots);
            }
        }
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
        // Interval features as float, whatever their storage type (a BF16
        // value is the upper half of the float with the same value).
        auto copy_features = [&](float* dst, size_t count) {
            if (input_bf16) {
                std::vector<uint16_t> raw(count);
                cudaMemcpy(raw.data(), data<dev_pvfinder_interval_features_t>(arguments),
                           count * sizeof(uint16_t), cudaMemcpyDeviceToHost);
                constexpr size_t row = N_BATCH_CHANNELS * W_IN;
                for (size_t i = 0; i < count; ++i) {
                    // channels-last rows back to [channel][bin]
                    const size_t r = i / row, e = i % row;
                    const size_t src = input_nwc ? r * row + (e % W_IN) * N_BATCH_CHANNELS + e / W_IN : i;
                    const uint32_t bits = (uint32_t)raw[src] << 16;
                    std::memcpy(dst + i, &bits, sizeof(float));
                }
            } else {
                cudaMemcpy(dst, data<dev_pvfinder_interval_features_t>(arguments),
                           count * sizeof(float), cudaMemcpyDeviceToHost);
            }
        };
        if (compact) {
            // Compact rows back to [event][interval]; skipped intervals as zeros.
            std::vector<int> h_slot_row(n_events * N_INTERVALS);
            std::vector<float> h_rows((size_t)n_rows * N_BATCH_CHANNELS * W_IN);
            cudaMemcpy(h_slot_row.data(), data<dev_pvfinder_slot_row_t>(arguments),
                       h_slot_row.size() * sizeof(int), cudaMemcpyDeviceToHost);
            copy_features(h_rows.data(), h_rows.size());
            constexpr size_t row_floats = N_BATCH_CHANNELS * W_IN;
            for (size_t slot = 0; slot < h_slot_row.size(); ++slot) {
                if (h_slot_row[slot] >= 0) {
                    std::memcpy(h_ncw.data() + slot * row_floats,
                                h_rows.data() + (size_t)h_slot_row[slot] * row_floats, row_floats * sizeof(float));
                }
            }
        } else {
            copy_features(h_ncw.data(), ncw_elems);
        }
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
