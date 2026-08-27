#pragma once

#include "AlgorithmTypes.cuh"
#ifdef ALLEN_CUDNN_BACKEND_CUDA
#include "AllenCuDNN.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

// ---------------------------------------------------------------------------
// PVFinderUNet: cuDNN-backed UNet inference algorithm.
//
// Input:  dev_pvfinder_interval_features  [n_events, 40, C=8, W=100]
// Output: dev_pvfinder_kde_output  [n_events, 40, 100]  (flat: n_events*4000)
//
// cuDNN integration design:
//   - All tensor shapes are compile-time constants — descriptors created once globally.
//   - One thread_local cudnnHandle_t per OS thread, created lazily via
//     Allen::CuDNN::get_thread_local_handle(stream) — no per-instance handle.
//   - IMPLICIT_GEMM algorithm pinned everywhere → zero workspace.
//   - Weight tensors loaded once into WeightRegistry via std::call_once.
// ---------------------------------------------------------------------------

namespace pvfinder_unet {

// UNet architecture constants (default weights: HDplusUNet100 iter12Ca)
// Override N_FEAT at build time with -DPVFINDER_UNET_N_FEAT=<n> (e.g. 16 for the lighter model).
// Override N_BATCH_CHANNELS (the FC/UNet handoff's latentChannels) at build
// time with -DPVFINDER_UNET_N_BATCH_CHANNELS=<n> to run a model trained
// with a different latent-channel count.
#ifdef PVFINDER_UNET_N_BATCH_CHANNELS
static constexpr int N_BATCH_CHANNELS = PVFINDER_UNET_N_BATCH_CHANNELS;
#else
static constexpr int N_BATCH_CHANNELS = 8;   // input latent channels
#endif
#ifdef PVFINDER_UNET_N_FEAT
static constexpr int N_FEAT    = PVFINDER_UNET_N_FEAT;
#else
static constexpr int N_FEAT    = 64;          // feature maps throughout
#endif
static constexpr int W_IN      = 100;         // input width
static constexpr int W_HALF    = 50;          // after first MaxPool
static constexpr int W_QTR     = 25;          // after second MaxPool
static constexpr int N_INTERVALS = 40;
static constexpr float KDE_SCALE = 0.001f;

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;

    // Interval features from aggregation: [n_events, 40, C=8, W=100]
    DEVICE_INPUT(dev_pvfinder_interval_features_t, float) dev_pvfinder_interval_features;

    // Scratch intermediate buffers (Allen pool, fixed size for ONE event reused each iteration)
    DEVICE_OUTPUT(dev_unet_x1_t,      float) dev_unet_x1;    // [N, 64, 100]
    DEVICE_OUTPUT(dev_unet_x2_t,      float) dev_unet_x2;    // [N, 64, 50]
    DEVICE_OUTPUT(dev_unet_x3_t,      float) dev_unet_x3;    // [N, 64, 100] (also logits)
    DEVICE_OUTPUT(dev_unet_up1_t,     float) dev_unet_up1;   // [N, 64, 50]
    DEVICE_OUTPUT(dev_unet_cat2_t,    float) dev_unet_cat2;  // [N, 128, 50] (also up2[N,64,100])
    // conv_ws: IMPLICIT_GEMM needs 0 workspace; allocate 1 float as Allen requires non-zero size.
    DEVICE_OUTPUT(dev_unet_conv_ws_t, float) dev_unet_conv_ws;

    // Final KDE output: [n_events * 40 * 100] floats
    DEVICE_OUTPUT(dev_pvfinder_kde_output_t, float) dev_pvfinder_kde_output;
};

struct pvfinder_unet_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
        ArgumentReferences<Parameters> arguments,
        const RuntimeOptions&,
        const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) const;

    void init();

    ~pvfinder_unet_t() = default;

private:
    Allen::Property<std::string> m_weight_file {
        this, "weight_file", "/data/home/melashri/iris/inference/cnn_weights.bin",
        "path to cnn_weights.bin produced by convert_cnn_weights.py"};

    Allen::Property<dim3> m_block_dim {
        this, "block_dim", {256, 1, 1}, "CUDA block dim for element-wise kernels"};

    Allen::Property<std::string> m_dump_dir {
        this, "dump_validation", "",
        "if non-empty, dump NCW input + KDE output of the first slice to this directory"};

    // Which operator() call (0-indexed) to dump on. Default 0 preserves prior
    // behaviour (dump the first call). A later index is needed to validate the
    // CUDA-graph path specifically: Allen's memory manager reshuffles argument
    // addresses every repetition, so the graph path's real correctness risk
    // (replaying against a stale/patched pointer after that reshuffle) only
    // manifests on calls after the first -- dumping only the first call cannot
    // exercise it.
    Allen::Property<unsigned> m_dump_repetition {
        this, "dump_repetition", 0u,
        "0-indexed operator() call to dump on, when dump_validation is set"};

    Allen::Property<bool> m_use_fp16 {
        this, "use_fp16", false,
        "use FP16 Tensor Core path for CBR layers (physics approximate)"};

    // BF16 Tensor Core path for CBR layers, same structure as m_use_fp16
    // (ConvTranspose and the output stage stay FP32 either way -- see the
    // eager-path comment in the .cu). Motivated by a confirmed FP16 bug:
    // input values up to ~109,000 exceed FP16's ~65504 max representable
    // magnitude at the very first f32->half cast, producing real NaN on
    // ~0.5% of real intervals. BF16 shares FP32's exponent range, so that
    // specific overflow cannot recur here -- still physics-approximate
    // (reduced mantissa, like FP16), not physics-exact. If both use_fp16 and
    // use_bf16 are set, BF16 takes precedence (see operator()'s branch
    // order) -- an arbitrary but deterministic choice, not expected to
    // matter since setting both is not a supported configuration.
    Allen::Property<bool> m_use_bf16 {
        this, "use_bf16", false,
        "BF16 Tensor Core path for CBR layers (physics approximate, but "
        "avoids the FP16 path's confirmed overflow-to-NaN failure mode). "
        "Takes precedence over use_fp16 if both are set."};

    // True single-pass Conv+Bias+ReLU for rcbn1, via the cuDNN backend graph API
    // (Allen::CuDNN::ConvBiasReluGraph), instead of a cuDNN conv followed by a
    // separate bias_relu_kernel pass. Physics-identical (BN is already folded
    // into the weights either way); this only changes how many times the
    // conv output round-trips through DRAM. FP32 only — ConvBiasReluGraph has
    // no FP16 variant, so this flag has no effect when use_fp16=true. Falls
    // back to the existing two-pass path at runtime if the backend graph API
    // finds no usable engine for this shape on the current GPU (see
    // ConvBiasReluGraph::create()'s doc comment).
    Allen::Property<bool> m_use_fused_cbr {
        this, "use_fused_cbr", false,
        "rcbn1 only, FP32 only: use single-pass cuDNN backend-graph Conv+Bias+ReLU "
        "instead of conv + separate bias/ReLU kernel (falls back automatically if "
        "unsupported on this GPU)"};

    // Forward-conv algorithm selection, bounded to a workspace budget instead
    // of an unrestricted Find/heur_v7 search (see CuDNNDescriptors.h's
    // ConvDescriptors class comment for why unrestricted search is not used
    // by default). 0 (default) keeps every forward conv pinned to
    // IMPLICIT_GEMM, bit-for-bit identical to current production behaviour.
    // A nonzero value applies to ALL forward ConvDescriptors (rcbn1/2/3,
    // up1_c, up2_c, oint_half, outc, and their FP16 counterparts) uniformly.
    Allen::Property<unsigned> m_fwd_algo_ws_budget_bytes {
        this, "fwd_algo_ws_budget_bytes", 0u,
        "0 (default): pin IMPLICIT_GEMM everywhere (no search). Nonzero: bounded "
        "cudnnGetConvolutionForwardAlgorithm_v7 search, adopting the top candidate "
        "whose workspace fits this many bytes (falls back to IMPLICIT_GEMM if none "
        "fits or the query fails)"};

    // Hand-written fused Conv+Bias+ReLU for rcbn3 only (the eager FP32 path
    // only -- not FP16, not the CUDA-graph path), replacing cuDNN + a
    // separate bias_relu_kernel pass with one
    // kernel that keeps the activation slice in shared memory instead of
    // round-tripping the raw conv output through DRAM. Physics-identical
    // (same weights/bias, same cross-correlation math) when correct; default
    // off pending the accompanying correctness sanity check.
    Allen::Property<bool> m_use_fused_rcbn3 {
        this, "use_fused_rcbn3", false,
        "rcbn3 only, eager FP32 path only: use a hand-written shared-memory "
        "fused Conv+Bias+ReLU kernel instead of cuDNN + separate bias/ReLU kernel"};

    // Merge out_intermediate+outc into a single Conv1d(k=9) per branch,
    // exact everywhere (interior via the merged kernel, boundary via the
    // original nested formula -- see oint_outc_merged_kernel). Eager FP32
    // path, skip_mode=="concat" only; ignored otherwise (falls back to the
    // existing unmerged path). Measured as a net throughput regression (the
    // hand-written merged kernel loses to tuned cuDNN despite fewer FLOPs) --
    // kept for reference, default off.
    Allen::Property<bool> m_use_merged_oint_outc {
        this, "use_merged_oint_outc", false,
        "eager FP32 path, skip_mode=concat only: replace the two-branch "
        "oint_half + bias_add + outc sequence with a single merged "
        "Conv1d(k=9)-per-branch kernel (exact, incl. boundary)"};

    // Merge up1's ConvTranspose1d(k=2,s=2) + Conv1d(k=5,pad=2) into one
    // phase-dependent kernel (exact, incl. boundary -- see up1_merge_kernel's
    // comment). Eager FP32 path only. Same result as
    // use_merged_oint_outc above: correct but a net throughput regression
    // versus the tuned cuDNN path it replaces -- kept for reference, default
    // off.
    Allen::Property<bool> m_use_merged_up1 {
        this, "use_merged_up1", false,
        "eager FP32 path only: replace up1's ConvTranspose+Conv+BiasReLU "
        "sequence with a single merged kernel (exact, incl. boundary)"};

    // Skip-connection ablation (FP32 path only; ignored when use_fp16=true).
    // "concat" — current physics-validated behaviour (default).
    // "add"    — replace both channel-concat skips with element-wise add
    //            (halves the channel count feeding the merge conv/deconv);
    //            reuses one arbitrary half of the concat-trained weights,
    //            so output is NOT physics-valid — throughput only.
    // "none"   — drop both skip connections entirely; decoder only sees the
    //            upsampled main path. Also throughput only.
    Allen::Property<std::string> m_skip_mode {
        this, "skip_mode", "concat",
        "Skip-connection ablation for throughput testing: concat | add | none "
        "(add/none are not physics-valid, benchmark only)"};

    // CUDA graph capture: captures the per-chunk pipeline once (thread_local) and
    // replays it via cudaGraphLaunch instead of ~15-20 separate host API calls per
    // chunk. Covers both use_fp16=false and use_fp16=true (separate captured graphs,
    // separate thread_local scratch pools -- see get_or_capture_cuda_graph and
    // get_or_capture_cuda_graph_fp16). Only active when skip_mode=="concat" (checked
    // fresh every call); add/none keep using the eager path (skip-mode ablation is
    // FP16-incompatible by design regardless of graphs, so this restriction isn't
    // graph-specific).
    Allen::Property<bool> m_use_cuda_graph {
        this, "use_cuda_graph", false,
        "capture+replay the concat-mode UNet pipeline (FP32 or FP16) as a CUDA graph "
        "(only active when skip_mode=concat)"};

    // m_init_done: set to true after init() completes. Guards call_once.
    mutable bool m_init_done = false;
    mutable bool m_dump_done = false;
    mutable unsigned m_call_count = 0;

#ifdef ALLEN_CUDNN_BACKEND_CUDA
    // Per-layer helpers — use global descriptor set + thread_local handle
    void run_convbnrelu(
        const Allen::CuDNN::ConvDescriptors& desc,
        const float* input, float* output,
        const float* w_fused, const float* b_fused,
        int K, int W_out, int N,
        cudnnHandle_t handle,
        const dim3& block, const Allen::Context& ctx) const;

    void run_convbnrelu_half(
        const Allen::CuDNN::ConvDescriptors& desc,
        const __half* input, __half* output,
        const __half* w_fused, const __half* b_fused,
        int K, int W_out, int N,
        cudnnHandle_t handle,
        const dim3& block, const Allen::Context& ctx) const;

    // BF16 counterpart of run_convbnrelu_half.
    void run_convbnrelu_bf16(
        const Allen::CuDNN::ConvDescriptors& desc,
        const __nv_bfloat16* input, __nv_bfloat16* output,
        const __nv_bfloat16* w_fused, const __nv_bfloat16* b_fused,
        int K, int W_out, int N,
        cudnnHandle_t handle,
        const dim3& block, const Allen::Context& ctx) const;

    void run_conv(
        const Allen::CuDNN::ConvDescriptors& desc,
        const float* input,  float* output,
        const float* w_ptr,  const float* bias_ptr,
        int N, int C_out, int W,
        const dim3& block, const Allen::Context& ctx,
        cudnnHandle_t handle,
        float beta_val = 0.f) const;

    void run_conv_transpose(
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
        void* workspace, size_t ws_bytes) const;

    // CUDA graph capture (thread_local, lazy): captures the FP32/concat pipeline
    // once per OS thread against the fixed graph-scratch pool, and returns the
    // graph exec + the two shuttle-kernel node handles so the caller can patch
    // their pointer arguments before each cudaGraphLaunch. seed_ncw/seed_kde only
    // need to be valid device pointers at capture time (the very first chunk of
    // the very first call on this thread) -- they are unconditionally patched
    // before every replay, including the first.
    void get_or_capture_cuda_graph(
        cudnnHandle_t handle,
        const dim3& block,
        const Allen::Context& ctx,
        const float* seed_ncw,
        float* seed_kde,
        cudaGraphExec_t& out_exec,
        cudaGraphNode_t& out_copy_in_node,
        cudaGraphNode_t& out_copy_out_node) const;

    // FP16 counterpart of get_or_capture_cuda_graph: captures the FP16 Tensor-Core
    // op sequence (f32_to_f16 -> rcbn*_h -> ... -> outc -> softplus -> copy) against
    // its own thread_local FP16 scratch pool. The leading f32_to_f16 conversion
    // kernel doubles as the input shuttle (its src argument is what gets patched
    // per replay); the output shuttle reuses the same squeeze_copy_kernel pattern
    // as the FP32 graph.
    void get_or_capture_cuda_graph_fp16(
        cudnnHandle_t handle,
        const dim3& block,
        const Allen::Context& ctx,
        const float* seed_ncw,
        float* seed_kde,
        cudaGraphExec_t& out_exec,
        cudaGraphNode_t& out_copy_in_node,
        cudaGraphNode_t& out_copy_out_node) const;
#endif
};

} // namespace pvfinder_unet
