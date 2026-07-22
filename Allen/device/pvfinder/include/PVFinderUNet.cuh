#pragma once

#include "AlgorithmTypes.cuh"
#ifdef ALLEN_CUDNN_BACKEND_CUDA
#include "AllenCuDNN.h"
#include <cuda_fp16.h>
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
static constexpr int N_BATCH_CHANNELS = 8;   // input latent channels
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
        "Phase M benchmark: use FP16 Tensor Core path for CBR layers (physics approximate)"};

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
