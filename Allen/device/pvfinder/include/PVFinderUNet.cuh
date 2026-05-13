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
//   - cuDNN plans make algorithm selection and workspace ownership explicit.
//     The current production path uses timed forward-conv selection, heuristic
//     backward-data selection, and init-time owned workspaces.
//   - Weight tensors loaded once through the Allen::CuDNN::DeviceWeights facade via std::call_once.
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
    // conv_ws: retained as a parser-visible Allen buffer for compatibility.
    // Current cuDNN plans own any init-time workspace internally.
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

    Allen::Property<bool> m_use_fp16 {
        this, "use_fp16", false,
        "Experimental validation-gated FP16 Tensor Core path for CBR layers (physics approximate)"};

    // m_init_done: set to true after init() completes. Guards call_once.
    mutable bool m_init_done = false;
    mutable bool m_dump_done = false;

#ifdef ALLEN_CUDNN_BACKEND_CUDA
    // Per-layer helpers — use global descriptor set + thread_local handle
    void run_convbnrelu(
        const Allen::CuDNN::ForwardConvPlan& desc,
        const float* input, float* output,
        const float* w_fused, const float* b_fused,
        int K, int W_out, int N,
        cudnnHandle_t handle,
        const dim3& block, const Allen::Context& ctx) const;

    void run_convbnrelu_half(
        const Allen::CuDNN::ForwardConvPlan& desc,
        const __half* input, __half* output,
        const __half* w_fused, const __half* b_fused,
        int K, int W_out, int N,
        cudnnHandle_t handle,
        const dim3& block, const Allen::Context& ctx) const;

    void run_conv(
        const Allen::CuDNN::ForwardConvPlan& desc,
        const float* input,  float* output,
        const float* w_ptr,  const float* bias_ptr,
        int N, int C_out, int W,
        const dim3& block, const Allen::Context& ctx,
        cudnnHandle_t handle,
        float beta_val = 0.f) const;

    void run_conv_transpose(
        const float* input, float* output,
        const Allen::CuDNN::BackwardDataConvPlan& plan,
        const float* w_ptr, const float* bias_ptr,
        int N, int C_out, int W_out,
        const dim3& block, const Allen::Context& ctx,
        cudnnHandle_t handle) const;
#endif
};

} // namespace pvfinder_unet
