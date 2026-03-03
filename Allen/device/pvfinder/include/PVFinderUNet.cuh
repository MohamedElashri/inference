#pragma once

#include "AlgorithmTypes.cuh"
#include "AllenCuDNN.h"

// ---------------------------------------------------------------------------
// PVFinderUNet: cuDNN-backed UNet inference algorithm.
//
// Reads the NCW tensor produced by PVFinderNCWLayout and runs the full
// pvfinder unet forward pass, producing per-event
// per-interval Histogram.
//
// Input:  dev_pvfinder_ncw_tensor  [N = n_events*40, C=8,  W=100]
// Output: dev_pvfinder_kde_output  [n_events, 40, 100]  (flat: n_events*4000)
//
// Weight binary layout (cnn_weights.bin):
//   magic uint32 = 0xCAFE0001
//   rcbn1: Conv(8→64,k=25) + BN(64)
//   rcbn2: Conv(64→64,k=7) + BN(64)
//   rcbn3: Conv(64→64,k=5) + BN(64)
//   up1:   ConvT(64→64,k=2,s=2) + Conv(64→64,k=5) + BN(64)
//   up2:   ConvT(128→64,k=2,s=2) + Conv(64→64,k=5) + BN(64)
//   out_intermediate: Conv(128→64,k=5)
//   outc:  Conv(64→1,k=5)
// ---------------------------------------------------------------------------

namespace pvfinder_unet {

// UNet architecture constants (default weights: HDplusUNet100 iter12Ca)
static constexpr int N_BATCH_CHANNELS = 8;   // input latent channels
static constexpr int N_FEAT    = 64;          // feature maps throughout
static constexpr int W_IN      = 100;         // input width
static constexpr int W_HALF    = 50;          // after first MaxPool
static constexpr int W_QTR     = 25;          // after second MaxPool
static constexpr int N_INTERVALS = 40;
static constexpr float KDE_SCALE = 0.001f;

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;

    // NCW input from PVFinderNCWLayout: [N_events*40, C=8, W=100]
    DEVICE_INPUT(dev_pvfinder_ncw_tensor_t, float) dev_pvfinder_ncw_tensor;

    // Scratch intermediate buffers (allocated from Allen pool)
    // Buffer reuse to minimise peak device memory:
    //   x1  [N,64,100]  — skip1; reused as oint [N,64,100] after step 9
    //   x2  [N,64,50]   — skip2
    //   x3  [N,64,100]  — sized for max(x3_raw[N,64,25], logits[N,1,100]); reused as logits
    //   up1 [N,64,50]   — upsampled step1
    //   up2 [N,64,100]  — upsampled step2; same #elems as cat2[N,128,50] → up2 reuses cat2 ptr
    //   cat2[N,128,50]  — concat of up1+x2 (skip); also holds up2 data (same size)
    //   cat_out[N,128,100] — concat of up2+x1 (largest buffer, unavoidable)
    //   conv_ws         — cuDNN workspace (shared across all layers)
    DEVICE_OUTPUT(dev_unet_x1_t,     float) dev_unet_x1;       // [N, 64, 100]
    DEVICE_OUTPUT(dev_unet_x2_t,     float) dev_unet_x2;       // [N, 64, 50]
    DEVICE_OUTPUT(dev_unet_x3_t,     float) dev_unet_x3;       // [N, 64, 100] (oversized; also logits)
    DEVICE_OUTPUT(dev_unet_up1_t,    float) dev_unet_up1;      // [N, 64, 50]
    DEVICE_OUTPUT(dev_unet_cat2_t,   float) dev_unet_cat2;     // [N, 128, 50] (also up2[N,64,100])
    // NOTE: no cat_out buffer — out_intermediate uses split half-convolutions with beta accumulation
    DEVICE_OUTPUT(dev_unet_conv_ws_t, float) dev_unet_conv_ws; // cuDNN workspace

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

    // Called once at startup to load weights from file into WeightRegistry
    void init();

    // NOTE: No explicit cuDNN cleanup needed in destructor.
    // Allen calls cudaDeviceReset() at shutdown which reclaims all CUDA/cuDNN
    // resources. Explicit cudnnDestroy after cudaDeviceReset crashes.
    ~pvfinder_unet_t() = default;

private:
    // Path to binary weight file (configurable property)
    Allen::Property<std::string> m_weight_file {
        this, "weight_file", "cnn_weights.bin",
        "path to cnn_weights.bin produced by convert_cnn_weights.py"};

    Allen::Property<dim3> m_block_dim {
        this, "block_dim", {256, 1, 1}, "CUDA block dim for element-wise kernels"};

    // Validation dump: if non-empty, write NCW input and KDE output for the first slice
    // to <dump_dir>/allen_ncw_input.bin and <dump_dir>/allen_kde_output.bin.
    Allen::Property<std::string> m_dump_dir {
        this, "dump_validation", "",
        "if non-empty, dump NCW input + KDE output of the first slice to this directory"};

    // cuDNN descriptors — one per conv layer (RAII, initialised in init())
    mutable Allen::CuDNN::Handle     m_handle;
    mutable Allen::CuDNN::ConvDescriptors m_desc_rcbn1;   // Conv(8→64, k=25, pad=12)
    mutable Allen::CuDNN::ConvDescriptors m_desc_rcbn2;   // Conv(64→64, k=7, pad=3)
    mutable Allen::CuDNN::ConvDescriptors m_desc_rcbn3;   // Conv(64→64, k=5, pad=2)
    mutable Allen::CuDNN::ConvDescriptors m_desc_up1_t;   // ConvT(64→64, k=2, s=2) as back-data
    mutable Allen::CuDNN::ConvDescriptors m_desc_up1_c;   // Conv(64→64, k=5, pad=2) after transpose
    mutable Allen::CuDNN::ConvDescriptors m_desc_up2_t;   // ConvT(128→64, k=2, s=2)
    mutable Allen::CuDNN::ConvDescriptors m_desc_up2_c;   // Conv(64→64, k=5, pad=2)
    mutable Allen::CuDNN::ConvDescriptors m_desc_oint_half; // Conv(64→64, k=5, pad=2) — two halves accumulate
    mutable Allen::CuDNN::ConvDescriptors m_desc_outc;      // Conv(64→1,  k=5, pad=2)

    // Transpose conv filter+conv descriptors (read-only after init, safe to share)
    mutable cudnnFilterDescriptor_t       m_filter_up1_t = nullptr;
    mutable cudnnFilterDescriptor_t       m_filter_up2_t = nullptr;
    mutable cudnnConvolutionDescriptor_t  m_conv_up1_t   = nullptr;
    mutable cudnnConvolutionDescriptor_t  m_conv_up2_t   = nullptr;
    // NOTE: tensor descriptors for transpose-conv are created/destroyed inside operator()
    // as local variables because operator() is called concurrently from multiple threads
    // and cudnnSetTensor4dDescriptor is not thread-safe on shared objects.

    mutable bool m_init_done  = false;
    mutable bool m_dump_done  = false;  // true after first validation dump is written

    void init_descriptors() const;
    size_t max_workspace_bytes(unsigned N) const;

    // Per-layer forward helpers
    void run_convbnrelu(
        const Allen::CuDNN::ConvDescriptors& desc,
        const float* input,  float* output,
        const float* w_key_ptr,
        const float* bias_ptr,
        const float* bn_gamma, const float* bn_beta,
        const float* bn_mean,  const float* bn_var, float bn_eps,
        void* workspace, size_t ws_bytes,
        int N, int C_in, int C_out, int W,
        const dim3& block, const Allen::Context& ctx,
        cudnnHandle_t handle) const;

    void run_conv(
        const Allen::CuDNN::ConvDescriptors& desc,
        const float* input, float* output,
        const float* w_ptr, const float* bias_ptr,
        void* workspace, size_t ws_bytes,
        int N, int C_in, int C_out, int W,
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
        void* workspace, size_t ws_bytes,
        int N, int C_out, int W_out,
        const dim3& block, const Allen::Context& ctx,
        cudnnHandle_t handle) const;
};

} // namespace pvfinder_unet
