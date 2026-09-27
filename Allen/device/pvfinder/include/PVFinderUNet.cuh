/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "PVFinderConstants.cuh"
#include <memory>
#ifdef ALLEN_CUDNN_BACKEND_CUDA
#include "AllenCuDNN.h"
#endif

// ---------------------------------------------------------------------------
// pvfinder_unet: the UNet from the FC stage's interval features to the KDE.
//
// Input:  dev_pvfinder_interval_features, one row per interval with tracks
//         (see pvfinder_fc_aggregation), N_BATCH_CHANNELS x W_IN per row.
// Output: dev_pvfinder_kde_output [n_events, 40, 100]: each row's KDE at its
//         (event, interval), the UNet's response to an all-zero input at every
//         interval without tracks.
//
// Two precisions (precision, must match pvfinder_fc_aggregation's):
//   float32   cuDNN convolutions (IMPLICIT_GEMM, no workspace) and small FP32
//             kernels, batches of unet_batch_events * 40 rows; reproduces the
//             trained model to float32 rounding. Any CUDA GPU.
//   bfloat16  the whole UNet in one kernel (PVFinderUNetFused.cuh):
//             activations in shared memory, convolutions on BF16 tensor cores
//             with FP32 accumulation. Compute capability 8.0 or newer, and the
//             shapes it is written for (16 feature maps, 4 input channels).
// ---------------------------------------------------------------------------

namespace pvfinder_unet {

// UNet architecture constants. The UNet has no skip connections: the decoder
// only sees the upsampled main path (rcbn1-3 -> up1 -> up2 -> out_intermediate
// -> outc), matching checkpoints trained with sc_mode=none. N_FEAT and
// N_BATCH_CHANNELS (the FC/UNet handoff's latentChannels) are fixed by the
// build (-DPVFINDER_UNET_N_FEAT, -DPVFINDER_UNET_N_BATCH_CHANNELS).
#ifdef PVFINDER_UNET_N_BATCH_CHANNELS
static constexpr int N_BATCH_CHANNELS = PVFINDER_UNET_N_BATCH_CHANNELS;
#else
static constexpr int N_BATCH_CHANNELS = 8; // input latent channels
#endif
#ifdef PVFINDER_UNET_N_FEAT
static constexpr int N_FEAT = PVFINDER_UNET_N_FEAT;
#else
static constexpr int N_FEAT = 64; // feature maps throughout
#endif
static constexpr int W_IN = PVFinderConstants::KDE::n_bins_per_interval; // input width
static constexpr int W_HALF = W_IN / 2;                                   // after first MaxPool
static constexpr int W_QTR = W_IN / 4;                                    // after second MaxPool
static constexpr int N_INTERVALS = PVFinderConstants::KDE::n_intervals;
static constexpr float KDE_SCALE = 0.001f;

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;

    // From pvfinder_fc_aggregation: the rows, their layout
    // (host_pvfinder_unet_rows: [1] rows in use, [2] bfloat16), each
    // (event, interval)'s row (-1 without tracks) and each row's (event, interval).
    DEVICE_INPUT(dev_pvfinder_interval_features_t, float) dev_pvfinder_interval_features;
    HOST_INPUT(host_pvfinder_unet_rows_t, unsigned) host_pvfinder_unet_rows;
    DEVICE_INPUT(dev_pvfinder_slot_row_t, int) dev_pvfinder_slot_row;
    DEVICE_INPUT(dev_pvfinder_row_slot_t, int) dev_pvfinder_row_slot;

    // float32 only: activations of one batch of N = unet_batch_events * 40 rows
    // (reused by every batch) and the KDE of each row, [rows][W_IN], spread to
    // dev_pvfinder_kde_output afterwards. Empty for bfloat16.
    DEVICE_OUTPUT(dev_unet_x1_t, float) dev_unet_x1;   // [N, N_FEAT, W_IN]
    DEVICE_OUTPUT(dev_unet_x2_t, float) dev_unet_x2;   // [N, N_FEAT, W_HALF]
    DEVICE_OUTPUT(dev_unet_x3_t, float) dev_unet_x3;   // [N, N_FEAT, W_IN]
    DEVICE_OUTPUT(dev_unet_up1_t, float) dev_unet_up1; // [N, N_FEAT, W_HALF]
    DEVICE_OUTPUT(dev_unet_up2_t, float) dev_unet_up2; // [N, N_FEAT, W_IN]
    DEVICE_OUTPUT(dev_unet_kde_rows_t, float) dev_unet_kde_rows;

    // The KDE: [n_events * 40 * 100] floats
    DEVICE_OUTPUT(dev_pvfinder_kde_output_t, float) dev_pvfinder_kde_output;
};

struct pvfinder_unet_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions&,
        const Constants&,
        const Allen::Context& context) const;

    // Loads the weights and prepares the configured precision's path.
    void init();

private:
    // Required, no default: Allen does not assume where weights live. The
    // repository's weights/ pipeline produces cnn_weights.bin
    // (make -C weights convert MODEL=<name>), and AllenConf fills this in from
    // PVFINDER_WEIGHTS_DIR when the sequence configuration is generated.
    Allen::Property<std::string> m_weight_file {
        this, "weight_file", "",
        "path to cnn_weights.bin (required; produced by the weights/ pipeline, "
        "set by AllenConf from PVFINDER_WEIGHTS_DIR)"};

    Allen::Property<std::string> m_precision {
        this, "precision", "float32", "float32 or bfloat16; must match pvfinder_fc_aggregation.precision"};

    // float32: events per cuDNN batch (N = this * 40 rows). Must match
    // pvfinder_fc_aggregation.unet_batch_events, which pads the rows to a
    // multiple of it (AllenConf sets both).
    Allen::Property<unsigned> m_unet_batch_events {
        this, "unet_batch_events", 20u, "float32: events per cuDNN batch; must match pvfinder_fc_aggregation"};

    // bfloat16: fraction of the full-occupancy grid the fused kernel is
    // launched with (one 4-warp block of about 59 KB per SM at full occupancy);
    // below 1, other streams' kernels keep SMs while it runs. 1/4 is the best
    // measured at 16 streams on the RTX 3090.
    Allen::Property<float> m_fused_grid_fraction {
        this, "fused_grid_fraction", 0.25f, "bfloat16: fraction of the full-occupancy grid for the fused UNet kernel"};

    Allen::Property<dim3> m_block_dim {
        this, "block_dim", {256, 1, 1}, "float32: block dimensions of the element-wise kernels"};

    Allen::Property<std::string> m_dump_dir {
        this, "dump_validation", "",
        "if non-empty, dump the input rows and the KDE of the first slice to this directory"};
    mutable bool m_dump_done = false;

    // Per-instance state (weights, cuDNN descriptors, the fused kernel's weight
    // image), defined in PVFinderUNet.cu so the cuDNN types stay out of this
    // header; created in init(). shared_ptr keeps the algorithm copyable.
    struct UNetState;
    std::shared_ptr<UNetState> m_state;
    bool m_bf16 = false;

#ifdef ALLEN_CUDNN_BACKEND_CUDA
    void run_fp32_batch(const float* rows, float* kde, float* const scratch[5], cudnnHandle_t handle,
                        const Allen::Context& context) const;
    void dump(const ArgumentReferences<Parameters>& arguments, const Allen::Context& context) const;
#endif
};

} // namespace pvfinder_unet
