#pragma once

#include "AlgorithmTypes.cuh"

namespace pvfinder_ncw_layout {

// Constants describing the layout
static constexpr unsigned N_INTERVALS   = 40;
static constexpr unsigned N_CHANNELS    = 8;
static constexpr unsigned N_BINS        = 100;
// Elements per event in the input: [40 intervals x 8 channels x 100 bins]
static constexpr unsigned ELEMS_PER_EVENT_IN  = N_INTERVALS * N_CHANNELS * N_BINS; // 32000
// Elements per (event x interval) NCW tensor slice: [C=8, W=100]
static constexpr unsigned ELEMS_PER_SLICE     = N_CHANNELS * N_BINS;               //   800

/**
 * @brief Reorders PVFinder interval features into a contiguous NCW tensor for cuDNN.
 *
 * Input layout (from PVFinderTrackAggregation):
 *   dev_pvfinder_interval_features[event][interval][channel][bin]
 *   Total: n_events * 40 * 8 * 100 floats
 *
 * Output layout (cuDNN NCW):
 *   dev_pvfinder_ncw_tensor[n * C * W]  where N = n_events * 40
 *   i.e. dev_pvfinder_ncw_tensor[(event*40 + interval)][channel][bin]
 *   Total: n_events * 40 * 8 * 100 floats  (same size, different logical batch dim)
 *
 * The memory layout of input and output is identical since the input is already
 * stored contiguously as [event][interval][channel][bin].  This algorithm therefore
 * performs a logical reinterpretation — merging the (event, interval) dimensions
 * into a single batch dimension N = n_events * N_INTERVALS — plus an explicit copy
 * so downstream algorithms receive an independent, properly-named Allen buffer.
 */
struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;

    // Input: 8-channel interval features from aggregation step
    DEVICE_INPUT(dev_pvfinder_interval_features_t, float) dev_pvfinder_interval_features;

    // Output: contiguous NCW tensor [N = n_events*40, C=8, W=100]
    DEVICE_OUTPUT(dev_pvfinder_ncw_tensor_t, float) dev_pvfinder_ncw_tensor;
};

struct pvfinder_ncw_layout_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(
        ArgumentReferences<Parameters> arguments,
        const RuntimeOptions&,
        const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) const;

private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1},
        "block dimensions for NCW layout copy kernel"};
};

} // namespace pvfinder_ncw_layout
