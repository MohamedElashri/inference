#pragma once

#include "AlgorithmTypes.cuh"
#include "PV_Definitions.cuh"
#include <string>

namespace pvfinder_kde_peak_finder {

// KDE histogram geometry constants (must match PVFinderFCAggregation and PVFinderUNet)
static constexpr int   KDE_N_BINS = 4000;    // 40 intervals x 100 bins
static constexpr float KDE_ZMIN   = -100.f;  // mm -- start of NN z coverage
static constexpr float KDE_DZ     =   0.1f;  // mm per bin

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    // Flat KDE output from PVFinderUNet: [N_events * 40 * 100] = [N_events * 4000]
    DEVICE_INPUT(dev_pvfinder_kde_output_t, float) dev_pvfinder_kde_output;
    // Outputs: z-seed positions and count per event
    DEVICE_OUTPUT(dev_nn_zpeaks_t, float)              dev_nn_zpeaks;            // [N_events * max_number_vertices]
    DEVICE_OUTPUT(dev_nn_number_of_zpeaks_t, unsigned) dev_nn_number_of_zpeaks;  // [N_events]
};

__global__ void pvfinder_kde_peak_finder(
    Parameters,
    const float kde_peak_threshold,
    const float min_integral_tracks,
    const unsigned min_width);

struct pvfinder_kde_peak_finder_t : public DeviceAlgorithm, Parameters {
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
    Allen::Property<float> m_kde_peak_threshold {
        this, "kde_peak_threshold", 1e-2f,
        "KDE bin threshold -- bins below this are treated as empty"};

    Allen::Property<float> m_min_integral_tracks {
        this, "min_integral_tracks", 0.5f,
        "Minimum cluster integral (units of track weight) to accept a z-seed; "
        "matches the reference pvfinder_pytorch post-processing"};

    Allen::Property<unsigned> m_min_width {
        this, "min_width", 2u,
        "Minimum width in bins for a KDE feature to become a z-seed"};

    Allen::Property<dim3> m_block_dim {
        this, "block_dim", {32, 1, 1},
        "CUDA block dimensions -- must stay at {32,1,1} (single warp) for warp-ballot logic"};

    Allen::Property<std::string> m_dump_dir {
        this, "dump_validation", "",
        "if non-empty, dump NN z-peaks binary to this directory on the first call"};

    mutable bool m_dump_done = false;
};

} // namespace pvfinder_kde_peak_finder
