#pragma once

#include "VeloConsolidated.cuh"
#include "AlgorithmTypes.cuh"

namespace pvfinder_fc_aggregation {

#ifndef PVFINDER_FC_HIDDEN_DIM
#define PVFINDER_FC_HIDDEN_DIM 32
#endif

#ifndef PVFINDER_FC_HIDDEN_LAYERS
#define PVFINDER_FC_HIDDEN_LAYERS 3
#endif

#ifndef PVFINDER_FC_OUTPUT_DIM
#define PVFINDER_FC_OUTPUT_DIM 400
#endif

static constexpr int FC_INPUT_DIM = 9;
static constexpr int FC_HIDDEN_DIM = PVFINDER_FC_HIDDEN_DIM;
static constexpr int FC_HIDDEN_LAYERS = PVFINDER_FC_HIDDEN_LAYERS;
static constexpr int FC_OUTPUT_DIM = PVFINDER_FC_OUTPUT_DIM;
static constexpr int FC_BINS = 100;
static constexpr int FC_INTERVALS = 40;
static constexpr int FC_LATENT_CHANNELS = FC_OUTPUT_DIM / FC_BINS;

static_assert(FC_HIDDEN_LAYERS >= 1, "PVFinder FC requires at least one hidden layer");
static_assert(FC_OUTPUT_DIM % FC_BINS == 0, "PVFinder FC output must be latentChannels * 100");

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_pvfinder_track_features_t, float) dev_pvfinder_track_features;      // Track size x 9
    
    // Output array: [events x 40 intervals x 100 bins] -> 4000 floats per event mapping the KDE layout targets
    DEVICE_OUTPUT(dev_pvfinder_output_histogram_t, float) dev_pvfinder_output_histogram;
    // Interval features: [events x 40 intervals x latent channels x 100 bins].
    // Preserved un-collapsed for UNet NCW input: channel c of interval i = latent dim c summed over tracks in i
    DEVICE_OUTPUT(dev_pvfinder_interval_features_t, float) dev_pvfinder_interval_features;
    // CSR index structure for interval-sorted track access:
    //   interval_start[n_events * 42]: start offset into track_idx for each interval + 1 sentinel
    //   track_idx[total_tracks * 2]:   track indices sorted by interval (boundary tracks appear twice)
    DEVICE_OUTPUT(dev_pvfinder_interval_start_t, int) dev_pvfinder_interval_start;
    DEVICE_OUTPUT(dev_pvfinder_track_idx_t,      int) dev_pvfinder_track_idx;
    // Compact non-empty slot list (used so the aggregation kernel skips empty intervals):
    //   nonempty_slots[i] = (event_number << 6) | interval — packed uint32
    //   nonempty_count[0] = number of valid entries in nonempty_slots
    DEVICE_OUTPUT(dev_pvfinder_nonempty_slots_t, unsigned) dev_pvfinder_nonempty_slots;
    DEVICE_OUTPUT(dev_pvfinder_nonempty_count_t, int)      dev_pvfinder_nonempty_count;
    // cuBLAS final-layer GEMM intermediate buffers, sized per chunk and reused across chunks.
    //   dev_pvfinder_fc_input: CSR-ordered track inputs, shape [T_chunk_max x FC_INPUT_DIM].
    //   dev_pvfinder_l5_output: final hidden states, shape [T_chunk_max x FC_HIDDEN_DIM] row-major.
    //   dev_pvfinder_hidden_tmp: second hidden-state buffer for the all-cuBLAS hidden-stack experiment.
    //   dev_pvfinder_l6a_output: raw final GEMM output, shape [FC_OUTPUT_DIM x T_chunk_max] col-major.
    //   Both are allocated only when ALLEN_WITH_CUBLAS is defined; zero-sized otherwise.
    DEVICE_OUTPUT(dev_pvfinder_fc_input_t,   float) dev_pvfinder_fc_input;
    DEVICE_OUTPUT(dev_pvfinder_l5_output_t,  float) dev_pvfinder_l5_output;
    DEVICE_OUTPUT(dev_pvfinder_hidden_tmp_t, float) dev_pvfinder_hidden_tmp;
    DEVICE_OUTPUT(dev_pvfinder_l6a_output_t, float) dev_pvfinder_l6a_output;
};

struct pvfinder_fc_aggregation_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) const;

private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
    Allen::Property<std::string> m_weight_file {
        this, "weight_file", "/data/home/melashri/iris/inference/fc_weights_32x3_400.bin",
        "path to PVFinder FC weights binary"};
    Allen::Property<bool> m_use_cublas_hidden_stack {
        this, "use_cublas_hidden_stack", false,
        "Experimental performance mode: run hidden FC layers with cuBLAS GEMM instead of register kernels"};
};

} // namespace pvfinder_fc_aggregation
