#pragma once

#include "VeloConsolidated.cuh"
#include "AlgorithmTypes.cuh"

namespace pvfinder_track_aggregation {

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_pvfinder_track_features_t, float) dev_pvfinder_track_features;      // Track size x 9
    
    // Output array: [events x 40 intervals x 100 bins] -> 4000 floats per event mapping the KDE layout targets
    DEVICE_OUTPUT(dev_pvfinder_output_histogram_t, float) dev_pvfinder_output_histogram;
    // 8-channel interval features: [events x 40 intervals x 8 channels x 100 bins] = 32000 floats per event
    // Preserved un-collapsed for UNet NCW input: channel c of interval i = latent dim c summed over tracks in i
    DEVICE_OUTPUT(dev_pvfinder_interval_features_t, float) dev_pvfinder_interval_features;
};

struct pvfinder_track_aggregation_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) const;

private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
};

} // namespace pvfinder_track_aggregation
