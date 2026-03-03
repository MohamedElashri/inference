#pragma once

#include "VeloConsolidated.cuh"
#include "AlgorithmTypes.cuh"

namespace pvfinder_fc_engine {

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_pvfinder_track_features_t, float) dev_pvfinder_track_features;

    // Output array: [total_velo_tracks x 800 latent features]
    DEVICE_OUTPUT(dev_pvfinder_latent_features_t, float) dev_pvfinder_latent_features;
};

struct pvfinder_fc_engine_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) const;

private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
};

} // namespace pvfinder_fc_engine
