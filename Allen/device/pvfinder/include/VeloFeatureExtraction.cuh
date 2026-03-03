#pragma once

#include "VeloConsolidated.cuh"
#include "VeloEventModel.cuh"
#include "KalmanParametrizations.cuh"
#include "AlgorithmTypes.cuh"
#include "BeamlinePVConstants.cuh"
#include "ParticleTypes.cuh"

namespace pvfinder_velo_feature_extraction {

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_velo_states_view_t, Allen::Views::Physics::KalmanStates) dev_velo_states_view;
    
    // Output array: [total_velo_tracks x 9 features]
    DEVICE_OUTPUT(dev_pvfinder_track_features_t, float) dev_pvfinder_track_features;
};

struct pvfinder_velo_feature_extraction_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions& runtime_options,
        const Constants& constants,
        const Allen::Context& context) const;

private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
};

} // namespace pvfinder_velo_feature_extraction
