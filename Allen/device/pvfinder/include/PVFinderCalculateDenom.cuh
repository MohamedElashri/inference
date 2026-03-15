#pragma once

#include "BeamlinePVConstants.cuh"
#include "Common.h"
#include "AlgorithmTypes.cuh"
#include "TrackBeamLineVertexFinder.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "FloatOperations.cuh"
#include <cstdint>

// ---------------------------------------------------------------------------
// PVFinderCalculateDenom: NN-chain equivalent of pv_beamline_calculate_denom.
//
// For each Velo track, pre-computes the sum of Gaussian weights over all NN
// z-seeds (from PVFinderKDEPeakFinder).  This denominator is consumed by
// PVFinderVertexFitter to implement the adaptive multi-vertex weight:
//
//   track_weight = exp(-chi2/2) / (chi2CutExp + exp(-chi2/2) + denom - exp(-chi2_seed/2))
//
// The NN z-seeds all lie in [-100, +300] mm (pp luminous region), which is
// always above the SMOG2/pp separation at -334 mm.  Therefore no SMOG2
// branching is needed; dev_beamline.tx is used unconditionally.
// ---------------------------------------------------------------------------

namespace pvfinder_nn_calculate_denom {

struct Parameters {
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_pvtracks_t, PVTrack) dev_pvtracks;
    // NN z-seeds from PVFinderKDEPeakFinder
    DEVICE_INPUT(dev_nn_zpeaks_t, float)              dev_nn_zpeaks;
    DEVICE_INPUT(dev_nn_number_of_zpeaks_t, unsigned) dev_nn_number_of_zpeaks;
    // Output: per-track denominator for the adaptive fitter weight
    DEVICE_OUTPUT(dev_nn_pvtracks_denom_t, float) dev_nn_pvtracks_denom;
};

__global__ void pvfinder_nn_calculate_denom(Parameters);

struct pvfinder_nn_calculate_denom_t : public DeviceAlgorithm, Parameters {
    // update() loads the beamline constant into __constant__ dev_beamline.
    void update(const Constants& constants) const;

    void set_arguments_size(
        ArgumentReferences<Parameters> arguments,
        const RuntimeOptions&,
        const Constants&) const;

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions&,
        const Constants&,
        const Allen::Context& context) const;

private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
};

} // namespace pvfinder_nn_calculate_denom
