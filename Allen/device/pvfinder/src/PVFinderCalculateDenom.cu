#include "PVFinderCalculateDenom.cuh"

INSTANTIATE_ALGORITHM(pvfinder_nn_calculate_denom::pvfinder_nn_calculate_denom_t)

namespace pvfinder_nn_calculate_denom {

void pvfinder_nn_calculate_denom_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    // One denom value per reconstructed Velo track (global offset array).
    set_size<dev_nn_pvtracks_denom_t>(
        arguments, first<host_number_of_reconstructed_velo_tracks_t>(arguments));
}

void pvfinder_nn_calculate_denom_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    global_function(pvfinder_nn_calculate_denom)(
        dim3(size<dev_event_list_t>(arguments)),
        m_block_dim,
        context)(arguments);
}

void pvfinder_nn_calculate_denom_t::update(const Constants& constants) const
{
    // Load the beamline position into __constant__ dev_beamline so the kernel
    // can compute seed x/y positions and residuals.
    updateCommon(constants);
}

// ---------------------------------------------------------------------------
// Kernel -- direct adaptation of pv_beamline_calculate_denom.
//
// The only differences from the classical version:
//   1. Reads from dev_nn_zpeaks / dev_nn_number_of_zpeaks (NN seeds).
//   2. No SMOG2 branching: all NN seeds are in [-100, +300] mm, well above
//      the SMOG2/pp boundary (-334 mm), so dev_beamline.tx is always used.
// ---------------------------------------------------------------------------
__global__ void pvfinder_nn_calculate_denom(
    pvfinder_nn_calculate_denom::Parameters parameters)
{
    const unsigned event_number = parameters.dev_event_list[blockIdx.x];
    const auto velo_tracks      = parameters.dev_velo_tracks_view[event_number];

    const float*    zseeds          = parameters.dev_nn_zpeaks + event_number * PV::max_number_vertices;
    const unsigned  number_of_seeds = parameters.dev_nn_number_of_zpeaks[event_number];

    const PVTrack* tracks         = parameters.dev_pvtracks       + velo_tracks.offset();
    float*         pvtracks_denom = parameters.dev_nn_pvtracks_denom + velo_tracks.offset();

    // Pre-compute the denominator for each track: sum of exp(-chi2_seed / 2)
    // over all NN z-seeds.  Used in the adaptive fitter weight normalisation.
    for (unsigned i = threadIdx.x; i < velo_tracks.size(); i += blockDim.x) {
        float track_denom = 0.f;
        const PVTrack& track = tracks[i];

        // All NN seeds are in the pp region (z > -100 mm > SMOG2_pp_separation).
        // Use the pp beamline slope unconditionally.
        const float tx_beam = dev_beamline.tx.x;
        const float ty_beam = dev_beamline.tx.y;

        for (unsigned j = 0; j < number_of_seeds; ++j) {
            const float2 seed_pos_xy {
                dev_beamline.pos.x + tx_beam * zseeds[j],
                dev_beamline.pos.y + ty_beam * zseeds[j]};

            const float  dz  = zseeds[j] - track.z;
            const float2 res = track.x + track.tx * dz - seed_pos_xy;
            const float  chi2 = res.x * res.x * track.W_00
                              + res.y * res.y * track.W_11;
            track_denom += expf(chi2 * (-0.5f));
        }

        pvtracks_denom[i] = track_denom;
    }
}

} // namespace pvfinder_nn_calculate_denom
