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
#include <string>

// ---------------------------------------------------------------------------
// PVFinderVertexFitter: NN-chain equivalent of pv_beamline_multi_fitter.
//
// Runs the same adaptive Tukey-bisquare weighted Kalman fit over PVTrack
// objects (from pv_beamline_extrapolate), but uses z-seeds from
// PVFinderKDEPeakFinder instead of the beamline histogram peaks.
//
// Coverage: z in [-100, +300] mm (pp luminous region only).
// SMOG2 region is not covered by the NN KDE -- no SMOG2 branching is needed.
// ---------------------------------------------------------------------------

namespace pvfinder_vertex_fitter {

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    // PVTrack objects from pv_beamline_extrapolate (shared with classical chain)
    DEVICE_INPUT(dev_pvtracks_t, PVTrack) dev_pvtracks;
    // Per-track denominators from PVFinderCalculateDenom
    DEVICE_INPUT(dev_nn_pvtracks_denom_t, float) dev_nn_pvtracks_denom;
    // NN z-seeds from PVFinderKDEPeakFinder
    DEVICE_INPUT(dev_nn_zpeaks_t, float)              dev_nn_zpeaks;
    DEVICE_INPUT(dev_nn_number_of_zpeaks_t, unsigned) dev_nn_number_of_zpeaks;
    // Fitted vertex outputs
    DEVICE_OUTPUT(dev_nn_multi_fit_vertices_t, PV::Vertex) dev_nn_multi_fit_vertices;
    DEVICE_OUTPUT(dev_nn_number_of_multi_fit_vertices_t, unsigned) dev_nn_number_of_multi_fit_vertices;
};

__global__ void pvfinder_vertex_fitter(
    Parameters,
    const unsigned pp_minNumTracksPerVertex,
    const unsigned maxFitIter,
    const float    zmin,
    const float    zmax,
    const float    maxChi2,
    const float    chi2CutExp,
    const float    minWeight,
    const float    maxDeltaZConverged,
    const float    maxVertexRho2);

struct pvfinder_vertex_fitter_t : public DeviceAlgorithm, Parameters {
    void update(const Constants& constants) const;

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
    Allen::Property<unsigned> m_block_dim_y {
        this, "block_dim_y", 4u, "block dimension Y (threads per z-seed)"};

    // NN coverage defaults -- narrower than the classical beamline range.
    Allen::Property<float> m_zmin {
        this, "zmin", -100.f, "minimum z for track acceptance in the fit (mm)"};
    Allen::Property<float> m_zmax {
        this, "zmax",  300.f, "maximum z for track acceptance in the fit (mm)"};

    // Fit quality cuts -- match classical defaults from BeamlinePVConstants.
    Allen::Property<unsigned> m_pp_minNumTracksPerVertex {
        this, "pp_minNumTracksPerVertex",
        BeamlinePVConstants::MultiFitter::pp_minNumTracksPerVertex,
        "minimum number of tracks to accept a vertex"};
    Allen::Property<float> m_maxVertexRho2 {
        this, "maxVertexRho2",
        BeamlinePVConstants::MultiFitter::maxVertexRho2,
        "maximum squared transverse distance from the beamline (mm^2)"};
    Allen::Property<unsigned> m_maxFitIter {
        this, "maxFitIter",
        BeamlinePVConstants::MultiFitter::maxFitIter,
        "maximum number of fit iterations"};
    Allen::Property<float> m_chi2CutExp {
        this, "chi2CutExp",
        BeamlinePVConstants::MultiFitter::chi2CutExp,
        "exp(-chi2_cut/2) -- Tukey weight denominator offset"};
    Allen::Property<float> m_minWeight {
        this, "minWeight",
        BeamlinePVConstants::MultiFitter::minWeight,
        "minimum track weight to contribute to the fit"};
    Allen::Property<float> m_maxChi2 {
        this, "maxChi2",
        BeamlinePVConstants::MultiFitter::maxChi2,
        "maximum track chi2 to enter the fit"};
    Allen::Property<float> m_maxDeltaZConverged {
        this, "maxDeltaZConverged",
        BeamlinePVConstants::MultiFitter::maxDeltaZConverged,
        "convergence criterion: |delta_z| < this value (mm)"};

    Allen::Property<std::string> m_dump_dir {
        this, "dump_validation", "",
        "if non-empty, dump fitted NN vertices to this directory on the first call"};

    mutable bool m_dump_done = false;
};

} // namespace pvfinder_vertex_fitter
