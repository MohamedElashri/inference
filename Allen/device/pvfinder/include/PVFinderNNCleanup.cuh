#pragma once

#include "BeamlinePVConstants.cuh"
#include "Common.h"
#include "AlgorithmTypes.cuh"
#include "TrackBeamLineVertexFinder.cuh"
#include "VeloConsolidated.cuh"
#include "VeloDefinitions.cuh"
#include "VeloEventModel.cuh"
#include "FloatOperations.cuh"
#include "AllenMonitoring.h"
#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// PVFinderNNCleanup: NN-chain equivalent of pv_beamline_cleanup.
//
// Deduplicates fitted vertices (chi2-distance cut) and z-sorts the survivors,
// producing the final PV::Vertex array consumed by downstream HLT1 selections
// and the PVChecker.
//
// Parameter tag names are kept identical to pv_beamline_cleanup_t so that
// the Python sequence wiring and all downstream consumers (PVChecker,
// GaudiAllenPVsToPrimaryVertexContainer) are transparent to which chain
// produced the vertices.
//
// SMOG2 monitoring histograms are omitted -- the NN chain covers pp only
// (z in [-100, +300] mm).
// ---------------------------------------------------------------------------

namespace pvfinder_nn_cleanup {

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    // Inputs from PVFinderVertexFitter
    DEVICE_INPUT(dev_multi_fit_vertices_t, PV::Vertex)    dev_multi_fit_vertices;
    DEVICE_INPUT(dev_number_of_multi_fit_vertices_t, unsigned) dev_number_of_multi_fit_vertices;
    // Outputs -- identical type/layout to pv_beamline_cleanup outputs
    DEVICE_OUTPUT(dev_multi_final_vertices_t, PV::Vertex)    dev_multi_final_vertices;
    DEVICE_OUTPUT(dev_number_of_multi_final_vertices_t, unsigned) dev_number_of_multi_final_vertices;
};

__global__ void pvfinder_nn_cleanup(
    Parameters,
    const float minChi2Dist,
    Allen::Monitoring::AveragingCounter<>::DeviceType,
    Allen::Monitoring::Histogram<>::DeviceType,
    Allen::Monitoring::Histogram<>::DeviceType,
    Allen::Monitoring::Histogram<>::DeviceType,
    Allen::Monitoring::Histogram<>::DeviceType,
    Allen::Monitoring::Histogram<>::DeviceType);

struct pvfinder_nn_cleanup_t : public DeviceAlgorithm, Parameters {
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
    Allen::Property<dim3> m_block_dim {
        this, "block_dim", {32, 1, 1}, "block dimensions"};
    Allen::Property<float> m_minChi2Dist {
        this, "minChi2Dist",
        BeamlinePVConstants::CleanUp::minChi2Dist,
        "minimum chi2 distance between two PVs for them to be considered unique"};

    // Monitoring -- pp-only (no SMOG2 histograms)
    Allen::Monitoring::AveragingCounter<> m_pvs {this, "n_PVs"};
    Allen::Monitoring::Histogram<> m_histogram_n_pvs {
        this, "n_pvs_event", "n_pvs_event", {21u, -0.5f, 20.5f}};
    Allen::Monitoring::Histogram<> m_histogram_pv_x {
        this, "pv_x", "pv_x", {1000u, -2.f, 2.f}};
    Allen::Monitoring::Histogram<> m_histogram_pv_y {
        this, "pv_y", "pv_y", {1000u, -2.f, 2.f}};
    Allen::Monitoring::Histogram<> m_histogram_pv_z {
        this, "pv_z", "pv_z", {2000u, -200.f, 400.f}};  // NN range: [-100, +300]
    Allen::Monitoring::Histogram<> m_histogram_pv_z_only_pp {
        this, "pv_z_only_pp", "pv_z_only_pp", {2000u, -200.f, 400.f}};

    // Binary dump of final vertices for offline validation
    Allen::Property<std::string> m_dump_dir {
        this, "dump_dir", "",
        "directory to write binary vertex dump (0xAB21 format); empty disables"};
    Allen::Property<std::string> m_output_file {
        this, "output_file", "allen_nn_final_vertices.bin",
        "filename inside dump_dir"};
};

} // namespace pvfinder_nn_cleanup
