#pragma once

#include "AlgorithmTypes.cuh"
#include "PV_Definitions.cuh"
#include "BackendCommon.h"
#include <string>

// ---------------------------------------------------------------------------
// PVFinderDumpVertices: generic side-effect algorithm that dumps any PV chain's
// final vertex array to a binary file.
//
// Intended use: attach after pv_beamline_cleanup_t (classical chain) or
// pvfinder_nn_cleanup_t (NN chain) to produce a reference dump for
// validate_vertices.py.  The algorithm has no device outputs; it only writes
// to disk.
//
// Binary layout (identical to PVFinderNNCleanup dump, magic 0xAB21):
//   uint32  magic = 0xAB21
//   uint32  n_events
//   for each event e:
//       uint32  n_vertices
//       for each vertex v:
//           float32  x, y, z
//           float32  cov00, cov10, cov11, cov20, cov21, cov22
//           float32  chi2
//           int32    ndof
//           float32  nTracks
// ---------------------------------------------------------------------------

namespace pvfinder_dump_vertices {

struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    // Accepts output of either pv_beamline_cleanup_t or pvfinder_nn_cleanup_t --
    // both use the same parameter tag names.
    DEVICE_INPUT(dev_multi_final_vertices_t, PV::Vertex) dev_multi_final_vertices;
    DEVICE_INPUT(dev_number_of_multi_final_vertices_t, unsigned) dev_number_of_multi_final_vertices;
    // Dummy output to force this algorithm into the Allen dependency graph
    HOST_OUTPUT(host_pv_dump_done_t, bool) host_pv_dump_done;
};

struct pvfinder_dump_vertices_t : public HostAlgorithm, Parameters {
    void set_arguments_size(
        ArgumentReferences<Parameters> arguments,
        const RuntimeOptions&,
        const Constants&) const {
        set_size<host_pv_dump_done_t>(arguments, 1);
    }

    void operator()(
        const ArgumentReferences<Parameters>& arguments,
        const RuntimeOptions&,
        const Constants&,
        const Allen::Context& context) const;

private:
    Allen::Property<std::string> m_dump_dir {
        this, "dump_dir", "",
        "directory to write the vertex dump; empty string disables the dump"};

    Allen::Property<std::string> m_output_file {
        this, "output_file", "allen_vertices.bin",
        "filename inside dump_dir (default: allen_vertices.bin)"};

};

} // namespace pvfinder_dump_vertices
