#include "PVFinderDumpVertices.cuh"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include "ArgumentOps.cuh"

INSTANTIATE_ALGORITHM(pvfinder_dump_vertices::pvfinder_dump_vertices_t)

namespace pvfinder_dump_vertices {

void pvfinder_dump_vertices_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    const std::string& dump_dir  = m_dump_dir.value();
    printf("[pvfinder_dump_vertices] operator() called, dump_dir='%s'\n", dump_dir.c_str());
    if (dump_dir.empty()) return;

    // Copy device arrays to host using Allen's portable helper
    const auto h_vertices  = Allen::ArgumentOperations::make_host_buffer<dev_multi_final_vertices_t>(arguments, context);
    const auto h_nvertices = Allen::ArgumentOperations::make_host_buffer<dev_number_of_multi_final_vertices_t>(arguments, context);

    const unsigned n_events = first<host_number_of_events_t>(arguments);

    // Binary layout (magic 0xAB21 -- "any-chain final vertices").
    // Append to file so multiple slices accumulate into one file.
    const std::string path = dump_dir + "/" + m_output_file.value();
    const bool file_exists = std::ifstream(path).good();
    std::ofstream f(path, std::ios::binary | std::ios::app);
    if (!f) {
        printf("[pvfinder_dump_vertices] WARNING: could not open %s for writing\n",
               path.c_str());
        return;
    }

    if (!file_exists) {
        // Write header only once, on first open
        const uint32_t magic = 0xAB21u;
        const uint32_t ne    = n_events;
        f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        f.write(reinterpret_cast<const char*>(&ne),    sizeof(ne));
    }

    for (unsigned e = 0; e < n_events; ++e) {
        const uint32_t nv = h_nvertices[e];
        f.write(reinterpret_cast<const char*>(&nv), sizeof(nv));
        const PV::Vertex* verts = h_vertices.data() + e * PV::max_number_vertices;
        for (unsigned v = 0; v < nv; ++v) {
            f.write(reinterpret_cast<const char*>(&verts[v].position.x), sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].position.y), sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].position.z), sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].cov00), sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].cov10), sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].cov11), sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].cov20), sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].cov21), sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].cov22), sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].chi2),     sizeof(float));
            f.write(reinterpret_cast<const char*>(&verts[v].ndof),     sizeof(int32_t));
            f.write(reinterpret_cast<const char*>(&verts[v].nTracks),  sizeof(float));
        }
    }
    printf("[pvfinder_dump_vertices] Written slice to %s (%u events)\n",
           path.c_str(), n_events);
}

} // namespace pvfinder_dump_vertices
