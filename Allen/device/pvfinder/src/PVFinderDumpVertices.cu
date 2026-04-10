#include "PVFinderDumpVertices.cuh"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_dump_vertices::pvfinder_dump_vertices_t)

namespace pvfinder_dump_vertices {

void pvfinder_dump_vertices_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    const std::string& dump_dir  = m_dump_dir.value();
    if (dump_dir.empty() || m_dump_done) return;

    cudaStreamSynchronize(context.stream());
    std::error_code ec;
    std::filesystem::create_directories(dump_dir, ec);

    const unsigned n_events = first<host_number_of_events_t>(arguments);
    std::vector<PV::Vertex> h_vertices(n_events * PV::max_number_vertices);
    std::vector<unsigned>   h_nvertices(n_events);

    cudaMemcpy(
        h_vertices.data(),
        data<dev_multi_final_vertices_t>(arguments),
        n_events * PV::max_number_vertices * sizeof(PV::Vertex),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        h_nvertices.data(),
        data<dev_number_of_multi_final_vertices_t>(arguments),
        n_events * sizeof(unsigned),
        cudaMemcpyDeviceToHost);

    // Binary layout (magic 0xAB21 -- "any-chain final vertices"):
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
    const std::string path = dump_dir + "/" + m_output_file.value();
    std::ofstream f(path, std::ios::binary);
    if (f) {
        const uint32_t magic = 0xAB21u;
        const uint32_t ne    = n_events;
        f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        f.write(reinterpret_cast<const char*>(&ne),    sizeof(ne));
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
        printf("[pvfinder_dump_vertices] Written to %s (%u events)\n",
               path.c_str(), n_events);
    } else {
        printf("[pvfinder_dump_vertices] WARNING: could not open %s for writing\n",
               path.c_str());
    }
    m_dump_done = true;
}

} // namespace pvfinder_dump_vertices
