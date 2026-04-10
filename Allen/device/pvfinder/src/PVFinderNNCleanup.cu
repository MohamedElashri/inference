#include "PVFinderNNCleanup.cuh"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_nn_cleanup::pvfinder_nn_cleanup_t)

namespace pvfinder_nn_cleanup {

// ---------------------------------------------------------------------------
// Insertion-sort device helper (inlined -- avoids cross-TU __device__ linkage).
// Identical logic to pv_beamline_cleanup::sort_pvs_by_z.
// ---------------------------------------------------------------------------
__device__ static void sort_pvs_by_z(PV::Vertex* final_vertices, unsigned n_vertices)
{
    if (n_vertices <= 1u) return;

    if (blockDim.x <= n_vertices) {
        // Parallel rank sort: each thread holds one PV, counts how many are smaller.
        auto pv = final_vertices[threadIdx.x];
        __syncthreads();

        unsigned smaller = 0u;
        for (unsigned i = 0u; i < n_vertices; ++i) {
            smaller += (final_vertices[i].position.z < pv.position.z) ? 1u : 0u;
        }
        final_vertices[smaller] = pv;
        return;
    }

    // Serial insertion sort (thread 0 only) -- used when n_vertices < blockDim.x.
    if (threadIdx.x == 0) {
        for (unsigned i = 1u; i < n_vertices; ++i) {
            PV::Vertex pv = final_vertices[i];
            int j = static_cast<int>(i) - 1;
            while (j >= 0 && final_vertices[j].position.z > pv.position.z) {
                final_vertices[j + 1] = final_vertices[j];
                --j;
            }
            final_vertices[j + 1] = pv;
        }
    }
}

// ---------------------------------------------------------------------------
// set_arguments_size
// ---------------------------------------------------------------------------
void pvfinder_nn_cleanup_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    set_size<dev_multi_final_vertices_t>(arguments, n_events * PV::max_number_vertices);
    set_size<dev_number_of_multi_final_vertices_t>(arguments, n_events);
}

// ---------------------------------------------------------------------------
// operator()
// ---------------------------------------------------------------------------
void pvfinder_nn_cleanup_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    Allen::memset_async<dev_number_of_multi_final_vertices_t>(arguments, 0, context);

    global_function(pvfinder_nn_cleanup)(
        dim3(size<dev_event_list_t>(arguments)),
        m_block_dim,
        context)(
        arguments,
        static_cast<float>(m_minChi2Dist),
        m_pvs.data(context),
        m_histogram_n_pvs.data(context),
        m_histogram_pv_x.data(context),
        m_histogram_pv_y.data(context),
        m_histogram_pv_z.data(context),
        m_histogram_pv_z_only_pp.data(context));

    // Validation dump -- fires once when dump_dir property is non-empty.
    // Writes post-deduplication final vertices (magic 0xAB20).
    // Binary layout matches validate_vertices.py reader (same as 0xAB1F from
    // PVFinderVertexFitter, but produced after the chi2-distance cleanup pass).
    const std::string& dump_dir = m_dump_dir.value();
    if (!dump_dir.empty() && !m_dump_done) {
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

        const std::string path = dump_dir + "/allen_nn_final_vertices.bin";
        std::ofstream f(path, std::ios::binary);
        if (f) {
            const uint32_t magic = 0xAB20u;
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
            printf("[pvfinder_nn_cleanup] Validation dump written to %s (%u events)\n",
                   path.c_str(), n_events);
        } else {
            printf("[pvfinder_nn_cleanup] WARNING: could not open %s for writing\n",
                   path.c_str());
        }
        m_dump_done = true;
    }
}

// ---------------------------------------------------------------------------
// Kernel -- adapted from pv_beamline_cleanup.
//
// Differences from the classical version:
//   1. Monitoring: SMOG2 histograms removed (NN covers pp region only).
//   2. sort_pvs_by_z inlined (avoids cross-TU __device__ linkage).
//   3. No SMOG2 vertex counting.
// ---------------------------------------------------------------------------
__global__ void pvfinder_nn_cleanup(
    pvfinder_nn_cleanup::Parameters parameters,
    const float minChi2Dist,
    Allen::Monitoring::AveragingCounter<>::DeviceType dev_n_pvs_counter,
    Allen::Monitoring::Histogram<>::DeviceType dev_n_pvs_histo,
    Allen::Monitoring::Histogram<>::DeviceType dev_pv_x_histo,
    Allen::Monitoring::Histogram<>::DeviceType dev_pv_y_histo,
    Allen::Monitoring::Histogram<>::DeviceType dev_pv_z_histo,
    Allen::Monitoring::Histogram<>::DeviceType dev_pv_z_only_pp_histo)
{
    __shared__ unsigned tmp_number_vertices[1];
    if (threadIdx.x == 0) *tmp_number_vertices = 0u;
    __syncthreads();

    const unsigned event_number = parameters.dev_event_list[blockIdx.x];

    const PV::Vertex* vertices        = parameters.dev_multi_fit_vertices
                                        + event_number * PV::max_number_vertices;
    PV::Vertex*       final_vertices  = parameters.dev_multi_final_vertices
                                        + event_number * PV::max_number_vertices;
    const unsigned n_fit = parameters.dev_number_of_multi_fit_vertices[event_number];

    // Deduplication: keep a vertex only if no other vertex with more tracks
    // is within minChi2Dist (in units of combined z-variance).
    for (unsigned i_pv = threadIdx.x; i_pv < n_fit; i_pv += blockDim.x) {
        bool unique = true;
        const PV::Vertex v1 = vertices[i_pv];
        for (unsigned j_pv = 0u; j_pv < n_fit; ++j_pv) {
            if (i_pv == j_pv) continue;
            const PV::Vertex v2 = vertices[j_pv];
            const float dz      = v1.position.z - v2.position.z;
            const float chi2_dist = dz * dz / (v1.cov22 + v2.cov22);
            if (chi2_dist < minChi2Dist && v1.nTracks < v2.nTracks) {
                unique = false;
                break;
            }
        }
        if (unique) {
            const unsigned slot = atomicAdd(tmp_number_vertices, 1u);
            final_vertices[slot] = v1;

            // Monitoring
            dev_pv_z_histo.increment(v1.position.z);
            dev_pv_x_histo.increment(v1.position.x);
            dev_pv_y_histo.increment(v1.position.y);
            dev_pv_z_only_pp_histo.increment(v1.position.z);
        }
    }
    __syncthreads();

    sort_pvs_by_z(final_vertices, *tmp_number_vertices);

    if (threadIdx.x == 0) {
        parameters.dev_number_of_multi_final_vertices[event_number] = *tmp_number_vertices;
        dev_n_pvs_histo.increment(*tmp_number_vertices);
        dev_n_pvs_counter.add(*tmp_number_vertices);
    }
}

} // namespace pvfinder_nn_cleanup
