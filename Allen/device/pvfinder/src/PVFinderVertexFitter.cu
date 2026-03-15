#include "PVFinderVertexFitter.cuh"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_vertex_fitter::pvfinder_vertex_fitter_t)

namespace pvfinder_vertex_fitter {

// ---------------------------------------------------------------------------
// set_arguments_size
// ---------------------------------------------------------------------------
void pvfinder_vertex_fitter_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    set_size<dev_nn_multi_fit_vertices_t>(arguments, n_events * PV::max_number_vertices);
    set_size<dev_nn_number_of_multi_fit_vertices_t>(arguments, n_events);
}

// ---------------------------------------------------------------------------
// operator()
// ---------------------------------------------------------------------------
void pvfinder_vertex_fitter_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    Allen::memset_async<dev_nn_number_of_multi_fit_vertices_t>(arguments, 0, context);

    const auto block_dim = dim3(warp_size, m_block_dim_y);
    global_function(pvfinder_vertex_fitter)(
        dim3(size<dev_event_list_t>(arguments)),
        block_dim,
        context)(
        arguments,
        static_cast<unsigned>(m_pp_minNumTracksPerVertex),
        static_cast<unsigned>(m_maxFitIter),
        static_cast<float>(m_zmin),
        static_cast<float>(m_zmax),
        static_cast<float>(m_maxChi2),
        static_cast<float>(m_chi2CutExp),
        static_cast<float>(m_minWeight),
        static_cast<float>(m_maxDeltaZConverged),
        static_cast<float>(m_maxVertexRho2));

    // Validation dump -- fires once when dump_dir property is non-empty.
    const std::string& dump_dir = m_dump_dir.value();
    if (!dump_dir.empty() && !m_dump_done) {
        cudaStreamSynchronize(context.stream());

        const unsigned n_events = first<host_number_of_events_t>(arguments);
        std::vector<PV::Vertex> h_vertices(n_events * PV::max_number_vertices);
        std::vector<unsigned>   h_nvertices(n_events);

        cudaMemcpy(
            h_vertices.data(),
            data<dev_nn_multi_fit_vertices_t>(arguments),
            n_events * PV::max_number_vertices * sizeof(PV::Vertex),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(
            h_nvertices.data(),
            data<dev_nn_number_of_multi_fit_vertices_t>(arguments),
            n_events * sizeof(unsigned),
            cudaMemcpyDeviceToHost);

        // Binary layout (matches validate_vertices.py reader):
        //   uint32  magic = 0xAB1F
        //   uint32  n_events
        //   for each event e:
        //       uint32  n_vertices
        //       for each vertex v:
        //           float32  x, y, z
        //           float32  cov00, cov10, cov11, cov20, cov21, cov22
        //           float32  chi2
        //           int32    ndof
        //           float32  nTracks
        const std::string path = dump_dir + "/allen_nn_vertices.bin";
        std::ofstream f(path, std::ios::binary);
        if (f) {
            const uint32_t magic = 0xAB1Fu;
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
            printf("[pvfinder_vertex_fitter] Validation dump written to %s (%u events)\n",
                   dump_dir.c_str(), n_events);
        } else {
            printf("[pvfinder_vertex_fitter] WARNING: could not open %s for writing\n",
                   path.c_str());
        }
        m_dump_done = true;
    }
}

void pvfinder_vertex_fitter_t::update(const Constants& constants) const
{
    updateCommon(constants);
}

// ---------------------------------------------------------------------------
// Kernel -- direct adaptation of pv_beamline_multi_fitter.
//
// Differences from the classical version:
//   1. Reads from dev_nn_zpeaks / dev_nn_number_of_zpeaks.
//   2. Reads per-track denominators from dev_nn_pvtracks_denom.
//   3. No SMOG2 branching: all NN seeds are in [-100, +300] mm (pp region).
//      dev_beamline.tx is used unconditionally for both seed x/y and residuals.
//   4. minTracks is always pp_minNumTracksPerVertex (no SMOG2 min tracks).
//   5. zmin/zmax defaults are -100/+300 mm (NN coverage, not full beamline).
// ---------------------------------------------------------------------------
__global__ void pvfinder_vertex_fitter(
    pvfinder_vertex_fitter::Parameters parameters,
    const unsigned pp_minNumTracksPerVertex,
    const unsigned maxFitIter,
    const float    zmin,
    const float    zmax,
    const float    maxChi2,
    const float    chi2CutExp,
    const float    minWeight,
    const float    maxDeltaZConverged,
    const float    maxVertexRho2)
{
    const unsigned event_number = parameters.dev_event_list[blockIdx.x];
    unsigned* number_of_multi_fit_vertices =
        parameters.dev_nn_number_of_multi_fit_vertices + event_number;

    const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];

    const float*    zseeds          = parameters.dev_nn_zpeaks + event_number * PV::max_number_vertices;
    const unsigned  number_of_seeds = parameters.dev_nn_number_of_zpeaks[event_number];

    const PVTrack* tracks         = parameters.dev_pvtracks          + velo_tracks_view.offset();
    const float*   pvtracks_denom = parameters.dev_nn_pvtracks_denom + velo_tracks_view.offset();

    PV::Vertex* vertices = parameters.dev_nn_multi_fit_vertices + event_number * PV::max_number_vertices;
    PV::Vertex vertex;

    // All NN seeds are in the pp region: use dev_beamline.tx unconditionally.
    const float2 tx_beam = dev_beamline.tx;

    // One thread-group per z-seed (threadIdx.y indexes the seed).
    for (unsigned i_thisseed = threadIdx.y; i_thisseed < number_of_seeds; i_thisseed += blockDim.y) {
        bool converged = false;
        bool accept    = true;
        float vtxcov[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

        const float seed_pos_z = zseeds[i_thisseed];
        const float2 seed_pos_xy {
            dev_beamline.pos.x + tx_beam.x * seed_pos_z,
            dev_beamline.pos.y + tx_beam.y * seed_pos_z};

        float2   vtxpos_xy = seed_pos_xy;
        float    vtxpos_z  = seed_pos_z;
        float    chi2tot   = 0.f;
        float    sum_weights = 0.f;
        unsigned nselectedtracks = 0u;

        for (unsigned iter = 0; iter < maxFitIter && !converged; ++iter) {
            auto halfD2Chi2DX2_00 = 0.f;
            auto halfD2Chi2DX2_11 = 0.f;
            auto halfD2Chi2DX2_20 = 0.f;
            auto halfD2Chi2DX2_21 = 0.f;
            auto halfD2Chi2DX2_22 = 0.f;
            float3 halfDChi2DX {0.f, 0.f, 0.f};
            sum_weights      = 0.f;
            nselectedtracks  = 0u;
            chi2tot          = 0.f;
            float local_chi2tot    = 0.f;
            float local_sum_weights = 0.f;

            for (unsigned i = threadIdx.x; i < velo_tracks_view.size(); i += blockDim.x) {
                const PVTrackInVertex& trk = tracks[i];
                if (trk.z < zmin || trk.z >= zmax) continue;

                const float2 tx       = trk.tx - tx_beam;
                const float  dz       = vtxpos_z - trk.z;
                const float2 res      = vtxpos_xy - (trk.x + tx * dz);
                const float  chi2     = res.x * res.x * trk.W_00
                                      + res.y * res.y * trk.W_11;

                if (chi2 < maxChi2) {
                    ++nselectedtracks;

                    // Adaptive weight: nominator uses current vertex position,
                    // denominator uses seed position (avoids double-counting).
                    const float  dz_seed   = seed_pos_z - trk.z;
                    const float2 res_seed  = seed_pos_xy - (trk.x + tx * dz_seed);
                    const float  chi2_seed = res_seed.x * res_seed.x * trk.W_00
                                           + res_seed.y * res_seed.y * trk.W_11;
                    const float  exp_chi2_0   = expf(chi2_seed * (-0.5f));
                    const float  nom          = expf(chi2   * (-0.5f));
                    const float  denom        = chi2CutExp + nom;
                    const float  track_weight = nom / (denom + pvtracks_denom[i] - exp_chi2_0);

                    if (track_weight > minWeight) {
                        const float3 HWr {
                            res.x * trk.W_00,
                            res.y * trk.W_11,
                            -tx.x * res.x * trk.W_00 - tx.y * res.y * trk.W_11};

                        halfDChi2DX       = halfDChi2DX + HWr * track_weight;
                        halfD2Chi2DX2_00 += track_weight * trk.HWH_00;
                        halfD2Chi2DX2_11 += track_weight * trk.HWH_11;
                        halfD2Chi2DX2_20 += track_weight * trk.HWH_20;
                        halfD2Chi2DX2_21 += track_weight * trk.HWH_21;
                        halfD2Chi2DX2_22 += track_weight * trk.HWH_22;

                        local_chi2tot      += track_weight * chi2;
                        local_sum_weights  += track_weight;
                    }
                }
            } // track loop

#if defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_HIP)
            for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
                halfD2Chi2DX2_00  += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_00,  offset);
                halfD2Chi2DX2_11  += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_11,  offset);
                halfD2Chi2DX2_20  += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_20,  offset);
                halfD2Chi2DX2_21  += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_21,  offset);
                halfD2Chi2DX2_22  += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_22,  offset);
                halfDChi2DX.x     += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX.x,     offset);
                halfDChi2DX.y     += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX.y,     offset);
                halfDChi2DX.z     += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX.z,     offset);
                local_chi2tot     += __shfl_down_sync(0xFFFFFFFF, local_chi2tot,      offset);
                local_sum_weights += __shfl_down_sync(0xFFFFFFFF, local_sum_weights,  offset);
                nselectedtracks   += __shfl_down_sync(0xFFFFFFFF, nselectedtracks,    offset);
            }
#endif

            if (threadIdx.x == 0) {
                chi2tot     += local_chi2tot;
                sum_weights += local_sum_weights;

                if (nselectedtracks >= pp_minNumTracksPerVertex) {
                    // Analytical 3x3 symmetric matrix inversion for the vertex covariance.
                    const auto a00 = halfD2Chi2DX2_00;
                    const auto a11 = halfD2Chi2DX2_11;
                    const auto a20 = halfD2Chi2DX2_20;
                    const auto a21 = halfD2Chi2DX2_21;
                    const auto a22 = halfD2Chi2DX2_22;

                    const auto det     = a00 * (a22 * a11 - a21 * a21) + a20 * (-a11 * a20);
                    const auto inv_det = 1.f / det;

                    vtxcov[0] =  (a22 * a11 - a21 * a21) * inv_det;
                    vtxcov[1] = -(-a20 * a21)             * inv_det;
                    vtxcov[2] =  (a22 * a00 - a20 * a20)  * inv_det;
                    vtxcov[3] =  (-a20 * a11)             * inv_det;
                    vtxcov[4] = -(a21 * a00)              * inv_det;
                    vtxcov[5] =  (a11 * a00)              * inv_det;

                    const float2 delta_xy {
                        -1.f * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z),
                        -1.f * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z)};
                    const float delta_z =
                        -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z);

                    chi2tot += delta_xy.x * halfDChi2DX.x
                             + delta_xy.y * halfDChi2DX.y
                             + delta_z    * halfDChi2DX.z;

                    vtxpos_xy = vtxpos_xy + delta_xy;
                    vtxpos_z  = vtxpos_z  + delta_z;
                    converged = fabsf(delta_z) < maxDeltaZConverged;
                } else {
                    converged = true;
                    accept    = false;
                }
            }

#if defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_HIP)
            vtxpos_xy.x = __shfl_sync(0xFFFFFFFF, vtxpos_xy.x, 0);
            vtxpos_xy.y = __shfl_sync(0xFFFFFFFF, vtxpos_xy.y, 0);
            vtxpos_z    = __shfl_sync(0xFFFFFFFF, vtxpos_z,    0);
            converged   = __shfl_sync(0xFFFFFFFF, converged,   0);
#endif
        } // iteration loop

        if (accept && threadIdx.x == 0) {
            vertex.chi2 = chi2tot;
            vertex.setPosition(vtxpos_xy, vtxpos_z);
            vertex.setCovMatrix(vtxcov);
            vertex.nTracks = sum_weights;

            const float beamlinedx  = vertex.position.x - dev_beamline.pos.x;
            const float beamlinedy  = vertex.position.y - dev_beamline.pos.y;
            const float beamlinerho2 = beamlinedx * beamlinedx + beamlinedy * beamlinedy;

            if (nselectedtracks >= pp_minNumTracksPerVertex && beamlinerho2 < maxVertexRho2) {
                unsigned vertex_index = atomicAdd(number_of_multi_fit_vertices, 1u);
                vertices[vertex_index] = vertex;
            }
        }
    } // seed loop
}

} // namespace pvfinder_vertex_fitter
