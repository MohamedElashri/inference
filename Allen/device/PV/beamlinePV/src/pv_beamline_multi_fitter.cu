/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "pv_beamline_multi_fitter.cuh"

INSTANTIATE_ALGORITHM(pv_beamline_multi_fitter::pv_beamline_multi_fitter_t)

void pv_beamline_multi_fitter::pv_beamline_multi_fitter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_multi_fit_vertices_t>(arguments, first<host_number_of_events_t>(arguments) * PV::max_number_vertices);
  set_size<dev_number_of_multi_fit_vertices_t>(arguments, first<host_number_of_events_t>(arguments));
}

void pv_beamline_multi_fitter::pv_beamline_multi_fitter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_number_of_multi_fit_vertices_t>(arguments, 0, context);

  const auto block_dimension = dim3(warp_size, m_block_dim_y);
  global_function(pv_beamline_multi_fitter)(dim3(size<dev_event_list_t>(arguments)), block_dimension, context)(
    arguments,
    m_SMOG2_pp_separation,
    m_SMOG2_minNumTracksPerVertex,
    m_pp_minNumTracksPerVertex,
    m_maxFitIter,
    m_zmin,
    m_zmax,
    m_maxChi2,
    m_chi2CutExp,
    m_minWeight,
    m_maxDeltaZConverged,
    m_maxVertexRho2);
}

void pv_beamline_multi_fitter::pv_beamline_multi_fitter_t::update(const Constants& constants) const
{
  updateCommon(constants);
}

namespace {
  // Fixed-point quantum for the multi-vertex fit's per-track chi2/Hessian accumulation. Integer
  // addition is associative regardless of order, unlike the float addition it replaces -- so the
  // fitted vertex no longer depends on which physical VELO track lands on which GPU thread/warp-lane,
  // which is itself not run-to-run reproducible (track storage order is assigned via an atomicAdd
  // counter during search_by_triplet's track-finding).
  //
  // Chosen to comfortably cover the physical range of these quantities: track_weight is in (0,1],
  // the chi2 contribution is bounded below maxChi2 (an O(10) property, via the `chi2 < maxChi2` cut
  // above) and the Hessian terms (trk.HWH_*, built from trk.W_00/W_11 -- inverse VELO position-
  // covariance terms) are bounded in practice by realistic VELO hit resolution (a few microns at
  // best), giving per-contribution magnitudes up to ~1e6 in the most extreme case. A 2^-20 quantum
  // with a 64-bit accumulator keeps quantisation error far below float32's own ~2^-23 relative
  // precision at these magnitudes, while leaving several orders of magnitude of headroom before the
  // sum (over at most a few hundred contributing tracks) could approach the 64-bit range.
  constexpr float fit_fixed_point_inv_precision = 1048576.f; // 2^20
  constexpr float fit_fixed_point_precision = 1.f / fit_fixed_point_inv_precision;

  __device__ long long to_fixed_point(float value) { return llrintf(value * fit_fixed_point_inv_precision); }

  __device__ float from_fixed_point(long long value) { return static_cast<float>(value) * fit_fixed_point_precision; }
} // namespace

__global__ void pv_beamline_multi_fitter::pv_beamline_multi_fitter(
  pv_beamline_multi_fitter::Parameters parameters,
  const float SMOG2_pp_separation,
  const unsigned SMOG2_minNumTracksPerVertex,
  const unsigned pp_minNumTracksPerVertex,
  const unsigned maxFitIter,
  const float zmin,
  const float zmax,
  const float maxChi2,
  const float chi2CutExp,
  const float minWeight,
  const float maxDeltaZConverged,
  const float maxVertexRho2)
{
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  unsigned* number_of_multi_fit_vertices = parameters.dev_number_of_multi_fit_vertices + event_number;

  const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];

  const float* zseeds = parameters.dev_zpeaks + event_number * PV::max_number_vertices;
  const unsigned number_of_seeds = parameters.dev_number_of_zpeaks[event_number];

  const PVTrack* tracks = parameters.dev_pvtracks + velo_tracks_view.offset();

  PV::Vertex* vertices = parameters.dev_multi_fit_vertices + event_number * PV::max_number_vertices;
  PV::Vertex vertex;
  const float* pvtracks_denom = parameters.dev_pvtracks_denom + velo_tracks_view.offset();

  // make sure that we have one thread per seed
  for (unsigned i_thisseed = threadIdx.y; i_thisseed < number_of_seeds; i_thisseed += blockDim.y) {
    bool converged = false;
    bool accept = true;
    float vtxcov[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const float seed_pos_z = zseeds[i_thisseed];
    float tx_beam_seed;
    float ty_beam_seed;
    if (seed_pos_z > SMOG2_pp_separation) {
      tx_beam_seed = dev_beamline.tx.x;
      ty_beam_seed = dev_beamline.tx.y;
    }
    else {
      tx_beam_seed = dev_beamline.tx_SMOG.x;
      ty_beam_seed = dev_beamline.tx_SMOG.y;
    }
    const float2 seed_pos_xy {
      dev_beamline.pos.x + tx_beam_seed * zseeds[i_thisseed], dev_beamline.pos.y + ty_beam_seed * zseeds[i_thisseed]};

    float2 vtxpos_xy = seed_pos_xy;

    auto vtxpos_z = seed_pos_z;
    float chi2tot = 0.f;
    float sum_weights = 0.f;
    unsigned nselectedtracks = 0;
    const unsigned minTracks =
      seed_pos_z <= SMOG2_pp_separation ? SMOG2_minNumTracksPerVertex : pp_minNumTracksPerVertex;

    for (unsigned iter = 0; iter < maxFitIter && !converged; ++iter) {
      long long halfD2Chi2DX2_00_fixed = 0;
      long long halfD2Chi2DX2_11_fixed = 0;
      long long halfD2Chi2DX2_20_fixed = 0;
      long long halfD2Chi2DX2_21_fixed = 0;
      long long halfD2Chi2DX2_22_fixed = 0;
      long long halfDChi2DX_x_fixed = 0;
      long long halfDChi2DX_y_fixed = 0;
      long long halfDChi2DX_z_fixed = 0;
      sum_weights = 0.f;

      nselectedtracks = 0;
      chi2tot = 0.f;
      long long local_chi2tot_fixed = 0;
      long long local_sum_weights_fixed = 0;

      for (unsigned i = threadIdx.x; i < velo_tracks_view.size(); i += blockDim.x) {
        // compute the chi2
        const PVTrackInVertex& trk = tracks[i];
        if (trk.z < zmin || trk.z >= zmax) continue;
        float2 tx_beam;
        if (seed_pos_z > SMOG2_pp_separation) {
          tx_beam = dev_beamline.tx;
        }
        else {
          tx_beam = dev_beamline.tx_SMOG;
        }

        const auto dz = vtxpos_z - trk.z;
        const float2 tx = trk.tx - tx_beam;
        const float2 res = vtxpos_xy - (trk.x + tx * dz);
        const auto chi2 = res.x * res.x * trk.W_00 + res.y * res.y * trk.W_11;

        // compute the weight.
        if (chi2 < maxChi2) {
          ++nselectedtracks;
          // for more information on the weighted fitting, see e.g.
          // Adaptive Multi-vertex fitting, R. Frühwirth, W. Waltenberger
          // https://cds.cern.ch/record/803519/files/p280.pdf
          // use seed position for chi2 calculation of nominator

          const float dz_seed = seed_pos_z - trk.z;
          const float2 res_seed = seed_pos_xy - (trk.x + tx * dz_seed);
          const float chi2_seed = res_seed.x * res_seed.x * trk.W_00 + res_seed.y * res_seed.y * trk.W_11;
          const float exp_chi2_0 = expf(chi2_seed * (-0.5f));
          // calculating chi w.r.t to vtx position and not seed posotion very important for resolution of high mult
          // vetices
          const auto nom = expf(chi2 * (-0.5f));

          const auto denom = chi2CutExp + nom;
          // substract this term to avoid double counting

          const auto track_weight = nom / (denom + pvtracks_denom[i] - exp_chi2_0);

          // unfortunately branchy, but reduces fake rate
          // not cutting on the weights seems to be important for resolution of high multiplicity tracks
          if (track_weight > minWeight) {
            const float3 HWr {res.x * trk.W_00, res.y * trk.W_11, -tx.x * res.x * trk.W_00 - tx.y * res.y * trk.W_11};

            halfDChi2DX_x_fixed += to_fixed_point(HWr.x * track_weight);
            halfDChi2DX_y_fixed += to_fixed_point(HWr.y * track_weight);
            halfDChi2DX_z_fixed += to_fixed_point(HWr.z * track_weight);
            halfD2Chi2DX2_00_fixed += to_fixed_point(track_weight * trk.HWH_00);
            halfD2Chi2DX2_11_fixed += to_fixed_point(track_weight * trk.HWH_11);
            halfD2Chi2DX2_20_fixed += to_fixed_point(track_weight * trk.HWH_20);
            halfD2Chi2DX2_21_fixed += to_fixed_point(track_weight * trk.HWH_21);
            halfD2Chi2DX2_22_fixed += to_fixed_point(track_weight * trk.HWH_22);

            local_chi2tot_fixed += to_fixed_point(track_weight * chi2);
            local_sum_weights_fixed += to_fixed_point(track_weight);
          }
        }
      }

#if defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_HIP)
      // Use CUDA warp-level primitives for adding up some numbers onto a single
      // thread without using any shared memory.
      // See https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
      // Integer addition is associative, unlike the float addition it replaces, so this reduction no
      // longer depends on which physical track's contribution landed on which lane.
      for (int i = warp_size / 2; i > 0; i = i / 2) {
        halfD2Chi2DX2_00_fixed += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_00_fixed, i);
        halfD2Chi2DX2_11_fixed += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_11_fixed, i);
        halfD2Chi2DX2_20_fixed += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_20_fixed, i);
        halfD2Chi2DX2_21_fixed += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_21_fixed, i);
        halfD2Chi2DX2_22_fixed += __shfl_down_sync(0xFFFFFFFF, halfD2Chi2DX2_22_fixed, i);
        halfDChi2DX_x_fixed += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX_x_fixed, i);
        halfDChi2DX_y_fixed += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX_y_fixed, i);
        halfDChi2DX_z_fixed += __shfl_down_sync(0xFFFFFFFF, halfDChi2DX_z_fixed, i);
        local_chi2tot_fixed += __shfl_down_sync(0xFFFFFFFF, local_chi2tot_fixed, i);
        local_sum_weights_fixed += __shfl_down_sync(0xFFFFFFFF, local_sum_weights_fixed, i);
        nselectedtracks += __shfl_down_sync(0xFFFFFFFF, nselectedtracks, i);
      }
#endif

      if (threadIdx.x == 0) {
        const float local_chi2tot = from_fixed_point(local_chi2tot_fixed);
        const float local_sum_weights = from_fixed_point(local_sum_weights_fixed);
        const float3 halfDChi2DX {from_fixed_point(halfDChi2DX_x_fixed),
                                  from_fixed_point(halfDChi2DX_y_fixed),
                                  from_fixed_point(halfDChi2DX_z_fixed)};

        chi2tot += local_chi2tot;
        sum_weights += local_sum_weights;
        if (nselectedtracks >= minTracks) {
          // compute the new vertex covariance using analytical inversion
          // dividing matrix elements not important for resoltuon of high mult pvs
          const auto a00 = from_fixed_point(halfD2Chi2DX2_00_fixed);
          const auto a11 = from_fixed_point(halfD2Chi2DX2_11_fixed);
          const auto a20 = from_fixed_point(halfD2Chi2DX2_20_fixed);
          const auto a21 = from_fixed_point(halfD2Chi2DX2_21_fixed);
          const auto a22 = from_fixed_point(halfD2Chi2DX2_22_fixed);

          const auto det = a00 * (a22 * a11 - a21 * a21) + a20 * (-a11 * a20);
          const auto inv_det = 1.f / det;

          // maybe we should catch the case when det = 0
          // if (det == 0) return false;

          vtxcov[0] = (a22 * a11 - a21 * a21) * inv_det;
          vtxcov[1] = -(-a20 * a21) * inv_det;
          vtxcov[2] = (a22 * a00 - a20 * a20) * inv_det;
          vtxcov[3] = (-a20 * a11) * inv_det;
          vtxcov[4] = -(a21 * a00) * inv_det;
          vtxcov[5] = (a11 * a00) * inv_det;

          const float2 delta_xy {
            -1.f * (vtxcov[0] * halfDChi2DX.x + vtxcov[1] * halfDChi2DX.y + vtxcov[3] * halfDChi2DX.z),
            -1.f * (vtxcov[1] * halfDChi2DX.x + vtxcov[2] * halfDChi2DX.y + vtxcov[4] * halfDChi2DX.z)};

          const auto delta_z =
            -1.f * (vtxcov[3] * halfDChi2DX.x + vtxcov[4] * halfDChi2DX.y + vtxcov[5] * halfDChi2DX.z);
          chi2tot += delta_xy.x * halfDChi2DX.x + delta_xy.y * halfDChi2DX.y + delta_z * halfDChi2DX.z;

          // update the position
          vtxpos_xy = vtxpos_xy + delta_xy;
          vtxpos_z = vtxpos_z + delta_z;
          converged = fabsf(delta_z) < maxDeltaZConverged;
        }
        else {
          // Finish loop and do not accept vertex
          converged = true;
          accept = false;
        }
      }

#if defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_HIP)
      // Synchronize the value of thread 0 in the warp across the entire warp
      // See https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
      vtxpos_xy.x = __shfl_sync(0xFFFFFFFF, vtxpos_xy.x, 0);
      vtxpos_xy.y = __shfl_sync(0xFFFFFFFF, vtxpos_xy.y, 0);
      vtxpos_z = __shfl_sync(0xFFFFFFFF, vtxpos_z, 0);
      converged = __shfl_sync(0xFFFFFFFF, converged, 0);
#endif
    } // end iteration loop

    if (accept && threadIdx.x == 0) {
      vertex.chi2 = chi2tot;
      vertex.setPosition(vtxpos_xy, vtxpos_z);
      vertex.setCovMatrix(vtxcov);
      vertex.nTracks = sum_weights;

      float2 tx_beam;
      if (vertex.position.z > SMOG2_pp_separation) {
        tx_beam = dev_beamline.tx;
      }
      else {
        tx_beam = dev_beamline.tx_SMOG;
      }

      const auto beamlinedx = vertex.position.x - dev_beamline.pos.x - tx_beam.x * vertex.position.z;
      const auto beamlinedy = vertex.position.y - dev_beamline.pos.y - tx_beam.y * vertex.position.z;
      const auto beamlinerho2 = beamlinedx * beamlinedx + beamlinedy * beamlinedy;
      const auto minTracks =
        vertex.position.z <= SMOG2_pp_separation ? SMOG2_minNumTracksPerVertex : pp_minNumTracksPerVertex;
      if (nselectedtracks >= minTracks && beamlinerho2 < maxVertexRho2) {
        unsigned vertex_index = atomicAdd(number_of_multi_fit_vertices, 1);
        vertices[vertex_index] = vertex;
      }
    }
  }
}
