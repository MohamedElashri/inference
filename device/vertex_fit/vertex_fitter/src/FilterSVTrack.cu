/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "FilterSVTrack.cuh"
#include "States.cuh"
#include "WarpIntrinsicsTools.cuh"
#include <PrefixSum.cuh>
#include <BinarySearch.cuh>

INSTANTIATE_ALGORITHM(FilterSVTrack::filter_sv_track_t)

__global__ void filter_svs_k(
  FilterSVTrack::Parameters parameters,
  const unsigned total_number_of_svs,
  const float SV_VZ_min,
  const float SV_VZ_max,
  const float SV_FD_min,
  const float SV_FD_max,
  const float SV_BPVIP_min,
  const float SV_BPVVDZ_min,
  const float SV_BPVVDRHO_min,
  const float SV_BPVDIRA_min,
  const float SV_MCORR_min,
  const float SV_MCORR_max,
  const float T_SV_MIPCHI2_min,
  const bool require_os_pair)
{
  const Allen::Views::Physics::CompositeParticle* svs = parameters.dev_svs;
  const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i_sv = threadID; i_sv < total_number_of_svs; i_sv += stride) {
    const auto& sv = svs[i_sv];
    const auto sv_fd = sv.fd();
    const auto sv_vxz = sv.vertex().z();
    const auto dira = sv.dira();
    bool sv_decision = SV_VZ_min < sv_vxz && sv_vxz < SV_VZ_max && sv.dz() > SV_BPVVDZ_min &&
                       sv.drho() > SV_BPVVDRHO_min && sv.ip() > SV_BPVIP_min && dira > SV_BPVDIRA_min &&
                       sv_fd > SV_FD_min && sv_fd < SV_FD_max && sv.mcor() > SV_MCORR_min && sv.mcor() < SV_MCORR_max;
    // Apply IP_chi2 constraint on the tracks of the input SV. Very powerful. Only if this variable is passed, it has
    // been seen it kills all signal for some lines.
    sv_decision &= (sv.minipchi2() >= T_SV_MIPCHI2_min);

    if (require_os_pair && sv.charge() != 0) {
      sv_decision = false;
    }
    if (sv_decision) atomicOr(parameters.dev_sv_decision + (i_sv / 32), 1 << (i_sv % 32));
  }
}

__global__ void filter_sv_track(
  FilterSVTrack::Parameters parameters,
  const unsigned number_of_events,
  const unsigned total_number_of_tracks,
  const float SV_T_BPV_NEWDIRA_min,
  const float T_CHI2NDF_max,
  const float T_PT_min,
  const float T_P_min,
  const float T_MIPCHI2_min,
  const float T_MIPCHI2_max,
  const float T_MIP_min,
  const float T_MIP_max,
  const float SV_T_DOCA_max,
  const float opening_angle_min,
  const bool require_same_pv)
{
  const Allen::Views::Physics::BasicParticle* tracks = parameters.dev_tracks;
  const unsigned* offsets_tracks = parameters.dev_offsets_tracks;
  const Allen::Views::Physics::CompositeParticle* svs = parameters.dev_svs;
  const unsigned* offsets_svs = parameters.dev_sv_offsets;
  const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i_track = threadID; i_track < total_number_of_tracks; i_track += stride) {

    unsigned event_number = binary_search_rightmost(offsets_tracks, number_of_events + 1, i_track);
    const unsigned sv_idx_offset = event_number * VertexFit::max_sv_track_combinations;

    unsigned* event_sv_idx = parameters.dev_sv_idx + sv_idx_offset;
    unsigned* event_track_idx = parameters.dev_track_idx + sv_idx_offset;
    unsigned* event_combination_number = parameters.dev_combination_offsets + event_number;

    const unsigned ev_i_track = i_track - offsets_tracks[event_number];
    const auto& track = tracks[i_track];
    if (!track.has_states()) continue; // TODO: this should never be false, but it is sometime...
    const auto t_s = track.state();
    const bool track_decision = t_s.pt() > T_PT_min && track.ip_chi2() > T_MIPCHI2_min &&
                                track.ip_chi2() < T_MIPCHI2_max && track.ip() > T_MIP_min && track.ip() < T_MIP_max &&
                                track.chi2() / static_cast<float>(track.ndof()) < T_CHI2NDF_max && t_s.p() > T_P_min;

    if (!track_decision) continue;

    const MiniState t_ministate = t_s;

    const unsigned svs_start_idx = offsets_svs[event_number];
    const unsigned svs_end_idx = offsets_svs[event_number + 1];

    // skip filtered svs (go directly to next valid):
    const unsigned word_begin = svs_start_idx / 32;
    const unsigned word_end = (svs_end_idx + 31) / 32;

    bool overflow = false;

    for (unsigned w = word_begin; w < word_end && !overflow; w++) {

      // get the mask
      uint32_t m = parameters.dev_sv_decision[w];

      // remove bits below start
      if (w == word_begin) {
        const unsigned start_bit = svs_start_idx % 32;
        const uint32_t start_mask = (start_bit == 0) ? 0xffffffffu : (0xffffffffu << start_bit);
        m &= start_mask;
      }

      // remove bits beyond the end
      if (w == (word_end - 1)) {
        const unsigned end_bit = svs_end_idx & 31u;
        const uint32_t end_mask = (end_bit == 0) ? 0xffffffffu : ((1u << end_bit) - 1u);
        m &= end_mask;
      }
      while (m) {
        // get next bit and clear it from the mask
        const int bit = __ffs(m) - 1;
        const unsigned i_sv = (w * 32) + static_cast<unsigned>(bit);
        m &= (m - 1);

        const unsigned ev_i_sv = i_sv - svs_start_idx;
        const auto& sv = svs[i_sv];

        // load the Mini State
        const auto sv_ministate = sv.get_state();
        // Check is the specific SV was created using this track
        if (sv.child_in_tree(&track)) continue;
        // Check that, if required, the track and SV come from the same PV
        if (require_same_pv && &(track.pv()) != &(sv.pv())) continue;
        // check DOCA within limit
        if (Allen::Views::Physics::state_doca(sv_ministate, t_ministate) > SV_T_DOCA_max) continue;
        // check opening angle within limit
        if (sv.min_opening_angle_in_tree(t_ministate) < opening_angle_min) continue;

        if (sv.newdira(t_s) < SV_T_BPV_NEWDIRA_min) continue;

        // oportunistic warp level aggregation
        unsigned cmb_idx = Allen::warp::atomic_increment(event_combination_number);
        // Leave the loop if the maximum number of combinations is exceeded.

        if (cmb_idx >= VertexFit::max_sv_track_combinations) {
          overflow = true;
          break;
        }
        event_sv_idx[cmb_idx] = ev_i_sv;
        event_track_idx[cmb_idx] = ev_i_track;
      }
    }
  }
}

// If there are too many combinations in an event, set the number of combinations to 0. We count the number of events
// reaching the cutoff
__global__ void clamp_counts(
  FilterSVTrack::Parameters parameters,
  const unsigned number_of_events,
  Allen::Monitoring::AveragingCounter<>::DeviceType dev_n_events_reaching_cutoff)
{
  const unsigned threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;

  for (unsigned e = threadID; e < number_of_events; e += stride) {
    auto* count = parameters.dev_combination_offsets + e;
    if (count[0] >= VertexFit::max_sv_track_combinations) {
      count[0] = 0;
      dev_n_events_reaching_cutoff.add(1);
    }
  }
}

void FilterSVTrack::filter_sv_track_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_events = first<host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);
  const unsigned number_of_svs = first<host_number_of_svs_t>(arguments);

  set_size<dev_track_decision_t>(arguments, (number_of_tracks + 31) / 32);
  set_size<dev_sv_decision_t>(arguments, (number_of_svs + 31) / 32);

  set_size<dev_combination_offsets_t>(arguments, number_of_events + 1);
  set_size<host_number_of_combinations_t>(arguments, 1);
  set_size<dev_sv_idx_t>(arguments, VertexFit::max_sv_track_combinations * number_of_events);
  set_size<dev_track_idx_t>(arguments, VertexFit::max_sv_track_combinations * number_of_events);
}

void FilterSVTrack::filter_sv_track_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_track_decision_t>(arguments, 0, context);
  Allen::memset_async<dev_sv_decision_t>(arguments, 0, context);
  Allen::memset_async<dev_combination_offsets_t>(arguments, 0, context);

  const unsigned number_of_events = first<host_number_of_events_t>(arguments);
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);
  const unsigned number_of_svs = first<host_number_of_svs_t>(arguments);

  global_function(filter_svs_k)(dim3(32), m_block_dim, context)(
    arguments,
    number_of_svs,
    m_SV_VZ_min,
    m_SV_VZ_max,
    m_SV_FD_min,
    m_SV_FD_max,
    m_SV_BPVIP_min,
    m_SV_BPVVDZ_min,
    m_SV_BPVVDRHO_min,
    m_SV_BPVDIRA_min,
    m_SV_MCORR_min,
    m_SV_MCORR_max,
    m_T_SV_MIPCHI2_min,
    m_require_os_pair.value());

  global_function(filter_sv_track)(dim3(32), m_block_dim, context)(
    arguments,
    number_of_events,
    number_of_tracks,
    m_SV_T_BPV_NEWDIRA_min,
    m_T_CHI2NDF_max,
    m_T_PT_min,
    m_T_P_min,
    m_T_MIPCHI2_min,
    m_T_MIPCHI2_max,
    m_T_MIP_min,
    m_T_MIP_max,
    m_SV_T_DOCA_max,
    m_opening_angle_min,
    m_require_same_pv.value());

  global_function(clamp_counts)(dim3(32), m_block_dim, context)(
    arguments, number_of_events, m_events_reaching_cutoff.data(context));

  PrefixSum::prefix_sum<dev_combination_offsets_t, host_number_of_combinations_t>(*this, arguments, context);
}
