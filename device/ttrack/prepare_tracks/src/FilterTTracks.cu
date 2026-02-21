/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration          *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// This
#include "FilterTTracks.cuh"

#include "SciFiDefinitions.cuh"
#include "PrefixSum.cuh"

#include "TTrackDefinitions.cuh"
#include "TrackTrajectory.cuh"

#include "KalmanFilterHelper.cuh"

#include <fstream>

INSTANTIATE_ALGORITHM(filter_ttracks::filter_ttracks_t);

void filter_ttracks::filter_ttracks_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  namespace Tracks = TTracks::Tracks;

  const auto n_scifi_tracks = first<host_number_of_scifi_tracks_t>(arguments);
  const auto soa_size = Tracks::PackedTrackStateVar::COUNT * n_scifi_tracks;
  const auto n_events = first<host_number_of_events_t>(arguments);

  set_size<dev_offsets_filtered_seeding_tracks_t>(arguments, n_events + 1);
  set_size<host_number_of_filtered_scifi_tracks_t>(arguments, 1);

  set_size<dev_filtered_seeding_states_t>(arguments, soa_size);
  set_size<dev_filtered_seeding_qop_t>(arguments, n_scifi_tracks);
  set_size<dev_filtered_state_indexes_t>(arguments, n_scifi_tracks);

  set_size<dev_buffer_filtered_state_indexes_t>(arguments, n_scifi_tracks);
  set_size<dev_buffer_filtered_seeding_states_t>(arguments, soa_size);
  set_size<dev_buffer_filtered_seeding_qop_t>(arguments, n_scifi_tracks);
  set_size<dev_buffer_used_tracks_t>(arguments, n_scifi_tracks);
}

void filter_ttracks::filter_ttracks_t::update(const Constants& constants) const
{
  float host_average_dxdy[SciFi::Constants::n_layers];
  const SciFi::SciFiGeometry scifi_geometry {constants.host_scifi_geometry};
  UNROLL(SciFi::Constants::n_layers)
  for (uint8_t i = 0; i < SciFi::Constants::n_layers; i++) {
    host_average_dxdy[i] = scifi_geometry.average_dxdy[i];
  }

  Allen::memcpyToSymbol(dev_average_dxdy, &host_average_dxdy, SciFi::Constants::n_layers * sizeof(float));
}

void filter_ttracks::filter_ttracks_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_offsets_filtered_seeding_tracks_t>(arguments, 0, context);

  Allen::copy_async<dev_buffer_used_tracks_t, dev_matched_is_scifi_track_used_t>(arguments, context);

  global_function(filter_ttracks)(size<dev_event_list_t>(arguments), m_block_dim, context)(
    arguments, constants.host_magnet_polarity.data()[0], DeviceProperties(*this));

  PrefixSum::prefix_sum<dev_offsets_filtered_seeding_tracks_t, host_number_of_filtered_scifi_tracks_t>(
    *this, arguments, context);

  global_function(copy_ttracks)(size<dev_event_list_t>(arguments), m_block_dim, context)(arguments);

  const int grid_dim =
    (first<host_number_of_filtered_scifi_tracks_t>(arguments) + m_block_dim.value().x - 1) / m_block_dim.value().x;
  if (m_with_kalman_filter) {
    global_function(kalman_filter)(grid_dim, m_block_dim, context)(arguments, *constants.magnetic_field);
  }

  resize<dev_muon_states_t>(arguments, first<host_number_of_filtered_scifi_tracks_t>(arguments));
  global_function(fill_muon_states)(grid_dim, m_block_dim, context)(arguments, *constants.magnetic_field);
}

namespace {
  __device__ float inline estimate_qop(const float magSign, const SciFi::Seeding::Track& track)
  {
    constexpr float slope = 2. * 20.13629722595215;
    return -magSign * slope * track.cx / powf((1.f + track.bx * track.bx), 1.5f);
  }

  __device__ inline bool areClones(
    const MiniState& stateA,
    const MiniState& stateB,
    const Allen::Views::SciFi::Consolidated::Track& trackViewA,
    const Allen::Views::SciFi::Consolidated::Track& trackViewB,
    const filter_ttracks::filter_ttracks_t::DeviceProperties& props)
  {
    if (!((fabsf(stateA.x() - stateB.x()) < props.max_clonekiller_dx) &&
          (fabsf(stateA.y() - stateB.y()) < props.max_clonekiller_dy) &&
          (fabsf(stateA.tx() - stateB.tx()) < props.max_clonekiller_dtx) &&
          (fabsf(stateA.ty() - stateB.ty()) < props.max_clonekiller_dty))) {
      return false;
    }
    unsigned hit_ids[SciFi::Constants::n_layers];

    UNROLL(SciFi::Constants::n_layers)
    for (uint8_t i_hit = 0; i_hit < SciFi::Constants::n_layers; ++i_hit) {
      hit_ids[i_hit] = std::numeric_limits<unsigned>::max();
    }

    for (uint8_t i_hit = 0; i_hit < trackViewA.number_of_scifi_hits(); ++i_hit) {
      const auto hit = trackViewA.hit(i_hit);
      hit_ids[hit.planeCode() / 2] = hit.id();
    }

    short coincidences = 0;
    for (uint8_t i_hit = 0; i_hit < trackViewB.number_of_scifi_hits(); ++i_hit) {
      const auto hit = trackViewB.hit(i_hit);
      coincidences += ((hit_ids[hit.planeCode() / 2]) == hit.id());
    }
    return coincidences > props.max_clonekiller_allowed_shared_hits;
  }
} // namespace

__global__ void filter_ttracks::filter_ttracks(
  Parameters parameters,
  float magnet_polarity,
  const filter_ttracks_t::DeviceProperties props)
{
  namespace Tracks = TTracks::Tracks;
  const auto event_number = parameters.dev_event_list[blockIdx.x];
  const auto offset = parameters.dev_offsets_seeding_tracks[event_number];
  const auto n_tracks = parameters.dev_offsets_seeding_tracks[event_number + 1] - offset;
  const auto input_states = parameters.dev_seeding_states + offset;
  const SciFi::Seeding::Track* event_scifi_seeds =
    parameters.dev_seeding_tracks + event_number * SciFi::Constants::Nmax_seeds;

  const auto downstream_tracks_view = parameters.dev_multi_event_downstream_tracks_view->container(event_number);

  auto buffer_qop = parameters.dev_buffer_filtered_seeding_qop + offset;
  auto buffer_indexes = parameters.dev_buffer_filtered_state_indexes + offset;
  Tracks::PackedTrackState buffer_states(
    parameters.dev_buffer_filtered_seeding_states + offset * Tracks::PackedTrackStateVar::COUNT, n_tracks);

  // This buffer already contains the information on whether a track is used in Velo-SciFi matching
  // Therefore, here we need to add only the Downstream VETO & decloning information
  auto isClone = parameters.dev_buffer_used_tracks + offset;

  __shared__ unsigned cached_idxs[SciFi::Constants::Nmax_seeds];
  __shared__ unsigned preselected_cnt;
  preselected_cnt = 0;

  if (n_tracks < 1) return;

  for (unsigned i = threadIdx.x; i < downstream_tracks_view.size(); i += blockDim.x) {
    isClone
      [downstream_tracks_view.track(i).track_segment<Allen::Views::Physics::Track::segment::scifi>().track_index()] =
        true;
  }
  __syncthreads();

  for (unsigned i = threadIdx.x; i < n_tracks; i += blockDim.x) {
    if (!isClone[i]) {
      const auto idx = atomicAdd(&preselected_cnt, 1);
      cached_idxs[idx] = i;
    }
  }

  __syncthreads();
  if (preselected_cnt < 1) return;

  for (unsigned combIdx = threadIdx.x; combIdx < TTracks::Combinations::number_of_combinations(preselected_cnt);
       combIdx += blockDim.x) {
    const auto [i_track, j_track] = TTracks::Combinations::convert_idx(combIdx, preselected_cnt);
    if (areClones(
          input_states[cached_idxs[i_track]],
          input_states[cached_idxs[j_track]],
          parameters.dev_scifi_track_view[offset + cached_idxs[i_track]],
          parameters.dev_scifi_track_view[offset + cached_idxs[j_track]],
          props)) {
      isClone[cached_idxs[j_track]] = true;
    }
  }

  __syncthreads();

  for (unsigned i = threadIdx.x; i < n_tracks; i += blockDim.x) {
    const auto qop = estimate_qop(magnet_polarity, event_scifi_seeds[i]);
    if (!(isClone[i]) && (fabsf(qop) < props.max_abs_qop)) {
      const auto idx = atomicAdd(&parameters.dev_offsets_filtered_seeding_tracks[event_number], 1);
      buffer_qop[idx] = qop;
      buffer_states.set(idx, input_states[i]);
      buffer_indexes[idx] = offset + i;
    }
  }
}

__global__ void filter_ttracks::copy_ttracks(Parameters parameters)
{
  namespace Tracks = TTracks::Tracks;
  const auto event_number = parameters.dev_event_list[blockIdx.x];

  const auto input_offset = parameters.dev_offsets_seeding_tracks[event_number];
  const auto output_offset = parameters.dev_offsets_filtered_seeding_tracks[event_number];

  const auto input_n_tracks = parameters.dev_offsets_seeding_tracks[event_number + 1] - input_offset;
  const auto output_n_tracks = parameters.dev_offsets_filtered_seeding_tracks[event_number + 1] - output_offset;

  const auto input_qop = parameters.dev_buffer_filtered_seeding_qop + input_offset;
  auto output_qop = parameters.dev_filtered_seeding_qop + output_offset;

  Tracks::PackedTrackStateConst buffer_states(
    parameters.dev_buffer_filtered_seeding_states + input_offset * Tracks::PackedTrackStateVar::COUNT, input_n_tracks);
  const auto total_output_tracks = parameters.dev_offsets_filtered_seeding_tracks[parameters.dev_number_of_events[0]];
  Tracks::PackedTrackState output_states(parameters.dev_filtered_seeding_states, total_output_tracks);

  for (unsigned i = threadIdx.x; i < output_n_tracks; i += blockDim.x) {
    using Var = Tracks::PackedTrackStateVar;
    output_states.set<Var::x>(output_offset + i, buffer_states.get<Var::x>(i));
    output_states.set<Var::y>(output_offset + i, buffer_states.get<Var::y>(i));
    output_states.set<Var::z>(output_offset + i, buffer_states.get<Var::z>(i));
    output_states.set<Var::tx>(output_offset + i, buffer_states.get<Var::tx>(i));
    output_states.set<Var::ty>(output_offset + i, buffer_states.get<Var::ty>(i));
    output_qop[i] = input_qop[i];
    parameters.dev_filtered_state_indexes[output_offset + i] =
      parameters.dev_buffer_filtered_state_indexes[input_offset + i];
  }
}

__global__ void filter_ttracks::kalman_filter(Parameters parameters, const MagneticField::Magfield field)
{
  namespace Tracks = TTracks::Tracks;

  const auto total_tracks = parameters.dev_offsets_filtered_seeding_tracks[parameters.dev_number_of_events[0]];
  Tracks::PackedTrackState states(parameters.dev_filtered_seeding_states, total_tracks);

  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < total_tracks; i += blockDim.x * gridDim.x) {

    const auto scifi_track_view = parameters.dev_scifi_track_view[parameters.dev_filtered_state_indexes[i]];

    // Fill hits information
    float x_pos[SciFi::Constants::n_layers];
    float z_pos[SciFi::Constants::n_layers];
    bool has_hit[SciFi::Constants::n_layers];

    UNROLL(SciFi::Constants::n_layers)
    for (unsigned i_hit = 0; i_hit < SciFi::Constants::n_layers; ++i_hit) {
      has_hit[i_hit] = false;
    }

    UNROLL(9)
    for (unsigned i_hit = 0; i_hit < scifi_track_view.number_of_scifi_hits(); ++i_hit) {
      const auto hit = scifi_track_view.hit(i_hit);
      const auto layer = hit.planeCode() / 2;
      x_pos[layer] = hit.x0();
      z_pos[layer] = hit.z0();
      has_hit[layer] = true;
    }

    // Initialize state and covariance
    Extrapolators::State state;
    state.x = states.get<Tracks::PackedTrackStateVar::x>(i);
    state.y = states.get<Tracks::PackedTrackStateVar::y>(i);
    state.z = states.get<Tracks::PackedTrackStateVar::z>(i);
    state.tx = states.get<Tracks::PackedTrackStateVar::tx>(i);
    state.ty = states.get<Tracks::PackedTrackStateVar::ty>(i);
    state.qop = parameters.dev_filtered_seeding_qop[i] * Extrapolators::c_light * Extrapolators::eplus;

    TTracks::KalmanFilter::SymmMatrix<5> cov;
    cov(0, 0) = 100.f * 25.f;                // x
    cov(1, 1) = 100.f * 625.f;               // y
    cov(2, 2) = 100.f * (0.0025f * 0.0025f); // tx
    cov(3, 3) = 100.f * (0.0125f * 0.0125f); // ty
    cov(4, 4) =
      100.f * 1e-8f * Extrapolators::c_light * Extrapolators::eplus * Extrapolators::c_light * Extrapolators::eplus;
    cov(0, 1) = 0.f;
    cov(0, 2) = 0.f;
    cov(0, 3) = 0.f;
    cov(0, 4) = 0.f;
    cov(1, 2) = 0.f;
    cov(1, 3) = 0.f;
    cov(1, 4) = 0.f;
    cov(2, 3) = 0.f;
    cov(2, 4) = 0.f;
    cov(3, 4) = 0.f;

    // Run the Kalman filter over the SciFi layers (from last to first)
    for (short int layer = SciFi::Constants::n_layers - 1; layer >= 0; --layer) {
      if (has_hit[layer]) {
        TTracks::KalmanFilter::makeStep(
          state, cov, x_pos[layer], 0.25f * 0.25f, z_pos[layer], dev_average_dxdy[layer], field);
      }
    }

    using Var = Tracks::PackedTrackStateVar;

    states.set<Var::x>(i, state.x);
    states.set<Var::y>(i, state.y);
    states.set<Var::z>(i, state.z);
    states.set<Var::tx>(i, state.tx);
    states.set<Var::ty>(i, state.ty);
    parameters.dev_filtered_seeding_qop[i] = state.qop / (Extrapolators::c_light * Extrapolators::eplus);
  }
}

__global__ void filter_ttracks::fill_muon_states(Parameters parameters, const MagneticField::Magfield field)
{
  namespace Tracks = TTracks::Tracks;

  const auto total_tracks = parameters.dev_offsets_filtered_seeding_tracks[parameters.dev_number_of_events[0]];
  Tracks::PackedTrackStateConst states(parameters.dev_filtered_seeding_states, total_tracks);

  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < total_tracks; i += blockDim.x * gridDim.x) {
    Extrapolators::State state;
    state.x = states.get<Tracks::PackedTrackStateVar::x>(i);
    state.y = states.get<Tracks::PackedTrackStateVar::y>(i);
    state.z = states.get<Tracks::PackedTrackStateVar::z>(i);
    state.tx = states.get<Tracks::PackedTrackStateVar::tx>(i);
    state.ty = states.get<Tracks::PackedTrackStateVar::ty>(i);
    state.qop = parameters.dev_filtered_seeding_qop[i] * Extrapolators::c_light * Extrapolators::eplus;

    using RK = Extrapolators::RungeKuttaNystromExtrapolator;
    RK::propagate(state, 15000.f, field, 1000.f);
    parameters.dev_muon_states[i] = MiniState {state.x, state.y, state.z, state.tx, state.ty};
  }
}