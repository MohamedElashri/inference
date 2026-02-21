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
#include "CreateTTrackViews.cuh"

#include "SciFiDefinitions.cuh"
#include "PrefixSum.cuh"

#include "TTrackDefinitions.cuh"

#include <fstream>

INSTANTIATE_ALGORITHM(create_ttrack_views::create_ttrack_views_t);

void create_ttrack_views::create_ttrack_views_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const auto n_events = first<host_number_of_events_t>(arguments);
  const auto n_tracks = first<host_number_of_filtered_scifi_tracks_t>(arguments);

  set_size<dev_track_kalman_states_t>(arguments, n_tracks * Allen::Views::Physics::KalmanState::size());
  set_size<dev_track_kalman_states_view_t>(arguments, n_events);

  set_size<dev_ttrack_view_t>(arguments, n_tracks);
  set_size<dev_ttracks_view_t>(arguments, n_events);

  set_size<dev_multi_event_ttracks_view_t>(arguments, 1);
  set_size<dev_multi_event_ttracks_view_ptr_t>(arguments, 1);
}

void create_ttrack_views::create_ttrack_views_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  const int block_dim = 256;
  const int grid_dim = (first<host_number_of_filtered_scifi_tracks_t>(arguments) + block_dim - 1) / block_dim;

  Allen::memset_async<dev_track_kalman_states_t>(arguments, 0, context);

  global_function(create_ttrack_views)(grid_dim, block_dim, context)(arguments);

  global_function(create_ttrack_event_views)(first<host_number_of_events_t>(arguments), block_dim, context)(arguments);
}

__global__ void create_ttrack_views::create_ttrack_views(Parameters parameters)
{
  const auto n_states = parameters.dev_offsets_filtered_seeding_tracks[parameters.dev_number_of_events[0]];
  namespace Tracks = TTracks::Tracks;
  Tracks::PackedTrackStateConst input_states(parameters.dev_filtered_seeding_states, n_states);

  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n_states; i += blockDim.x * gridDim.x) {
    Allen::Views::Physics::KalmanStateMutable kalman_state_view(parameters.dev_track_kalman_states, i, n_states);

    using Var = Tracks::PackedTrackStateVar;
    kalman_state_view.x() = input_states.get<Var::x>(i);
    kalman_state_view.y() = input_states.get<Var::y>(i);
    kalman_state_view.z() = input_states.get<Var::z>(i);
    kalman_state_view.tx() = input_states.get<Var::tx>(i);
    kalman_state_view.ty() = input_states.get<Var::ty>(i);
    kalman_state_view.qop() = parameters.dev_filtered_seeding_qop[i];

    new (parameters.dev_ttrack_view + i) Allen::Views::Physics::TTrack(
      // *(parameters.dev_scifi_multi_event_tracks_view->container(0).track(i)),
      parameters.dev_scifi_track_view + parameters.dev_filtered_state_indexes[i],
      parameters.dev_filtered_seeding_qop + i);
  }
}

__global__ void create_ttrack_views::create_ttrack_event_views(Parameters parameters)
{
  const auto event_number = blockIdx.x;

  if (threadIdx.x == 0) {
    new (parameters.dev_ttracks_view + event_number) Allen::Views::Physics::TTracks {
      parameters.dev_ttrack_view, parameters.dev_offsets_filtered_seeding_tracks, event_number};

    new (parameters.dev_track_kalman_states_view + event_number)
      Allen::Views::Physics::KalmanStates {parameters.dev_track_kalman_states,
                                           parameters.dev_offsets_filtered_seeding_tracks,
                                           event_number,
                                           parameters.dev_number_of_events[0]};
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (parameters.dev_multi_event_ttracks_view)
      Allen::Views::Physics::MultiEventTTracks {parameters.dev_ttracks_view, parameters.dev_number_of_events[0]};

    parameters.dev_multi_event_ttracks_view_ptr[0] = parameters.dev_multi_event_ttracks_view;
  }
}
