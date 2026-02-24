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
#include "DownstreamKalmanValidator.cuh"
#include "CopyTrackParameters.cuh"

INSTANTIATE_ALGORITHM(downstream_kalman_validator::downstream_kalman_validator_t)

__global__ void create_downstream_kalman_tracks_for_checker(downstream_kalman_validator::Parameters parameters)
{
  const unsigned event_number = blockIdx.x;
  const auto event_downstream_tracks = parameters.dev_multi_event_downstream_tracks_view->container(event_number);
  const auto number_of_tracks_event = event_downstream_tracks.size();
  const auto offset_kalman_tracks = event_downstream_tracks.offset();
  const auto downstream_kf_states = parameters.dev_downstream_kf_track_states_view[event_number];
  Checker::Track* downstream_kalman_checker_tracks_event =
    parameters.dev_downstream_kalman_checker_tracks + offset_kalman_tracks;
  const ParKalmanFilter::FittedTrack* kf_tracks_event = parameters.dev_downstream_kf_tracks + offset_kalman_tracks;
  const PV::Vertex* rec_vertices_event =
    parameters.dev_multi_final_vertices + event_number * PatPV::max_number_vertices;

  const auto number_of_vertices_event = parameters.dev_number_of_multi_final_vertices[event_number];

  // Step 1: Prepare hit information using existing helper
  prepare_downstream_tracks(event_downstream_tracks, downstream_kf_states, downstream_kalman_checker_tracks_event);

  __syncthreads();

  // Step 2: Prepare Kalman information using new helper (will be added to CopyTrackParameters.cuh)
  prepare_downstream_kalman_tracks(
    number_of_tracks_event,
    number_of_vertices_event,
    rec_vertices_event,
    downstream_kf_states,
    kf_tracks_event,
    downstream_kalman_checker_tracks_event);

  __syncthreads();
}

void downstream_kalman_validator::downstream_kalman_validator_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_downstream_kalman_checker_tracks_t>(arguments, first<host_number_of_downstream_tracks_t>(arguments));
}

void downstream_kalman_validator::downstream_kalman_validator_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  global_function(create_downstream_kalman_tracks_for_checker)(first<host_number_of_events_t>(arguments), 256, context)(
    arguments);

  const auto event_list = make_host_buffer<dev_event_list_t>(arguments, context);
  const auto downstream_kalman_tracks_for_checker =
    make_host_buffer<dev_downstream_kalman_checker_tracks_t>(arguments, context);
  const auto event_tracks_offsets = make_host_buffer<dev_downstream_track_offsets_t>(arguments, context);
  std::vector<std::vector<Checker::Track>> tracks;
  tracks.resize(event_list.size());
  for (size_t i = 0; i < event_list.size(); ++i) {
    const auto evnum = event_list[i];
    const auto event_offset = event_tracks_offsets[evnum];
    const auto n_tracks = event_tracks_offsets[evnum + 1] - event_offset;
    std::vector<Checker::Track> event_tracks = {downstream_kalman_tracks_for_checker.begin() + event_offset,
                                                downstream_kalman_tracks_for_checker.begin() + event_offset + n_tracks};
    tracks[i] = event_tracks;
  }

  // Use KalmanChecker with a different name to create a separate TDirectory
  auto& checker = runtime_options.checker_invoker->checker<KalmanChecker>(name(), m_root_output_filename, false);
  checker.accumulate(*first<host_mc_events_t>(arguments), tracks, event_list);
}
