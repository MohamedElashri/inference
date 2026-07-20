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
#include "CreateTTrackParticleViews.cuh"

#include "SciFiDefinitions.cuh"
#include "PrefixSum.cuh"

#include "TTrackDefinitions.cuh"

#include <fstream>

INSTANTIATE_ALGORITHM(create_ttrack_particle_views::create_ttrack_particle_views_t);

void create_ttrack_particle_views::create_ttrack_particle_views_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const auto n_events = first<host_number_of_events_t>(arguments);
  const auto n_tracks = first<host_number_of_filtered_scifi_tracks_t>(arguments);

  set_size<dev_track_particle_view_t>(arguments, n_tracks);
  set_size<dev_track_particles_view_t>(arguments, n_events);
  set_size<dev_multi_event_basic_particles_view_t>(arguments, 1);
  set_size<dev_multi_event_basic_particles_view_ptr_t>(arguments, 1);
}

void create_ttrack_particle_views::create_ttrack_particle_views_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  global_function(create_ttrack_event_views)(first<host_number_of_events_t>(arguments), m_block_dim, context)(
    arguments);
}

__global__ void create_ttrack_particle_views::create_ttrack_event_views(Parameters parameters)
{
  const auto event_number = blockIdx.x;
  const auto offset = parameters.dev_offsets_filtered_seeding_tracks[event_number];
  const auto n_states = parameters.dev_offsets_filtered_seeding_tracks[event_number + 1] - offset;

  for (unsigned i = threadIdx.x; i < n_states; i += blockDim.x) {

    new (parameters.dev_track_particle_view + offset + i) Allen::Views::Physics::BasicParticle {
      parameters.dev_ttrack_view + offset + i,
      parameters.dev_track_kalman_states_view + event_number,
      nullptr, // no PV
      i,       // index
      parameters.dev_lepton_id[offset + i]};
  }

  if (threadIdx.x == 0) {
    new (parameters.dev_track_particles_view + event_number) Allen::Views::Physics::BasicParticles {
      parameters.dev_track_particle_view, parameters.dev_offsets_filtered_seeding_tracks, event_number};
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (parameters.dev_multi_event_basic_particles_view) Allen::Views::Physics::MultiEventBasicParticles {
      parameters.dev_track_particles_view, parameters.dev_number_of_events[0]};

    parameters.dev_multi_event_basic_particles_view_ptr[0] = parameters.dev_multi_event_basic_particles_view;
  }
}
