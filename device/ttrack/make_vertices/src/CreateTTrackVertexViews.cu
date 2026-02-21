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
#include "CreateTTrackVertexViews.cuh"

#include "SciFiDefinitions.cuh"
#include "PrefixSum.cuh"

#include "TTrackDefinitions.cuh"

#include <fstream>

INSTANTIATE_ALGORITHM(create_ttrack_vertex_views::create_ttrack_vertex_views_t);

void create_ttrack_vertex_views::create_ttrack_vertex_views_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const auto n_events = first<host_number_of_events_t>(arguments);
  const auto n_vertices = first<host_number_of_tt_vertices_t>(arguments);

  set_size<dev_sv_fit_results_t>(arguments, n_vertices * Allen::Views::Physics::SecondaryVertex::size());
  set_size<dev_sv_fit_results_view_t>(arguments, n_events);

  set_size<dev_two_track_sv_track_pointers_t>(arguments, n_vertices);
  set_size<dev_two_track_composite_view_t>(arguments, n_vertices);
  set_size<dev_two_track_composites_view_t>(arguments, n_events);
  set_size<dev_multi_event_composites_view_t>(arguments, 1);
  set_size<dev_multi_event_composites_ptr_t>(arguments, 1);
}

void create_ttrack_vertex_views::create_ttrack_vertex_views_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_sv_fit_results_t>(arguments, 0, context);

  global_function(create_vertex_views)(first<host_number_of_events_t>(arguments), m_block_dim, context)(arguments);

  // global_function(validate)(1, 1, context)(
  //     arguments);
}

__global__ void create_ttrack_vertex_views::create_vertex_views(Parameters parameters)
{
  using VertexVar = TTracks::Vertex::PackedMiniVertexVar;

  const auto event_number = blockIdx.x;
  const auto number_of_events = gridDim.x;

  const auto output_offset = parameters.dev_offsets_tt_vertices[event_number];
  const auto n_vertices = parameters.dev_offsets_tt_vertices[event_number + 1] - output_offset;

  const auto total_num_vertices = parameters.dev_offsets_tt_vertices[parameters.dev_number_of_events[0]];
  const TTracks::Vertex::PackedMiniVertex input_vertices(
    parameters.dev_tt_vertices.data() + event_number * TTracks::Constants::MAX_VERTICES_PER_EVENT * VertexVar::COUNT,
    TTracks::Constants::MAX_VERTICES_PER_EVENT);

  const auto& basic_particles = parameters.dev_multi_event_basic_particles_view->container(event_number);

  for (unsigned i = threadIdx.x; i < n_vertices; i += blockDim.x) {
    Allen::Views::Physics::SecondaryVertexMutable vertex_state(
      parameters.dev_sv_fit_results, output_offset + i, total_num_vertices);

    vertex_state.x() = input_vertices.template get<VertexVar::x, float>(i);
    vertex_state.y() = input_vertices.template get<VertexVar::y, float>(i);
    vertex_state.z() = input_vertices.template get<VertexVar::z, float>(i);
    vertex_state.px() = input_vertices.template get<VertexVar::px, float>(i);
    vertex_state.py() = input_vertices.template get<VertexVar::py, float>(i);
    vertex_state.pz() = input_vertices.template get<VertexVar::pz, float>(i);
    vertex_state.ttracks_doca() = input_vertices.template get<VertexVar::doca, float>(i);

    parameters.dev_two_track_sv_track_pointers[output_offset + i] = {
      basic_particles.particle_pointer(input_vertices.template get<VertexVar::trk1, unsigned>(i)),
      basic_particles.particle_pointer(input_vertices.template get<VertexVar::trk2, unsigned>(i)),
      nullptr,
      nullptr};

    new (parameters.dev_two_track_composite_view + output_offset + i) Allen::Views::Physics::CompositeParticle(
      parameters.dev_two_track_sv_track_pointers[output_offset + i],
      parameters.dev_sv_fit_results_view + event_number,
      nullptr,
      2,
      i,
      nullptr);
  }

  if (threadIdx.x == 0) {
    // Create event views

    new (parameters.dev_sv_fit_results_view + event_number) Allen::Views::Physics::SecondaryVertices(
      parameters.dev_sv_fit_results, parameters.dev_offsets_tt_vertices, event_number, number_of_events);

    new (parameters.dev_two_track_composites_view + event_number) Allen::Views::Physics::CompositeParticles(
      parameters.dev_two_track_composite_view, parameters.dev_offsets_tt_vertices, event_number);
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // Create global views

    new (parameters.dev_multi_event_composites_view)
      Allen::Views::Physics::MultiEventCompositeParticles(parameters.dev_two_track_composites_view, number_of_events);

    parameters.dev_multi_event_composites_ptr[0] = parameters.dev_multi_event_composites_view;
  }
}
