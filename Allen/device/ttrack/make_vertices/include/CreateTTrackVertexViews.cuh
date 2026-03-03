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
#pragma once

// Basic
#include "AlgorithmTypes.cuh"
// Common
#include "Common.h"
#include "States.cuh"
#include "SciFiConsolidated.cuh"
#include "ParticleTypes.cuh"

namespace create_ttrack_vertex_views {

  struct Parameters {
    // Basic
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;

    // Vertices
    HOST_INPUT(host_number_of_tt_vertices_t, unsigned) host_number_of_tt_vertices;
    DEVICE_INPUT(dev_offsets_tt_vertices_t, unsigned) dev_offsets_tt_vertices;
    DEVICE_INPUT(dev_tt_vertices_t, float) dev_tt_vertices;

    // Basic particles
    DEVICE_INPUT(dev_multi_event_basic_particles_view_t, Allen::Views::Physics::MultiEventBasicParticles)
    dev_multi_event_basic_particles_view;

    // Vertex info
    DEVICE_OUTPUT(dev_sv_fit_results_t, char) dev_sv_fit_results;

    // Views
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_sv_fit_results_view_t,
      DEPENDENCIES(dev_sv_fit_results_t),
      Allen::Views::Physics::SecondaryVertices)
    dev_sv_fit_results_view;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_two_track_sv_track_pointers_t,
      DEPENDENCIES(dev_multi_event_basic_particles_view_t),
      std::array<const Allen::Views::Physics::IParticle*, 4>)
    dev_two_track_sv_track_pointers;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_two_track_composite_view_t,
      DEPENDENCIES(
        dev_two_track_sv_track_pointers_t,
        dev_multi_event_basic_particles_view_t,
        dev_sv_fit_results_view_t),
      Allen::Views::Physics::CompositeParticle)
    dev_two_track_composite_view;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_two_track_composites_view_t,
      DEPENDENCIES(dev_two_track_composite_view_t),
      Allen::Views::Physics::CompositeParticles)
    dev_two_track_composites_view;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_multi_event_composites_view_t,
      DEPENDENCIES(dev_two_track_composites_view_t),
      Allen::Views::Physics::MultiEventCompositeParticles)
    dev_multi_event_composites_view;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_multi_event_composites_ptr_t,
      DEPENDENCIES(dev_multi_event_composites_view_t),
      Allen::IMultiEventContainer*)
    dev_multi_event_composites_ptr;
  };

  struct create_ttrack_vertex_views_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants& constants,
      const Allen::Context& context) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {64, 1, 1}, "block dimensions"};

    Allen::Property<float> m_max_abs_qop {this, "max_abs_qop", 1.f / 600.f, "maximum absolute value of qop (in 1/MeV)"};
  };

  __global__ void create_vertex_views(Parameters);

  __global__ void validate(Parameters);
} // namespace create_ttrack_vertex_views
