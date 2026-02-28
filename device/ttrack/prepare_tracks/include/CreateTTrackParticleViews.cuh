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

namespace create_ttrack_particle_views {

  struct Parameters {
    // Basic
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;

    // Inputs
    HOST_INPUT(host_number_of_filtered_scifi_tracks_t, unsigned) host_number_of_filtered_scifi_tracks;
    DEVICE_INPUT(dev_offsets_filtered_seeding_tracks_t, unsigned) dev_offsets_filtered_seeding_tracks;

    DEVICE_INPUT(dev_ttrack_view_t, Allen::Views::Physics::TTrack) dev_ttrack_view;
    DEVICE_INPUT(dev_ttracks_view_t, Allen::Views::Physics::TTracks) dev_ttracks_view;
    DEVICE_INPUT(dev_track_kalman_states_view_t, Allen::Views::Physics::KalmanStates) dev_track_kalman_states_view;

    DEVICE_INPUT(dev_lepton_id_t, uint8_t) dev_lepton_id;

    // Output
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_track_particle_view_t,
      DEPENDENCIES(dev_ttracks_view_t, dev_track_kalman_states_view_t),
      Allen::Views::Physics::BasicParticle)
    dev_track_particle_view;

    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_track_particles_view_t,
      DEPENDENCIES(dev_track_particle_view_t),
      Allen::Views::Physics::BasicParticles)
    dev_track_particles_view;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_multi_event_basic_particles_view_t,
      DEPENDENCIES(dev_track_particles_view_t),
      Allen::Views::Physics::MultiEventBasicParticles)
    dev_multi_event_basic_particles_view;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_multi_event_basic_particles_view_ptr_t,
      DEPENDENCIES(dev_multi_event_basic_particles_view_t),
      Allen::IMultiEventContainer*)
    dev_multi_event_basic_particles_view_ptr;
  };

  struct create_ttrack_particle_views_t : public DeviceAlgorithm, Parameters {
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

  __global__ void create_ttrack_particle_views(Parameters);
  __global__ void create_ttrack_event_views(Parameters);

  __global__ void validate(Parameters);
} // namespace create_ttrack_particle_views
