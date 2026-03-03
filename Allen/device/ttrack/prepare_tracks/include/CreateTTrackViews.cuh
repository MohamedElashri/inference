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

namespace create_ttrack_views {

  struct Parameters {
    // Basic
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;

    // Filtered tracks input
    HOST_INPUT(host_number_of_filtered_scifi_tracks_t, unsigned) host_number_of_filtered_scifi_tracks;
    DEVICE_INPUT(dev_offsets_filtered_seeding_tracks_t, unsigned) dev_offsets_filtered_seeding_tracks;
    DEVICE_INPUT(dev_filtered_seeding_qop_t, float) dev_filtered_seeding_qop;
    DEVICE_INPUT(dev_filtered_seeding_states_t, float) dev_filtered_seeding_states;
    DEVICE_INPUT(dev_filtered_state_indexes_t, unsigned) dev_filtered_state_indexes;

    // SciFi input
    DEVICE_INPUT(dev_scifi_track_view_t, Allen::Views::SciFi::Consolidated::Track)
    dev_scifi_track_view;

    // Output
    DEVICE_OUTPUT(dev_track_kalman_states_t, char) dev_track_kalman_states;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_track_kalman_states_view_t,
      DEPENDENCIES(dev_track_kalman_states_t),
      Allen::Views::Physics::KalmanStates)
    dev_track_kalman_states_view;

    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_ttrack_view_t,
      DEPENDENCIES(dev_scifi_track_view_t, dev_filtered_seeding_qop_t),
      Allen::Views::Physics::TTrack)
    dev_ttrack_view;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(dev_ttracks_view_t, DEPENDENCIES(dev_ttrack_view_t), Allen::Views::Physics::TTracks)
    dev_ttracks_view;

    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_multi_event_ttracks_view_t,
      DEPENDENCIES(dev_ttracks_view_t),
      Allen::Views::Physics::MultiEventTTracks)
    dev_multi_event_ttracks_view;

    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_multi_event_ttracks_view_ptr_t,
      DEPENDENCIES(dev_multi_event_ttracks_view_t),
      Allen::IMultiEventContainer*)
    dev_multi_event_ttracks_view_ptr;
  };

  struct create_ttrack_views_t : public DeviceAlgorithm, Parameters {
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

  __global__ void create_ttrack_views(Parameters);
  __global__ void create_ttrack_event_views(Parameters);

  __global__ void validate(Parameters);
} // namespace create_ttrack_views
