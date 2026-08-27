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
#pragma once

#include "ParKalmanFilter.cuh"
#include "AlgorithmTypes.cuh"
#include "States.cuh"

namespace downstream_kalman_filter {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_downstream_tracks_t, unsigned) host_number_of_downstream_tracks;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    // Downstream tracks input (from consolidate - for hit access)
    DEVICE_INPUT(dev_downstream_track_view_t, Allen::Views::Physics::DownstreamTrack)
    dev_downstream_track_view;
    DEVICE_INPUT(dev_downstream_track_offsets_t, unsigned) dev_downstream_track_offsets;

    // Polynomial fit outputs (to initialize KF)
    DEVICE_INPUT(dev_downstream_track_states_t, char) // Velo::Consolidated::States
    dev_downstream_track_states;

    // Kalman filter outputs (CompactState + qop + chi2 from ParKF fit)
    DEVICE_OUTPUT(dev_downstream_kf_tracks_t, ParKalmanFilter::FittedTrack) dev_downstream_kf_tracks;
    // State at MidUTz
    DEVICE_OUTPUT(dev_downstream_kf_track_states_t, char) dev_downstream_kf_track_states;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_downstream_kf_track_states_view_t,
      DEPENDENCIES(dev_downstream_kf_track_states_t),
      Allen::Views::Physics::KalmanStates)
    dev_downstream_kf_track_states_view;
  };

  //--------------------------------------------------
  // Main execution of the parametrized Kalman Filter for downstream tracks.
  //--------------------------------------------------
  __global__ void downstream_kalman_filter(
    Parameters,
    const float magnet_polarity,
    const ParKalmanFilter::KalmanParametrizations* dev_kalman_params);

  // Create KalmanStates views for the output (per-event structure)
  __global__ void consolidate_kalman_tracks(Parameters, const unsigned n_events);

  struct downstream_kalman_filter_t : public DeviceAlgorithm, Parameters {
    void update(const Constants& constants) const;
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {128, 1, 1}, "block dimensions"};
  };
} // namespace downstream_kalman_filter
