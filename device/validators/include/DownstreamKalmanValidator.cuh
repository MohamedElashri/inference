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

#include "States.cuh"
#include "AlgorithmTypes.cuh"
#include "LookingForwardConstants.cuh"
#include "ParKalmanFittedTrack.cuh"
#include "ParKalmanMath.cuh"
#include "PV_Definitions.cuh"
#include "patPV_Definitions.cuh"
#include "ParticleTypes.cuh"
#include "CheckerTracks.cuh"
#include "CheckerInvoker.h"
#include "KalmanChecker.h"

namespace downstream_kalman_validator {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_downstream_tracks_t, unsigned) host_number_of_downstream_tracks;
    HOST_INPUT(host_mc_events_t, const MCEvents*) host_mc_events;
    DEVICE_INPUT(dev_multi_event_downstream_tracks_view_t, Allen::Views::Physics::MultiEventDownstreamTracks)
    dev_multi_event_downstream_tracks_view;
    DEVICE_INPUT(dev_downstream_track_offsets_t, unsigned) dev_downstream_track_offsets;
    DEVICE_INPUT(dev_downstream_kf_tracks_t, ParKalmanFilter::FittedTrack) dev_downstream_kf_tracks;
    DEVICE_INPUT(dev_downstream_kf_track_states_view_t, Allen::Views::Physics::KalmanStates)
    dev_downstream_kf_track_states_view;
    DEVICE_INPUT(dev_multi_final_vertices_t, PV::Vertex) dev_multi_final_vertices;
    DEVICE_INPUT(dev_number_of_multi_final_vertices_t, unsigned) dev_number_of_multi_final_vertices;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_OUTPUT(dev_downstream_kalman_checker_tracks_t, Checker::Track) dev_downstream_kalman_checker_tracks;
  };

  __global__ void downstream_kalman_validator(
    Parameters,
    const LookingForward::Constants* dev_looking_forward_constants,
    const float* dev_magnet_polarity);

  struct downstream_kalman_validator_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
    Allen::Property<std::string> m_root_output_filename {this,
                                                         "root_output_filename",
                                                         "PrCheckerPlots.root",
                                                         "root output filename"};
  };
} // namespace downstream_kalman_validator
