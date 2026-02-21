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
#include "States.cuh"
#include "RungeKuttaExtrapolator.cuh"
#include "VertexDefinitions.cuh"

namespace extrapolate_ttracks {

  struct DumpedObject {
    constexpr static unsigned NEVENTS = 500;

    unsigned number_of_scifi_states[NEVENTS];
    MiniState scifi_states[SciFi::Constants::Nmax_seeds * NEVENTS];
    float qop[SciFi::Constants::Nmax_seeds * NEVENTS];

    MiniState extr_states[5 * SciFi::Constants::Nmax_seeds * NEVENTS];
  };

  struct Parameters {
    // Basic
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;

    // SciFi input
    HOST_INPUT(host_number_of_scifi_tracks_t, unsigned) host_number_of_scifi_tracks;
    DEVICE_INPUT(dev_offsets_scifi_tracks_t, unsigned) dev_offsets_scifi_tracks;
    DEVICE_INPUT(dev_scifi_track_qop_t, float) dev_scifi_track_qop;
    DEVICE_INPUT(dev_scifi_track_states_t, float) dev_scifi_track_states;

    // Extrapolated states output
    DEVICE_OUTPUT(dev_extrapolated_ttrack_states_t, float) dev_extrapolated_ttrack_states;
  };

  struct extrapolate_ttracks_t : public DeviceAlgorithm, Parameters {
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

  __global__ void extrapolate_ttracks(Parameters, const MagneticField::Magfield);
} // namespace extrapolate_ttracks
