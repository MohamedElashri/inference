/*****************************************************************************\
* (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include <RichSmartID.cuh>
#include "RichDefinitions.cuh"

namespace rich_init_pid {
  struct Parameters {
    HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;
    DEVICE_OUTPUT(dev_pid_t, Allen::Rich::ParticleIDType) dev_pid;
  };
  struct rich_init_pid_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    /// The default PID to initialize all tracks with
    // Retain the integer property for Gaudi compatibility and convert it at the algorithm boundary.
    Allen::Property<int> m_default_pid {this, "defaultPID", static_cast<int>(Allen::Rich::ParticleIDType::Pion), ""};
  };
} // namespace rich_init_pid
