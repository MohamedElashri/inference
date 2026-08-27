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
#include "RichDefinitions.cuh"
#include "RichParticleHypos.cuh"

namespace rich_simple_pid {
  struct Parameters {
    HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;
    DEVICE_INPUT(dev_track_total_signals_r1_t, Allen::Rich::HypoData<float>) dev_track_total_signals_r1;
    DEVICE_INPUT(dev_track_total_signals_r2_t, Allen::Rich::HypoData<float>) dev_track_total_signals_r2;
    DEVICE_OUTPUT(dev_pid_t, Allen::Rich::ParticleIDType) dev_pid;
  };
  struct rich_simple_pid_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
  };
} // namespace rich_simple_pid
