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
#include "AllenMonitoring.h"
#include "States.cuh"
#include "RichParticleHypos.cuh"

namespace rich_make_hypos {
  struct Parameters {
    HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;
    DEVICE_INPUT(dev_rich_entry_states_t, SimpleKalmanState) dev_rich_entry_states;
    DEVICE_INPUT(dev_rich_exit_states_t, SimpleKalmanState) dev_rich_exit_states;
    DEVICE_INPUT(dev_offsets_rich_states_t, unsigned) dev_offsets_rich_states;
    DEVICE_OUTPUT(dev_segs_best_point_t, float3) dev_segs_best_point;
    DEVICE_OUTPUT(dev_segs_best_momentum_t, float3) dev_segs_best_momentum;
    DEVICE_OUTPUT(dev_segs_point_at_panel_t, float2) dev_segs_point_at_panel;
    DEVICE_OUTPUT(dev_rich_hypos_t, Allen::Rich::ParticleHypos) dev_rich_hypos;
  };

  struct rich_make_hypos_t : public DeviceAlgorithm, Parameters {
    void update(const Constants&) const;

    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    template<Allen::Rich::Detector::DetectorType richIdx>
    void updateRich(const Allen::Rich::RichDetector<richIdx>*) const;

    template<Allen::Rich::Detector::DetectorType richIdx>
    void launchForRich(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&,
      const Allen::Rich::RichDetector<richIdx>*) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};
    Allen::Property<unsigned> m_current_rich {this, "current_rich", 1, "current rich"};

    Allen::Property<Allen::Rich::DetectorArray<float>> m_mirrShift {
      this,
      "MirrorShiftCorr",
      {350, 1500},
      "z shift values to move state to be on the 'inside' of the primary mirrors."};
    Allen::Property<Allen::Rich::DetectorArray<float>> m_radScale {
      this,
      "RadScale",
      {0.0117f, -0.021f},
      "Rich1Gas, Rich2Gas"};
    Allen::Property<Allen::Rich::DetectorArray<float>> m_minRadLength {
      this,
      "MinRadiatorPathLength",
      {500.f, 1500.f},
      "Minimum path length in each RICH radiator volume."};

    /// CK angle computation strategy
    Allen::Property<bool> m_useYieldWeightedAngles {
      this,
      "UseYieldWeightedAngles",
      true,
      "Use yield-weighted Cherenkov angles with tabulated n(E) instead of constant refractive index"};

    /// Cached values:
    mutable float m_refIndexTheta {};
    mutable float m_refIndexYield {};
    mutable float m_deltaE {};
    mutable std::array<float, Allen::Rich::NPhotonSpectraBins> m_paraWDiff {};
    Allen::Monitoring::Counter<> m_failed_ray_traces {this, "n_failed_ray_traces"};
  };
} // namespace rich_make_hypos
