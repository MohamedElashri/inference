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
#include "States.cuh"
#include "RichPhoton.cuh"
#include "RichTrackSegment.cuh"
#include "RichParticleHypos.cuh"

namespace rich_photon_predicted_pixel_signal {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;
    HOST_INPUT(host_number_of_photons_t, unsigned) host_number_of_photons;
    DEVICE_INPUT(dev_rich_hypos_t, Allen::Rich::ParticleHypos) dev_rich_hypos;
    DEVICE_INPUT(dev_rich_pd_offsets_t, unsigned) dev_rich_pd_offsets;
    DEVICE_INPUT(dev_offsets_rich_photons_t, unsigned) dev_offsets_rich_photons;
    DEVICE_INPUT(dev_rich_photons_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons;
    DEVICE_OUTPUT(dev_photon_pix_signals_t, Allen::Rich::HypoData<float>) dev_photon_pix_signals;
  };
  struct rich_photon_predicted_pixel_signal_t : public DeviceAlgorithm, Parameters {
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

    /// The minimum expected track Cherenkov angle to be considered 'Above Threshold'
    Allen::Property<Allen::Rich::DetectorArray<float>> m_minExpCKT {
      this,
      "MinExpTrackCKTheta",
      {0.0f, 0.0f}, // was { 1e-6f, 1e-6f }
      "The minimum expected track Cherenkov angle for each RICH (R1Gas/R2Gas)"};

    /// The minimum cut value for photon probability
    Allen::Property<Allen::Rich::DetectorArray<float>> m_minPhotonProb {
      this,
      "MinPhotonProbability",
      {1e-15f, 1e-15f},
      "The minimum allowed photon probability values for each RICH (R1Gas/R2Gas)"};

    mutable float m_factor {};
  };
} // namespace rich_photon_predicted_pixel_signal
