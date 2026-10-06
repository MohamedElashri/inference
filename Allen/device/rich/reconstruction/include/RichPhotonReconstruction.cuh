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

namespace rich_photon_reconstruction {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;
    DEVICE_INPUT(dev_rich_pixels_lpos_t, short2) dev_rich_pixels_lpos;
    DEVICE_INPUT(dev_rich_pixels_gpos_t, float3) dev_rich_pixels_gpos;
    DEVICE_INPUT(dev_rich_pd_offsets_t, unsigned) dev_rich_pd_offsets;
    DEVICE_INPUT(dev_offsets_rich_states_t, unsigned) dev_offsets_rich_states;
    DEVICE_INPUT(dev_segs_best_point_t, float3) dev_segs_best_point;
    DEVICE_INPUT(dev_segs_best_momentum_t, float3) dev_segs_best_momentum;
    DEVICE_INPUT(dev_segs_point_at_panel_t, float2) dev_segs_point_at_panel;
    DEVICE_INPUT(dev_rich_hypos_t, Allen::Rich::ParticleHypos) dev_rich_hypos;
    HOST_OUTPUT(host_total_number_of_photons_t, unsigned) host_total_number_of_photons;
    DEVICE_OUTPUT(dev_offsets_rich_photons_prefilter_t, unsigned) dev_offsets_rich_photons_prefilter;
    DEVICE_OUTPUT(dev_rich_photons_prefilter_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons_prefilter;
    DEVICE_OUTPUT(dev_offsets_rich_photons_t, unsigned) dev_offsets_rich_photons;
    DEVICE_OUTPUT(dev_rich_photons_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons;
  };

  struct rich_photon_reconstruction_t : public DeviceAlgorithm, Parameters {

    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

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

    // Pre-filtering:

    Allen::Property<Allen::Rich::DetectorArray<float>> m_radScale {
      this,
      "RadScale",
      {0.0117f, -0.021f},
      "Rich1Gas, Rich2Gas"};
    Allen::Property<Allen::Rich::DetectorArray<float>> m_minROIPreSel {this, "PreSelMinTrackROI", {0.f, 0.f}, ""};
    Allen::Property<Allen::Rich::DetectorArray<float>> m_maxROIPreSel {this, "PreSelMaxTrackROI", {110.f, 165.f}, ""};

    // The CK theta value
    Allen::Property<Allen::Rich::DetectorArray<float>> m_ckThetaScale {
      this,
      "ScaleFactorCKTheta",
      {0.045f, 0.024f},
      ""};

    // The seperation the scale factors apply to
    Allen::Property<Allen::Rich::DetectorArray<float>> m_sepGScale {this, "ScaleFactorSepG", {83.f, 107.f}, ""};

    /// N sigma for acceptance bands for preselection
    Allen::Property<Allen::Rich::DetectorArray<float>> m_nSigmaPreSel {this, "PreSelNSigma", {5.f, 24.f}, ""};

#ifdef USE_DD4HEP
    Allen::Property<Allen::Rich::DetectorArray<float>> m_ckBiasCorrs {this, "CKThetaBiasCorr", {2.15e-4, 1.8e-5}, ""};
#else
    Allen::Property<Allen::Rich::DetectorArray<float>> m_ckBiasCorrs {this, "CKThetaBiasCorr", {0.6e-4, 0.5e-5}, ""};
#endif

    /// Absolute minimum allowed Cherenkov Angle
    Allen::Property<Allen::Rich::DetectorArray<float>> m_minCKtheta {
      this,
      "MinAllowedCherenkovTheta",
      {0.005, 0.005},
      "The minimum allowed CK theta values for each RICH (R1Gas/R2Gas)"};

    /// Absolute maximum allowed Cherenkov Angle
    Allen::Property<Allen::Rich::DetectorArray<float>> m_maxCKtheta {
      this,
      "MaxAllowedCherenkovTheta",
      {0.055, 0.032},
      "The maximum allowed CK theta values for each RICH (R1Gas/R2Gas)"};

    /// N sigma for acceptance bands
    Allen::Property<Allen::Rich::DetectorArray<float>> m_nSigma {
      this,
      "NSigma",
      {3.6f, 4.5f},
      "The CK theta # sigma selection range for each RICH (R1Gas/R2Gas)"};
  };
} // namespace rich_photon_reconstruction
