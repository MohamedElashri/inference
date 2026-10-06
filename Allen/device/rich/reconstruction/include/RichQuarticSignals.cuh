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

namespace rich_quartic_signals {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;

    // Track data
    HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;
    DEVICE_INPUT(dev_offsets_rich_states_t, unsigned) dev_offsets_rich_states;
    DEVICE_INPUT(dev_segs_best_point_t, float3) dev_segs_best_point;
    DEVICE_INPUT(dev_segs_point_at_panel_t, float2) dev_segs_point_at_panel;
    DEVICE_INPUT(dev_segs_best_momentum_t, float3) dev_segs_best_momentum;
    DEVICE_INPUT(dev_rich_hypos_t, Allen::Rich::ParticleHypos) dev_rich_hypos;

    // Pixel data
    DEVICE_INPUT(dev_rich_pd_offsets_t, unsigned) dev_rich_pd_offsets;
    DEVICE_INPUT(dev_pd_pixels_t, uint64_t) dev_pd_pixels;

    // Temporaries
    DEVICE_OUTPUT(dev_pd_photon_dir_corners_t, std::array<float2, 4>) dev_pd_photon_dir_corners;

    // Signal from each track hypo (\mu_j)
    HOST_OUTPUT(host_total_number_of_geomeffs_t, unsigned) host_total_number_of_geomeffs;
    DEVICE_OUTPUT(dev_rich_geomeff_offsets_t, unsigned) dev_rich_geomeff_offsets;
    DEVICE_OUTPUT(dev_rich_geomeff_pd_ids_t, int) dev_rich_geomeff_pd_ids;
    DEVICE_OUTPUT(dev_rich_geomeff_pd_fractions_t, Allen::Rich::HypoData<float>) dev_rich_geomeff_pd_fractions;
    DEVICE_OUTPUT(dev_rich_geomeff_fractions_t, Allen::Rich::HypoData<float>) dev_rich_geomeff_fractions;

    // Photon observed signals (a_{ij})
    HOST_OUTPUT(host_total_number_of_photons_t, unsigned) host_total_number_of_photons;
    DEVICE_OUTPUT(dev_offsets_rich_photons_prefilter_t, unsigned) dev_offsets_rich_photons_prefilter;
    DEVICE_OUTPUT(dev_rich_photons_prefilter_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons_prefilter;
    DEVICE_OUTPUT(dev_offsets_rich_photons_t, unsigned) dev_offsets_rich_photons;
    DEVICE_OUTPUT(dev_rich_photons_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons;
  };

  struct PDShortInfo {
    PDShortInfo(uint16_t index, bool isLarge) : m_data(index | (isLarge << 15)) {}
    __device__ uint16_t index() const { return m_data & 0x7FFF; }
    __device__ bool isLarge() const { return m_data >> 15; }

  private:
    uint16_t m_data {};
  };

  struct rich_quartic_signals_t : public DeviceAlgorithm, Parameters {
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

    // Pre-filtering:

    Allen::Property<Allen::Rich::DetectorArray<float>> m_radScale {
      this,
      "RadScale",
      {0.0117f, -0.021f},
      "Rich1Gas, Rich2Gas"};
    Allen::Property<Allen::Rich::DetectorArray<float>> m_minROIPreSel {this, "PreSelMinTrackROI", {0.f, 0.f}, ""};
    Allen::Property<Allen::Rich::DetectorArray<float>> m_maxROIPreSel {this, "PreSelMaxTrackROI", {110.f, 165.f}, ""};

    Allen::Property<Allen::Rich::DetectorArray<float>> m_minRadLength {
      this,
      "MinRadiatorPathLength",
      {500.f, 1500.f},
      "Minimum path length in each RICH radiator volume."};

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

    // Precomputed values
    mutable std::array<float, 2> m_factor {};
    mutable unsigned m_n_pds {};
    mutable PDShortInfo* m_pd_infos {nullptr};
    mutable short2* m_pd_lpos16 {nullptr};
    mutable float3* m_pd_corners {nullptr};
    mutable float* m_pd_diag {nullptr}; // TODO: this shouldnt be an aray..
  };
} // namespace rich_quartic_signals
