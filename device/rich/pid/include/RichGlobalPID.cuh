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

namespace rich_global_pid {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;
    HOST_INPUT(host_number_of_pixels_r1_t, unsigned) host_number_of_pixels_r1;
    HOST_INPUT(host_number_of_pixels_r2_t, unsigned) host_number_of_pixels_r2;
    HOST_INPUT(host_number_of_photons_r1_t, unsigned) host_number_of_photons_r1;
    HOST_INPUT(host_number_of_photons_r2_t, unsigned) host_number_of_photons_r2;

    DEVICE_INPUT(dev_offsets_tracks_t, unsigned) dev_offsets_tracks;
    DEVICE_INPUT(dev_pid_in_t, Allen::Rich::ParticleIDType) dev_pid_in;

    // For "FromCones" background estimation, optionals are buggy so commented for now:
    /*DEVICE_INPUT_OPTIONAL(dev_rich_geomeff_offsets_r1_t, unsigned) dev_rich_geomeff_offsets_r1;
    DEVICE_INPUT_OPTIONAL(dev_rich_geomeff_pd_ids_r1_t, int) dev_rich_geomeff_pd_ids_r1;
    DEVICE_INPUT_OPTIONAL(dev_rich_geomeff_fractions_r1_t, float) dev_rich_geomeff_fractions_r1;
    DEVICE_INPUT_OPTIONAL(dev_rich_geomeff_offsets_r2_t, unsigned) dev_rich_geomeff_offsets_r2;
    DEVICE_INPUT_OPTIONAL(dev_rich_geomeff_pd_ids_r2_t, int) dev_rich_geomeff_pd_ids_r2;
    DEVICE_INPUT_OPTIONAL(dev_rich_geomeff_fractions_r2_t, float) dev_rich_geomeff_fractions_r2;*/

    DEVICE_INPUT(dev_rich_pd_offsets_r1_t, unsigned) dev_rich_pd_offsets_r1;
    DEVICE_INPUT(dev_offsets_rich_photons_r1_t, unsigned) dev_offsets_rich_photons_r1;
    DEVICE_INPUT(dev_rich_photons_r1_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons_r1;
    DEVICE_INPUT(dev_photon_pix_signals_r1_t, Allen::Rich::HypoData<float>) dev_photon_pix_signals_r1;
    DEVICE_INPUT(dev_track_total_signals_r1_t, Allen::Rich::HypoData<float>) dev_track_total_signals_r1;

    DEVICE_INPUT(dev_rich_pd_offsets_r2_t, unsigned) dev_rich_pd_offsets_r2;
    DEVICE_INPUT(dev_offsets_rich_photons_r2_t, unsigned) dev_offsets_rich_photons_r2;
    DEVICE_INPUT(dev_rich_photons_r2_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons_r2;
    DEVICE_INPUT(dev_photon_pix_signals_r2_t, Allen::Rich::HypoData<float>) dev_photon_pix_signals_r2;
    DEVICE_INPUT(dev_track_total_signals_r2_t, Allen::Rich::HypoData<float>) dev_track_total_signals_r2;

    // pix2track map
    DEVICE_INPUT(dev_pix2track_offsets_r1_t, unsigned) dev_pix2track_offsets_r1;
    DEVICE_INPUT(dev_pix2track_r1_t, unsigned) dev_pix2track_r1;
    DEVICE_INPUT(dev_pix2track_offsets_r2_t, unsigned) dev_pix2track_offsets_r2;
    DEVICE_INPUT(dev_pix2track_r2_t, unsigned) dev_pix2track_r2;

    DEVICE_OUTPUT(dev_pixel_signals_r1_t, int) dev_pixel_signals_r1;
    DEVICE_OUTPUT(dev_pixel_signals_r2_t, int) dev_pixel_signals_r2;

    DEVICE_OUTPUT(dev_pix_bkg_r1_t, float) dev_pix_bkg_r1;
    DEVICE_OUTPUT(dev_pix_bkg_r2_t, float) dev_pix_bkg_r2;

    DEVICE_OUTPUT(dev_pid_out_t, Allen::Rich::ParticleIDType) dev_pid_out;
    DEVICE_OUTPUT(dev_dll_out_t, Allen::Rich::HypoData<float>) dev_dll_out;
  };
  struct rich_global_pid_t : public DeviceAlgorithm, Parameters {
    void update(const Constants&) const;

    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    template<Allen::Rich::Detector::DetectorType richIdx>
    void updateRich(const Allen::Rich::RichDetector<richIdx>*) const;

    template<Allen::Rich::Detector::DetectorType richIdx>
    void pixelSignalsForRich(
      const ArgumentReferences<Parameters>&,
      const Allen::Context&,
      const Allen::Rich::ParticleIDType*) const;

    template<Allen::Rich::Detector::DetectorType richIdx>
    void backgroundsForRichFromReco(const ArgumentReferences<Parameters>&, const Allen::Context&, const unsigned) const;

    void initDLLs(
      const ArgumentReferences<Parameters>&,
      const Allen::Context&,
      const Allen::Rich::ParticleIDType*,
      Allen::Rich::ParticleIDType*) const;

    void doIterations(
      const ArgumentReferences<Parameters>&,
      const Allen::Context&,
      Allen::Rich::ParticleIDType*,  // pids
      Allen::Rich::HypoData<float>*, // dlls
      float*,                        // s_old_r1
      float*,                        // s_old_r2
      const unsigned*,               // pix2track_offsets_r1
      const unsigned*,               // pix2track_r1
      const unsigned*,               // pix2track_offsets_r2
      const unsigned*                // pix2track_r2
    ) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {256, 1, 1}, "block dimensions"};

    Allen::Property<unsigned> m_nLikelihoodIterations {this, "nLikelihoodIterations", 2, ""};

    /** Ignore the expected signal when computing the background terms.
        Effectively, will assume all observed hits are background */
    Allen::Property<std::vector<bool>> m_ignoreExpSignal {
      this,
      "IgnoreExpectedSignals",
      {{true, false}},
      "Ignore track expectations when calculating backgrounds"};

    Allen::Property<float> m_epsilon {this, "LikelihoodThreshold", -1e-3f, "Threshold for likelihood maximisation"};

    Allen::Property<unsigned> m_maxEventIterations {this, "MaxEventIterations", 2000u, "Maximum globalPID iterations"};

    mutable Allen::Rich::DetectorArray<uint16_t*> m_cached_effNumPixsEC {nullptr, nullptr};
  };
} // namespace rich_global_pid
