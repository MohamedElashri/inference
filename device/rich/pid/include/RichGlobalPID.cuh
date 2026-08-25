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

  // How the per-pixel background is estimated. FromReco derives it from the observed
  // (decoded) pixel occupancy; FromCones will derive it from the ray-traced geometrical
  // efficiency (rich_raytrace_cherenkov_cones_t's geomeff outputs)
  enum class BackgroundEstimationMethod { FromReco, FromCones };

  // Parameters<method> is reflected via Boost.PFR (structured bindings) by Allen's
  // Store/ArgumentReferences machinery. Boost.PFR refuses to reflect any type with a
  // base class -- static_assert(!std::is_base_of<...>) -- regardless of whether that
  // base is templated or empty; this is a structured-bindings language restriction,
  // not a library quirk, so inheritance (the idiomatic way to share these fields
  // between the two Parameters specializations) isn't an option here. This macro is
  // the workaround: a plain textual expansion, so the two Parameters specializations
  // stay flat structs with no base class, while avoiding hand-duplicating the ~20
  // fields shared between them.
  // Can be removed if these algorithms are migrated off the Allen standalone
  // Store/PFR reflection and onto the Gaudi Functional-style (via the multi-event
  // scheduler MES) syntax, which doesn't have this constraint
#define RICH_GLOBAL_PID_COMMON_PARAMETERS                                                              \
  HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;                                 \
  HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;                                 \
  HOST_INPUT(host_number_of_pixels_r1_t, unsigned) host_number_of_pixels_r1;                           \
  HOST_INPUT(host_number_of_pixels_r2_t, unsigned) host_number_of_pixels_r2;                           \
  HOST_INPUT(host_number_of_photons_r1_t, unsigned) host_number_of_photons_r1;                         \
  HOST_INPUT(host_number_of_photons_r2_t, unsigned) host_number_of_photons_r2;                         \
                                                                                                       \
  DEVICE_INPUT(dev_offsets_tracks_t, unsigned) dev_offsets_tracks;                                     \
  DEVICE_INPUT(dev_pid_in_t, Allen::Rich::ParticleIDType) dev_pid_in;                                  \
                                                                                                       \
  DEVICE_INPUT(dev_rich_pd_offsets_r1_t, unsigned) dev_rich_pd_offsets_r1;                             \
  DEVICE_INPUT(dev_offsets_rich_photons_r1_t, unsigned) dev_offsets_rich_photons_r1;                   \
  DEVICE_INPUT(dev_rich_photons_r1_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons_r1;            \
  DEVICE_INPUT(dev_photon_pix_signals_r1_t, Allen::Rich::HypoData<float>) dev_photon_pix_signals_r1;   \
  DEVICE_INPUT(dev_track_total_signals_r1_t, Allen::Rich::HypoData<float>) dev_track_total_signals_r1; \
                                                                                                       \
  DEVICE_INPUT(dev_rich_pd_offsets_r2_t, unsigned) dev_rich_pd_offsets_r2;                             \
  DEVICE_INPUT(dev_offsets_rich_photons_r2_t, unsigned) dev_offsets_rich_photons_r2;                   \
  DEVICE_INPUT(dev_rich_photons_r2_t, Allen::Rich::PhotonReco::Photon) dev_rich_photons_r2;            \
  DEVICE_INPUT(dev_photon_pix_signals_r2_t, Allen::Rich::HypoData<float>) dev_photon_pix_signals_r2;   \
  DEVICE_INPUT(dev_track_total_signals_r2_t, Allen::Rich::HypoData<float>) dev_track_total_signals_r2; \
                                                                                                       \
  DEVICE_INPUT(dev_pix2track_offsets_r1_t, unsigned) dev_pix2track_offsets_r1;                         \
  DEVICE_INPUT(dev_pix2track_r1_t, unsigned) dev_pix2track_r1;                                         \
  DEVICE_INPUT(dev_pix2photon_r1_t, unsigned) dev_pix2photon_r1;                                       \
  DEVICE_INPUT(dev_pix2track_offsets_r2_t, unsigned) dev_pix2track_offsets_r2;                         \
  DEVICE_INPUT(dev_pix2track_r2_t, unsigned) dev_pix2track_r2;                                         \
  DEVICE_INPUT(dev_pix2photon_r2_t, unsigned) dev_pix2photon_r2;                                       \
                                                                                                       \
  DEVICE_OUTPUT(dev_pixel_signals_r1_t, int) dev_pixel_signals_r1;                                     \
  DEVICE_OUTPUT(dev_pixel_signals_r2_t, int) dev_pixel_signals_r2;                                     \
                                                                                                       \
  DEVICE_OUTPUT(dev_pix_bkg_r1_t, float) dev_pix_bkg_r1;                                               \
  DEVICE_OUTPUT(dev_pix_bkg_r2_t, float) dev_pix_bkg_r2;                                               \
                                                                                                       \
  DEVICE_OUTPUT(dev_pid_out_t, Allen::Rich::ParticleIDType) dev_pid_out;                               \
  DEVICE_OUTPUT(dev_dll_out_t, Allen::Rich::HypoData<float>) dev_dll_out;

  template<BackgroundEstimationMethod bkg_method>
  struct Parameters;

  // using bkg from quartic signals method
  template<>
  struct Parameters<BackgroundEstimationMethod::FromReco> {
    RICH_GLOBAL_PID_COMMON_PARAMETERS

    DEVICE_INPUT(dev_rich_geomeff_offsets_r1_t, unsigned) dev_rich_geomeff_offsets_r1;
    DEVICE_INPUT(dev_rich_geomeff_pd_ids_r1_t, int) dev_rich_geomeff_pd_ids_r1;
    DEVICE_INPUT(dev_rich_geomeff_pd_fractions_r1_t, Allen::Rich::HypoData<float>) dev_rich_geomeff_pd_fractions_r1;

    DEVICE_INPUT(dev_rich_geomeff_offsets_r2_t, unsigned) dev_rich_geomeff_offsets_r2;
    DEVICE_INPUT(dev_rich_geomeff_pd_ids_r2_t, int) dev_rich_geomeff_pd_ids_r2;
    DEVICE_INPUT(dev_rich_geomeff_pd_fractions_r2_t, Allen::Rich::HypoData<float>) dev_rich_geomeff_pd_fractions_r2;

    DEVICE_OUTPUT(dev_exp_signal_ec_r1_t, int) dev_exp_signal_ec_r1;
    DEVICE_OUTPUT(dev_exp_signal_ec_r2_t, int) dev_exp_signal_ec_r2;
  };

  // using bkg from CK Cones method (HLT2 like)
  template<>
  struct Parameters<BackgroundEstimationMethod::FromCones> {
    RICH_GLOBAL_PID_COMMON_PARAMETERS

    DEVICE_INPUT(dev_rich_geomeff_offsets_r1_t, unsigned) dev_rich_geomeff_offsets_r1;
    DEVICE_INPUT(dev_rich_geomeff_pd_ids_r1_t, int) dev_rich_geomeff_pd_ids_r1;
    DEVICE_INPUT(dev_rich_geomeff_fractions_r1_t, float) dev_rich_geomeff_fractions_r1;
    DEVICE_INPUT(dev_rich_geomeff_fractions_per_hypo_r1_t, Allen::Rich::HypoData<float>)
    dev_rich_geomeff_fractions_per_hypo_r1;

    DEVICE_INPUT(dev_rich_geomeff_offsets_r2_t, unsigned) dev_rich_geomeff_offsets_r2;
    DEVICE_INPUT(dev_rich_geomeff_pd_ids_r2_t, int) dev_rich_geomeff_pd_ids_r2;
    DEVICE_INPUT(dev_rich_geomeff_fractions_r2_t, float) dev_rich_geomeff_fractions_r2;
    DEVICE_INPUT(dev_rich_geomeff_fractions_per_hypo_r2_t, Allen::Rich::HypoData<float>)
    dev_rich_geomeff_fractions_per_hypo_r2;

    DEVICE_INPUT(dev_rich_hypos_r1_t, Allen::Rich::ParticleHypos) dev_rich_hypos_r1;
    DEVICE_INPUT(dev_rich_hypos_r2_t, Allen::Rich::ParticleHypos) dev_rich_hypos_r2;

    DEVICE_OUTPUT(dev_exp_signal_ec_r1_t, int) dev_exp_signal_ec_r1;
    DEVICE_OUTPUT(dev_exp_signal_ec_r2_t, int) dev_exp_signal_ec_r2;
  };

#undef RICH_GLOBAL_PID_COMMON_PARAMETERS

  template<BackgroundEstimationMethod bkg_method>
  struct rich_global_pid_t : public DeviceAlgorithm, Parameters<bkg_method> {
    void update(const Constants&) const;

    void set_arguments_size(ArgumentReferences<Parameters<bkg_method>>, const RuntimeOptions&, const Constants&) const;

    template<Allen::Rich::Detector::DetectorType richIdx>
    void updateRich(const Allen::Rich::RichDetector<richIdx>*) const;

    template<Allen::Rich::Detector::DetectorType richIdx>
    void pixelSignalsForRich(
      const ArgumentReferences<Parameters<bkg_method>>&,
      const Allen::Context&,
      const Allen::Rich::ParticleIDType*) const;

    template<Allen::Rich::Detector::DetectorType richIdx>
    void backgroundsForRich(
      const ArgumentReferences<Parameters<bkg_method>>&,
      const Allen::Context&,
      const Allen::Rich::ParticleIDType*,
      const unsigned) const;

    void initDLLs(
      const ArgumentReferences<Parameters<bkg_method>>&,
      const Allen::Context&,
      const Allen::Rich::ParticleIDType*,
      Allen::Rich::ParticleIDType*) const;

    void doIterations(
      const ArgumentReferences<Parameters<bkg_method>>&,
      const Allen::Context&,
      Allen::Rich::ParticleIDType*,  // pids
      Allen::Rich::HypoData<float>*, // dlls
      const unsigned*,               // pix2track_offsets_r1
      const unsigned*,               // pix2track_r1
      const unsigned*,               // pix2photon_r1
      const unsigned*,               // pix2track_offsets_r2
      const unsigned*,               // pix2track_r2
      const unsigned*                // pix2photon_r2
    ) const;

    void operator()(
      const ArgumentReferences<Parameters<bkg_method>>&,
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

    // Background properties indexed as [iter0_rich1, iter0_rich2, iter1_rich1, iter1_rich2, ...])
    Allen::Property<std::vector<float>> m_bkgWeight {this, "PDBckWeights", {1., 1., 1., 1., 1., 1., 1., 1.}, ""};
    Allen::Property<std::vector<float>> m_bkgThreshold {this, "PDBackThresholds", {0., 0., 0., 0., 0., 0., 0., 0.}, ""};
    Allen::Property<std::vector<float>> m_bkgMin {this, "PDBackMinPixBackground", {0., 0., 0., 0., 0., 0., 0., 0.}, ""};
    Allen::Property<std::vector<float>> m_bkgMax {
      this,
      "PDBackMaxPixBackground",
      {999., 999., 999., 999., 999., 999., 999., 999.},
      ""};

    mutable Allen::Rich::DetectorArray<uint16_t*> m_cached_effNumPixsEC {nullptr, nullptr};
  };

  using rich_global_pid_from_reco_t = rich_global_pid_t<BackgroundEstimationMethod::FromReco>;
  using rich_global_pid_from_cones_t = rich_global_pid_t<BackgroundEstimationMethod::FromCones>;

} // namespace rich_global_pid
