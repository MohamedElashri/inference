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

namespace rich_raytrace_cherenkov_cones {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_tracks_t, unsigned) host_number_of_tracks;
    DEVICE_INPUT(dev_offsets_rich_states_t, unsigned) dev_offsets_rich_states;
    DEVICE_INPUT(dev_segs_best_point_t, float3) dev_segs_best_point;
    DEVICE_INPUT(dev_segs_best_momentum_t, float3) dev_segs_best_momentum;
    DEVICE_INPUT(dev_rich_hypos_t, Allen::Rich::ParticleHypos) dev_rich_hypos;
    HOST_OUTPUT(host_total_number_of_photons_t, unsigned) host_total_number_of_photons;
    DEVICE_OUTPUT(dev_rich_photons_offsets_t, unsigned) dev_rich_photons_offsets;
    DEVICE_OUTPUT(dev_rich_cone_pd_ids_t, int) dev_rich_cone_pd_ids;
    HOST_OUTPUT(host_total_number_of_geomeffs_t, unsigned) host_total_number_of_geomeffs;
    DEVICE_OUTPUT(dev_rich_geomeff_offsets_t, unsigned) dev_rich_geomeff_offsets;
    DEVICE_OUTPUT(dev_rich_geomeff_pd_ids_t, int) dev_rich_geomeff_pd_ids;
    DEVICE_OUTPUT(dev_rich_geomeff_fractions_t, float) dev_rich_geomeff_fractions;
    DEVICE_OUTPUT(dev_rich_geomeff_fractions_per_hypo_t, Allen::Rich::HypoData<float>)
    dev_rich_geomeff_fractions_per_hypo;
    DEVICE_OUTPUT(dev_track_total_signals_t, Allen::Rich::HypoData<float>) dev_track_total_signals;
  };
  struct rich_raytrace_cherenkov_cones_t : public DeviceAlgorithm, Parameters {
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

    /// Minimum number of points to ray trace on each ring, for each RICH
    Allen::Property<Allen::Rich::DetectorArray<unsigned>> m_nPointsMin {this, "NRingPointsMin", {16u, 16u}, ""};
    /// Maximum number of points to ray trace on each ring, for each RICH
    Allen::Property<Allen::Rich::DetectorArray<unsigned>> m_nPointsMax {this, "NRingPointsMax", {96u, 96u}, ""};
  };
} // namespace rich_raytrace_cherenkov_cones
