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

#include "KalmanParametrizations.cuh"
#include "ParKalmanDefinitions.cuh"
#include "ParKalmanMath.cuh"
#include "ParKalmanMethods.cuh"
#include "ParKalmanFittedTrack.cuh"

#include "SciFiConsolidated.cuh"
#include "UTConsolidated.cuh"
#include "VeloConsolidated.cuh"
#include "AssociateConsolidated.cuh"

#include "States.cuh"
#include "SciFiDefinitions.cuh"

#include "AlgorithmTypes.cuh"
#include "PV_Definitions.cuh"
#include "ParticleTypes.cuh"

namespace ParKalmanFilter {

  //----------------------------------------------------------------------
  // Create the output track.
  __device__ inline void MakeTrack(
    const KalmanFloat& init_qop,
    const Vector5& x,
    const SymMatrix5x5& C,
    const trackInfo& tI,
    FittedTrack& track,
    const unsigned& velo_hits,
    unsigned& ut_hits,
    unsigned& scifi_hits)
  {
    track.chi2 = tI.m_chi2V + tI.m_chi2T + tI.m_chi2UT;
    track.chi2V = tI.m_chi2V;
    track.chi2T = tI.m_chi2T;
    track.chi2UT = tI.m_chi2UT;
    track.ndof = scifi_hits + ut_hits + 2 * velo_hits - 5;
    track.ndofV = velo_hits == 0 ? 0 : 2 * velo_hits - 5;
    track.ndofT = scifi_hits - 5;
    track.state = x;
    track.cov = C;
    track.z = tI.m_Lastz;
    track.first_qop = init_qop;
    track.best_qop = tI.m_BestMomEst;
    track.nhits = velo_hits + ut_hits + scifi_hits;
    track.nhitsV = velo_hits;
    track.nhitsUT = ut_hits;
    track.nhitsT = scifi_hits;
  }

  //----------------------------------------------------------------------
  // Create UT hit map: encode for each UT layer [0-3] which hit index
  // corresponds to that layer. Uses 4 bits per layer, 0xf means no hit.
  __device__ inline unsigned make_ut_hitmap(const Allen::Views::UT::Consolidated::Track& ut_track, unsigned& n_layers)
  {
    const unsigned n_ut_hits = ut_track.number_of_ut_hits();
    unsigned hit_map = 0xffff; // [1111 1111 1111 1111] - 4 layers * 4 bits
    n_layers = 0;

    for (unsigned hit_counter = 0; hit_counter < n_ut_hits; hit_counter++) {
      const unsigned layer = ut_track.hit(hit_counter).plane_code();
      // Check that there is not already a hit in that layer
      // If there already is one, the second hit in that layer will be discarded
      const bool no_hit_in_layer = ((hit_map >> (layer * 4)) & 0xf) == 0xf;
      hit_map -= no_hit_in_layer * (0xf << (layer * 4)); // Clear the placeholder
      hit_map += no_hit_in_layer * (hit_counter << (layer * 4));
      n_layers += no_hit_in_layer;
    }
    return hit_map;
  }

  //----------------------------------------------------------------------
  // Create SciFi hit maps: two separate maps for the first and last 6 T layers.
  // Uses 4 bits per layer, 0xf means no hit.
  __device__ inline void make_scifi_hitmaps(
    const Allen::Views::SciFi::Consolidated::Track& scifi_track,
    unsigned& hit_map0,
    unsigned& hit_map1,
    unsigned& n_layers)
  {
    const unsigned n_scifi_hits = scifi_track.number_of_scifi_hits();
    hit_map0 = 0xffffff; // [1111 1111 1111 1111 1111 1111] - 6 layers * 4 bits
    hit_map1 = 0xffffff;
    n_layers = 0;

    for (unsigned hit_counter = 0; hit_counter < n_scifi_hits; hit_counter++) {
      unsigned layer = scifi_track.hit(hit_counter).planeCode() / 2;
      // Check that there is not already a hit in that layer
      unsigned& hit_map = (layer < 6) ? hit_map0 : hit_map1;
      layer = layer % 6; // Position relative to front of map (0 or 6)
      const bool no_hit_in_layer = ((hit_map >> (layer * 4)) & 0xf) == 0xf;
      hit_map -= no_hit_in_layer * (0xf << (layer * 4)); // Clear the placeholder
      hit_map += no_hit_in_layer * (hit_counter << (layer * 4));
      n_layers += no_hit_in_layer;
    }
  }

  //----------------------------------------------------------------------
  // Run the Kalman filter on a track.
  __device__ void fit(
    const Allen::Views::Velo::Consolidated::Track& velo_track,
    const Allen::Views::UT::Consolidated::Track& ut_track,
    const Allen::Views::SciFi::Consolidated::Track& scifi_track,
    const KalmanFloat init_qop,
    const KalmanParametrizations* kalman_params,
    FittedTrack& track,
    SimpleKalmanState& r1_f_state,
    SimpleKalmanState& r1_b_state,
    SimpleKalmanState& r2_f_state,
    SimpleKalmanState& r2_b_state,
    const float* dev_UT_lay,
    const float* dev_T_lay,
    const float* dev_V_pars,
    const float* dev_VUT_pars,
    const float* dev_UT_pars,
    const float* dev_UTTF_pars,
    const float* dev_T_pars,
    const float* dev_TFT_pars,
    const float* dev_UTT_META,
    const KalmanFloat magSign);

  // Set the fit results
  __host__ __device__ void set_result(
    const unsigned track_number,
    const ParKalmanFilter::FittedTrack& track,
    Velo::Consolidated::States& states);

} // namespace ParKalmanFilter

namespace kalman_filter {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_number_of_reconstructed_scifi_tracks_t, unsigned) host_number_of_reconstructed_scifi_tracks;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    DEVICE_INPUT(dev_multi_event_long_tracks_view_t, Allen::Views::Physics::MultiEventLongTracks)
    dev_multi_event_long_tracks_view;
    DEVICE_INPUT(dev_long_track_view_t, Allen::Views::Physics::LongTrack) dev_long_track_view;
    DEVICE_INPUT(dev_offsets_long_tracks_t, unsigned) dev_atomics_scifi;
    DEVICE_INPUT(dev_multi_final_vertices_t, PV::Vertex) dev_multi_final_vertices;
    DEVICE_INPUT(dev_number_of_multi_final_vertices_t, unsigned) dev_number_of_multi_final_vertices;
    DEVICE_INPUT(dev_is_muon_t, bool) dev_is_muon;
    DEVICE_OUTPUT(dev_kf_tracks_t, ParKalmanFilter::FittedTrack) dev_kf_tracks;
    DEVICE_OUTPUT(dev_kalman_pv_ip_t, char) dev_kalman_pv_ip;
    DEVICE_OUTPUT(dev_kalman_fit_results_t, char) dev_kalman_fit_results;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_kalman_states_view_t,
      DEPENDENCIES(dev_kalman_fit_results_t),
      Allen::Views::Physics::KalmanStates)
    dev_kalman_states_view;
    DEVICE_OUTPUT_WITH_DEPENDENCIES(
      dev_kalman_pv_tables_t,
      DEPENDENCIES(dev_kalman_pv_ip_t),
      Allen::Views::Physics::PVTable)
    dev_kalman_pv_tables;

    // all the RICH states:
    DEVICE_OUTPUT(dev_kalman_R1_F_view_t, SimpleKalmanState) dev_kalman_R1_F_view;
    DEVICE_OUTPUT(dev_kalman_R1_B_view_t, SimpleKalmanState) dev_kalman_R1_B_view;
    DEVICE_OUTPUT(dev_kalman_R2_F_view_t, SimpleKalmanState) dev_kalman_R2_F_view;
    DEVICE_OUTPUT(dev_kalman_R2_B_view_t, SimpleKalmanState) dev_kalman_R2_B_view;
  };

  //--------------------------------------------------
  // Main execution of the parametrized Kalman Filter.
  //--------------------------------------------------
  __global__ void kalman_filter(
    Parameters,
    const float magnet_polarity,
    const ParKalmanFilter::KalmanParametrizations* dev_kalman_params);

  __global__ void kalman_pv_ip(Parameters parameters);

  struct kalman_filter_t : public DeviceAlgorithm, Parameters {
    void update(const Constants& constants) const;
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {128, 1, 1}, "block dimensions"};
  };
} // namespace kalman_filter
