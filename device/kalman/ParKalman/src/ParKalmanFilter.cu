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
#include "ParKalmanFilter.cuh"
#include "ParKalmanSharedConstants.cuh"

INSTANTIATE_ALGORITHM(kalman_filter::kalman_filter_t)

__global__ void refit_outliers(
  kalman_filter::Parameters parameters,
  const float magnet_polarity,
  const ParKalmanFilter::KalmanParametrizations* dev_kalman_params,
  const KalmanFloat outlier_threshold,
  const unsigned max_outlier_iterations);

void kalman_filter::kalman_filter_t::update(const Constants& constants) const
{
  updateCommon(constants);
  // Load shared ParKF parameters
  parkalman_shared::update_shared_constants(constants);
}

void kalman_filter::kalman_filter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  auto n_scifi_tracks = first<host_number_of_reconstructed_scifi_tracks_t>(arguments);
  set_size<dev_kf_tracks_t>(arguments, n_scifi_tracks);
  set_size<dev_n_outlier_tracks_t>(arguments, 1);
  set_size<dev_outlier_track_indices_t>(arguments, n_scifi_tracks);
  set_size<dev_kalman_pv_ip_t>(arguments, Associate::Consolidated::table_size(n_scifi_tracks));
  set_size<dev_kalman_fit_results_t>(arguments, n_scifi_tracks * Velo::Consolidated::States::size);
  set_size<dev_kalman_states_view_t>(arguments, first<host_number_of_events_t>(arguments));
  set_size<dev_kalman_pv_tables_t>(arguments, first<host_number_of_events_t>(arguments));

  set_size<dev_kalman_R1_F_view_t>(arguments, n_scifi_tracks);
  set_size<dev_kalman_R1_B_view_t>(arguments, n_scifi_tracks);
  set_size<dev_kalman_R2_F_view_t>(arguments, n_scifi_tracks);
  set_size<dev_kalman_R2_B_view_t>(arguments, n_scifi_tracks);
}

void kalman_filter::kalman_filter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  dim3 block_dim = m_block_dim;
  int _gridDim = (first<host_number_of_reconstructed_scifi_tracks_t>(arguments) + (block_dim.x) - 1) / (block_dim.x);

  // Zero-initialize outlier counter before main fit
  Allen::memset_async<dev_n_outlier_tracks_t>(arguments, 0, context);

  // Main Kalman filter fit on all tracks
  global_function(kalman_filter)(dim3(_gridDim), m_block_dim, context)(
    arguments, constants.magnet_polarity, constants.dev_kalman_params, m_outlier_chi2_threshold);

  // Refit outlier tracks excluding their worst hit (only if enabled).
  if (m_outlier_chi2_threshold > 0.0f) {
    unsigned n_outlier_tracks = 0;
    Allen::memcpy_async(
      &n_outlier_tracks, data<dev_n_outlier_tracks_t>(arguments), sizeof(unsigned), Allen::memcpyDeviceToHost, context);
    Allen::synchronize(context);

    if (n_outlier_tracks > 0) {
      _gridDim = (n_outlier_tracks + block_dim.x - 1) / block_dim.x;
      global_function(refit_outliers)(dim3(_gridDim), m_block_dim, context)(
        arguments,
        constants.magnet_polarity,
        constants.dev_kalman_params,
        m_outlier_chi2_threshold,
        m_max_outlier_iterations);
    }
  }

  global_function(kalman_pv_ip)(dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(arguments);
}

namespace ParKalmanFilter {
  //----------------------------------------------------------------------
  // Run the Kalman filter.
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
    const KalmanFloat magSign,
    const KalmanFloat outlier_chi2_threshold,
    const uint64_t skip_mask)
  {
    using namespace parkalman_shared;
    // Fit information.
    trackInfo tI;
    tI.m_BestMomEst = init_qop;
    tI.m_polarity = magSign;

    // Get Velo Hits
    const unsigned n_velo_hits = velo_track.number_of_hits();

    // Run the fit.
    tI.m_Lastz = -1;
    Vector5 x;
    SymMatrix5x5 C;
    x[4] = init_qop;

    // Initilise the Velo state using the first and last Velo hit
    CreateVeloSeedState(velo_track, n_velo_hits, x, C, tI, skip_mask);
    // sets the state at the position of the first (in z last in index) hit.
    // initialises a very large covariance matrix
    tI.m_chi2V = 0;
    tI.m_chi2T = 0;
    tI.m_chi2UT = 0;

    // Count previously removed hits per detector from the accumulated skip_mask
    unsigned velo_removed = 0, ut_removed = 0, scifi_removed = 0;
    if (skip_mask != 0) {
      for (unsigned b = 0; b < n_velo_hits; ++b)
        velo_removed += (skip_mask >> b) & 1;
      for (unsigned b = n_velo_hits; b < n_velo_hits + 4; ++b)
        ut_removed += (skip_mask >> b) & 1;
      for (unsigned b = n_velo_hits + 4; b < n_velo_hits + 4 + 12; ++b)
        scifi_removed += (skip_mask >> b) & 1;
    }

    OutlierContext oc {outlier_chi2_threshold > 0.0f, outlier_chi2_threshold, skip_mask};

    //------------------------------ Start forward fit.
    // Velo loop.
    // Update on the first hit
    if (oc.should_process(true))
      oc.consider(
        UpdateStateV(velo_track, 1, n_velo_hits - 1, x, C, tI), n_velo_hits - velo_removed, minVeloHitsForOutlier);
    else
      oc.skip();

    // have to iterate down from `n_velo_hits - 1` to `0`
    for (unsigned i_hit = 1; i_hit < n_velo_hits; i_hit++) {
      PredictStateV(velo_track, dev_V_pars, n_velo_hits - 1 - i_hit, x, C, tI);
      if (oc.should_process(true))
        oc.consider(
          UpdateStateV(velo_track, 1, n_velo_hits - 1 - i_hit, x, C, tI),
          n_velo_hits - velo_removed,
          minVeloHitsForOutlier);
      else
        oc.skip();
    }

    KalmanFloat endVeloZ = tI.m_Lastz; // z position if the last Velo hit
    // Define UT layer counter (avoid counting two hits in one layer)
    unsigned n_ut_layers = 0;
    // Create UT hit map
    unsigned layer;
    unsigned hit_counter;
    unsigned hit_map0 = make_ut_hitmap(ut_track, n_ut_layers);

    // Velo -> UT.
    hit_counter = (hit_map0 & 0xf); // Checks for hit in first layer
    PredictStateVUT(ut_track, dev_UT_lay, dev_VUT_pars, x, C, tI, hit_counter);
    // m_RefStateForwardV was saved at the last Velo hit.

    // Update the first UT layer if there is a hit.
    if (oc.should_process(hit_counter != 0xf))
      oc.consider(UpdateStateUT(ut_track, x, C, tI, hit_counter), n_ut_layers - ut_removed, minUTLayersForOutlier);
    else
      oc.skip();

    // Iterate over the remaining UT layers
    for (layer = 1; layer < 4; layer++) {
      hit_counter = ((hit_map0 >> (layer * 4)) & 0xf);
      PredictStateUT(ut_track, dev_UT_lay, dev_UT_pars, x, C, tI, layer, hit_counter);
      if (oc.should_process(hit_counter != 0xf))
        oc.consider(UpdateStateUT(ut_track, x, C, tI, hit_counter), n_ut_layers - ut_removed, minUTLayersForOutlier);
      else
        oc.skip();
    }

    layer = 3; // needed because `PredictStateUTT` calls `ExtrapolateInUT` again
    PredictStateUTT(dev_UT_pars, dev_TFT_pars, dev_UTTF_pars, dev_UTT_META, dev_T_lay, kalman_params, x, C, tI, layer);

    // Define SciFi layer counter (avoid counting two hits in one layer)
    unsigned n_scifi_layers = 0;
    // Get SciFi Hits
    unsigned hit_map1;
    make_scifi_hitmaps(scifi_track, hit_map0, hit_map1, n_scifi_layers);

    // Predict State UTT already does all the necessary extrapolation to the first layer of FT
    // Update in first T layer if there is a hit.
    // TODO: this could probably use and extrapolation considering the hit in FT.
    // ----- would possibly make the TFT step redundant and improve on it.
    hit_counter = (hit_map0 & 0xf);
    layer = 0;
    if (oc.should_process(hit_counter != 0xf))
      oc.consider(
        UpdateStateT(scifi_track, dev_T_lay, x, C, tI, hit_counter, layer),
        n_scifi_layers - scifi_removed,
        minSciFiLayersForOutlier);
    else
      oc.skip();

    for (layer = 1; layer < 6; layer++) {
      hit_counter = ((hit_map0 >> (4 * layer)) & 0xf);
      PredictStateT(scifi_track, dev_T_lay, dev_T_pars, x, C, tI, layer, hit_counter);
      if (oc.should_process(hit_counter != 0xf))
        oc.consider(
          UpdateStateT(scifi_track, dev_T_lay, x, C, tI, hit_counter, layer),
          n_scifi_layers - scifi_removed,
          minSciFiLayersForOutlier);
      else
        oc.skip();
    }
    for (layer = 6; layer < 12; layer++) {
      hit_counter = ((hit_map1 >> (4 * (layer - 6))) & 0xf);
      PredictStateT(scifi_track, dev_T_lay, dev_T_pars, x, C, tI, layer, hit_counter);
      if (oc.should_process(hit_counter != 0xf))
        oc.consider(
          UpdateStateT(scifi_track, dev_T_lay, x, C, tI, hit_counter, layer),
          n_scifi_layers - scifi_removed,
          minSciFiLayersForOutlier);
      else
        oc.skip();
    }
    // Extrapolate to R2 states
    Vector5 x_tmp = x;
    ExtrapolateToR2(PAR_RICH2_F, RICH2_F_zTo, x_tmp, tI);
    r2_f_state = SimpleKalmanState(x_tmp[0], x_tmp[1], RICH2_F_zTo, x_tmp[2], x_tmp[3], x_tmp[4]);

    x_tmp = x;
    ExtrapolateToR2(PAR_RICH2_B, RICH2_B_zTo, x_tmp, tI);
    r2_b_state = SimpleKalmanState(x_tmp[0], x_tmp[1], RICH2_B_zTo, x_tmp[2], x_tmp[3], x_tmp[4]);
    oc.commit(tI);
    //------------------------------ End forward fit.

    // Set state and covariance for VELO-only backward fit
    tI.m_BestMomEst = x[4];
    x[0] = tI.m_RefStateForward[0];
    x[1] = tI.m_RefStateForward[1];
    x[2] = tI.m_RefStateForward[2];
    x[3] = tI.m_RefStateForward[3];
    tI.m_Lastz = endVeloZ;

    C = similarity_5_5(inverse(tI.m_RefPropForwardTotal), C);

    // get the RICH1 states
    x_tmp = x;
    ExtrapolateVR1(PAR_RICH1_F, RICH1_F_zTo, x_tmp, tI);
    r1_f_state = SimpleKalmanState(x_tmp[0], x_tmp[1], RICH1_F_zTo, x_tmp[2], x_tmp[3], x_tmp[4]);

    x_tmp = x;
    ExtrapolateVR1(PAR_RICH1_B, RICH1_B_zTo, x_tmp, tI);
    r1_b_state = SimpleKalmanState(x_tmp[0], x_tmp[1], RICH1_B_zTo, x_tmp[2], x_tmp[3], x_tmp[4]);

    const unsigned n_velo_hits2 = velo_track.number_of_hits();

    //------------------------------ Start backward fit.
    // Velo loop.
    // Update again on the hit in the last layer
    if (!oc.is_masked(n_velo_hits - 1)) {
      UpdateStateV(velo_track, -1, 0, x, C, tI);
    }
    for (unsigned i_hit = 1; i_hit < n_velo_hits; i_hit++) { // Velo hits are sorted from large z to small z
      PredictStateV(velo_track, dev_V_pars, i_hit, x, C, tI);
      if (!oc.is_masked(n_velo_hits - 1 - i_hit)) {
        UpdateStateV(velo_track, -1, i_hit, x, C, tI);
      }
    }
    //------------------------------ End backward fit.

    unsigned eff_ut_layers = n_ut_layers - ut_removed;
    unsigned eff_scifi_layers = n_scifi_layers - scifi_removed;
    MakeTrack(init_qop, x, C, tI, track, n_velo_hits2 - velo_removed, eff_ut_layers, eff_scifi_layers);

    // Straight line extrapolation to the closest point to the beamline.

    // The Rec version uses TrackMasterExtrapolator for this
    // step. Because that isn't available here, just use the same
    // propagation as the VELO-only Kalman Filter.
    propagate_to_beamline(track, dev_beamline, false);
  }

  __host__ __device__ void
  set_result(const unsigned track_number, const ParKalmanFilter::FittedTrack& track, Velo::Consolidated::States& states)
  {
    states.x(track_number) = track.state[0];
    states.y(track_number) = track.state[1];
    states.tx(track_number) = track.state[2];
    states.ty(track_number) = track.state[3];
    states.qop(track_number) = track.state[4];
    states.z(track_number) = track.z;

    states.c00(track_number) = track.cov(0, 0);
    // states.c10(track_number) = track.cov(1, 0);
    states.c11(track_number) = track.cov(1, 1);
    states.c20(track_number) = track.cov(2, 0);
    // states.c21(track_number) = track.cov(2, 1);
    states.c22(track_number) = track.cov(2, 2);
    states.c31(track_number) = track.cov(3, 1);
    states.c33(track_number) = track.cov(3, 3);
    states.chi2(track_number) = track.chi2;
    states.ndof(track_number) = track.ndof;
    return;
  }
} // End namespace ParKalmanFilter.

//----------------------------------------------------------------------
// Refit outlier tracks excluding their worst hit.
// This kernel only processes tracks identified as outliers during the main fit.
__global__ void refit_outliers(
  kalman_filter::Parameters parameters,
  const float magnet_polarity,
  const ParKalmanFilter::KalmanParametrizations* dev_kalman_params,
  const KalmanFloat outlier_threshold,
  const unsigned max_outlier_iterations)
{
  const KalmanFloat magSign = magnet_polarity;
  const unsigned n_outlier_tracks = *parameters.dev_n_outlier_tracks;
  const unsigned total_tracks = parameters.dev_long_track_view.size();
  const Allen::Views::Physics::LongTrack* track_base = parameters.dev_long_track_view.data();
  Velo::Consolidated::States kalman_states {parameters.dev_kalman_fit_results, total_tracks};

  for (unsigned consolidated_track_idx = blockIdx.x * blockDim.x + threadIdx.x;
       consolidated_track_idx < n_outlier_tracks;
       consolidated_track_idx += blockDim.x * gridDim.x) {
    const unsigned track_idx = parameters.dev_outlier_track_indices[consolidated_track_idx];
    auto& kf_track = parameters.dev_kf_tracks[track_idx];

    const auto& long_track = track_base[track_idx];
    const auto ut_track_ptr = long_track.track_segment_ptr<Allen::Views::Physics::Track::segment::ut>();
    const auto ut_track = ut_track_ptr != nullptr ? *ut_track_ptr : Allen::Views::UT::Consolidated::Track();

    for (unsigned outlier_pass = 0; outlier_pass < max_outlier_iterations; ++outlier_pass) {
      // Skip tracks that have already converged (no outlier above threshold)
      if (kf_track.worst_chi2 <= outlier_threshold) continue;
      if (kf_track.worst_hit_global_id >= 64) continue;

      const KalmanFloat current_chi2 = kf_track.chi2;
      const unsigned current_ndof = kf_track.ndof;

      // Accumulate skip mask: add the worst hit from the previous pass
      uint64_t new_mask = kf_track.skip_mask | (uint64_t(1) << kf_track.worst_hit_global_id);

      // Refit the track excluding all accumulated bad hits
      ParKalmanFilter::FittedTrack refit_track;
      SimpleKalmanState r1_f, r1_b, r2_f, r2_b;
      ParKalmanFilter::fit(
        long_track.track_segment<Allen::Views::Physics::Track::segment::velo>(),
        ut_track,
        long_track.track_segment<Allen::Views::Physics::Track::segment::scifi>(),
        kf_track.first_qop,
        dev_kalman_params,
        refit_track,
        r1_f,
        r1_b,
        r2_f,
        r2_b,
        magSign,
        outlier_threshold,
        new_mask);

      refit_track.skip_mask = new_mask;

      const bool improved =
        current_ndof > 0 && refit_track.ndof > 0 && refit_track.chi2 * current_ndof < current_chi2 * refit_track.ndof;

      if (improved) {
        kf_track = refit_track;
        ParKalmanFilter::set_result(track_idx, refit_track, kalman_states);
        parameters.dev_kalman_R1_F_view[track_idx] = r1_f;
        parameters.dev_kalman_R1_B_view[track_idx] = r1_b;
        parameters.dev_kalman_R2_F_view[track_idx] = r2_f;
        parameters.dev_kalman_R2_B_view[track_idx] = r2_b;
      }
      else {
        // Mark as converged so further passes do not recompute the same rejected refit.
        kf_track.worst_chi2 = 0;
      }
    }
  }
}

//----------------------------------------------------------------------
// Kalman filter kernel.
__global__ void kalman_filter::kalman_filter(
  kalman_filter::Parameters parameters,
  const float magnet_polarity,
  const ParKalmanFilter::KalmanParametrizations* dev_kalman_params,
  const float outlier_chi2_threshold)
{
  const KalmanFloat magSign = magnet_polarity;

  // Base pointer for the list of all tracks (contiguous in memory), regardless of events boundaries
  const Allen::Views::Physics::LongTrack* track_base = parameters.dev_long_track_view.data();
  const unsigned total_number_of_tracks = parameters.dev_long_track_view.size();

  Velo::Consolidated::States kalman_states {parameters.dev_kalman_fit_results, total_number_of_tracks};

  for (unsigned track_id = blockIdx.x * blockDim.x + threadIdx.x; track_id < total_number_of_tracks;
       track_id += blockDim.x * gridDim.x) {
    // Prepare fit input.
    const Allen::Views::Physics::LongTrack& long_track = track_base[track_id];
    // subdetector tracks which will give us access to the hits.
    const auto velo_track = long_track.track_segment<Allen::Views::Physics::Track::segment::velo>();
    const auto ut_track_ptr = long_track.track_segment_ptr<Allen::Views::Physics::Track::segment::ut>();
    const auto ut_track = ut_track_ptr != nullptr ? *ut_track_ptr : Allen::Views::UT::Consolidated::Track();
    const auto scifi_track = long_track.track_segment<Allen::Views::Physics::Track::segment::scifi>();
    const KalmanFloat init_qop = (KalmanFloat) long_track.qop(); // Tracking estimate of qop
    ParKalmanFilter::FittedTrack kalman_track;
    SimpleKalmanState r1_f_state;
    SimpleKalmanState r1_b_state;
    SimpleKalmanState r2_f_state;
    SimpleKalmanState r2_b_state;
    fit(
      velo_track,
      ut_track,
      scifi_track,
      init_qop,
      dev_kalman_params,
      kalman_track,
      r1_f_state,
      r1_b_state,
      r2_f_state,
      r2_b_state,
      magSign,
      outlier_chi2_threshold);
    kalman_track.skip_mask = 0; // No hits removed yet; refit_outliers accumulates into this
    set_result(track_id, kalman_track, kalman_states);
    parameters.dev_kf_tracks[track_id] = kalman_track;
    parameters.dev_kalman_R1_F_view[track_id] = r1_f_state;
    parameters.dev_kalman_R1_B_view[track_id] = r1_b_state;
    parameters.dev_kalman_R2_F_view[track_id] = r2_f_state;
    parameters.dev_kalman_R2_B_view[track_id] = r2_b_state;

    // If outlier removal is enabled, add tracks with outliers to the compact index list
    if (outlier_chi2_threshold > 0.0f && kalman_track.worst_chi2 > outlier_chi2_threshold) {
      auto idx = atomicAdd(&parameters.dev_n_outlier_tracks[0], 1u);
      parameters.dev_outlier_track_indices[idx] = track_id;
    }
  }
}
