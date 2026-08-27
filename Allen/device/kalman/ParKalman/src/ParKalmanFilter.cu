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
  global_function(kalman_filter)(dim3(_gridDim), m_block_dim, context)(
    arguments, constants.magnet_polarity, constants.dev_kalman_params);

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
    const float* dev_UT_lay,
    const float* dev_T_lay,
    const float* dev_V_pars,
    const float* dev_VUT_pars,
    const float* dev_UT_pars,
    const float* dev_UTTF_pars,
    const float* dev_T_pars,
    const float* dev_TFT_pars,
    const float* dev_UTT_META,
    const KalmanFloat magSign)
  {
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
    CreateVeloSeedState(velo_track, n_velo_hits, x, C, tI);
    // sets the state at the position of the first (in z last in index) hit.
    // initialises a very large covariance matrix
    tI.m_chi2V = 0;
    tI.m_chi2T = 0;
    tI.m_chi2UT = 0;

    //------------------------------ Start forward fit.
    // Velo loop.
    // Update on the first hit
    UpdateStateV(velo_track, 1, n_velo_hits - 1, x, C, tI);
    // have to iterate down from `n_velo_hits - 1` to `0`
    for (unsigned i_hit = 1; i_hit < n_velo_hits; i_hit++) {
      PredictStateV(velo_track, dev_V_pars, n_velo_hits - 1 - i_hit, x, C, tI);
      UpdateStateV(velo_track, 1, n_velo_hits - 1 - i_hit, x, C, tI);
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

    //  and update the first ut layer if there is a hit.
    if (hit_counter != 0xf) {
      UpdateStateUT(ut_track, x, C, tI, hit_counter);
    }

    // iterater over the remaining UT layers
    for (layer = 1; layer < 4; layer++) {
      hit_counter = ((hit_map0 >> (layer * 4)) & 0xf);
      PredictStateUT(ut_track, dev_UT_lay, dev_UT_pars, x, C, tI, layer, hit_counter);
      if (hit_counter != 0xf) {
        UpdateStateUT(ut_track, x, C, tI, hit_counter);
      }
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
    if (hit_counter != 0xf) {
      UpdateStateT(scifi_track, dev_T_lay, x, C, tI, hit_counter, layer);
    }
    for (layer = 1; layer < 6; layer++) {
      hit_counter = ((hit_map0 >> (4 * layer)) & 0xf);
      PredictStateT(scifi_track, dev_T_lay, dev_T_pars, x, C, tI, layer, hit_counter);
      if (hit_counter != 0xf) {
        UpdateStateT(scifi_track, dev_T_lay, x, C, tI, hit_counter, layer);
      }
    }
    for (layer = 6; layer < 12; layer++) {
      hit_counter = ((hit_map1 >> (4 * (layer - 6))) & 0xf);
      PredictStateT(scifi_track, dev_T_lay, dev_T_pars, x, C, tI, layer, hit_counter);
      if (hit_counter != 0xf) {
        UpdateStateT(scifi_track, dev_T_lay, x, C, tI, hit_counter, layer);
      }
    }
    // Extrapolate to R2 states
    Vector5 x_tmp = x;
    ExtrapolateToR2(PAR_RICH2_F, RICH2_F_zTo, x_tmp, tI);
    r2_f_state = SimpleKalmanState(x_tmp[0], x_tmp[1], RICH2_F_zTo, x_tmp[2], x_tmp[3], x_tmp[4]);

    x_tmp = x;
    ExtrapolateToR2(PAR_RICH2_B, RICH2_B_zTo, x_tmp, tI);
    r2_b_state = SimpleKalmanState(x_tmp[0], x_tmp[1], RICH2_B_zTo, x_tmp[2], x_tmp[3], x_tmp[4]);
    __syncthreads();
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
    UpdateStateV(velo_track, -1, 0, x, C, tI);
    for (unsigned i_hit = 1; i_hit < n_velo_hits; i_hit++) { // Velo hits are sorted from large z to small z
      PredictStateV(velo_track, dev_V_pars, i_hit, x, C, tI);
      UpdateStateV(velo_track, -1, i_hit, x, C, tI);
    }
    //------------------------------ End backward fit.

    MakeTrack(init_qop, x, C, tI, track, n_velo_hits2, n_ut_layers, n_scifi_layers);

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
// Kalman filter kernel.
__global__ void kalman_filter::kalman_filter(
  kalman_filter::Parameters parameters,
  const float magnet_polarity,
  const ParKalmanFilter::KalmanParametrizations* dev_kalman_params)
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
      parkalman_shared::dev_UT_lay,
      parkalman_shared::dev_T_lay,
      parkalman_shared::dev_V_pars,
      parkalman_shared::dev_VUT_pars,
      parkalman_shared::dev_UT_pars,
      parkalman_shared::dev_UTTF_pars,
      parkalman_shared::dev_T_pars,
      parkalman_shared::dev_TFT_pars,
      parkalman_shared::dev_UTT_META,
      magSign);
    set_result(track_id, kalman_track, kalman_states);
    parameters.dev_kf_tracks[track_id] = kalman_track;
    parameters.dev_kalman_R1_F_view[track_id] = r1_f_state;
    parameters.dev_kalman_R1_B_view[track_id] = r1_b_state;
    parameters.dev_kalman_R2_F_view[track_id] = r2_f_state;
    parameters.dev_kalman_R2_B_view[track_id] = r2_b_state;
  }
}
