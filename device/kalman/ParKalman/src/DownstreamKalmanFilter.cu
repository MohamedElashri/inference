/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "DownstreamKalmanFilter.cuh"
#include "ParKalmanSharedConstants.cuh"
#include "States.cuh"

INSTANTIATE_ALGORITHM(downstream_kalman_filter::downstream_kalman_filter_t)

void downstream_kalman_filter::downstream_kalman_filter_t::update(const Constants& constants) const
{
  // Load shared ParKF parameters (shared with kalman_filter)
  parkalman_shared::update_shared_constants(constants);
}

void downstream_kalman_filter::downstream_kalman_filter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const auto number_of_tracks = first<host_number_of_downstream_tracks_t>(arguments);
  const auto number_of_events = first<host_number_of_events_t>(arguments);

  set_size<dev_downstream_kf_track_states_t>(arguments, number_of_tracks * Velo::Consolidated::States::size);
  set_size<dev_downstream_kf_tracks_t>(arguments, number_of_tracks);
  // Views (states)
  set_size<dev_downstream_kf_track_states_view_t>(arguments, number_of_events);
}

void downstream_kalman_filter::downstream_kalman_filter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  dim3 block_dim = m_block_dim;
  int _gridDim = (first<host_number_of_downstream_tracks_t>(arguments) + (block_dim.x) - 1) / (block_dim.x);

  global_function(downstream_kalman_filter)(dim3(_gridDim), m_block_dim, context)(
    arguments, constants.dev_magnet_polarity.data(), constants.dev_kalman_params);

  // Create KalmanStates views (per-event structure matching input, so we don't have to touch the Particle maker)
  const unsigned n_events = first<host_number_of_events_t>(arguments);
  _gridDim = (n_events + (block_dim.x) - 1) / (block_dim.x);
  global_function(consolidate_kalman_tracks)(dim3(first<host_number_of_events_t>(arguments)), m_block_dim, context)(
    arguments, n_events);
}

namespace {
  using namespace ParKalmanFilter;

  //----------------------------------------------------------------------
  // Run the Kalman filter for downstream tracks (UT + SciFi only).
  __device__ void fit_downstream(
    const Allen::Views::UT::Consolidated::Track& ut_track,
    const Allen::Views::SciFi::Consolidated::Track& scifi_track,
    const MiniState ut_mini_state,
    const KalmanFloat init_qop,
    const KalmanParametrizations* kalman_params,
    FittedTrack& track,
    const float* dev_UT_lay,
    const float* dev_T_lay,
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

    // Run the fit.
    Vector5 x;
    SymMatrix5x5 C;

    // Initialize slopes from UT CompactState
    x[2] = ut_mini_state.m_tx;
    x[3] = ut_mini_state.m_ty;
    x[4] = init_qop;

    // Initialize covariance (larger values as no Velo constraint)
    C.SetZero();
    C(0, 0) = 100.0f;                             // x uncertainty
    C(1, 1) = 100.0f;                             // y uncertainty
    C(2, 2) = 0.01f;                              // tx uncertainty
    C(3, 3) = 0.01f;                              // ty uncertainty
    C(4, 4) = ((KalmanFloat) 0.09) * x(4) * x(4); // qop uncertainty

    tI.m_chi2V = 0; // No Velo contribution
    tI.m_chi2T = 0;
    tI.m_chi2UT = 0;

    //------------------------------ Start forward fit (UT + SciFi only).

    // Create UT hit map
    unsigned n_ut_layers = 0;
    unsigned hit_mapUT = make_ut_hitmap(ut_track, n_ut_layers);

    unsigned layer;
    unsigned hit_counter;
    // find the first hit and set the state based on this
    layer = (__ffs(~hit_mapUT) - 1) / 4;
    hit_counter = ((hit_mapUT >> (layer * 4)) & 0xf);
    KalmanFloat dxDy = (KalmanFloat) ut_track.hit(hit_counter).dxDy();
    KalmanFloat y0 = (KalmanFloat) ut_track.hit(hit_counter).yBegin();
    KalmanFloat y1 = (KalmanFloat) ut_track.hit(hit_counter).yEnd();
    KalmanFloat x0 = (KalmanFloat) ut_track.hit(hit_counter).xAtYEq0() + y0 * dxDy;
    KalmanFloat x1 = (KalmanFloat) ut_track.hit(hit_counter).xAtYEq0() + y1 * dxDy;
    x[0] = (x0 + x1) * 0.5f;
    x[1] = (y0 + y1) * 0.5f;
    tI.m_Lastz = (KalmanFloat) ut_track.hit(hit_counter).zAtYEq0();
    UpdateStateUT(ut_track, x, C, tI, hit_counter);

    // Iterate over the remaining UT layers
    for (layer = layer + 1; layer < 4; layer++) {
      hit_counter = ((hit_mapUT >> (layer * 4)) & 0xf);
      PredictStateUT(ut_track, dev_UT_lay, dev_UT_pars, x, C, tI, layer, hit_counter, false);
      if (hit_counter != 0xf) {
        UpdateStateUT(ut_track, x, C, tI, hit_counter);
      }
    }

    tI.m_RefPropForwardTotal.SetDiag();
    tI.m_RefStateForward[0] = x[0];
    tI.m_RefStateForward[1] = x[1];
    tI.m_RefStateForward[2] = x[2];
    tI.m_RefStateForward[3] = x[3];

    layer = 3;
    PredictStateUTT(dev_UT_pars, dev_TFT_pars, dev_UTTF_pars, dev_UTT_META, dev_T_lay, kalman_params, x, C, tI, layer);

    // Create SciFi hitmaps
    unsigned hit_mapT0, hit_mapT1;
    unsigned n_scifi_layers = 0;
    make_scifi_hitmaps(scifi_track, hit_mapT0, hit_mapT1, n_scifi_layers);

    // Update first SciFi layer if there is a hit
    hit_counter = (hit_mapT0 & 0xf);
    layer = 0;
    if (hit_counter != 0xf) {
      UpdateStateT(scifi_track, dev_T_lay, x, C, tI, hit_counter, layer);
    }

    // Process remaining SciFi layers
    for (layer = 1; layer < 6; layer++) {
      hit_counter = ((hit_mapT0 >> (4 * layer)) & 0xf);
      PredictStateT(scifi_track, dev_T_lay, dev_T_pars, x, C, tI, layer, hit_counter);
      if (hit_counter != 0xf) {
        UpdateStateT(scifi_track, dev_T_lay, x, C, tI, hit_counter, layer);
      }
    }
    for (layer = 6; layer < 12; layer++) {
      hit_counter = ((hit_mapT1 >> (4 * (layer - 6))) & 0xf);
      PredictStateT(scifi_track, dev_T_lay, dev_T_pars, x, C, tI, layer, hit_counter);
      if (hit_counter != 0xf) {
        UpdateStateT(scifi_track, dev_T_lay, x, C, tI, hit_counter, layer);
      }
    }
    //------------------------------ End forward fit.

    // Set state and covariance for UT backward fit
    tI.m_BestMomEst = x[4];
    x[0] = tI.m_RefStateForward[0];
    x[1] = tI.m_RefStateForward[1];
    x[2] = tI.m_RefStateForward[2];
    x[3] = tI.m_RefStateForward[3];
    tI.m_chi2UT = 0; // backwards pass gives the better chi2

    C = similarity_5_5(inverse(tI.m_RefPropForwardTotal), C);

    //------------------------------ Start backward fit (UT only).
    // UT backwards loop:
    // last layer update is used.
    layer = 3;
    hit_counter = ((hit_mapUT >> (4 * (layer))) & 0xf);
    if (hit_counter != 0xf) {
      UpdateStateUT(ut_track, x, C, tI, hit_counter);
      tI.m_Lastz = ut_track.hit(hit_counter).zAtYEq0();
    }
    else {
      tI.m_Lastz = dev_UT_lay[layer];
    }

    for (layer = 6; layer >= 4; layer--) { // parameters are accessed with offset (layer - 1) * 12
      hit_counter = ((hit_mapUT >> ((layer - 4) * 4)) & 0xf);
      PredictStateUT(ut_track, dev_UT_lay, dev_UT_pars, x, C, tI, layer, hit_counter, false);
      if (hit_counter != 0xf) {
        UpdateStateUT(ut_track, x, C, tI, hit_counter);
      }
    }
    //------------------------------ End backward fit.

    // Update best momentum estimate
    tI.m_BestMomEst = x[4];

    // Create final track (velo_hits = 0)
    MakeTrack(init_qop, x, C, tI, track, 0, n_ut_layers, n_scifi_layers);
  }
} // anonymous namespace

//----------------------------------------------------------------------
// Downstream Kalman filter kernel.
__global__ void downstream_kalman_filter::downstream_kalman_filter(
  downstream_kalman_filter::Parameters parameters,
  const float* dev_magnet_polarity,
  const ParKalmanFilter::KalmanParametrizations* dev_kalman_params)
{
  const KalmanFloat magSign = dev_magnet_polarity[0];

  const unsigned total_number_of_tracks = parameters.dev_downstream_track_view.size();

  // Get tracks and states views
  const Allen::Views::Physics::DownstreamTrack* track_base = parameters.dev_downstream_track_view.data();
  // Wrap input/output packed states
  Velo::Consolidated::ConstStates input_states(parameters.dev_downstream_track_states, total_number_of_tracks);
  Velo::Consolidated::States midUT_states(parameters.dev_downstream_kf_track_states, total_number_of_tracks);

  // Loop over all downstream tracks
  for (unsigned track_id = blockIdx.x * blockDim.x + threadIdx.x; track_id < total_number_of_tracks;
       track_id += blockDim.x * gridDim.x) {
    // const unsigned global_track_id = tracks_offset + track_index;
    // Get track from view
    const Allen::Views::Physics::DownstreamTrack& downstream_track = track_base[track_id];

    // Extract UT and SciFi track segments
    const auto ut_track = downstream_track.track_segment<Allen::Views::Physics::Track::segment::ut>();
    const auto scifi_track = downstream_track.track_segment<Allen::Views::Physics::Track::segment::scifi>();

    // Get state and qop (from polynomial fit)
    const float init_qop = (KalmanFloat) input_states.qop(track_id);
    const MiniState ut_mini_state = input_states.get(track_id); // no need to drag all these zeros around

    // Create temporary FittedTrack
    ParKalmanFilter::FittedTrack kalman_track;

    // Run the Kalman filter
    fit_downstream(
      ut_track,
      scifi_track,
      ut_mini_state,
      init_qop,
      dev_kalman_params,
      kalman_track,
      parkalman_shared::dev_UT_lay,
      parkalman_shared::dev_T_lay,
      parkalman_shared::dev_UT_pars,
      parkalman_shared::dev_UTTF_pars,
      parkalman_shared::dev_T_pars,
      parkalman_shared::dev_TFT_pars,
      parkalman_shared::dev_UTT_META,
      magSign);

    parameters.dev_downstream_kf_tracks[track_id] = kalman_track;

    // make midUT a copy of final-fit state
    midUT_states.x(track_id) = input_states.x(track_id);
    midUT_states.y(track_id) = input_states.y(track_id);
    midUT_states.tx(track_id) = input_states.tx(track_id);
    midUT_states.ty(track_id) = input_states.ty(track_id);
    midUT_states.qop(track_id) = kalman_track.state[4]; // update qop
    midUT_states.z(track_id) = input_states.z(track_id);
    midUT_states.c00(track_id) = input_states.c00(track_id);
    midUT_states.c11(track_id) = input_states.c11(track_id);
    midUT_states.c20(track_id) = input_states.c20(track_id);
    midUT_states.c22(track_id) = input_states.c22(track_id);
    midUT_states.c31(track_id) = input_states.c31(track_id);
    midUT_states.c33(track_id) = input_states.c33(track_id);
    midUT_states.chi2(track_id) = kalman_track.chi2; // forward pass
    midUT_states.ndof(track_id) = kalman_track.ndof; // chi2 / ndof
  }
}

//----------------------------------------------------------------------
// Create KalmanStates views - per-event structure matching consolidate output.
__global__ void downstream_kalman_filter::consolidate_kalman_tracks(
  downstream_kalman_filter::Parameters parameters,
  const unsigned n_events)
{
  for (unsigned event_id = blockIdx.x * blockDim.x + threadIdx.x; event_id < n_events;
       event_id += blockDim.x * gridDim.x) {
    parameters.dev_downstream_kf_track_states_view[event_id] = Allen::Views::Physics::KalmanStates {
      parameters.dev_downstream_kf_track_states, parameters.dev_downstream_track_offsets, event_id, n_events};
  }
}
