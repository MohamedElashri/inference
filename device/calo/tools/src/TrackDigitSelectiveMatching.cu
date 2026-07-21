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
#include "TrackDigitSelectiveMatching.cuh"
#include "EcalScan.cuh"

INSTANTIATE_ALGORITHM_WITH_ID(
  track_digit_selective_matching::track_digit_selective_matching_t<Allen::Views::Physics::MultiEventLongTracks>,
  "track_digit_selective_matching_long_t")
INSTANTIATE_ALGORITHM_WITH_ID(
  track_digit_selective_matching::track_digit_selective_matching_t<Allen::Views::Physics::MultiEventDownstreamTracks>,
  "track_digit_selective_matching_downstream_t")

template<typename MultiEventTracks>
void track_digit_selective_matching::track_digit_selective_matching_t<MultiEventTracks>::set_arguments_size(
  ArgumentReferences<Parameters<MultiEventTracks>> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  using P = Parameters<MultiEventTracks>;
  const unsigned number_of_tracks = first<typename P::host_number_of_reconstructed_scifi_tracks_t>(arguments);
  const unsigned number_of_digits = first<typename P::host_ecal_number_of_digits_t>(arguments);

  set_size<typename P::dev_matched_ecal_energy_t>(arguments, number_of_tracks);
  set_size<typename P::dev_matched_ecal_digits_t>(arguments, number_of_tracks);
  set_size<typename P::dev_matched_ecal_digits_size_t>(arguments, number_of_tracks);
  set_size<typename P::dev_track_inEcalAcc_t>(arguments, number_of_tracks);
  set_size<typename P::dev_track_Eop_t>(arguments, number_of_tracks);
  set_size<typename P::dev_track_Eop3x3_t>(arguments, number_of_tracks);
  set_size<typename P::dev_delta_barycenter_t>(arguments, number_of_tracks);
  set_size<typename P::dev_delta_barycenter_x_t>(arguments, number_of_tracks);
  set_size<typename P::dev_region_t>(arguments, number_of_tracks);
  set_size<typename P::dev_delta_barycenter_y_t>(arguments, number_of_tracks);
  set_size<typename P::dev_dispersion_x_t>(arguments, number_of_tracks);
  set_size<typename P::dev_dispersion_y_t>(arguments, number_of_tracks);
  set_size<typename P::dev_dispersion_xy_t>(arguments, number_of_tracks);
  set_size<typename P::dev_track_local_max_t>(arguments, number_of_tracks);
  set_size<typename P::dev_track_isElectron_t>(arguments, number_of_tracks);
  set_size<typename P::dev_ecal_digits_isTrackMatched_t>(arguments, number_of_digits);
}

template<typename MultiEventTracks>
void track_digit_selective_matching::track_digit_selective_matching_t<MultiEventTracks>::operator()(
  const ArgumentReferences<Parameters<MultiEventTracks>>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  Allen::Context const& context) const
{
  using P = Parameters<MultiEventTracks>;

  Allen::memset_async<typename P::dev_ecal_digits_isTrackMatched_t>(arguments, 0, context);

  const auto kernel = track_digit_selective_matching<MultiEventTracks>;
  global_function(kernel)(dim3(size<typename P::dev_event_list_t>(arguments)), m_block_dim, context)(
    arguments, constants.dev_ecal_geometry);
}

template<typename MultiEventTracks>
__global__ void track_digit_selective_matching::track_digit_selective_matching(
  track_digit_selective_matching::Parameters<MultiEventTracks> parameters,
  const char* raw_ecal_geometry)
{
  const MultiEventTracks* dev_long_tracks_view = parameters.dev_tracks_view;

  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  // Long tracks.
  const auto long_tracks = dev_long_tracks_view->container(event_number);

  const unsigned n_long_tracks = long_tracks.size();
  const unsigned event_offset = long_tracks.offset();

  // Get ECAL digits
  auto ecal_geometry = CaloGeometry(raw_ecal_geometry);
  const unsigned digits_offset = parameters.dev_ecal_digits_offsets[event_number];
  auto const* digits = parameters.dev_ecal_digits + digits_offset;
  auto* digits_isTrackMatched = parameters.dev_ecal_digits_isTrackMatched + digits_offset;

  // Loop over the long tracks in parallel
  for (unsigned track_index = threadIdx.x; track_index < n_long_tracks; track_index += blockDim.x) {
    const auto long_track = long_tracks.track(track_index);
    // SciFi state
    const auto& scifi_state = parameters.dev_scifi_states[event_offset + track_index];

    // Get z positions of intersection of the track and front, showermax and back planes
    float z_front = ecal_geometry.getZFromTrackToCaloplaneIntersection(scifi_state, 0);
    float z_showermax = ecal_geometry.getZFromTrackToCaloplaneIntersection(scifi_state, 1);
    float z_back = ecal_geometry.getZFromTrackToCaloplaneIntersection(scifi_state, 2);

    // Delta z traversed inside the ECAL
    float ecal_delta_z = z_back - z_front;

    // Define 6 z positions along the track inside the ECAL
    constexpr unsigned N_ecal_positions = 6;
    const float ecal_positions[N_ecal_positions] = {
      z_front,
      z_showermax,
      z_front + 0.25f * ecal_delta_z,
      z_front + 0.5f * ecal_delta_z,
      z_front + 0.75f * ecal_delta_z,
      z_back};

    std::array<unsigned, N_ecal_positions> digit_indices = {9999, 9999, 9999, 9999, 9999, 9999};
    unsigned N_matched_digits {0};
    bool inAcc = false;
    float sum_cell_E = 0.f, xbar = 0.f, ybar = 0.f, xdispersion = 0.f, ydispersion = 0.f, xydispersion = 0.f;
    float barycenter_E = 0.f;
    float ecal_z;
    bool localmax = false;
    // Loop over six z positions in the ECAL to check which cell is traversed by the track
    ecal_scan_local_max(
      N_ecal_positions,
      ecal_positions,
      scifi_state,
      ecal_geometry,
      inAcc,
      digits,
      N_matched_digits,
      sum_cell_E,
      digit_indices,
      localmax);

    for (unsigned j = 0; j < N_matched_digits; ++j) {
      const unsigned& digit_index = digit_indices[j];
      digits_isTrackMatched[digit_index] = true;
    }

    int region = 0;
    cluster_shape_scan(
      N_ecal_positions,
      ecal_positions,
      scifi_state,
      ecal_geometry,
      inAcc,
      digits,
      barycenter_E,
      xbar,
      ybar,
      xdispersion,
      ydispersion,
      xydispersion,
      ecal_z,
      region);

    const float dz_ecal = z_showermax - scifi_state.z();
    float xV = scifi_state.x() + scifi_state.tx() * dz_ecal;
    float yV = scifi_state.y() + scifi_state.ty() * dz_ecal;
    const auto delta2 = (xV - xbar) * (xV - xbar) + (yV - ybar) * (yV - ybar);
    parameters.dev_matched_ecal_energy[track_index + event_offset] = sum_cell_E;
    parameters.dev_matched_ecal_digits[track_index + event_offset] = digit_indices;
    parameters.dev_matched_ecal_digits_size[track_index + event_offset] = N_matched_digits;
    parameters.dev_track_inEcalAcc[track_index + event_offset] = inAcc;
    const float gamma = N_matched_digits > 0 ? ecal_geometry.getGamma(digit_indices[0]) : 0.f;
    const float eop_estimate = (sum_cell_E + gamma) * fabsf(long_track.qop());
    const float gamma_corrected = gamma * ((eop_estimate < 1) ? eop_estimate : 1.f);
    parameters.dev_track_Eop[track_index + event_offset] = (sum_cell_E + gamma_corrected) * fabsf(long_track.qop());
    parameters.dev_track_Eop3x3[track_index + event_offset] =
      (barycenter_E + gamma_corrected) * fabsf(long_track.qop());
    parameters.dev_track_isElectron[track_index + event_offset] =
      parameters.dev_track_Eop[track_index + event_offset] > 0.7f;
    parameters.dev_delta_barycenter_x[track_index + event_offset] = (xV - xbar);
    parameters.dev_delta_barycenter_y[track_index + event_offset] = (yV - ybar);
    parameters.dev_delta_barycenter[track_index + event_offset] = delta2;
    parameters.dev_region[track_index + event_offset] = region;
    parameters.dev_dispersion_x[track_index + event_offset] = xdispersion;
    parameters.dev_dispersion_y[track_index + event_offset] = ydispersion;
    parameters.dev_dispersion_xy[track_index + event_offset] = xydispersion;
    parameters.dev_track_local_max[track_index + event_offset] = localmax;
  }
}

// These declarations need to be at the end of the file otherwise clang don't export them:
template struct track_digit_selective_matching::track_digit_selective_matching_t<
  Allen::Views::Physics::MultiEventLongTracks>;
template struct track_digit_selective_matching::track_digit_selective_matching_t<
  Allen::Views::Physics::MultiEventDownstreamTracks>;
