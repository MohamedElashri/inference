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
#include "RichValidator.cuh"
#include "CopyTrackParameters.cuh"

INSTANTIATE_ALGORITHM(rich_validator::rich_validator_t)

template<unsigned richIdx>
__global__ void rich_validator_k(rich_validator::Parameters parameters)
{
  const unsigned event_number = blockIdx.x;
  const auto event_long_tracks = parameters.dev_multi_event_long_tracks_view->container(event_number);
  const auto endvelo_states = parameters.dev_velo_states_view[event_number];
  const unsigned offset_long_tracks = event_long_tracks.offset();
  const Allen::Rich::ParticleIDType* pid = parameters.dev_pid + offset_long_tracks;
  Checker::Track* rich_checker_tracks_event = parameters.dev_rich_checker_tracks + offset_long_tracks;

  prepare_long_tracks(event_long_tracks, endvelo_states, rich_checker_tracks_event);

  __syncthreads();

  for (unsigned i = threadIdx.x; i < event_long_tracks.size(); i += blockDim.x) {
    unsigned trackId = offset_long_tracks + i;
    const auto hypos = parameters.dev_rich_hypos[trackId];

    UNROLL(Allen::Rich::NRealParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NRealParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      rich_checker_tracks_event[i].ckThetaExp[hypo] = hypos.ckTheta[hypo];
    }

    rich_checker_tracks_event[i].pid = pid[i];
  }
}

void rich_validator::rich_validator_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_rich_checker_tracks_t>(arguments, first<host_number_of_reconstructed_long_tracks_t>(arguments));
}

void rich_validator::rich_validator_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  const unsigned richValue = m_current_rich;
  const auto& rich_validator_kernel = richValue == 1 ? rich_validator_k<0> : rich_validator_k<1>;
  global_function(rich_validator_kernel)(first<host_number_of_events_t>(arguments), 256, context)(arguments);

  const auto event_list = make_host_buffer<dev_event_list_t>(arguments, context);
  const auto rich_tracks_for_checker = make_host_buffer<dev_rich_checker_tracks_t>(arguments, context);
  const auto event_tracks_offsets = make_host_buffer<dev_offsets_long_tracks_t>(arguments, context);
  const auto rich_photons = make_host_buffer<dev_rich_photons_t>(arguments, context);
  const auto rich_photons_offsets = make_host_buffer<dev_offsets_rich_photons_t>(arguments, context);
  std::vector<Checker::Tracks> tracks;
  std::vector<std::vector<std::vector<Allen::Rich::PhotonReco::Photon>>> photons;
  tracks.resize(event_list.size());
  photons.resize(event_list.size());
  for (size_t i = 0; i < event_list.size(); ++i) {
    const auto evnum = event_list[i];
    const auto event_offset = event_tracks_offsets[evnum];
    const auto n_tracks = event_tracks_offsets[evnum + 1] - event_offset;
    std::vector<Checker::Track> event_tracks = {
      rich_tracks_for_checker.begin() + event_offset, rich_tracks_for_checker.begin() + event_offset + n_tracks};
    tracks[i] = event_tracks;

    photons[i].resize(n_tracks);
    for (size_t j = 0; j < n_tracks; j++) {
      unsigned start = rich_photons_offsets[event_offset + j];
      unsigned end = rich_photons_offsets[event_offset + j + 1];
      std::vector<Allen::Rich::PhotonReco::Photon> track_photons = {
        rich_photons.begin() + start, rich_photons.begin() + end};
      photons[i][j] = track_photons;
    }
  }

  auto& checker = runtime_options.checker_invoker->checker<TrackCheckerRich>(name(), m_root_output_filename);
  checker.accumulate(*first<host_mc_events_t>(arguments), tracks, photons, event_list);
}
