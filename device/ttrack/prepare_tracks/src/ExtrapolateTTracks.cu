/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration          *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// This
#include "ExtrapolateTTracks.cuh"

#include "SciFiDefinitions.cuh"
#include "PrefixSum.cuh"

#include "TTrackDefinitions.cuh"
#include "TrackTrajectory.cuh"

#include <fstream>

INSTANTIATE_ALGORITHM(extrapolate_ttracks::extrapolate_ttracks_t);

void extrapolate_ttracks::extrapolate_ttracks_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const auto n_scifi_tracks = first<host_number_of_scifi_tracks_t>(arguments);
  const auto soa_size = Trajectories::StateVar::COUNT * TTracks::Constants::num_reference_states * n_scifi_tracks;

  set_size<dev_extrapolated_ttrack_states_t>(arguments, soa_size);
}

void extrapolate_ttracks::extrapolate_ttracks_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  const int block_dim = 256;
  const int grid_dim = (first<host_number_of_scifi_tracks_t>(arguments) + block_dim - 1) / block_dim;

  global_function(extrapolate_ttracks)(grid_dim, block_dim, context)(arguments, *constants.magnetic_field);
}

__global__ void extrapolate_ttracks::extrapolate_ttracks(Parameters parameters, const MagneticField::Magfield field)
{
  namespace Constants = TTracks::Constants;
  namespace Tracks = TTracks::Tracks;

  const auto n_states = parameters.dev_offsets_scifi_tracks[parameters.dev_number_of_events[0]];

  Trajectories::GlobalMemoryManager output(parameters.dev_extrapolated_ttrack_states, n_states);
  Tracks::PackedTrackStateConst input_states(parameters.dev_scifi_track_states, n_states);

  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n_states; i += blockDim.x * gridDim.x) {
    // const auto input = parameters.dev_scifi_track_states[i];
    const auto qop = parameters.dev_scifi_track_qop[i];

    using Var = Tracks::PackedTrackStateVar;
    Extrapolators::State baseState {
      // input.x(), input.y(), input.z(), input.tx(), input.ty(),
      input_states.get<Var::x>(i),
      input_states.get<Var::y>(i),
      input_states.get<Var::z>(i),
      input_states.get<Var::tx>(i),
      input_states.get<Var::ty>(i),
      qop * Extrapolators::c_light * Extrapolators::eplus // Don't forget to do this conversion !
    };

    for (int s = Constants::num_reference_states - 1; s >= 0; s--) {
      Extrapolators::RungeKuttaNystromExtrapolator::propagate(baseState, Constants::z_reference_states[s], field);
      output.set_state(i, s, baseState);
    }
  }
}
