/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "VeloLargeClustersLine.cuh"
#include "Event/ODIN.h"
#include "ODINBank.cuh"

// Explicit instantiation
INSTANTIATE_LINE(velo_large_clusters_line::velo_large_clusters_line_t, velo_large_clusters_line::Parameters)

__device__ std::tuple<const float, const Allen::Views::Velo::Consolidated::Track>
velo_large_clusters_line::velo_large_clusters_line_t::get_input(
  const Parameters& parameters,
  const unsigned event_number,
  const unsigned i)
{
  const auto track = parameters.dev_velo_tracks_view[event_number].track(i);
  const auto state = parameters.dev_velo_states_view[event_number].state(i);
  const float eta = state.eta();
  return std::forward_as_tuple(eta, track);
}

__device__ bool velo_large_clusters_line::velo_large_clusters_line_t::select(
  const Parameters&,
  const DeviceProperties& properties,
  std::tuple<const float, const Allen::Views::Velo::Consolidated::Track> input)
{
  const auto [eta, track] = input;
  if (eta <= properties.min_eta) {
    return false;
  }
  unsigned large_cluster_count = 0;
  for (unsigned i = 0; i < track.number_of_hits(); ++i) {
    if (track.hit(i).cluster_size() >= properties.min_cluster_size) {
      ++large_cluster_count;
    }
  }
  return large_cluster_count >= properties.min_n_hits;
}
