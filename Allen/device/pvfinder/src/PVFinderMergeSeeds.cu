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
#include "PVFinderMergeSeeds.cuh"
#include "PV_Definitions.cuh"

INSTANTIATE_ALGORITHM(pvfinder_merge_seeds::pvfinder_merge_seeds_t)

void pvfinder_merge_seeds::pvfinder_merge_seeds_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_zpeaks_t>(arguments, first<host_number_of_events_t>(arguments) * PV::max_number_vertices);
  set_size<dev_number_of_zpeaks_t>(arguments, first<host_number_of_events_t>(arguments));
}

void pvfinder_merge_seeds::pvfinder_merge_seeds_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_number_of_zpeaks_t>(arguments, 0, context);
  global_function(pvfinder_merge_seeds)(dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
    arguments, PVFinderConstants::KDE::z_min, PVFinderConstants::KDE::z_max);
}

// One block per event; the first thread copies (at most 2 x 32 seeds).
__global__ void
pvfinder_merge_seeds::pvfinder_merge_seeds(pvfinder_merge_seeds::Parameters parameters, const float z_min, const float z_max)
{
  if (threadIdx.x != 0) return;
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const unsigned offset = event_number * PV::max_number_vertices;
  float* out = parameters.dev_zpeaks + offset;
  unsigned n = 0;

  // PVFinder's seeds, all inside [z_min, z_max).
  const unsigned n_pvfinder = parameters.dev_pvfinder_number_of_zpeaks[event_number];
  for (unsigned i = 0; i < n_pvfinder && n < PV::max_number_vertices; ++i) {
    out[n++] = parameters.dev_pvfinder_zpeaks[offset + i];
  }
  // The other finder's seeds outside PVFinder's range.
  const unsigned n_other = parameters.dev_other_number_of_zpeaks[event_number];
  for (unsigned i = 0; i < n_other && n < PV::max_number_vertices; ++i) {
    const float z = parameters.dev_other_zpeaks[offset + i];
    if (z < z_min || z >= z_max) out[n++] = z;
  }
  parameters.dev_number_of_zpeaks[event_number] = n;
}
