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

#include "RichSimplePID.cuh"

INSTANTIATE_ALGORITHM(rich_simple_pid::rich_simple_pid_t);

__global__ void rich_simple_pid_k(
  const unsigned number_of_tracks,
  const Allen::Rich::HypoData<float>* track_total_signals_r1,
  const Allen::Rich::HypoData<float>* track_total_signals_r2,
  Allen::Rich::ParticleIDType* pids)
{
  const unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned stride = gridDim.x * blockDim.x;
  for (unsigned i = threadId; i < number_of_tracks; i += stride) {
    // Find PID
    float max_bin = 0.f;
    auto pid = Allen::Rich::ParticleIDType::Unknown;
    UNROLL(Allen::Rich::NParticleTypes)
    for (unsigned hypo_index = 0; hypo_index < Allen::Rich::NParticleTypes; ++hypo_index) {
      const auto hypo = static_cast<Allen::Rich::ParticleIDType>(hypo_index);
      float h = track_total_signals_r1[i][hypo] + track_total_signals_r2[i][hypo];
      if (h > max_bin) {
        max_bin = h;
        pid = hypo;
      }
    }

    pids[i] = pid;
  }
}

void rich_simple_pid::rich_simple_pid_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);
  set_size<dev_pid_t>(arguments, number_of_tracks);
}

void rich_simple_pid::rich_simple_pid_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  const unsigned number_of_tracks = first<host_number_of_tracks_t>(arguments);

  global_function(rich_simple_pid_k)(dim3(32), m_block_dim, context)(
    number_of_tracks,
    data<dev_track_total_signals_r1_t>(arguments),
    data<dev_track_total_signals_r2_t>(arguments),
    data<dev_pid_t>(arguments));
}
