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
#include "MakeTTrackVertices.cuh"

#include "TrackTrajectory.cuh"
#include "TrajectoryPoca.cuh"
#include "SciFiDefinitions.cuh"
#include "PrefixSum.cuh"

#include "TTrackDefinitions.cuh"

#include <fstream>

INSTANTIATE_ALGORITHM(make_ttrack_vertices::make_ttrack_vertices_t);

void make_ttrack_vertices::make_ttrack_vertices_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const auto n_events = first<host_number_of_events_t>(arguments);
  set_size<dev_tt_vertices_t>(
    arguments, n_events * TTracks::Constants::MAX_VERTICES_PER_EVENT * TTracks::Vertex::PackedMiniVertexVar::COUNT);
  set_size<dev_offsets_tt_vertices_t>(arguments, n_events + 1);
  set_size<host_number_of_tt_vertices_t>(arguments, 1);
}

void make_ttrack_vertices::make_ttrack_vertices_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_offsets_tt_vertices_t>(arguments, 0, context);

  // Shared memory layout:
  // unsigned candidates_wptr[n_warps] - FIFO write pointers (per warp)
  // unsigned candidates_rptr[n_warps] - FIFO read pointers (per warp)
  // unsigned candidates[n_warps * 2 * warp_size] - FIFO buffer of size 2 * warp_size (per warp)
  //    - At any given time only warp_size (32) vertex candidates can be added to the FIFO
  //    - If we have >= warp_size candidates, we process those (with DOCA minimization algorithm).
  //      The processed candidates are removed from the FIFO.
  const auto n_warps = (m_block_dim.value().x - 1) / warp_size + 1;
  const unsigned shared_mem_bytes = sizeof(unsigned) * 2 * n_warps * (1 + warp_size);
  global_function(make_ttrack_vertices)(
    dim3(size<dev_event_list_t>(arguments)), m_block_dim, context, shared_mem_bytes)(
    arguments, DeviceProperties(*this, context));

  // print<dev_offsets_tt_vertices_t>(arguments);

  PrefixSum::prefix_sum<dev_offsets_tt_vertices_t, host_number_of_tt_vertices_t>(*this, arguments, context);
}

__global__ void make_ttrack_vertices::make_ttrack_vertices(
  Parameters parameters,
  make_ttrack_vertices_t::DeviceProperties props)
{
  namespace Constants = TTracks::Constants;
  using VertexVar = TTracks::Vertex::PackedMiniVertexVar;

  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const auto offset = parameters.dev_offsets_seeding_tracks[event_number];
  const auto n_input_states = parameters.dev_offsets_seeding_tracks[event_number + 1] - offset;

  if (n_input_states < 2) return;

  const auto input_states = Trajectories::GlobalMemoryManagerConst(
    parameters.dev_filtered_ttracks_states.data(),
    parameters.dev_offsets_seeding_tracks[parameters.dev_number_of_events[0]]);
  const auto input_qop = parameters.dev_filtered_ttracks_qop + offset;

  // FIFO buffer
  const auto n_warps = (blockDim.x - 1) / warp_size + 1;
  const auto i_warp = threadIdx.x / warp_size;
  DYNAMIC_SHARED_MEMORY_BUFFER(unsigned, _shared_mem, parameters.config)
  unsigned* candidates_wptr = _shared_mem;
  unsigned* candidates_rptr = candidates_wptr + n_warps;
  unsigned* candidates = candidates_rptr + n_warps;
  candidates_wptr[i_warp] = 0;
  candidates_rptr[i_warp] = 0;
  __syncwarp();

  // Outputs
  auto output_vertices = TTracks::Vertex::PackedMiniVertexMutable(
    parameters.dev_tt_vertices.data() + event_number * Constants::MAX_VERTICES_PER_EVENT * VertexVar::COUNT,
    Constants::MAX_VERTICES_PER_EVENT);
  auto output_num_vertices = parameters.dev_offsets_tt_vertices + event_number;

  const auto process_combination = [&](const std::pair<unsigned, unsigned> comb) {
    const auto [i_track, j_track] = comb;

    const auto yA = input_states.get<Trajectories::y>(offset + i_track, Constants::num_reference_states - 1);
    const auto tyA = input_states.get<Trajectories::ty>(offset + i_track, Constants::num_reference_states - 1);
    const auto yB = input_states.get<Trajectories::y>(offset + j_track, Constants::num_reference_states - 1);
    const auto tyB = input_states.get<Trajectories::ty>(offset + j_track, Constants::num_reference_states - 1);

    const auto start_z =
      fabsf(tyA - tyB) < 1e-5f ?
        5000.f :
        (yB - yA + (tyA - tyB) * Constants::z_reference_states[Constants::num_reference_states - 1]) / (tyA - tyB);

    auto splineA = Trajectories::Spline(input_states.get_proxy(offset + i_track));
    auto splineB = Trajectories::Spline(input_states.get_proxy(offset + j_track));

    const auto traj_poca = Trajectories::TrajectoryPoca();
    const auto poca = traj_poca.minimize(splineA, splineB, start_z, start_z, props.min_vrtx_z, props.max_vrtx_z);

    const auto r_cut = fmaxf(props.min_distance_to_beamline, poca.pos.z * props.min_distance_to_beamline_slope);
    const auto r2 = poca.pos.x * poca.pos.x + poca.pos.y * poca.pos.y;

    if (
      poca.doca2 < props.max_doca * props.max_doca && poca.pos.z >= props.min_vrtx_z &&
      poca.pos.z <= props.max_vrtx_z && r2 > r_cut * r_cut) {
      const auto qopA = input_qop[i_track];
      const auto qopB = input_qop[j_track];
      const auto stateA = splineA.state(poca.muA);
      const auto stateB = splineB.state(poca.muB);

      const auto pzA = 1.f / (fabsf(qopA) * sqrtf(1.f + stateA.tx() * stateA.tx() + stateA.ty() * stateA.ty()));
      const auto pzB = 1.f / (fabsf(qopB) * sqrtf(1.f + stateB.tx() * stateB.tx() + stateB.ty() * stateB.ty()));

      const auto px = pzA * stateA.tx() + pzB * stateB.tx();
      const auto py = pzA * stateA.ty() + pzB * stateB.ty();
      const auto pz = pzA + pzB;
      const auto pt2 = px * px + py * py;

      if ((pt2 > props.min_pt2) && (pt2 + pz * pz > props.min_p2)) {
        const unsigned vertex_idx = atomicAdd(output_num_vertices, 1);
        if (vertex_idx < Constants::MAX_VERTICES_PER_EVENT) {
          output_vertices.set<VertexVar::trk1, unsigned>(vertex_idx, i_track);
          output_vertices.set<VertexVar::trk2, unsigned>(vertex_idx, j_track);
          output_vertices.set<VertexVar::x, float>(vertex_idx, poca.pos.x);
          output_vertices.set<VertexVar::y, float>(vertex_idx, poca.pos.y);
          output_vertices.set<VertexVar::z, float>(vertex_idx, poca.pos.z);
          output_vertices.set<VertexVar::px, float>(vertex_idx, px);
          output_vertices.set<VertexVar::py, float>(vertex_idx, py);
          output_vertices.set<VertexVar::pz, float>(vertex_idx, pz);
          output_vertices.set<VertexVar::doca, float>(vertex_idx, sqrtf(poca.doca2));
          props.histogram_xz_vrtx_pos.increment(poca.pos.z, poca.pos.x);
          props.histogram_yz_vrtx_pos.increment(poca.pos.z, poca.pos.y);
        }
      }
    }
  };

  const auto n_combinations = TTracks::Combinations::number_of_combinations(n_input_states);
  const auto n_iterations = ((n_combinations + warp_size - 1) / warp_size) * warp_size; // Multiple of warp_size
  for (unsigned combIdx = threadIdx.x; combIdx < n_iterations; combIdx += blockDim.x) {

    const auto [i_track, j_track] = TTracks::Combinations::convert_idx(combIdx, n_input_states);

    if (combIdx < n_combinations) {
      const auto yA = input_states.get<Trajectories::y>(offset + i_track, Constants::num_reference_states - 1);
      const auto tyA = input_states.get<Trajectories::ty>(offset + i_track, Constants::num_reference_states - 1);
      const auto yB = input_states.get<Trajectories::y>(offset + j_track, Constants::num_reference_states - 1);
      const auto tyB = input_states.get<Trajectories::ty>(offset + j_track, Constants::num_reference_states - 1);

      const auto start_z =
        fabsf(tyA - tyB) < 1e-5f ?
          5000.f :
          (yB - yA + (tyA - tyB) * Constants::z_reference_states[Constants::num_reference_states - 1]) / (tyA - tyB);

      if (start_z > props.min_yzinter_z && start_z < props.max_yzinter_z) {

        const auto idx = atomicAdd(candidates_wptr + i_warp, 1);
        candidates[2 * i_warp * warp_size + (idx % (2 * warp_size))] = combIdx;
      }
    }

    // If we have >= 32 elements in the buffer, we can process them
    __syncwarp();
    if ((candidates_wptr[i_warp] - candidates_rptr[i_warp]) >= warp_size) {
      const auto _combIdx =
        candidates[2 * i_warp * warp_size + (candidates_rptr[i_warp] % (2 * warp_size)) + threadIdx.x % warp_size];
      const auto comb = TTracks::Combinations::convert_idx(_combIdx, n_input_states);

      process_combination(comb);

      __syncwarp();
      if (threadIdx.x % 32 == 0) {
        candidates_rptr[i_warp] = candidates_rptr[i_warp] + warp_size;
      }
    }
  }

  __syncwarp();
  const auto cached_idx = candidates_rptr[i_warp] + threadIdx.x % warp_size;
  if (cached_idx < candidates_wptr[i_warp]) {
    const auto _combIdx = candidates[2 * i_warp * warp_size + (cached_idx % (2 * warp_size))];
    const auto comb = TTracks::Combinations::convert_idx(_combIdx, n_input_states);

    process_combination(comb);
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    parameters.dev_offsets_tt_vertices[event_number] =
      min(parameters.dev_offsets_tt_vertices[event_number], Constants::MAX_VERTICES_PER_EVENT);
  }
}
