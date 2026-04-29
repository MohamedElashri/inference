/*****************************************************************************\
* (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "CodexClustering.cuh"
#include "BankTypes.h"
#include <BackendCommon.h>
#include <MEPTools.h>
#include <PrefixSum.cuh>

INSTANTIATE_ALGORITHM(codex_clustering::codex_clustering_t)

__global__ void codex_clustering_kernel(codex_clustering::Parameters parameters)
{

  unsigned const event_number = parameters.dev_event_list[blockIdx.x];

  // required input
  const auto permutations = parameters.dev_codex_hits_permutations;
  const auto singlets_offsets = parameters.dev_codex_singlet_offsets + event_number * Codex::NumberOfSinglets * 2;
  const auto codex_hits = parameters.dev_codex_hits;

  // output
  const auto codex_all_clusters =
    parameters.dev_codex_all_clusters + event_number * Codex::MaxPhiClustersPerSinglet * Codex::NumberOfSinglets * 2;
  const auto codex_all_clusters_sizes =
    parameters.dev_codex_all_clusters_sizes + event_number * Codex::NumberOfSinglets * 2;

  for (unsigned singlet_id = threadIdx.x; singlet_id < Codex::NumberOfSinglets * 2; singlet_id += blockDim.x) {
    unsigned pos = singlets_offsets[singlet_id];
    unsigned end = singlets_offsets[singlet_id + 1];

    unsigned unique_cluster_id = 0;
    uint8_t prev_id = 100; // maximum strip_id is 96 from hardware. use 100 as default value
    uint8_t prev_time = 0;

    int counter = 0; // number of hits per cluster

    while (pos < end && unique_cluster_id < (Codex::MaxPhiClustersPerSinglet)) {

      const auto hit = codex_hits[permutations[pos]];

      if (counter == 0) // start new cluster
      {
        codex_all_clusters[singlet_id * Codex::MaxPhiClustersPerSinglet + unique_cluster_id] =
          CodexSideCluster(hit.strip_id, hit.time, hit.singlet_id, hit.strip_type);
        prev_id = hit.strip_id;
        prev_time = hit.time;
        ++unique_cluster_id;
        ++counter;
        ++pos;
        continue;
      }

      bool next_strip = (hit.strip_id == prev_id + 1);
      // Use integer absolute difference to avoid wrap-around and casting issues
      bool in_time = (std::abs(static_cast<int>(hit.time) - static_cast<int>(prev_time)) <= Codex::TimeWindow);

      if (next_strip && in_time) {
        codex_all_clusters[singlet_id * Codex::MaxPhiClustersPerSinglet + unique_cluster_id - 1].addHit(
          hit.strip_id, hit.time);

        prev_id = hit.strip_id;
        prev_time = hit.time;
        ++counter;
        ++pos;
        continue;
      }

      // hit was not added -> reset the values, hit will create cluster in next iteration
      counter = 0;
    }

    codex_all_clusters_sizes[singlet_id] = unique_cluster_id;
  }
  __syncthreads();
}

// Parallelized by phi singlets for now. Improve by parallezation by phi clusters in future
__global__ void codex_clustering_match_phi_and_eta_cluster(codex_clustering::Parameters parameters)
{
  unsigned const event_number = parameters.dev_event_list[blockIdx.x];
  const auto all_clusters =
    parameters.dev_codex_all_clusters + event_number * Codex::MaxPhiClustersPerSinglet * Codex::NumberOfSinglets * 2;

  const auto cluster_sizes = parameters.dev_codex_all_clusters_sizes + event_number * Codex::NumberOfSinglets * 2;

  // output
  auto draft_clusters_in_event =
    parameters.dev_codex_draft_clusters + event_number * Codex::MaxEndClustersPerSinglet * Codex::NumberOfSinglets;
  auto draft_clusters_size = parameters.dev_codex_cluster_size + event_number * Codex::NumberOfSinglets;

  for (unsigned singlet_id = threadIdx.x * 2; singlet_id < Codex::NumberOfSinglets * 2; singlet_id += blockDim.x * 2) {

    // counter of clusters per singlet
    unsigned unique_cluster_id = 0;

    const unsigned number_of_phi_clusters_at_singlet = cluster_sizes[singlet_id];

    // in input array two arrays per each singlet (phi and eta)
    const int original_singlet_id = singlet_id / 2;

    // we save cluster in format to parallel by clusters in first layer of each RTC at the next step
    // [0]  [1]  [2]
    // [3]  [4]  [5]
    // [6]  [7]  [8]
    //
    // -> 0 3 6 9  12 15 18 21 24 27 30 33 36 39
    // -> 1 4 7 10 13 16 19 22 25 28 31 34 37 40
    // -> 2 5 8 11 14 17 20 23 26 29 32 35 38 41

    const int singlet_id_in_output_array =
      static_cast<int>(original_singlet_id % 3) * Codex::NumberOfDCTs + static_cast<int>(original_singlet_id / 3);

    // move pointer to save output clusters
    auto possible_clusters_at_singlet =
      draft_clusters_in_event + singlet_id_in_output_array * Codex::MaxEndClustersPerSinglet;

    // get phi and eta cluster from the input array
    const auto phi_clusters = all_clusters + singlet_id * Codex::MaxPhiClustersPerSinglet;
    const auto eta_clusters = all_clusters + (singlet_id + 1) * Codex::MaxPhiClustersPerSinglet;

    const unsigned number_of_eta_clusters_at_singlet = cluster_sizes[singlet_id + 1];

    for (unsigned phi_cluster_id = 0; phi_cluster_id < number_of_phi_clusters_at_singlet; phi_cluster_id++) {
      const auto phi_cl = phi_clusters[phi_cluster_id];

      for (unsigned eta_cluster_id = 0; eta_cluster_id < number_of_eta_clusters_at_singlet; eta_cluster_id++) {
        const auto eta_cl = eta_clusters[eta_cluster_id];

        if (fabsf(phi_cl.get_cluster_time() - eta_cl.get_cluster_time()) < Codex::TimeWindow) {
          const float cluster_time = static_cast<float>((phi_cl.get_cluster_time() + eta_cl.get_cluster_time())) * 0.5f;

          possible_clusters_at_singlet[unique_cluster_id] = CodexCluster(
            eta_cl.strips_size,
            phi_cl.strips_size,
            phi_cl.get_cluster_strip_id(),
            eta_cl.get_cluster_strip_id(),
            cluster_time,
            original_singlet_id);
          ++unique_cluster_id;
        }

        if (unique_cluster_id >= Codex::MaxEndClustersPerSinglet) break;
      }

      if (unique_cluster_id >= Codex::MaxEndClustersPerSinglet) break;
    }

    draft_clusters_size[singlet_id_in_output_array] = unique_cluster_id;
  }
}

__global__ void consolidate_codex_clusters(codex_clustering::Parameters parameters)
{

  unsigned const event_number = parameters.dev_event_list[blockIdx.x];

  const auto output_offset = parameters.dev_codex_cluster_size + event_number * Codex::NumberOfSinglets;

  const auto clusters_input =
    parameters.dev_codex_draft_clusters + event_number * Codex::MaxEndClustersPerSinglet * Codex::NumberOfSinglets;

  const unsigned total_slots = Codex::NumberOfSinglets * Codex::MaxEndClustersPerSinglet;

  for (unsigned slot = threadIdx.x; slot < total_slots; slot += blockDim.x) {
    const unsigned singlet_id = slot / Codex::MaxEndClustersPerSinglet;
    const unsigned cluster_id = slot % Codex::MaxEndClustersPerSinglet;

    const unsigned clusters_size = output_offset[singlet_id + 1] - output_offset[singlet_id];
    if (cluster_id >= clusters_size) continue;

    const auto singlet_output_offset = output_offset[singlet_id];
    const auto singlet_input_clusters = clusters_input + singlet_id * Codex::MaxEndClustersPerSinglet;
    const auto clusters_output = parameters.dev_codex_clusters + singlet_output_offset;

    clusters_output[cluster_id] = singlet_input_clusters[cluster_id];
  }
}

void codex_clustering::codex_clustering_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_codex_all_clusters_t>(
    arguments,
    first<host_number_of_events_t>(arguments) * Codex::MaxPhiClustersPerSinglet * Codex::NumberOfSinglets * 2);
  set_size<dev_codex_all_clusters_sizes_t>(
    arguments, first<host_number_of_events_t>(arguments) * Codex::NumberOfSinglets * 2 + 1);

  set_size<dev_codex_draft_clusters_t>(
    arguments, first<host_number_of_events_t>(arguments) * Codex::MaxEndClustersPerSinglet * Codex::NumberOfSinglets);

  set_size<dev_codex_cluster_size_t>(
    arguments, first<host_number_of_events_t>(arguments) * Codex::NumberOfSinglets + 1);

  set_size<host_codex_num_clusters_t>(arguments, 1u);
}

void codex_clustering::codex_clustering_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{

  auto const num_hits = first<host_codex_num_hits_t>(arguments);

  if (num_hits == 0) { // no hits are present in events -> set cluster size to 0
    Allen::memset_async<host_codex_num_clusters_t>(arguments, 0, context);
    return;
  }

  global_function(codex_clustering_kernel)(dim3(size<dev_event_list_t>(arguments)), dim3(m_block_dim_x), context)(
    arguments);

  global_function(codex_clustering_match_phi_and_eta_cluster)(
    dim3(size<dev_event_list_t>(arguments)), dim3(m_block_dim_x), context)(arguments);

  PrefixSum::prefix_sum<dev_codex_cluster_size_t, host_codex_num_clusters_t>(*this, arguments, context);

  auto num_clusters = first<host_codex_num_clusters_t>(arguments);

  resize<dev_codex_clusters_t>(arguments, num_clusters);
  global_function(consolidate_codex_clusters)(dim3(size<dev_event_list_t>(arguments)), dim3(m_block_dim_x), context)(
    arguments);
}
