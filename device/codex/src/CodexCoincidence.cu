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

#include "CodexCoincidence.cuh"

#include "BankTypes.h"
#include <BackendCommon.h>
#include <MEPTools.h>
#include <PrefixSum.cuh>

INSTANTIATE_ALGORITHM(codex_coincidence::codex_coincidence_t)

__global__ void codex_coincidence_create_first_layer_candidates(codex_coincidence::Parameters parameters)
{

  unsigned const event_number = parameters.dev_event_list[blockIdx.x];
  const auto max_allowed = Codex::MaxCoincidencePerTriplet * Codex::NumberOfDCTs;
  const auto codex_clusters_sizes = parameters.dev_codex_cluster_size + event_number * Codex::NumberOfSinglets;
  const auto clusters_offset = codex_clusters_sizes[0];
  auto codex_clusters = parameters.dev_codex_clusters + clusters_offset;

  auto total_clusters_in_first_layers =
    codex_clusters_sizes[Codex::NumberOfDCTs] -
    codex_clusters_sizes[0]; // Total 42 singlets: 14 layer 0 + 14 layer 1 + 14 layer 2

  __shared__ unsigned int used[Codex::MaxEndClustersPerSinglet * Codex::NumberOfSinglets / 32];

  for (unsigned i = threadIdx.x; i < Codex::MaxEndClustersPerSinglet * Codex::NumberOfSinglets / 32; i += blockDim.x) {
    used[i] = 0u;
  }

  __syncthreads();

  auto coincidence_output =
    parameters.dev_codex_coincidences + event_number * Codex::MaxCoincidencePerTriplet * Codex::NumberOfDCTs;
  auto created_coincidence_sizes = parameters.dev_codex_created_coincidences_size + event_number;

  auto double_coincidence_sizes = parameters.dev_codex_double_coincidences_size + event_number;
  auto triple_coincidence_sizes = parameters.dev_codex_triple_coincidences_size + event_number;

  // ------------------------------------------------------------
  // layer 0 ↔ layer 1 coincidences
  // ------------------------------------------------------------
  for (unsigned cluster_id = threadIdx.x; cluster_id < total_clusters_in_first_layers; cluster_id += blockDim.x) {
    if (created_coincidence_sizes[0] >= max_allowed) continue;

    auto cluster_input = codex_clusters[cluster_id];

    auto cluster_singlet_id = cluster_input.singlet_id;
    auto cluster_singlet_index =
      static_cast<int>(cluster_singlet_id % 3) * Codex::NumberOfDCTs + static_cast<int>(cluster_singlet_id / 3);
    auto cluster_phi_strip_mean = cluster_input.phi_strip_mean;
    auto cluster_eta_strip_mean = cluster_input.eta_strip_mean;

    auto cluster_mean_time = cluster_input.cluster_mean_time;

    auto next_singlet_id = cluster_singlet_index + Codex::NumberOfDCTs;

    auto next_singlet_number_of_clusters =
      codex_clusters_sizes[next_singlet_id + 1] - codex_clusters_sizes[next_singlet_id];
    auto next_singlet_offset = codex_clusters_sizes[next_singlet_id] - codex_clusters_sizes[0];

    unsigned wordA = cluster_id / 32;
    unsigned bitA = 1u << (cluster_id % 32);

    for (unsigned possible_cluster_id = 0; possible_cluster_id < next_singlet_number_of_clusters;
         possible_cluster_id++) {
      auto possible_cluster = codex_clusters[next_singlet_offset + possible_cluster_id];
      const auto possible_cluster_phi_strip_mean = possible_cluster.phi_strip_mean;
      const auto possible_cluster_eta_strip_mean = possible_cluster.eta_strip_mean;
      const auto possible_cluster_mean_time = possible_cluster.cluster_mean_time;

      if (std::abs(possible_cluster_mean_time - cluster_mean_time) > Codex::TimeWindow) continue;

      auto phi_off = std::abs(cluster_phi_strip_mean - possible_cluster_phi_strip_mean);
      auto eta_off = std::abs(cluster_eta_strip_mean - possible_cluster_eta_strip_mean);

      if (phi_off + eta_off > Codex::DistanceWindow) continue;

      auto coincidence_time = (possible_cluster_mean_time + cluster_mean_time) / 2;
      auto coincidence_phi = (cluster_phi_strip_mean + possible_cluster_phi_strip_mean) / 2;
      auto coincidence_eta = (cluster_eta_strip_mean + possible_cluster_eta_strip_mean) / 2;

      unsigned coincidence_id = atomicAdd(created_coincidence_sizes, 1u);

      if (coincidence_id >= max_allowed) continue; // overfill protections

      coincidence_output[coincidence_id] = CodexCoincidence(
        1, coincidence_phi, coincidence_eta, coincidence_time, cluster_id, next_singlet_offset + possible_cluster_id);

      atomicOr(&used[wordA], bitA); // mark cluster as used
      atomicOr(
        &used[(next_singlet_offset + possible_cluster_id) / 32],
        1u << ((next_singlet_offset + possible_cluster_id) % 32));
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    if (created_coincidence_sizes[0] > max_allowed) {
      created_coincidence_sizes[0] = 0;
    }
  }

  __syncthreads();

  // ------------------------------------------------------------
  // update to layer 0 ↔ layer 1  ↔ layer 2  coincidences
  // ------------------------------------------------------------
  for (unsigned coincidence_id = threadIdx.x; coincidence_id < created_coincidence_sizes[0];
       coincidence_id += blockDim.x) {
    // update coincicdence with third layer_hit

    auto* coincidence = &coincidence_output[coincidence_id];
    auto first_singlet_id = codex_clusters[coincidence->layer_0_clusted_index].singlet_id;
    auto cluster_singlet_index =
      static_cast<int>(first_singlet_id % 3) * Codex::NumberOfDCTs + static_cast<int>(first_singlet_id / 3);

    auto coin_phi_strip_mean = coincidence->phi_strip_mean;
    auto coin_eta_strip_mean = coincidence->eta_strip_mean;

    auto coin_mean_time = coincidence->mean_time;

    auto third_singlet_id = cluster_singlet_index + 28;

    auto next_singlet_number_of_clusters =
      codex_clusters_sizes[third_singlet_id + 1] - codex_clusters_sizes[third_singlet_id];
    auto next_singlet_offset = codex_clusters_sizes[third_singlet_id] - codex_clusters_sizes[0];

    for (unsigned possible_cluster_id = 0; possible_cluster_id < next_singlet_number_of_clusters;
         possible_cluster_id++) {
      auto possible_cluster = codex_clusters[next_singlet_offset + possible_cluster_id];

      const auto possible_cluster_mean_time = possible_cluster.cluster_mean_time;

      const auto possible_cluster_phi_strip_mean = possible_cluster.phi_strip_mean;
      const auto possible_cluster_eta_strip_mean = possible_cluster.eta_strip_mean;

      auto phi_off = std::abs(coin_phi_strip_mean - possible_cluster_phi_strip_mean); // todo: fix
      auto eta_off = std::abs(coin_eta_strip_mean - possible_cluster_eta_strip_mean); // todo: fix

      if (std::abs(possible_cluster_mean_time - coin_mean_time) > Codex::TimeWindow) continue;

      if (phi_off + eta_off > Codex::DistanceWindow) continue;

      coincidence->update_coincidence(
        possible_cluster_phi_strip_mean,
        possible_cluster_eta_strip_mean,
        possible_cluster_mean_time,
        next_singlet_offset + possible_cluster_id);

      atomicOr(
        &used[(next_singlet_offset + possible_cluster_id) / 32],
        1u << ((next_singlet_offset + possible_cluster_id) % 32));

      break; // we do not check the rest of the clusters in third layer after updating the coincidence, because of
             // possible clones
    }
  }

  __syncthreads();

  auto third_layer_offset = codex_clusters_sizes[2 * Codex::NumberOfDCTs] -
                            codex_clusters_sizes[0]; // Total 42 singlets: 14 layer 0 + 14 layer 1 + 14 layer 2
  auto second_layer_offset = codex_clusters_sizes[Codex::NumberOfDCTs] - codex_clusters_sizes[0];

  // ------------------------------------------------------------
  // layer 1 ↔ layer 2 coincidences
  // ------------------------------------------------------------
  for (unsigned cluster_id = second_layer_offset + threadIdx.x; cluster_id < third_layer_offset;
       cluster_id += blockDim.x) {
    auto cluster_input = codex_clusters[cluster_id];

    auto cluster_singlet_id = cluster_input.singlet_id;
    auto cluster_singlet_index =
      static_cast<int>(cluster_singlet_id % 3) * Codex::NumberOfDCTs + static_cast<int>(cluster_singlet_id / 3);
    auto cluster_phi_strip_mean = cluster_input.phi_strip_mean;
    auto cluster_eta_strip_mean = cluster_input.eta_strip_mean;

    auto cluster_mean_time = cluster_input.cluster_mean_time;

    auto next_singlet_id = cluster_singlet_index + Codex::NumberOfDCTs;
    auto next_singlet_number_of_clusters =
      codex_clusters_sizes[next_singlet_id + 1] - codex_clusters_sizes[next_singlet_id];
    auto next_singlet_offset = codex_clusters_sizes[next_singlet_id] - codex_clusters_sizes[0];

    unsigned wordA = cluster_id / 32;
    unsigned bitA = 1u << (cluster_id % 32);

    // check if cluster is already used in a coincidence (keeping same priority as in clone removal)
    if (used[wordA] & bitA) {
      continue;
    }

    for (unsigned possible_cluster_id = 0; possible_cluster_id < next_singlet_number_of_clusters;
         possible_cluster_id++) {
      auto possible_cluster = codex_clusters[next_singlet_offset + possible_cluster_id];
      const auto possible_cluster_phi_strip_mean = possible_cluster.phi_strip_mean;
      const auto possible_cluster_eta_strip_mean = possible_cluster.eta_strip_mean;
      const auto possible_cluster_mean_time = possible_cluster.cluster_mean_time;

      if (std::abs(possible_cluster_mean_time - cluster_mean_time) > Codex::TimeWindow) continue;

      auto phi_off = std::abs(cluster_phi_strip_mean - possible_cluster_phi_strip_mean);
      auto eta_off = std::abs(cluster_eta_strip_mean - possible_cluster_eta_strip_mean);

      if (phi_off + eta_off > Codex::DistanceWindow) continue;

      if (created_coincidence_sizes[0] >= max_allowed) continue; // overfill protections

      auto coincidence_time = (possible_cluster_mean_time + cluster_mean_time) / 2;
      auto coincidence_phi = (cluster_phi_strip_mean + possible_cluster_phi_strip_mean) / 2;
      auto coincidence_eta = (cluster_eta_strip_mean + possible_cluster_eta_strip_mean) / 2;

      unsigned coincidence_id = atomicAdd(created_coincidence_sizes, 1u);

      if (coincidence_id >= max_allowed) continue; // overfill protections

      coincidence_output[coincidence_id] = CodexCoincidence(
        2, coincidence_phi, coincidence_eta, coincidence_time, cluster_id, next_singlet_offset + possible_cluster_id);

      atomicOr(
        &used[(next_singlet_offset + possible_cluster_id) / 32],
        1u << ((next_singlet_offset + possible_cluster_id) % 32)); // mark cluster as used
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    if (created_coincidence_sizes[0] > max_allowed) {
      created_coincidence_sizes[0] = 0;
    }
  }

  __syncthreads();

  // ------------------------------------------------------------
  // layer 0 ↔ layer 2 coincidences
  // ------------------------------------------------------------
  for (unsigned cluster_id = threadIdx.x; cluster_id < second_layer_offset; cluster_id += blockDim.x) {

    auto cluster_input = codex_clusters[cluster_id];
    auto cluster_singlet_id = cluster_input.singlet_id;
    auto cluster_singlet_index =
      static_cast<int>(cluster_singlet_id % 3) * Codex::NumberOfDCTs + static_cast<int>(cluster_singlet_id / 3);
    auto cluster_phi_strip_mean = cluster_input.phi_strip_mean;
    auto cluster_eta_strip_mean = cluster_input.eta_strip_mean;
    auto cluster_mean_time = cluster_input.cluster_mean_time;
    auto third_singlet_id = cluster_singlet_index + 28;
    auto third_singlet_number_of_clusters =
      codex_clusters_sizes[third_singlet_id + 1] - codex_clusters_sizes[third_singlet_id];
    auto third_singlet_offset = codex_clusters_sizes[third_singlet_id] - codex_clusters_sizes[0];

    unsigned wordA = cluster_id / 32;
    unsigned bitA = 1u << (cluster_id % 32);

    // check if cluster is already used in a coincidence (keeping same priority as in clone removal)
    if (used[wordA] & bitA) {
      continue;
    }

    for (unsigned possible_cluster_id = 0; possible_cluster_id < third_singlet_number_of_clusters;
         possible_cluster_id++) {

      // Check if cluster is already used in a high priority coincidence, no conflicts as we are not wrting in this loop
      unsigned wordB = (possible_cluster_id + third_singlet_offset) / 32;
      unsigned bitB = 1u << ((possible_cluster_id + third_singlet_offset) % 32);
      if (used[wordB] & bitB) continue;

      auto possible_cluster = codex_clusters[third_singlet_offset + possible_cluster_id];
      const auto possible_cluster_phi_strip_mean = possible_cluster.phi_strip_mean;
      const auto possible_cluster_eta_strip_mean = possible_cluster.eta_strip_mean;
      const auto possible_cluster_mean_time = possible_cluster.cluster_mean_time;

      if (std::abs(possible_cluster_mean_time - cluster_mean_time) > Codex::TimeWindow) continue;

      auto phi_off = std::abs(cluster_phi_strip_mean - possible_cluster_phi_strip_mean);
      auto eta_off = std::abs(cluster_eta_strip_mean - possible_cluster_eta_strip_mean);

      if (phi_off + eta_off > Codex::DistanceWindow) continue;

      auto coincidence_time = (possible_cluster_mean_time + cluster_mean_time) / 2;
      auto coincidence_phi = (cluster_phi_strip_mean + possible_cluster_phi_strip_mean) / 2;
      auto coincidence_eta = (cluster_eta_strip_mean + possible_cluster_eta_strip_mean) / 2;

      unsigned coincidence_id = atomicAdd(created_coincidence_sizes, 1u);

      if (coincidence_id >= max_allowed) continue; // overfill protections

      coincidence_output[coincidence_id] = CodexCoincidence(
        3, coincidence_phi, coincidence_eta, coincidence_time, cluster_id, third_singlet_offset + possible_cluster_id);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    if (created_coincidence_sizes[0] > max_allowed) {
      created_coincidence_sizes[0] = 0;
    }
  }
  __syncthreads();

  unsigned int* clone = used;

  for (unsigned i = threadIdx.x; i < (max_allowed / 32); i += blockDim.x) {
    clone[i] = 0u;
  }

  __syncthreads();

  for (unsigned coincidence_A_id = threadIdx.x; coincidence_A_id < created_coincidence_sizes[0];
       coincidence_A_id += blockDim.x) {
    auto coincidence_A = coincidence_output[coincidence_A_id];
    unsigned wordA = coincidence_A_id / 32;
    unsigned bitA = 1u << (coincidence_A_id % 32);

    for (unsigned coincidence_B_id = coincidence_A_id + 1; coincidence_B_id < created_coincidence_sizes[0];
         coincidence_B_id++) {
      auto coincidence_B = coincidence_output[coincidence_B_id];
      unsigned wordB = coincidence_B_id / 32;
      unsigned bitB = 1u << (coincidence_B_id % 32);

      int shared_clusters = 0;

      if (coincidence_A.layer_0_clusted_index == coincidence_B.layer_0_clusted_index) shared_clusters++;
      if (coincidence_A.layer_1_clusted_index == coincidence_B.layer_1_clusted_index) shared_clusters++;
      if (coincidence_A.layer_2_clusted_index == coincidence_B.layer_2_clusted_index) shared_clusters++;

      if (shared_clusters == 0) continue;

      if (coincidence_A.coincidence_type + coincidence_B.coincidence_type == 0) // if both are full coincidence
      {
        if (shared_clusters == 1) continue; // each of them at least two layers coincidence -> keep both

        if (shared_clusters > 1) {
          atomicOr(&clone[wordA], bitA); // assign first to be clone and remove it
          break;
        }
      }

      if ((coincidence_A.coincidence_type == 0) && (coincidence_B.coincidence_type != 0)) {
        atomicOr(&clone[wordB], bitB);
        continue; // remove the small one
      }

      if ((coincidence_A.coincidence_type != 0) && (coincidence_B.coincidence_type == 0)) {
        atomicOr(&clone[wordA], bitA);
        break; // remove the small one
      }

      if ((coincidence_A.coincidence_type != 0) && (coincidence_B.coincidence_type != 0)) {
        // Treat 0-2 as "lower priority" than 0-1 or 1-2
        if (coincidence_A.coincidence_type == 3 && coincidence_B.coincidence_type != 3) {
          atomicOr(&clone[wordA], bitA); // remove the 0-2
          break;
        }
        else if (coincidence_B.coincidence_type == 3 && coincidence_A.coincidence_type != 3) {
          atomicOr(&clone[wordB], bitB); // remove the 0-2
          continue;
        }
        else {
          // remove one of the two (arbitrarily choose A)
          atomicOr(&clone[wordA], bitA);
          break;
        }
      }
    }
  }

  __syncthreads();

  // Calculate number of coincidences by type
  unsigned local_triple = 0;
  unsigned local_double = 0;

  for (unsigned i = threadIdx.x; i < created_coincidence_sizes[0]; i += blockDim.x) {
    unsigned word = i / 32;
    unsigned bit = 1u << (i % 32);

    if (clone[word] & bit) continue;

    auto* coincidence = &coincidence_output[i];

    if (coincidence->coincidence_type == 0) {
      ++local_triple;
    }
    else {
      ++local_double;
    }
  }

  if (local_triple > 0) {
    atomicAdd(triple_coincidence_sizes, local_triple);
  }
  if (local_double > 0) {
    atomicAdd(double_coincidence_sizes, local_double);
  }
}

void codex_coincidence::codex_coincidence_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_codex_coincidences_t>(
    arguments, first<host_number_of_events_t>(arguments) * Codex::NumberOfDCTs * Codex::MaxCoincidencePerTriplet);
  set_size<dev_codex_created_coincidences_size_t>(arguments, first<host_number_of_events_t>(arguments) + 1);

  set_size<dev_codex_double_coincidences_size_t>(arguments, first<host_number_of_events_t>(arguments) + 1);
  set_size<dev_codex_triple_coincidences_size_t>(arguments, first<host_number_of_events_t>(arguments) + 1);
}

void codex_coincidence::codex_coincidence_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{

  auto const num_clusters = first<host_codex_num_clusters_t>(arguments);

  // fill with zeros before check, as only this output is used in trigger lines
  Allen::memset_async<dev_codex_created_coincidences_size_t>(arguments, 0, context);
  Allen::memset_async<dev_codex_double_coincidences_size_t>(arguments, 0, context);
  Allen::memset_async<dev_codex_triple_coincidences_size_t>(arguments, 0, context);

  // if there are clusters in event empty arrays will be used afterwards as well
  if (num_clusters == 0) { // no clusters are present in events -> return
    return;
  }

  global_function(codex_coincidence_create_first_layer_candidates)(
    dim3(size<dev_event_list_t>(arguments)), dim3(m_block_dim_x), context)(arguments);
}
