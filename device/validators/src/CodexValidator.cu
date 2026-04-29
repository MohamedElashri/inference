/*****************************************************************************\
* (c) Copyright 2022 CERN for the benefit of the LHCb Collaboration          *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "CodexValidator.cuh"
#include <fstream>

INSTANTIATE_ALGORITHM(codex_validator::codex_validator_t)

__global__ void codex_validator_hits(codex_validator::Parameters parameters)
{
  // 1. Single thread guard
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  // 2. Get total event count
  unsigned num_events = parameters.dev_number_of_events[0];

  const auto codex_hits_offsets = parameters.dev_codex_hits_size;
  const auto codex_hits = parameters.dev_codex_hits;
  const auto hits_permutation = parameters.dev_codex_hits_permutations;

  // 3. Loop over ALL events sequentially
  for (unsigned event_number = 0; event_number < num_events; ++event_number) {

    unsigned start = codex_hits_offsets[event_number];
    unsigned end = codex_hits_offsets[event_number + 1];
    unsigned hits_size = end - start;

    // WARNING: This 16KB array is still large.
    // Since we are now single-threaded, it is safer, but strictly speaking
    // this should be passed as a scratchpad argument to avoid stack overflow.
    constexpr int number_of_strip_per_event = 4032;
    unsigned number_of_hits_per_strip[number_of_strip_per_event];

    // Reset array for this event
    for (int i = 0; i < number_of_strip_per_event; ++i)
      number_of_hits_per_strip[i] = 0;

    printf("Event %u, hits = %u\n", event_number, hits_size);

    for (unsigned hitID = 0; hitID < hits_size; ++hitID) {
      const auto hit = codex_hits[hits_permutation[start + hitID]];

      unsigned strip_id_at_singlet = (hit.strip_type == 1) ? hit.strip_id : hit.strip_id + 64;
      unsigned unique_strip_id = hit.singlet_id * 96 + strip_id_at_singlet;

      if (unique_strip_id < number_of_strip_per_event) {
        number_of_hits_per_strip[unique_strip_id]++;
      }
    }

    for (unsigned strip_id = 0; strip_id < number_of_strip_per_event; strip_id++) {
      if (number_of_hits_per_strip[strip_id] == 0) continue;
      int dct_id = strip_id / 288;
      int layer = (strip_id % 288) / 96 + 1;
      int id_in_layer = (strip_id % 288) % 96;
      int type = id_in_layer > 64 ? 1 : 0;
      int type_strip_id = type ? id_in_layer - 64 : id_in_layer;

      printf(
        "DCT %i, Layer %i, Type %i, Strip %i, Hits %u\n",
        dct_id,
        layer,
        type,
        type_strip_id,
        number_of_hits_per_strip[strip_id]);
    }
    printf("\n");
  }
}

__global__ void codex_validator_clusters(codex_validator::Parameters parameters)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  unsigned num_events = parameters.dev_number_of_events[0];

  for (unsigned event_number = 0; event_number < num_events; ++event_number) {

    // Re-calculate offsets based on current event_number loop
    const auto codex_offsets = parameters.dev_codex_cluster_size + event_number * Codex::NumberOfSinglets;

    const auto codex_clusters = parameters.dev_codex_clusters;

    const auto codex_clusters_size = codex_offsets[Codex::NumberOfSinglets] - codex_offsets[0];

    for (unsigned cluster_id = 0; cluster_id < codex_clusters_size; cluster_id++) {
      printf(
        "Event %u: Cluster_id = %u, Phi mean = %u, Eta mean = %u\n",
        event_number,
        codex_offsets[0] + cluster_id,
        codex_clusters[codex_offsets[0] + cluster_id].phi_strip_mean,
        codex_clusters[codex_offsets[0] + cluster_id].eta_strip_mean);
    }
  }
}

__global__ void codex_validator_coincidence(codex_validator::Parameters parameters)
{
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  unsigned num_events = parameters.dev_number_of_events[0];

  for (unsigned event_number = 0; event_number < num_events; ++event_number) {

    auto coincidence_output =
      parameters.dev_codex_coincidences + event_number * Codex::MaxCoincidencePerTriplet * Codex::NumberOfDCTs;

    auto created_coincidence_sizes = parameters.dev_codex_created_coincidences_size + event_number;
    auto double_coincidence_sizes = parameters.dev_codex_double_coincidences_size + event_number;
    auto triple_coincidence_sizes = parameters.dev_codex_triple_coincidences_size + event_number;

    printf("\n\nEvent %u, Coincidence validator\n", event_number);
    printf("Created coincidences = %u\n", created_coincidence_sizes[0]);
    printf("Double coincidences = %u\n", double_coincidence_sizes[0]);
    printf("Triple coincidences = %u\n", triple_coincidence_sizes[0]);

    for (unsigned coin_idx = 0; coin_idx < created_coincidence_sizes[0]; coin_idx++) {
      printf(
        "Coincidence_idx = %u, Phi mean = %u, Eta mean = %u, Time mean = %f\n",
        coin_idx,
        coincidence_output[coin_idx].phi_strip_mean,
        coincidence_output[coin_idx].eta_strip_mean,
        coincidence_output[coin_idx].mean_time);
    }
  }
}

void codex_validator::codex_validator_t::set_arguments_size(
  ArgumentReferences<Parameters>,
  const RuntimeOptions&,
  const Constants&) const
{}

void codex_validator::codex_validator_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{

  global_function(codex_validator_hits)(1, 1, context)(arguments);
  global_function(codex_validator_clusters)(1, 1, context)(arguments);
  global_function(codex_validator_coincidence)(1, 1, context)(arguments);
}
