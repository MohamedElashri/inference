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
#include "CodexDecode.cuh"
#include "BankTypes.h"
#include <BackendCommon.h>
#include <MEPTools.h>
#include <PrefixSum.cuh>
#include <SegSort.h>

INSTANTIATE_ALGORITHM(codex_decode::codex_decode_t)

__device__ uint32_t build_sort_key(const CodexHit& hit)
{
  return (uint32_t(uint8_t(hit.singlet_id)) << 24) | (uint32_t(uint8_t(hit.strip_type)) << 16) |
         (uint32_t(uint8_t(hit.strip_id)) << 8) | (uint32_t(uint8_t(hit.time)));
}

__device__ unsigned inline calculate_singlet_id(int singlet_id, int strip_type) { return singlet_id * 2 + strip_type; }

__global__ void codex_decode_consolidate_hits(
  codex_decode::Parameters parameters,
  Allen::Monitoring::Histogram2D<>::DeviceType dev_histogram_n_hits_vs_RPC_id)
{
  unsigned const event_number = parameters.dev_event_list[blockIdx.x];

  const auto output_offset = parameters.dev_codex_all_hits_size + event_number;
  const auto hits_input = parameters.dev_codex_all_hits + event_number * Codex::MaxHitsPerEvent;

  const auto hits_output = parameters.dev_codex_hits + output_offset[0];
  const auto hits_keys = parameters.dev_codex_hits_keys + output_offset[0];

  const auto singlet_offsets = parameters.dev_codex_singlet_offsets + event_number * Codex::NumberOfSinglets * 2;

  const unsigned hits_size = output_offset[1] - output_offset[0];

  if (hits_size == 0) return;

  for (unsigned hit_id = threadIdx.x; hit_id < hits_size; hit_id += blockDim.x) {

    hits_output[hit_id] = hits_input[hit_id];
    hits_keys[hit_id] = build_sort_key(hits_input[hit_id]);

    unsigned unique_singlet_id = calculate_singlet_id(hits_input[hit_id].singlet_id, hits_input[hit_id].strip_type);
    atomicAdd(&singlet_offsets[unique_singlet_id], 1u);
  }

  __syncthreads();

  for (unsigned i = threadIdx.x; i < Codex::NumberOfSinglets * 2; i += blockDim.x) {
    dev_histogram_n_hits_vs_RPC_id.increment(i, singlet_offsets[i]);
  }
}

template<bool mep_layout>
__global__ void codex_decode_kernel(
  codex_decode::Parameters parameters,
  Allen::Monitoring::Counter<>::DeviceType dev_n_error_banks,
  Allen::Monitoring::Histogram<>::DeviceType dev_histogram_n_hits,
  Allen::Monitoring::Histogram<>::DeviceType dev_histogram_time_cycle_occupancy)
{
  unsigned const event_number = parameters.dev_event_list[blockIdx.x];

  const char* d = parameters.dev_codex_raw_input;
  const uint32_t* o = parameters.dev_codex_raw_input_offsets;
  const uint32_t* s = parameters.dev_codex_raw_input_sizes;
  const uint32_t* t = parameters.dev_codex_raw_input_types;

  auto raw_event = Codex::RawEvent<mep_layout>(d, o, s, t, event_number);

  auto number_of_raw_banks = raw_event.number_of_raw_banks;

  auto output_hits = parameters.dev_codex_all_hits + event_number * Codex::MaxHitsPerEvent;
  auto hits_size = parameters.dev_codex_all_hits_size + event_number;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    hits_size[0] = 0;
  }
  __syncthreads();

  uint8_t dct, strip_id, time0, time1;

  for (unsigned bank_number = threadIdx.y; bank_number < number_of_raw_banks; bank_number += blockDim.y) {
    auto raw_bank = raw_event.raw_bank(bank_number);

    if (raw_bank.type == (uint8_t) LHCb::Event::Enum::RawBank::BankType::CODEXError) {
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        dev_n_error_banks.increment();
      }
      continue;
    }

    const unsigned bank_size = raw_bank.end - raw_bank.data;
    const unsigned n_words = bank_size / 4; // 4 bytes per word

    if (n_words == 0) continue;
    if (bank_size % 4 != 0) continue;

    // Each thread handles multiple words
    for (unsigned word = threadIdx.x; word < n_words; word += blockDim.x) {
      unsigned base = word * 4;

      dct = raw_bank.data[base + 0];
      strip_id = raw_bank.data[base + 1];
      time0 = raw_bank.data[base + 2];
      time1 = raw_bank.data[base + 3];

      // store local hits for this word
      CodexHit local_hits[2];
      unsigned local_count = 0;

      // following the TELL40 mapping:
      // time 0 strip IDs 0 - 95: layer 0
      // time 0 strip IDs 96 - 143: layer 1
      // time 1 strip IDs 0 - 47: layer 1
      // time 1 strip IDs 48 - 143: layer 2

      // lambda to decode a single hit
      auto decode_hit = [&](uint8_t hit_time, uint8_t strip, uint8_t layer_offset) -> CodexHit {
        uint8_t layer = 0;
        uint8_t output_time = hit_time;

        if (layer_offset == 0) {
          layer = (strip < 96) ? 0 : 1;
          strip = (strip < 96) ? strip : (strip - 96);
        }
        else {
          layer = (strip < 48) ? 1 : 2;
          strip = (strip < 48) ? (strip + 48) : (strip - 48);
        }

        uint8_t singlet_id = dct * 3 + layer;
        uint8_t strip_type = (strip < 64) ? 0 : 1; // type 0 for phi strips (0-63), type 1 for eta strips (64-143)
        strip = (strip < 64) ? strip : strip - 64; // reindex within type

        return CodexHit(singlet_id, strip, strip_type, output_time);
      };

      if ((dct >= Codex::NumberOfDCTs) || (strip_id > Codex::MaxStripId)) { // sanity check on input data (skip dct > 13
                                                                            // and strip_id > 143)
        continue;
      }

      if (time0 != 0) {
        local_hits[0] = decode_hit(time0, strip_id, 0);
        local_count++;
        dev_histogram_time_cycle_occupancy.increment(time0);
      }
      if (time1 != 0) {
        local_hits[1] = decode_hit(time1, strip_id, 1);
        local_count++;
        dev_histogram_time_cycle_occupancy.increment(time1);
      }
      // write to global array using atomicAdd for exact hits count

      unsigned global_index = atomicAdd(hits_size, local_count);

      if (time0 != 0) {
        if (global_index < Codex::MaxHitsPerEvent) // memory extra protection
          output_hits[global_index] = local_hits[0];
      }
      if (time1 != 0) {
        if ((global_index + (time0 != 0)) < Codex::MaxHitsPerEvent) // memory extra protection
          output_hits[global_index + (time0 != 0)] = local_hits[1];
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    dev_histogram_n_hits.increment(hits_size[0]);
    // CLAMP THE SIZE for the next kernel
    if (hits_size[0] > Codex::MaxHitsPerEvent) {
      hits_size[0] = 0;
    }
  }
}

void codex_decode::codex_decode_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  // Temporary output
  set_size<dev_codex_all_hits_t>(arguments, first<host_number_of_events_t>(arguments) * Codex::MaxHitsPerEvent);
  set_size<dev_codex_all_hits_size_t>(arguments, first<host_number_of_events_t>(arguments) + 1);

  set_size<dev_codex_singlet_offsets_t>(
    arguments, first<host_number_of_events_t>(arguments) * Codex::NumberOfSinglets * 2 + 1); // Factor 2 for strip types

  set_size<host_codex_num_hits_t>(arguments, 1);
}

void codex_decode::codex_decode_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{

  auto const bank_version = first<host_raw_bank_version_t>(arguments);

  if (bank_version < 0) { // no CODEX banks present in data
    Allen::memset_async<host_codex_num_hits_t>(arguments, 0, context);
    return;
  }

  auto f_codex_decode_kernel = runtime_options.mep_layout ? codex_decode_kernel<true> : codex_decode_kernel<false>;

  global_function(f_codex_decode_kernel)(
    dim3(size<dev_event_list_t>(arguments)), dim3(m_block_dim_x, m_block_dim_y), context)(
    arguments,
    m_n_error_banks.data(context),
    m_histogram_n_hits.data(context),
    m_histogram_time_cycle_occupancy.data(context));

  PrefixSum::prefix_sum<dev_codex_all_hits_size_t, host_codex_num_hits_t>(*this, arguments, context);

  auto num_hits = first<host_codex_num_hits_t>(arguments);

  resize<dev_codex_hits_t>(arguments, num_hits);
  resize<dev_codex_hits_permutations_t>(arguments, num_hits);
  resize<dev_codex_hits_keys_t>(arguments, num_hits);

  Allen::memset_async<dev_codex_singlet_offsets_t>(arguments, 0, context);

  global_function(codex_decode_consolidate_hits)(dim3(size<dev_event_list_t>(arguments)), dim3(m_block_dim_x), context)(
    arguments, m_histogram_n_hits_vs_RPC_id.data(context));

  PrefixSum::prefix_sum<dev_codex_singlet_offsets_t>(*this, arguments, context);

  SegSort::segsort(
    *this,
    arguments,
    context,
    data<dev_codex_hits_keys_t>(arguments),
    data<dev_codex_all_hits_size_t>(arguments),
    size<dev_codex_all_hits_size_t>(arguments) - 1,
    data<dev_codex_hits_permutations_t>(arguments));
}
