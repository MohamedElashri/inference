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
#include "CodexPreparePassthrough.cuh"
#include "BankTypes.h"
#include <BackendCommon.h>
#include <MEPTools.h>
#include <PrefixSum.cuh>

INSTANTIATE_ALGORITHM(codex_prepare_passthrough::codex_prepare_passthrough_t)

template<bool mep_layout>
__global__ void codex_prepare_passthrough_kernel(
  codex_prepare_passthrough::Parameters parameters,
  unsigned number_of_events)
{
  for (unsigned idx = blockIdx.x * blockDim.x + threadIdx.x; idx < number_of_events; idx += blockDim.x * gridDim.x) {
    unsigned const event_number = parameters.dev_event_list[idx];
    const auto passthrough_decision = parameters.dev_codex_passthrough_decisions + event_number;

    const char* d = parameters.dev_codex_raw_input;
    const uint32_t* o = parameters.dev_codex_raw_input_offsets;
    const uint32_t* s = parameters.dev_codex_raw_input_sizes;
    const uint32_t* t = parameters.dev_codex_raw_input_types;

    auto raw_event = Codex::RawEvent<mep_layout>(d, o, s, t, event_number);
    bool has_nonempty_bank = false;

    for (uint32_t i = 0; i < raw_event.number_of_raw_banks; ++i) {
      auto raw_bank = raw_event.raw_bank(i);
      const unsigned bank_size = raw_bank.end - raw_bank.data;
      if (bank_size > 0) {
        has_nonempty_bank = true;
        break;
      }
    }
    passthrough_decision[0] = has_nonempty_bank;
  }
}

void codex_prepare_passthrough::codex_prepare_passthrough_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_codex_passthrough_decisions_t>(arguments, first<host_number_of_events_t>(arguments));
}

void codex_prepare_passthrough::codex_prepare_passthrough_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_codex_passthrough_decisions_t>(arguments, false, context);

  auto const bank_version = first<host_raw_bank_version_t>(arguments);

  if (bank_version < 0) { // no CODEX banks present in data
    return;
  }

  auto f_codex_prepare_passthrough_kernel =
    runtime_options.mep_layout ? codex_prepare_passthrough_kernel<true> : codex_prepare_passthrough_kernel<false>;

  auto const number_of_events = size<dev_event_list_t>(arguments);

  global_function(f_codex_prepare_passthrough_kernel)(
    dim3(size<dev_event_list_t>(arguments)), dim3(m_block_dim_x), context)(arguments, number_of_events);
}
