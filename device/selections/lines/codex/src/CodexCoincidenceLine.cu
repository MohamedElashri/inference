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
#include "CodexCoincidenceLine.cuh"

INSTANTIATE_LINE(codex_coincidence_line::codex_coincidence_line_t, codex_coincidence_line::Parameters)

__device__ std::tuple<unsigned> codex_coincidence_line::codex_coincidence_line_t::get_input(
  const Parameters&,
  const unsigned event_number,
  const unsigned)
{
  return std::make_tuple(event_number);
}

__device__ bool codex_coincidence_line::codex_coincidence_line_t::select(
  const Parameters& p,
  const DeviceProperties& properties,
  std::tuple<unsigned> input)
{
  const auto event_number = std::get<0>(input);
  int number_of_coincidences = 0;
  if (p.dev_codex_double_coincidences_size || p.dev_codex_triple_coincidences_size) {
    number_of_coincidences += p.dev_codex_double_coincidences_size[event_number];
    number_of_coincidences += p.dev_codex_triple_coincidences_size[event_number];
  }
  return number_of_coincidences >= properties.minCoinc;
}
