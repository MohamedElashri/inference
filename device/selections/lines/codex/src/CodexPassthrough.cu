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
#include "CodexPassthrough.cuh"

INSTANTIATE_LINE(codex_passthrough_line::codex_passthrough_line_t, codex_passthrough_line::Parameters)

__device__ std::tuple<const bool> codex_passthrough_line::codex_passthrough_line_t::get_input(
  const Parameters& p,
  const unsigned event_number,
  const unsigned)
{

  bool has_codex;

  const auto event_decision = p.dev_codex_passthrough_decisions + event_number;
  has_codex = event_decision[0];

  return std::make_tuple(has_codex);
}

__device__ bool codex_passthrough_line::codex_passthrough_line_t::select(
  const Parameters&,
  std::tuple<const bool> input)
{
  return std::get<0>(input);
}
