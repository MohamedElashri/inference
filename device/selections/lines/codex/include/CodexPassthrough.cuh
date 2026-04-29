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
#pragma once

#include "AlgorithmTypes.cuh"
#include "ParticleTypes.cuh"
#include "EventLine.cuh"

namespace codex_passthrough_line {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;

    DEVICE_INPUT(dev_codex_passthrough_decisions_t, bool) dev_codex_passthrough_decisions;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;

    HOST_OUTPUT(host_line_data_t, LineData) host_line_data;
    HOST_OUTPUT(host_fn_parameters_t, char) host_fn_parameters;
  };

  struct codex_passthrough_line_t : public SelectionAlgorithm,
                                    Parameters,
                                    EventLine<codex_passthrough_line_t, Parameters> {
    __device__ static std::tuple<const bool>
    get_input(const Parameters& parameters, const unsigned event_number, const unsigned);

    __device__ static bool select(const Parameters& parameters, std::tuple<const bool> input);

  private:
  };
} // namespace codex_passthrough_line