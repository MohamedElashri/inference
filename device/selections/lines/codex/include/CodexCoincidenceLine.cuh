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

namespace codex_coincidence_line {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_codex_double_coincidences_size_t, unsigned) dev_codex_double_coincidences_size;
    DEVICE_INPUT(dev_codex_triple_coincidences_size_t, unsigned) dev_codex_triple_coincidences_size;
    HOST_OUTPUT(host_line_data_t, LineData) host_line_data;
    HOST_OUTPUT(host_fn_parameters_t, char) host_fn_parameters;
  };

  struct codex_coincidence_line_t : public SelectionAlgorithm,
                                    Parameters,
                                    EventLine<codex_coincidence_line_t, Parameters> {

    struct DeviceProperties {
      int minCoinc;
      DeviceProperties(const codex_coincidence_line_t& algo, const Allen::Context&) : minCoinc(algo.m_minCoinc) {}
    };

    __device__ static std::tuple<unsigned> get_input(const Parameters&, const unsigned, const unsigned);

    __device__ static bool select(const Parameters&, const DeviceProperties&, std::tuple<unsigned> input);

  private:
    Allen::Property<int> m_minCoinc {this, "minCoinc", 1, "Minimum number of 2-cluster coincidences per event"};
  };
} // namespace codex_coincidence_line