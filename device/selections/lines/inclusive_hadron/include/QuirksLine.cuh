/*****************************************************************************\
 * (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
 \*****************************************************************************/
#pragma once

// Allen
#include "AlgorithmTypes.cuh"
#include "EventLine.cuh"
#include "Logger.h"
#include "AllenMonitoring.h"

namespace quirks_line {
  struct Parameters {
    // Commonly required inputs, outputs and properties
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    HOST_OUTPUT(host_line_data_t, LineData) host_line_data;
    HOST_OUTPUT(host_fn_parameters_t, char) host_fn_parameters;
    // Line-specific inputs and properties
    DEVICE_INPUT(dev_quirks_pairs_t, unsigned) dev_quirks_pairs;
    DEVICE_OUTPUT(evtNo_t, uint64_t) evtNo;
    DEVICE_OUTPUT(runNo_t, unsigned) runNo;
  };

  // SelectionAlgorithm definition
  struct quirks_line_t : public SelectionAlgorithm, Parameters, EventLine<quirks_line_t, Parameters> {

    // Get input function
    __device__ static unsigned get_input(const Parameters& parameters, const unsigned event_number, const unsigned i);

    // Selection function
    __device__ static bool select(const Parameters& parameters, unsigned input);

    __device__ static bool fill_tuples(const Parameters&, unsigned input, unsigned index, bool sel);

    using monitoring_types = std::tuple<evtNo_t, runNo_t>;
  };
} // namespace quirks_line
