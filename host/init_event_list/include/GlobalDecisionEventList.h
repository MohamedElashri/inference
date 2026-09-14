/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"

namespace global_decision_event_list {
  struct Parameters {
    HOST_INPUT(host_global_decision_t, bool) host_global_decision;
    HOST_OUTPUT(host_event_list_output_t, unsigned) host_event_list;
    MASK_OUTPUT(dev_event_list_output_t) dev_event_list;
  };

  struct global_decision_event_list_t : public HostAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions& runtime_options,
      const Constants& constants,
      const Allen::Context& context) const;
  };
} // namespace global_decision_event_list
