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

#include "CodexModel.cuh"

namespace codex_prepare_passthrough {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    HOST_INPUT(host_raw_bank_version_t, int) host_raw_bank_version;

    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_codex_raw_input_t, char) dev_codex_raw_input;
    DEVICE_INPUT(dev_codex_raw_input_offsets_t, unsigned) dev_codex_raw_input_offsets;
    DEVICE_INPUT(dev_codex_raw_input_sizes_t, unsigned) dev_codex_raw_input_sizes;
    DEVICE_INPUT(dev_codex_raw_input_types_t, unsigned) dev_codex_raw_input_types;

    DEVICE_OUTPUT(dev_codex_passthrough_decisions_t, bool) dev_codex_passthrough_decisions;
  };

  struct codex_prepare_passthrough_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<unsigned> m_block_dim_x {this, "block_dim_x", 32, "block dimension X"};
  };
} // namespace codex_prepare_passthrough
