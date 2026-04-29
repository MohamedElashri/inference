/*****************************************************************************\
* (c) Copyright 2021 CERN for the benefit of the LHCb Collaboration           *
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
#include "AllenMonitoring.h"
#include "CodexModel.cuh"

namespace codex_decode {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    HOST_INPUT(host_raw_bank_version_t, int) host_raw_bank_version;
    DEVICE_INPUT(dev_codex_raw_input_t, char) dev_codex_raw_input;
    DEVICE_INPUT(dev_codex_raw_input_offsets_t, unsigned) dev_codex_raw_input_offsets;
    DEVICE_INPUT(dev_codex_raw_input_sizes_t, unsigned) dev_codex_raw_input_sizes;
    DEVICE_INPUT(dev_codex_raw_input_types_t, unsigned) dev_codex_raw_input_types;

    // Temporary output (before consolidation)
    DEVICE_OUTPUT(dev_codex_all_hits_t, CodexHit) dev_codex_all_hits;
    DEVICE_OUTPUT(dev_codex_all_hits_size_t, unsigned) dev_codex_all_hits_size;
    DEVICE_OUTPUT(dev_codex_hits_keys_t, uint32_t) dev_codex_hits_keys;

    // Output
    HOST_OUTPUT(host_codex_num_hits_t, unsigned) host_codex_num_hits;
    DEVICE_OUTPUT(dev_codex_hits_t, CodexHit) dev_codex_hits;
    DEVICE_OUTPUT(dev_codex_singlet_offsets_t, unsigned) dev_codex_singlet_offsets;
    DEVICE_OUTPUT(dev_codex_hits_permutations_t, unsigned) dev_codex_hits_permutations;
  };

  struct codex_decode_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Monitoring::Counter<> m_n_error_banks {this, "n_error_banks"};
    Allen::Property<unsigned> m_block_dim_x {this, "block_dim_x", 32, "block dimension X"};
    Allen::Property<unsigned> m_block_dim_y {this, "block_dim_y", 2, "block dimension Y"};

    Allen::Monitoring::Histogram<> m_histogram_n_hits {
      this,
      "n_hits",
      "n_hits",
      {Codex::MaxHitsPerEvent + 1, -0.5, Codex::MaxHitsPerEvent + 0.5}};

    Allen::Monitoring::Histogram2D<> m_histogram_n_hits_vs_RPC_id {
      this,
      "n_hits_vs_RPC_id",
      "n_hits_vs_RPC_id",
      {Codex::NumberOfSinglets * 2 + 1, -0.5f, Codex::NumberOfSinglets * 2 + 0.5f},
      {Codex::MaxHitsPerEvent + 1, -0.5, Codex::MaxHitsPerEvent + 0.5}};

    Allen::Monitoring::Histogram<> m_histogram_time_cycle_occupancy {this,
                                                                     "n_hits_vs_time_cycle",
                                                                     "n_hits_vs_time_cycle",
                                                                     {31u, -0.5f, 30.5f}};
  };

} // namespace codex_decode
