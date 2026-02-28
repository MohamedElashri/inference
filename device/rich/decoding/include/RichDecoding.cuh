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

#include "AlgorithmTypes.cuh"
#include <RichSmartID.cuh>

#if defined(TARGET_DEVICE_CUDA)
#define RICH_DECODING_BLOCK_DIM_X 32
#define RICH_DECODING_BLOCK_DIM_Y 4
#else
// Default to CPU-like configuration
// Although Allen automatically sets this up for CPU,
// we need a variable for array sizes in compile-time
#define RICH_DECODING_BLOCK_DIM_X 1
#define RICH_DECODING_BLOCK_DIM_Y 1
#endif

namespace rich_decoding {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, uint) host_number_of_events;
    HOST_INPUT(host_raw_bank_version_t, int) host_raw_bank_version;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_rich_raw_input_t, char) dev_rich_raw_input;
    DEVICE_INPUT(dev_rich_raw_input_offsets_t, uint) dev_rich_raw_input_offsets;
    DEVICE_INPUT(dev_rich_raw_input_sizes_t, uint) dev_rich_raw_input_sizes;
    DEVICE_INPUT(dev_rich_raw_input_types_t, uint) dev_rich_raw_input_types;
    DEVICE_OUTPUT(dev_rich_hit_offsets_t, unsigned) dev_rich_hit_offsets;
    HOST_OUTPUT(host_rich_total_number_of_hits_t, unsigned) host_rich_total_number_of_hits;
    DEVICE_OUTPUT(dev_smart_ids_t, Allen::Rich::Decoding::SmartID) dev_smart_ids;
  };

  struct rich_decoding_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context&) const;

  private:
    // Changed to use 2D block dimensions for better parallelism
    // threads in x-dimension for warp-level operations
    // threads in y-dimension for bank-level parallelism
    Allen::Property<unsigned> m_block_dim_x {this, "block_dim_x", {RICH_DECODING_BLOCK_DIM_X}, "block x dimension"};
    Allen::Property<unsigned> m_block_dim_y {this, "block_dim_y", {RICH_DECODING_BLOCK_DIM_Y}, "block y dimension"};
    Allen::Property<dim3> m_block_dim {this,
                                       "block_dim",
                                       {RICH_DECODING_BLOCK_DIM_X, RICH_DECODING_BLOCK_DIM_Y, 1},
                                       "block dimensions"};
  };
} // namespace rich_decoding
