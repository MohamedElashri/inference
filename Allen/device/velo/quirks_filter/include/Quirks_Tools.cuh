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
#include "VeloConsolidated.cuh"
#include "VeloEventModel.cuh"
#include "ClusteringDefinitions.cuh"
#include "VeloTools.cuh"
#include "BinarySearch.cuh"

#include <cstdint>
#include <cmath>

namespace quirks_tools {

  struct Parameters {
    // Required inputs (same structure as codex example)
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    HOST_INPUT(host_total_number_of_velo_clusters_t, unsigned int) host_total_number_of_velo_clusters;

    DEVICE_INPUT(dev_module_cluster_num_t, unsigned) dev_module_cluster_num;
    DEVICE_INPUT(dev_velo_clusters_t, Velo::Clusters) dev_velo_clusters;
    DEVICE_INPUT(dev_sorted_velo_clusters_container_t, char) dev_sorted_velo_clusters_container;
    DEVICE_INPUT(dev_offsets_estimated_input_size_t, unsigned) dev_offsets_estimated_input_size;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;

    // Output: one int per event: 0 = fail selection, 1 = pass selection
    DEVICE_OUTPUT(dev_quirks_pairs_t, unsigned) dev_quirks_pairs;

    // per-module pair-list storage:
    // dev_pair_count: number of pairs produced for (event_idx* pairs + layer)
    // dev_pair_list: encoded uint32 pairs (i<<16 | j) stored with per-module stride = max_pairs_per_module
    DEVICE_OUTPUT(dev_pair_count_t, unsigned) dev_pair_count;
    DEVICE_OUTPUT(dev_pair_list_t, uint32_t) dev_pair_list;
  };

  // pair-list generator: one block per module, emits (i,j) seeds that pass cheap seed checks
  __global__ void generate_pair_list_kernel(
    Parameters parameters,
    const unsigned max_opposite_considered,
    const unsigned hit_threshold,
    const unsigned max_pairs_per_module,
    const int16_t host_maxPHI_i16,
    const int16_t host_window_start,
    const float maxR);

  // selection kernel prototype: now iterates over pair-lists (last arg = max_pairs_per_module)
  __global__ void quirks_selection_kernel(
    Parameters parameters,
    const int16_t host_maxPHIDF_i16,
    const unsigned minStations,
    const unsigned max_pairs_per_module);

  // Device algorithm wrapper (follows codex_cluster style)
  struct quirks_tools_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    // New tunables: explicit blockDim.x and blockDim.y (defaults multiply to 256 = 16*16).
    Allen::Property<unsigned> m_block_dim_x {this, "block_dim_x", 16u, "block dimension X (owners / lanes)"};
    Allen::Property<unsigned> m_block_dim_y {
      this,
      "block_dim_y",
      8u,
      "block dimension Y (helpers) - product X*Y should be 256"};

    // pair-list cap (per-module)
    Allen::Property<unsigned> m_max_pairs_per_module {
      this,
      "max_pairs_per_module",
      20u,
      "max seed pairs stored per module (cap)"};

    // other tunables
    Allen::Property<unsigned> m_max_opposite_considered {
      this,
      "max_opposite_considered",
      6,
      "max hits to search after binary search"};
    Allen::Property<unsigned> m_hit_threshold {this, "hit_threshold", 170, "max hits per module to process"};
    Allen::Property<unsigned> m_minStations {this, "minStations", 6u, "minimum consecutive stations required"};
    Allen::Property<float> m_maxR {this, "maxR", 4.5, "maxR (r-difference)"};
    Allen::Property<float> m_maxPHI {this, "maxPHI", 0.1, "maxPHI (deg)"};
    Allen::Property<float> m_maxPHIDF {this, "maxPHIDF", 0.13, "maxPHIDF (deg)"};
  };
} // namespace quirks_tools
