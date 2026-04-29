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
#include "CodexModel.cuh"

namespace codex_clustering {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;

    HOST_INPUT(host_codex_num_hits_t, unsigned) host_codex_num_hits;
    DEVICE_INPUT(dev_codex_hits_t, CodexHit) dev_codex_hits;
    DEVICE_INPUT(dev_codex_hits_permutations_t, unsigned) dev_codex_hits_permutations;
    DEVICE_INPUT(dev_codex_hits_size_t, unsigned) dev_codex_hits_size;
    DEVICE_INPUT(dev_codex_singlet_offsets_t, unsigned) dev_codex_singlet_offsets;

    // Temporary output before consolidating
    DEVICE_OUTPUT(dev_codex_all_clusters_t, CodexSideCluster) dev_codex_all_clusters;
    DEVICE_OUTPUT(dev_codex_all_clusters_sizes_t, unsigned) dev_codex_all_clusters_sizes;
    DEVICE_OUTPUT(dev_codex_draft_clusters_t, CodexCluster) dev_codex_draft_clusters;

    // Output
    DEVICE_OUTPUT(dev_codex_clusters_t, CodexCluster) dev_codex_clusters;
    DEVICE_OUTPUT(dev_codex_cluster_size_t, unsigned) dev_codex_cluster_size;

    HOST_OUTPUT(host_codex_num_clusters_t, unsigned) host_codex_num_clusters;
  };

  struct codex_clustering_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<unsigned> m_block_dim_x {this, "block_dim_x", 32, "block dimension X"};
  };

} // namespace codex_clustering
