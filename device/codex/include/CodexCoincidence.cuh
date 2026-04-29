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
#include "CodexModel.cuh"

namespace codex_coincidence {
  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    MASK_INPUT(dev_event_list_t) dev_event_list;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;

    HOST_INPUT(host_codex_num_clusters_t, unsigned) host_codex_num_clusters;
    DEVICE_INPUT(dev_codex_clusters_t, CodexCluster) dev_codex_clusters;
    DEVICE_INPUT(dev_codex_cluster_size_t, unsigned) dev_codex_cluster_size;

    // Used for validation
    DEVICE_OUTPUT(dev_codex_coincidences_t, CodexCoincidence) dev_codex_coincidences;
    DEVICE_OUTPUT(dev_codex_created_coincidences_size_t, unsigned) dev_codex_created_coincidences_size;

    // Used for selection
    DEVICE_OUTPUT(dev_codex_double_coincidences_size_t, unsigned) dev_codex_double_coincidences_size;
    DEVICE_OUTPUT(dev_codex_triple_coincidences_size_t, unsigned) dev_codex_triple_coincidences_size;
  };

  struct codex_coincidence_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters>, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>&,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<unsigned> m_block_dim_x {this, "block_dim_x", 32, "block dimension X"};
  };
} // namespace codex_coincidence
