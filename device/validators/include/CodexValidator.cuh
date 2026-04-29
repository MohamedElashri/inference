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

#include "SciFiEventModel.cuh"
#include "States.cuh"
#include "AlgorithmTypes.cuh"
#include "ParticleTypes.cuh"
#include "CheckerTracks.cuh"
#include "CheckerInvoker.h"
#include "TrackChecker.h"
#include "CodexModel.cuh"

namespace codex_validator {
  struct Parameters {

    // Basic
    MASK_INPUT(dev_event_list_t) dev_event_list;
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;

    // Hits
    DEVICE_INPUT(dev_codex_hits_t, CodexHit) dev_codex_hits;
    DEVICE_INPUT(dev_codex_hits_size_t, unsigned) dev_codex_hits_size;
    DEVICE_INPUT(dev_codex_hits_permutations_t, unsigned) dev_codex_hits_permutations;
    DEVICE_INPUT(dev_codex_singlet_offsets_t, unsigned) dev_codex_singlet_offsets;

    // Clusterization
    DEVICE_INPUT(dev_codex_clusters_t, CodexCluster) dev_codex_clusters;
    DEVICE_INPUT(dev_codex_cluster_size_t, unsigned) dev_codex_cluster_size;

    // Coincidence
    DEVICE_INPUT(dev_codex_coincidences_t, CodexCoincidence) dev_codex_coincidences;
    DEVICE_INPUT(dev_codex_created_coincidences_size_t, unsigned) dev_codex_created_coincidences_size;
    DEVICE_INPUT(dev_codex_double_coincidences_size_t, unsigned) dev_codex_double_coincidences_size;
    DEVICE_INPUT(dev_codex_triple_coincidences_size_t, unsigned) dev_codex_triple_coincidences_size;
  };

  __global__ void codex_validator(Parameters parameters);

  struct codex_validator_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;

    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Allen::Property<dim3> m_block_dim {this, "block_dim", {1, 1, 1}, "block dimensions"};
    Allen::Property<std::string> m_root_output_filename {this,
                                                         "root_output_filename",
                                                         "CodexCheckerPlots.root",
                                                         "root output filename"};
  };
} // namespace codex_validator
