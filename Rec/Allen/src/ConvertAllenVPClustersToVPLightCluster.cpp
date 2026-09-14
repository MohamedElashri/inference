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
/**
 * Convert Allen VP hits (raw buffers) into LHCb::VPLightClusters,
 * scattering per-event clusters to individual event stores.
 */

// Gaudi
#include <LHCbAlgs/Transformer.h>
// #include "GaudiKernel/StdArrayAsProperty.h"

// LHCb
#include "Event/VPLightCluster.h"
#include "Detector/VP/VPChannelID.h"
#include "Kernel/LHCbID.h"

// Allen
#include "VeloEventModel.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

class ConvertAllenVPClustersToVPLightCluster final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<LHCb::VPLightClusters>(
      const Allen::device_buffer<unsigned>&,
      const Allen::device_buffer<unsigned>&,
      const Allen::device_buffer<char>&)> {

public:
  /// Standard constructor
  ConvertAllenVPClustersToVPLightCluster(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      // Inputs
      {KeyValue {"vp_hits_num", ""}, KeyValue {"vp_hit_offsets", ""}, KeyValue {"vp_hits", ""}},
      // Outputs
      {KeyValue {"VPLightClustersFromAllen", ""}})
  {}

  /// Algorithm execution
  /// Returns a vector of VPLightClusters, one entry per sub-event in the slice
  std::tuple<std::vector<LHCb::VPLightClusters>> operator()(
    const EventContext&,
    const Allen::device_buffer<unsigned>& vp_hits_num,
    const Allen::device_buffer<unsigned>& vp_hit_offsets,
    const Allen::device_buffer<char>& vp_hits) const override
  {
    // Copy device buffers to host
    auto vp_hits_num_host = vp_hits_num.to_host();
    auto vp_hit_offsets_host = vp_hit_offsets.to_host();
    auto vp_hits_host = vp_hits.to_host();

    // Number of events: each event has n_module_pairs module pairs,
    // so the offsets array has N * n_module_pairs + 1 elements.
    const unsigned n_events = vp_hit_offsets_host.size() / Velo::Constants::n_module_pairs;

    std::vector<LHCb::VPLightClusters> all_clusters;
    all_clusters.reserve(n_events);

    // Total number of hits across all events determines the ConstClusters span
    const auto n_hits_total = vp_hit_offsets_host[vp_hit_offsets_host.size() - 1];
    Velo::ConstClusters vp_hit_container {vp_hits_host.data(), n_hits_total};

    for (unsigned evt = 0; evt < n_events; ++evt) {
      const unsigned base_offset = evt * Velo::Constants::n_module_pairs;

      const unsigned evt_total = vp_hit_offsets_host[base_offset + Velo::Constants::n_module_pairs];

      LHCb::VPLightClusters clusters;
      clusters.reserve(evt_total - vp_hit_offsets_host[base_offset]);

      for (unsigned i = 0; i < Velo::Constants::n_module_pairs; ++i) {
        const auto module_hit_start = vp_hit_offsets_host[base_offset + i];
        const auto module_hit_num = vp_hits_num_host[base_offset + i];

        for (unsigned hit_number = 0; hit_number < module_hit_num; ++hit_number) {
          const auto hit_index = module_hit_start + hit_number;

          // Strip detector type bits from Allen LHCbID
          const auto vpID = vp_hit_container.id(hit_index) & 0x0FFFFFFF;

          const float x = vp_hit_container.x(hit_index);
          const float y = vp_hit_container.y(hit_index);
          const float z = vp_hit_container.z(hit_index);

          // Create a light cluster with fraction = 1 (same as Rec light clusters)
          clusters.emplace_back(static_cast<unsigned char>(1), static_cast<unsigned char>(1), x, y, z, vpID);
        }
      }

      all_clusters.emplace_back(std::move(clusters));
    }

    return std::make_tuple(std::move(all_clusters));
  }
};

DECLARE_COMPONENT(ConvertAllenVPClustersToVPLightCluster)
