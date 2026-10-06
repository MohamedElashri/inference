/***************************************************************************** \
 * (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration          *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "GaudiAlg/Consumer.h"
#include <Gaudi/Accumulators.h>

#include "Event/VPFullCluster.h"

#include "VeloEventModel.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

#include <unordered_map>
#include <vector>

// ==================================================================
//  Multi-event converter: Allen VP raw buffers → per-event channelID+size vectors
// ==================================================================

struct VPClusterSize {
  unsigned channelID;
  int16_t clusterSize;
};
using VPClusterSizes = std::vector<VPClusterSize>;

class ConvertAllenVPClusterSizes final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<VPClusterSizes>(
      const Allen::device_buffer<unsigned>&, // vp_hits_num (N * n_module_pairs)
      const Allen::device_buffer<unsigned>&, // vp_hit_offsets (N * (n_module_pairs + 1))
      const Allen::device_buffer<char>&)> {  // vp_hits

public:
  ConvertAllenVPClusterSizes(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"vp_hits_num", ""}, KeyValue {"vp_hit_offsets", ""}, KeyValue {"vp_hits", ""}},
      {KeyValue {"VPClusterSizes", ""}})
  {}

  std::tuple<std::vector<VPClusterSizes>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<unsigned>& dev_hits_num,
    const Allen::device_buffer<unsigned>& dev_hit_offsets,
    const Allen::device_buffer<char>& dev_hits) const override
  {
    auto h_hits_num = dev_hits_num.to_host();
    auto h_hit_offsets = dev_hit_offsets.to_host();
    auto h_hits = dev_hits.to_host();

    const unsigned n_events = (h_hit_offsets.size() - 1) / Velo::Constants::n_module_pairs;
    const unsigned n_hits_total = h_hit_offsets[h_hit_offsets.size() - 1];
    Velo::ConstClusters all_hits {h_hits.data(), n_hits_total};

    std::vector<VPClusterSizes> all_sizes;
    all_sizes.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {
      const unsigned base_off = evt * Velo::Constants::n_module_pairs;

      VPClusterSizes sizes;
      sizes.reserve(h_hit_offsets[base_off + Velo::Constants::n_module_pairs] - h_hit_offsets[base_off]);

      for (unsigned i = 0; i < Velo::Constants::n_module_pairs; ++i) {
        const unsigned mod_start = h_hit_offsets[base_off + i];
        const unsigned mod_num = h_hits_num[base_off + i];
        for (unsigned j = 0; j < mod_num; ++j) {
          const unsigned idx = mod_start + j;
          sizes.push_back({all_hits.id(idx) & 0x0FFFFFFF, all_hits.cluster_size(idx)});
        }
      }
      all_sizes.emplace_back(std::move(sizes));
    }

    return std::make_tuple(std::move(all_sizes));
  }
};

DECLARE_COMPONENT(ConvertAllenVPClusterSizes)

// ==================================================================
//  Single-event comparison: Allen cluster sizes vs Rec VPFullClusters
// ==================================================================

class TestAllenRetinaClusterSize final
  : public Gaudi::Functional::Consumer<void(const VPClusterSizes&, const std::vector<LHCb::VPFullCluster>&)> {

public:
  TestAllenRetinaClusterSize(const std::string& name, ISvcLocator* pSvcLocator);

  void operator()(const VPClusterSizes& allen_sizes, const std::vector<LHCb::VPFullCluster>& rec_clusters)
    const override;

private:
  mutable Gaudi::Accumulators::Counter<> m_n_clusters {this, "Clusters compared"};
  mutable Gaudi::Accumulators::Counter<> m_n_size_mismatch {this, "Size mismatches"};
  mutable Gaudi::Accumulators::Counter<> m_n_unmatched {this, "Allen clusters not in Rec"};
  mutable Gaudi::Accumulators::Counter<> m_n_mismatch_edge {this, "Mismatches Allen>Rec (benign on MC)"};
  mutable Gaudi::Accumulators::Counter<> m_n_mismatch_other {this, "Mismatches Allen<Rec (unexpected)"};
};

DECLARE_COMPONENT(TestAllenRetinaClusterSize)

TestAllenRetinaClusterSize::TestAllenRetinaClusterSize(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"VPClusterSizes", ""}, KeyValue {"VPFullClustersLocation", LHCb::VPFullClusterLocation::Default}})
{}

void TestAllenRetinaClusterSize::operator()(
  const VPClusterSizes& allen_sizes,
  const std::vector<LHCb::VPFullCluster>& rec_clusters) const
{
  std::unordered_map<unsigned, const LHCb::VPFullCluster*> rec_map;
  rec_map.reserve(rec_clusters.size());
  for (const auto& rc : rec_clusters)
    rec_map.emplace(rc.channelID().channelID(), &rc);

  for (const auto& as : allen_sizes) {
    auto it = rec_map.find(as.channelID);
    if (it == rec_map.end()) {
      ++m_n_unmatched;
      error() << "Allen cluster not in Rec, channelID = 0x" << std::hex << as.channelID << std::dec << endmsg;
      continue;
    }

    const auto rec_size = static_cast<int16_t>(it->second->pixels().size());
    ++m_n_clusters;
    if (as.clusterSize == rec_size) continue;

    ++m_n_size_mismatch;
    if (as.clusterSize > rec_size)
      ++m_n_mismatch_edge;
    else {
      ++m_n_mismatch_other;
      error() << "Cluster size mismatch Allen < Rec:"
              << " channelID = 0x" << std::hex << as.channelID << std::dec << " Allen = " << as.clusterSize
              << " Rec = " << rec_size << endmsg;
    }
  }
}
