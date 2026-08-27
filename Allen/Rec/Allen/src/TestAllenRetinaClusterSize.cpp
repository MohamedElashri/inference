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
#include <Kernel/EventLocalAllocator.h>

#include "VeloEventModel.cuh"
#include "Logger.h"

#include <unordered_map>

class TestAllenRetinaClusterSize final : public Gaudi::Functional::Consumer<void(
                                           const std::vector<unsigned, LHCb::Allocators::EventLocal<unsigned>>&,
                                           const std::vector<unsigned, LHCb::Allocators::EventLocal<unsigned>>&,
                                           const std::vector<char, LHCb::Allocators::EventLocal<char>>&,
                                           const std::vector<LHCb::VPFullCluster>&)> {

public:
  TestAllenRetinaClusterSize(const std::string& name, ISvcLocator* pSvcLocator);

  void operator()(
    const std::vector<unsigned, LHCb::Allocators::EventLocal<unsigned>>&,
    const std::vector<unsigned, LHCb::Allocators::EventLocal<unsigned>>&,
    const std::vector<char, LHCb::Allocators::EventLocal<char>>&,
    const std::vector<LHCb::VPFullCluster>&) const override;

private:
  mutable Gaudi::Accumulators::Counter<> m_n_clusters {this, "Clusters compared"};
  mutable Gaudi::Accumulators::Counter<> m_n_size_mismatch {this, "Size mismatches"};
  mutable Gaudi::Accumulators::Counter<> m_n_unmatched {this, "Allen clusters not in Rec"};
  // Allen > Rec is benign on MC (HLT2 clips out-of-sensor pixels from wrong
  // TopologyIDs). Allen < Rec must never happen.
  mutable Gaudi::Accumulators::Counter<> m_n_mismatch_edge {this, "Mismatches Allen>Rec (benign on MC)"};
  mutable Gaudi::Accumulators::Counter<> m_n_mismatch_other {this, "Mismatches Allen<Rec (unexpected)"};
};

DECLARE_COMPONENT(TestAllenRetinaClusterSize)

TestAllenRetinaClusterSize::TestAllenRetinaClusterSize(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"vp_hits_num", ""},
     KeyValue {"vp_hit_offsets", ""},
     KeyValue {"vp_hits", ""},
     KeyValue {"VPFullClustersLocation", LHCb::VPFullClusterLocation::Default}})
{}

void TestAllenRetinaClusterSize::operator()(
  const std::vector<unsigned, LHCb::Allocators::EventLocal<unsigned>>& vp_hits_num,
  const std::vector<unsigned, LHCb::Allocators::EventLocal<unsigned>>& vp_hit_offsets,
  const std::vector<char, LHCb::Allocators::EventLocal<char>>& vp_hits,
  std::vector<LHCb::VPFullCluster> const& hit_handler) const
{
  const auto n_hits_total_allen = vp_hit_offsets[Velo::Constants::n_module_pairs];
  Velo::ConstClusters vp_hit_container_allen {vp_hits.data(), n_hits_total_allen};

  std::unordered_map<unsigned, const LHCb::VPFullCluster*> rec_by_channelid;
  rec_by_channelid.reserve(hit_handler.size());
  for (const auto& rec_cluster : hit_handler) {
    rec_by_channelid.emplace(rec_cluster.channelID().channelID(), &rec_cluster);
  }

  for (unsigned i = 0; i < Velo::Constants::n_module_pairs; ++i) {
    const auto module_hit_start = vp_hit_offsets[i];
    const auto module_hit_num = vp_hits_num[i];
    for (unsigned hit_number = 0; hit_number < module_hit_num; ++hit_number) {
      const auto hit_index = module_hit_start + hit_number;

      // Strip the LHCbID detector-type nibble to get the bare channelID.
      const auto allen_channelid = vp_hit_container_allen.id(hit_index) & 0x0FFFFFFFu;
      const auto allen_size = vp_hit_container_allen.cluster_size(hit_index);

      auto it = rec_by_channelid.find(allen_channelid);
      if (it == rec_by_channelid.end()) {
        ++m_n_unmatched;
        error() << "Allen cluster not in Rec, channelID = 0x" << std::hex << allen_channelid << std::dec << endmsg;
        continue;
      }

      const auto rec_size = static_cast<int16_t>(it->second->pixels().size());

      ++m_n_clusters;
      if (allen_size == rec_size) continue;

      ++m_n_size_mismatch;
      if (allen_size > rec_size) {
        // HLT2 additionally drops pixels that fall outside the sensor via
        // pixel_in_valid_region. That check only exists for back-compatibility
        // with MC samples produced with the wrong TopologyID, and is a no-op
        // on real data and on correct MC. We decided to not implement it in
        // Allen to avoid possible slow downs.
        // See LHCb/VP/VPDAQ/src/VPRetinaFullClusterDecoder.cpp L126-L135
        ++m_n_mismatch_edge;
      }
      else {
        ++m_n_mismatch_other;
        error() << "Cluster size mismatch Allen < Rec:"
                << " channelID = 0x" << std::hex << allen_channelid << std::dec << " Allen = " << allen_size
                << " Rec = " << rec_size << endmsg;
      }
    }
  }
}
