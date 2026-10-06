/***************************************************************************** \
 * (c) Copyright 2000-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
// Gaudi
#include "GaudiAlg/Consumer.h"
#include "Gaudi/Accumulators.h"

// Detector
#include "Detector/FT/FTChannelID.h"

// LHCb
#include "Event/FTLiteCluster.h"
#include "FTDAQ/FTInfo.h"
#include "Kernel/LHCbID.h"

// Allen
#include "SciFiEventModel.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

using AllenFTClusterIDs = std::vector<uint32_t>;

// ==================================================================
//  Multi-event converter: raw device buffers → per-event FT channel IDs
//
//  Offsets: N_events * (n_zones + 1) unsigned, flat-concatenated
//  per-event zone-offset blocks.
// ==================================================================

class ConvertAllenFTClusters final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<AllenFTClusterIDs>(
      const Allen::device_buffer<unsigned>&, // per-event-per-zone offsets
      const Allen::device_buffer<char>&)> {  // raw SciFi hit data

public:
  ConvertAllenFTClusters(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"scifi_offsets", ""}, KeyValue {"scifi_hits", ""}},
      {KeyValue {"AllenFTClusterIDs", ""}})
  {}

  std::tuple<std::vector<AllenFTClusterIDs>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<unsigned>& dev_offsets,
    const Allen::device_buffer<char>& dev_hits) const override
  {
    auto h_offsets = dev_offsets.to_host();
    auto h_hits = dev_hits.to_host();

    const unsigned n_zones = SciFi::Constants::n_zones;
    const unsigned n_events = (h_offsets.size() - 1) / n_zones;
    const unsigned n_hits_total = h_offsets[h_offsets.size() - 1];

    SciFi::ConstHits all_hits {h_hits.data(), n_hits_total};

    std::vector<AllenFTClusterIDs> all_events;
    all_events.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {
      const unsigned evt_off = evt * n_zones;
      const unsigned begin = h_offsets[evt_off];
      const unsigned end = h_offsets[evt_off + n_zones];

      AllenFTClusterIDs ids;
      ids.reserve(end - begin);
      for (unsigned i = begin; i < end; ++i)
        ids.push_back(all_hits.id(i));

      all_events.emplace_back(std::move(ids));
    }

    return std::make_tuple(std::move(all_events));
  }
};

DECLARE_COMPONENT(ConvertAllenFTClusters)

// ==================================================================
//  Single-event comparison: Allen FT IDs  vs  Rec FTLiteClusters
// ==================================================================

class CompareRecAllenFTClusters final
  : public Gaudi::Functional::Consumer<void(const AllenFTClusterIDs&, const LHCb::FTLiteCluster::FTLiteClusters&)> {

public:
  CompareRecAllenFTClusters(const std::string& name, ISvcLocator* pSvcLocator);

  void operator()(const AllenFTClusterIDs& allen_ids, const LHCb::FTLiteCluster::FTLiteClusters& ft_lite_clusters)
    const override;

private:
  mutable Gaudi::Accumulators::Counter<> m_lonelyAllen {this, "onlyAllen hits"};
  mutable Gaudi::Accumulators::Counter<> m_multipleAllen {this, "multAllen hits"};
  mutable Gaudi::Accumulators::Counter<> m_lonelyRec {this, "onlyRec hits"};
};

DECLARE_COMPONENT(CompareRecAllenFTClusters)

CompareRecAllenFTClusters::CompareRecAllenFTClusters(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"AllenFTClusterIDs", ""}, KeyValue {"FTClusterLocation", LHCb::FTLiteClusterLocation::Default}})
{}

void CompareRecAllenFTClusters::operator()(
  const AllenFTClusterIDs& allen_ids,
  const LHCb::FTLiteCluster::FTLiteClusters& ft_lite_clusters) const
{
  std::vector<uint32_t> scifi_ids_rec;
  std::vector<LHCb::Detector::FTChannelID> scifi_ft_channel_ids;

  debug() << "Number of FT clusters (Allen) in this event " << allen_ids.size() << endmsg;
  debug() << "Number of FT clusters (Rec) in this event   " << ft_lite_clusters.size() << endmsg;

  // Rec side: extract IDs
  for (unsigned i = 0; i < LHCb::Detector::FT::nZonesTotal; ++i) {
    for (int quarter = 0; quarter < 2; quarter++) {
      for (const auto& clus : ft_lite_clusters.range(i * 2 + quarter)) {
        const auto ft_channel_id = clus.channelID();
        scifi_ft_channel_ids.push_back(ft_channel_id);
        scifi_ids_rec.emplace_back(LHCb::LHCbID {LHCb::LHCbID::channelIDtype::FT, ft_channel_id}.lhcbID());
      }
    }
  }

  // Match Allen → Rec
  for (const auto& id_allen : allen_ids) {
    auto tmp_iter =
      std::remove_if(scifi_ids_rec.begin(), scifi_ids_rec.end(), [&](auto& id_rec) { return id_rec == id_allen; });
    const auto n_found = std::distance(tmp_iter, scifi_ids_rec.end());
    scifi_ids_rec.erase(tmp_iter, scifi_ids_rec.end());

    if (n_found == 0) {
      debug() << "Could not match this FT cluster decoded by Allen to a FT cluster decoded by Rec" << endmsg;
      debug() << id_allen << endmsg;
      ++m_lonelyAllen;
    }
    else if (n_found > 1) {
      debug() << "This FT cluster decoded by Allen has multiple FT clusters decoded by Rec" << endmsg;
      debug() << id_allen << endmsg;
      ++m_multipleAllen;
    }
  }

  // Report lonely Rec hits
  for (const auto& cid : scifi_ft_channel_ids) {
    debug() << cid << " in Allen "
            << static_cast<unsigned>(
                 std::find(
                   scifi_ids_rec.begin(),
                   scifi_ids_rec.end(),
                   static_cast<uint32_t>(LHCb::LHCbID {LHCb::LHCbID::channelIDtype::FT, cid}.lhcbID())) ==
                 scifi_ids_rec.end())
            << endmsg;
  }

  for (const auto& id_rec : scifi_ids_rec) {
    ++m_lonelyRec;
    debug() << "Lonely Rec hit " << id_rec << endmsg;
  }
}
