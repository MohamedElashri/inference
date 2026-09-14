/***************************************************************************** \
 * (c) Copyright 2000-2018 CERN for the benefit of the LHCb Collaboration      *
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

// LHCb
#include "Event/VPLightCluster.h"

class CompareRecAllenVPHits final
  : public Gaudi::Functional::Consumer<void(const LHCb::VPLightClusters&, const LHCb::VPLightClusters&)> {

public:
  /// Standard constructor
  CompareRecAllenVPHits(const std::string& name, ISvcLocator* pSvcLocator);

  /// Algorithm execution
  void operator()(const LHCb::VPLightClusters& clusters_allen, const LHCb::VPLightClusters& clusters_rec)
    const override;

private:
  mutable Gaudi::Accumulators::Counter<> m_allen_n_clusters {this, "Allen VP clusters"};
  mutable Gaudi::Accumulators::Counter<> m_rec_n_clusters {this, "Rec VP clusters"};
};

DECLARE_COMPONENT(CompareRecAllenVPHits)

CompareRecAllenVPHits::CompareRecAllenVPHits(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"VPLightClustersAllen", ""}, KeyValue {"VPHitsLocation", LHCb::VPClusterLocation::Light}})
{}

void CompareRecAllenVPHits::operator()(
  const LHCb::VPLightClusters& clusters_allen,
  const LHCb::VPLightClusters& clusters_rec) const
{
  // The goal is to compare channel IDs of individual clusters from data
  // decoded with HLT1 (Allen) and HLT2 (Rec).
  std::vector<uint32_t> vp_ids_allen, vp_ids_rec;

  const auto n_hits_total_allen = clusters_allen.size();
  const auto n_hits_total_rec = clusters_rec.size();

  debug() << "Number of VP clusters (Allen) in this event " << n_hits_total_allen << endmsg;
  debug() << "Number of VP clusters (Rec) in this event   " << n_hits_total_rec << endmsg;

  m_allen_n_clusters += n_hits_total_allen;
  m_rec_n_clusters += n_hits_total_rec;

  // Allen side
  for (const auto& cl : clusters_allen) {
    vp_ids_allen.emplace_back(cl.channelID().channelID());
  }

  // Rec side
  for (const auto& cl : clusters_rec) {
    vp_ids_rec.emplace_back(cl.channelID().channelID());
  }

  for (const auto& vp_id_allen : vp_ids_allen) {
    auto tmp_iter = std::remove_if(
      vp_ids_rec.begin(), vp_ids_rec.end(), [&vp_id_allen](auto& vp_id_rec) { return vp_id_rec == vp_id_allen; });
    const auto n_hits_found = std::distance(tmp_iter, vp_ids_rec.end());
    vp_ids_rec.erase(tmp_iter, vp_ids_rec.end());
    if (n_hits_found == 0) {
      error() << "Could not match this VP cluster decoded by Allen to a VP cluster decoded by Rec" << endmsg;
      error() << vp_id_allen << endmsg;
    }
    else if (n_hits_found > 1) {
      error() << "This VP cluster decoded by Allen has multiple VP clusters decoded by Rec" << endmsg;
      error() << vp_id_allen << endmsg;
    }
  }

  if (!vp_ids_rec.empty()) {
    for (const auto& vp_hit_rec : vp_ids_rec) {
      error() << "Lonely Rec hit " << vp_hit_rec << endmsg;
    }
  }
}
