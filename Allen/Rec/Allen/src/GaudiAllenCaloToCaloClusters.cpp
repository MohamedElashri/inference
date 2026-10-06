/*****************************************************************************\
* (c) Copyright 2008-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include <vector>

#include "GaudiAlg/Transformer.h"

#include "Event/Track.h"

#include "VeloConsolidated.cuh"
#include "CaloCluster.cuh"
#include "Event/CaloClusters_v2.h"
#include "Detector/Calo/CaloCellID.h"
#include "GaudiKernel/Point3DTypes.h"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

/**
 * Convert Allen CaloClusters (raw device buffers) into
 * LHCb::Event::Calo::Clusters, scattering per-event containers
 * to individual event stores.
 */

class ConvertAllenCaloToCaloClusters final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<LHCb::Event::Calo::Clusters>(
      const Allen::device_buffer<unsigned>&,       // cluster offsets (N+1)
      const Allen::device_buffer<CaloCluster>&)> { // all clusters (concatenated)

public:
  ConvertAllenCaloToCaloClusters(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"allen_ecal_cluster_offsets", ""}, KeyValue {"allen_ecal_clusters", ""}},
      {KeyValue {"AllenEcalClusters", "Allen/Calo/EcalCluster"}})
  {}

  std::tuple<std::vector<LHCb::Event::Calo::Clusters>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<unsigned>& dev_offsets,
    const Allen::device_buffer<CaloCluster>& dev_clusters) const override
  {
    auto h_offsets = dev_offsets.to_host();
    auto h_clusters = dev_clusters.to_host();

    const unsigned n_events = h_offsets.size() - 1;

    std::vector<LHCb::Event::Calo::Clusters> all_clusters;
    all_clusters.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {
      const unsigned begin = h_offsets[evt];
      const unsigned end = h_offsets[evt + 1];
      const unsigned n_clu = end - begin;

      LHCb::Event::Calo::Clusters out;
      out.reserve(n_clu);

      for (unsigned i = begin; i < end; ++i) {
        const auto& cluster = h_clusters[i];

        auto seedCellID = LHCb::Detector::Calo::DenseIndex::details::toCellID(cluster.center_id);
        if (!LHCb::Detector::Calo::isValid(seedCellID)) continue;

        auto clusterOut = out.emplace_back<SIMDWrapper::InstructionSet::Scalar>();

        auto entry = clusterOut.entries().emplace_back();
        entry.setCellID(seedCellID);
        entry.setEnergy(cluster.e);
        entry.setFraction(1.f);
        entry.setStatus(LHCb::CaloDigitStatus::Mask::UseForEnergy | LHCb::CaloDigitStatus::Mask::SeedCell);

        for (unsigned j = 0; j < Calo::Constants::max_neighbours; ++j) {
          if (cluster.digits[j] == USHRT_MAX) continue;
          auto cellID = LHCb::Detector::Calo::DenseIndex::details::toCellID(cluster.digits[j]);
          if (!LHCb::Detector::Calo::isValid(cellID)) continue;

          auto entry = clusterOut.entries().emplace_back();
          entry.setCellID(cellID);
          entry.setEnergy(0.f);
          entry.setFraction(1.f);
          entry.setStatus(LHCb::CaloDigitStatus::Mask::UseForEnergy | LHCb::CaloDigitStatus::Mask::OwnedCell);
        }

        clusterOut.setCellID(seedCellID);
        clusterOut.setType(LHCb::Event::Calo::Clusters::Type::Area3x3);
        clusterOut.setEnergy(cluster.e);
        clusterOut.setPosition({cluster.x, cluster.y, Calo::Constants::z});
      }

      all_clusters.emplace_back(std::move(out));
    }

    return std::make_tuple(std::move(all_clusters));
  }

private:
  Gaudi::Property<float> m_EtCalo {this, "EtCalo", 400 * Allen::Units::MeV, "Default ET for Calo Clusters"};
};

DECLARE_COMPONENT(ConvertAllenCaloToCaloClusters)
