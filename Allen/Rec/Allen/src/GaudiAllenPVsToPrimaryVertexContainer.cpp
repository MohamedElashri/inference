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

/**
 * Convert PV::Vertex (raw device buffers) into LHCb::Event::PV::PrimaryVertexContainer
 *
 * Multi-event: receives per-slice raw device buffers, copies to host,
 * scatters per-event vertex containers to event stores.
 */

// Gaudi
#include "GaudiAlg/Transformer.h"
#include "Gaudi/Accumulators.h"

// LHCb
#include "Event/PrimaryVertices.h"

// Allen
#include "PV_Definitions.cuh"
#include "patPV_Definitions.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

using Vertices = LHCb::Event::PV::PrimaryVertexContainer;

class GaudiAllenPVsToPrimaryVertexContainer final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<Vertices>(
      const Allen::device_buffer<unsigned>&,  // number of PVs per event (N elements)
      const Allen::device_buffer<PV::Vertex>& // all PVs: N * max PVs per event, flat
      )> {

public:
  GaudiAllenPVsToPrimaryVertexContainer(const std::string& name, ISvcLocator* pSvcLocator);

  StatusCode initialize() override;

  std::tuple<std::vector<Vertices>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<unsigned>& dev_n_pvs,
    const Allen::device_buffer<PV::Vertex>& dev_vertices) const override;

private:
  mutable Gaudi::Accumulators::SummingCounter<unsigned int> m_nbPVsCounter {this, "Nb PVs"};
};

DECLARE_COMPONENT(GaudiAllenPVsToPrimaryVertexContainer)

GaudiAllenPVsToPrimaryVertexContainer::GaudiAllenPVsToPrimaryVertexContainer(
  const std::string& name,
  ISvcLocator* pSvcLocator) :
  MultiTransformer(
    name,
    pSvcLocator,
    {KeyValue {"number_of_multivertex", ""}, KeyValue {"reconstructed_multi_pvs", ""}},
    {KeyValue {"OutputPVs", "Allen/PVs/PrimaryVertices"}})
{}

StatusCode GaudiAllenPVsToPrimaryVertexContainer::initialize()
{
  if (msgLevel(MSG::DEBUG)) debug() << "==> Initialize" << endmsg;
  return StatusCode::SUCCESS;
}

std::tuple<std::vector<Vertices>> GaudiAllenPVsToPrimaryVertexContainer::operator()(
  const EventContext& /*ctx*/,
  const Allen::device_buffer<unsigned>& dev_n_pvs,
  const Allen::device_buffer<PV::Vertex>& dev_vertices) const
{
  // Copy to host
  auto h_n_pvs = dev_n_pvs.to_host();
  auto h_vertices = dev_vertices.to_host();

  const unsigned n_events = h_n_pvs.size();
  constexpr unsigned max_pv = PatPV::max_number_vertices;

  std::vector<Vertices> all_containers;
  all_containers.reserve(n_events);

  for (unsigned evt = 0; evt < n_events; ++evt) {
    const unsigned n_pvs = h_n_pvs[evt];

    Vertices pvcontainer;
    auto& vertices = pvcontainer.vertices;
    vertices.reserve(n_pvs);

    const unsigned evt_offset = evt * max_pv;

    for (unsigned i = 0; i < n_pvs; ++i) {
      const PV::Vertex& vertex = h_vertices[evt_offset + i];

      Gaudi::SymMatrix3x3 poscov;
      poscov(0, 0) = static_cast<double>(vertex.cov00);
      poscov(1, 0) = static_cast<double>(vertex.cov10);
      poscov(1, 1) = static_cast<double>(vertex.cov11);
      poscov(2, 0) = static_cast<double>(vertex.cov20);
      poscov(2, 1) = static_cast<double>(vertex.cov21);
      poscov(2, 2) = static_cast<double>(vertex.cov22);

      auto& recvertex = vertices.emplace_back(Gaudi::XYZPoint {
        static_cast<double>(vertex.position.x),
        static_cast<double>(vertex.position.y),
        static_cast<double>(vertex.position.z)});
      recvertex.setCovMatrix(poscov);
      recvertex.setChi2(static_cast<double>(vertex.chi2));
      recvertex.setNDoF(std::lround(2 * (1 + 1.58 * static_cast<double>(vertex.nTracks)) - 3));
    }

    m_nbPVsCounter += vertices.size();
    all_containers.emplace_back(std::move(pvcontainer));
  }

  return std::make_tuple(std::move(all_containers));
}
