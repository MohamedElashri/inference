/*****************************************************************************\
* (c) Copyright 2021-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
/**
 * Convert Allen secondary vertices (raw device buffers) into
 * LHCb::Event::v2::RecVertices, with track association.
 *
 * Two steps:
 *   (1) ConvertAllenSVs   — multi-event → per-event raw SV vectors
 *   (2) AssociateAllenSVs — per-event → RecVertices with track linking
 */

#include <sstream>

#include "GaudiAlg/Transformer.h"

#include "Event/Track_v2.h"
#include "Event/RecVertex_v2.h"

#include "VertexDefinitions.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

// ================================================================
//  Raw per-SV data  (no track pointers)
// ================================================================

struct AllenSVData {
  float x, y, z;
  float cov00, cov10, cov11, cov20, cov21, cov22;
  float chi2;
  unsigned trk1, trk2;
};

using AllenSVs = std::vector<AllenSVData>;

// ================================================================
//  Step 1: Multi-event converter → per-event AllenSVs
// ================================================================

class ConvertAllenSVs final : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<AllenSVs>(
                                const Allen::device_buffer<unsigned>&,                     // SV offsets (N+1)
                                const Allen::device_buffer<VertexFit::TrackMVAVertex>&)> { // SV data (concatenated)

public:
  ConvertAllenSVs(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"allen_sv_offsets", ""}, KeyValue {"allen_secondary_vertices", ""}},
      {KeyValue {"AllenSVData", ""}})
  {}

  std::tuple<std::vector<AllenSVs>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<unsigned>& dev_offsets,
    const Allen::device_buffer<VertexFit::TrackMVAVertex>& dev_vertices) const override
  {
    auto h_offsets = dev_offsets.to_host();
    auto h_vertices = dev_vertices.to_host();

    const unsigned n_events = h_offsets.size() - 1;

    std::vector<AllenSVs> all_svs;
    all_svs.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {
      const unsigned begin = h_offsets[evt];
      const unsigned end = h_offsets[evt + 1];

      AllenSVs svs;
      svs.reserve(end - begin);

      for (unsigned i = begin; i < end; ++i) {
        const auto& v = h_vertices[i];
        svs.push_back({v.x, v.y, v.z, v.cov00, v.cov10, v.cov11, v.cov20, v.cov21, v.cov22, v.chi2, v.trk1, v.trk2});
      }

      all_svs.emplace_back(std::move(svs));
    }

    return std::make_tuple(std::move(all_svs));
  }
};

DECLARE_COMPONENT(ConvertAllenSVs)

// ================================================================
//  Step 2: Per-event transformer → RecVertices with track association
// ================================================================

class AssociateAllenSVs final
  : public Gaudi::Functional::Transformer<
      LHCb::Event::v2::RecVertices(const AllenSVs&, const std::vector<LHCb::Event::v2::Track>&)> {

public:
  AssociateAllenSVs(const std::string& name, ISvcLocator* pSvcLocator) :
    Transformer(
      name,
      pSvcLocator,
      {KeyValue {"AllenSVData", ""}, KeyValue {"InputTracks", "Allen/Out/ForwardTracks"}},
      {KeyValue {"OutputSVs", "Allen/Out/RecVertex"}})
  {}

  LHCb::Event::v2::RecVertices operator()(const AllenSVs& svs, const std::vector<LHCb::Event::v2::Track>& tracks)
    const override
  {
    LHCb::Event::v2::RecVertices sv_container;
    sv_container.reserve(svs.size());

    for (const auto& sv : svs) {
      Gaudi::SymMatrix3x3 poscov;
      poscov(0, 0) = static_cast<double>(sv.cov00);
      poscov(1, 0) = static_cast<double>(sv.cov10);
      poscov(1, 1) = static_cast<double>(sv.cov11);
      poscov(2, 0) = static_cast<double>(sv.cov20);
      poscov(2, 1) = static_cast<double>(sv.cov21);
      poscov(2, 2) = static_cast<double>(sv.cov22);

      Gaudi::XYZPoint position {static_cast<double>(sv.x), static_cast<double>(sv.y), static_cast<double>(sv.z)};

      auto& new_sv = sv_container.emplace_back(
        position, poscov, LHCb::Event::v2::Track::Chi2PerDoF {static_cast<double>(sv.chi2) / 2., 2});

      new_sv.addToTracks(&tracks[sv.trk1], 0.f);
      new_sv.addToTracks(&tracks[sv.trk2], 0.f);
    }

    return sv_container;
  }
};

DECLARE_COMPONENT(AssociateAllenSVs)
