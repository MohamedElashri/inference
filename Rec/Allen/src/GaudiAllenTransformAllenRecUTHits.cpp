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
 * Convert Allen (multi-event) and Rec (single-event) UT hits into
 * the same vector<UT::Hit> format.
 */

#include <sstream>
#include <vector>

#include <LHCbAlgs/Transformer.h>
#include "GaudiKernel/StdArrayAsProperty.h"

#include "Kernel/LHCbID.h"
#include "LHCbMath/SIMDWrapper.h"
#include "Event/PrHits.h"

#include "LHCbID.cuh"
#include "UTEventModel.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

using simd = SIMDWrapper::best::types;

// ==================================================================
//  Multi-event converter: raw Allen UT buffers → per-event UT::Hit vectors
//
//  Offsets: N_events * (n_groups + 1) unsigned, flat-concatenated
//  per-event sector-group offset blocks.
// ==================================================================

class ConvertAllenUTHits final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<std::vector<UT::Hit>>(
      const Allen::device_buffer<unsigned>&, // per-event-per-group offsets
      const Allen::device_buffer<char>&)> {  // raw UT hit data

public:
  ConvertAllenUTHits(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"ut_hit_offsets", ""}, KeyValue {"ut_hits", ""}},
      {KeyValue {"allen_ut_hits", ""}})
  {}

  std::tuple<std::vector<std::vector<UT::Hit>>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<unsigned>& dev_offsets,
    const Allen::device_buffer<char>& dev_hits) const override
  {
    auto h_offsets = dev_offsets.to_host();
    auto h_hits = dev_hits.to_host();

    const unsigned n_groups = UT::Constants::n_groups;
    const unsigned n_layers = UT::Constants::n_layers;
    const unsigned n_groups_in_layer = UT::Constants::n_groups_in_layer;
    const unsigned n_events = (h_offsets.size() - 1) / n_groups;
    const unsigned n_hits_total = h_offsets[h_offsets.size() - 1];

    UT::ConstHits all_hits {h_hits.data(), n_hits_total};

    std::vector<std::vector<UT::Hit>> all_events;
    all_events.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {
      const unsigned evt_off = evt * n_groups;

      std::vector<UT::Hit> hits;
      hits.reserve(h_offsets[evt_off + n_groups] - h_offsets[evt_off]);

      // Extract hits per sector group
      for (unsigned sg = 0; sg < n_groups; ++sg) {
        for (unsigned i = h_offsets[evt_off + sg]; i < h_offsets[evt_off + sg + 1]; ++i)
          hits.emplace_back(all_hits.getHit(i));
      }

      // Assign plane_code = layer
      for (unsigned layer = 0; layer < n_layers; ++layer) {
        const unsigned layer_off = h_offsets[evt_off + layer * n_groups_in_layer];
        const unsigned n_layer = h_offsets[evt_off + (layer + 1) * n_groups_in_layer] - layer_off;
        for (unsigned i = 0; i < n_layer; ++i)
          hits[layer_off - h_offsets[evt_off] + i].plane_code = layer;
      }

      all_events.emplace_back(std::move(hits));
    }

    return std::make_tuple(std::move(all_events));
  }
};

DECLARE_COMPONENT(ConvertAllenUTHits)

// ==================================================================
//  Single-event converter: Rec UT::Hits → vector<UT::Hit>
// ==================================================================

class ConvertRecUTHits final : public Gaudi::Functional::Transformer<std::vector<UT::Hit>(LHCb::Pr::UT::Hits const&)> {

public:
  ConvertRecUTHits(const std::string& name, ISvcLocator* pSvcLocator) :
    Transformer(name, pSvcLocator, {KeyValue {"UTHitsLocation", UTInfo::HitLocation}}, {KeyValue {"rec_ut_hits", ""}})
  {}

  std::vector<UT::Hit> operator()(LHCb::Pr::UT::Hits const& hit_handler) const override
  {
    const auto n_hits = hit_handler.nHits();
    const auto& simd_hits = hit_handler.simd();

    std::vector<UT::Hit> rec_hits;
    rec_hits.reserve(n_hits);

    for (int i = 0; i < n_hits; i += simd::size) {
      const auto mH = simd_hits[i];
      std::array<int, simd::size> channelIDs;
      mH.get<LHCb::Pr::UT::UTHitsTag::channelID>().store(channelIDs.data());
      std::array<float, simd::size> yBegins, yEnds, zAtYEq0s, xAtYEq0s, weights, dxDys;
      mH.get<LHCb::Pr::UT::UTHitsTag::yBegin>().store(yBegins.data());
      mH.get<LHCb::Pr::UT::UTHitsTag::yEnd>().store(yEnds.data());
      mH.get<LHCb::Pr::UT::UTHitsTag::zAtYEq0>().store(zAtYEq0s.data());
      mH.get<LHCb::Pr::UT::UTHitsTag::xAtYEq0>().store(xAtYEq0s.data());
      mH.get<LHCb::Pr::UT::UTHitsTag::weight>().store(weights.data());
      mH.get<LHCb::Pr::UT::UTHitsTag::dxDy>().store(dxDys.data());

      for (std::size_t j = 0; j < simd::size; ++j) {
        const auto channelID = LHCb::Detector::UT::ChannelID(channelIDs[j]);
        const auto lhcbID = bit_cast<int, unsigned int>(LHCb::LHCbID(channelID).lhcbID());
        const auto layer = channelID.layer();
        rec_hits.emplace_back(yBegins[j], yEnds[j], zAtYEq0s[j], xAtYEq0s[j], dxDys[j], weights[j], lhcbID, layer);
      }
    }

    rec_hits.resize(n_hits);
    return rec_hits;
  }
};

DECLARE_COMPONENT(ConvertRecUTHits)
