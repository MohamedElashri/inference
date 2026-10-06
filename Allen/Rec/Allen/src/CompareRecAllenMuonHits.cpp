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
#include <sstream>

// Gaudi
#include <GaudiAlg/Consumer.h>
#include <Gaudi/Accumulators.h>

// LHCb
#include <Event/PrHits.h>
#include <Event/ODIN.h>

// Allen
#include <MuonEventModel.cuh>
#include <MuonDefinitions.cuh>
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

using AllenMuonHits = std::vector<Muon::Hit>;

// ==================================================================
//  Multi-event converter: raw device buffers → per-event Muon::Hit vectors
//
//  Offsets layout: N_events * (n_stations + 1) unsigned values,
//  flat-concatenated per-event station-offset blocks.
//  Event e, station s: [offsets[e * (n_stations+1) + s],
//                        offsets[e * (n_stations+1) + s + 1])
// ==================================================================

class ConvertAllenMuonHits final : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<AllenMuonHits>(
                                     const Allen::device_buffer<unsigned>&, // per-event-per-station offsets
                                     const Allen::device_buffer<char>&)> {  // raw muon hit data

public:
  ConvertAllenMuonHits(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"muon_offsets", ""}, KeyValue {"muon_hits", ""}},
      {KeyValue {"AllenMuonHits", ""}})
  {}

  std::tuple<std::vector<AllenMuonHits>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<unsigned>& dev_offsets,
    const Allen::device_buffer<char>& dev_hits) const override
  {
    auto h_offsets = dev_offsets.to_host();
    auto h_hits = dev_hits.to_host();

    const unsigned n_stations = Muon::Constants::n_stations;
    const unsigned n_events = (h_offsets.size() - 1) / n_stations;
    const unsigned n_hits_total = h_offsets[h_offsets.size() - 1];

    Muon::ConstHits all_hits {h_hits.data(), n_hits_total};

    std::vector<AllenMuonHits> all_events;
    all_events.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {
      const unsigned evt_off = evt * n_stations;

      AllenMuonHits hits;
      for (unsigned s = 0; s < n_stations; ++s) {
        const unsigned sta_begin = h_offsets[evt_off + s];
        const unsigned sta_end = h_offsets[evt_off + s + 1];

        hits.reserve(hits.size() + (sta_end - sta_begin));
        for (unsigned i = sta_begin; i < sta_end; ++i) {
          hits.emplace_back(
            all_hits.x(i),
            all_hits.dx(i),
            all_hits.y(i),
            all_hits.dy(i),
            all_hits.z(i),
            all_hits.time(i),
            all_hits.tile(i),
            all_hits.uncrossed(i),
            all_hits.delta_time(i),
            all_hits.region(i));
        }
      }

      all_events.emplace_back(std::move(hits));
    }

    return std::make_tuple(std::move(all_events));
  }
};

DECLARE_COMPONENT(ConvertAllenMuonHits)

// ==================================================================
//  Single-event comparison: AllenMuonHits  vs  Rec MuonHitContainer
// ==================================================================

class CompareRecAllenMuonHits final
  : public Gaudi::Functional::Consumer<void(const LHCb::ODIN& odin, const AllenMuonHits&, const MuonHitContainer&)> {

public:
  CompareRecAllenMuonHits(const std::string& name, ISvcLocator* pSvcLocator);

  void operator()(const LHCb::ODIN& odin, const AllenMuonHits& allen_hits, const MuonHitContainer& rec_hits)
    const override;

private:
  mutable Gaudi::Accumulators::Counter<> m_matched {this, "Matched HLT1/HLT2 muon hits"};
  mutable Gaudi::Accumulators::Counter<> m_errors {this, "Not matched HLT1/HLT2 muon hits"};
};

DECLARE_COMPONENT(CompareRecAllenMuonHits)

CompareRecAllenMuonHits::CompareRecAllenMuonHits(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"ODIN", ""},
     KeyValue {"AllenMuonHits", ""},
     KeyValue {"MuonHitsLocation", MuonHitContainerLocation::Default}})
{}

void CompareRecAllenMuonHits::operator()(
  const LHCb::ODIN& odin,
  const AllenMuonHits& allen_hits,
  const MuonHitContainer& rec_hits) const
{
  // Build Rec hit vector (same as before)
  std::vector<Muon::Hit> muon_hits_rec;
  size_t n_hits_total_rec = 0;
  for (unsigned station = 0; station < Muon::Constants::n_stations; ++station) {
    n_hits_total_rec += rec_hits.station(station).hits().size();
    for (const auto& hit : rec_hits.station(station).hits()) {
      muon_hits_rec.emplace_back(
        hit.x(),
        hit.dx(),
        hit.y(),
        hit.dy(),
        hit.z(),
        hit.time(),
        hit.tile(),
        hit.uncrossed(),
        hit.deltaTime(),
        hit.region());
    }
  }

  debug() << "Number of Muon hits (Allen) in this event " << allen_hits.size() << endmsg;
  debug() << "Number of Muon hits (Rec)   in this event " << n_hits_total_rec << endmsg;

  std::vector<std::string> errors;
  errors.reserve(100);

  for (const auto& muon_hit_allen : allen_hits) {
    auto tmp_iter = std::remove_if(muon_hits_rec.begin(), muon_hits_rec.end(), [&](auto& r) {
      return r.tile == muon_hit_allen.tile && fabsf(r.x - muon_hit_allen.x) < 1e-3f &&
             fabsf(r.y - muon_hit_allen.y) < 1e-3f && fabsf(r.z - muon_hit_allen.z) < 1e-1f &&
             r.uncrossed == muon_hit_allen.uncrossed && r.time == muon_hit_allen.time &&
             fabsf(r.dx - muon_hit_allen.dx) < 1e-3f && fabsf(r.dy - muon_hit_allen.dy) < 1e-3f &&
             r.delta_time == muon_hit_allen.delta_time && r.region == muon_hit_allen.region;
    });
    const auto n_found = std::distance(tmp_iter, muon_hits_rec.end());
    muon_hits_rec.erase(tmp_iter, muon_hits_rec.end());
    if (n_found == 0) {
      std::stringstream msg;
      msg << "Lonely Allen hit            " << muon_hit_allen;
      errors.push_back(msg.str());
    }
    else if (n_found > 1) {
      std::stringstream msg;
      msg << "Multiply matched Allen hit  " << muon_hit_allen;
      errors.push_back(msg.str());
    }
    else {
      ++m_matched;
      debug() << "Successfully matched hit" << muon_hit_allen << endmsg;
    }
  }

  for (const auto& r : muon_hits_rec) {
    std::stringstream msg;
    msg << "Lonely Rec   hit            " << r;
    errors.push_back(msg.str());
  }

  if (!errors.empty()) {
    m_errors += errors.size();
    error() << std::setw(5) << errors.size() << " mismatches in event " << std::setw(8) << odin.runNumber()
            << std::setw(15) << odin.eventNumber() << endmsg;
    for (const auto& msg : errors)
      error() << msg << endmsg;
  }
}
