/*****************************************************************************\
 * (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
 *                                                                             *
 * This software is distributed under the terms of the Apache License          *
 * version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
 \*****************************************************************************/

// Gaudi
#include "LHCbAlgs/Consumer.h"
#include "Gaudi/Accumulators.h"
#include <Gaudi/Accumulators/Histogram.h>

// Rec
#include "RichFutureRecEvent/RichRecCherenkovPhotons.h"
#include "RichFutureRecEvent/RichRecRelations.h"
#include "RichFutureRecEvent/RichRecPhotonPredictedPixelSignals.h"
#include "RichUtils/FastMaths.h"
#include "RichUtils/ZipRange.h"

// Allen
#include "AlgorithmConversionTools.h"
#include "RichPhoton.cuh"
#include "RichParticleHypos.cuh"
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

// Shared with CompareRecAllenRichPixels
struct AllenRichPixel {
  float3 gpos;
  float2 lpos;
  Allen::Rich::Decoding::SmartID smartID;
};
using AllenRichPixels = std::vector<AllenRichPixel>;

// std
#include <iomanip>
#include <limits>
#include <map>
#include <vector>

namespace {

  using AllenRichPhoton = Allen::Rich::PhotonReco::Photon;

  struct RecPhotonIndividual {
    unsigned trackID {};
    uint64_t smartID {};
    float ckTheta {};
    float ckPhi {};
    Rich::DetectorType rich {Rich::InvalidDetector};
    Rich::Future::HypoData<float> signals {};

    RecPhotonIndividual(unsigned tid, uint64_t sid, float theta, float phi, const Rich::DetectorType r) :
      trackID(tid), smartID(sid), ckTheta(theta), ckPhi(phi), rich(r)
    {}
  };

  /// Per-event Allen Rich photon data for one detector
  struct AllenRichPhotonData {
    std::vector<AllenRichPhoton> photons;
    std::vector<uint64_t> pixelSmartIDs; // SmartID key per photon (looked up from pixel array)
    std::vector<unsigned> offsets;       // per-track photon offsets within this event
    std::vector<Allen::Rich::HypoData<float>> pixelSignals;
    unsigned n_tracks = 0;
  };

} // namespace

// ==================================================================
//  Multi-event converter: Allen photon device buffers →
//  per-event AllenRichPhotonData (Rich1 + Rich2)
// ==================================================================

class ConvertAllenRichPhotons final
  : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<AllenRichPhotonData, AllenRichPhotonData>(
      const Allen::device_buffer<AllenRichPhoton>&,                   // Rich1 photons
      const Allen::device_buffer<unsigned>&,                          // Rich1 photon offsets (per-track)
      const Allen::device_buffer<Allen::Rich::HypoData<float>>&,      // Rich1 signals
      const Allen::device_buffer<unsigned>&,                          // track offsets (N+1, shared)
      const Allen::device_buffer<Allen::Rich::Decoding::SmartID>&,    // Rich1 pixel SmartIDs
      const Allen::device_buffer<AllenRichPhoton>&,                   // Rich2 photons
      const Allen::device_buffer<unsigned>&,                          // Rich2 photon offsets
      const Allen::device_buffer<Allen::Rich::HypoData<float>>&,      // Rich2 signals
      const Allen::device_buffer<Allen::Rich::Decoding::SmartID>&)> { // Rich2 pixel SmartIDs

public:
  ConvertAllenRichPhotons(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"rich1_photons", ""},
       KeyValue {"rich1_photons_offsets", ""},
       KeyValue {"rich1_photons_pixel_signals", ""},
       KeyValue {"rich_photon_track_offsets", ""},
       KeyValue {"rich1_pixels_smartid", ""},
       KeyValue {"rich2_photons", ""},
       KeyValue {"rich2_photons_offsets", ""},
       KeyValue {"rich2_photons_pixel_signals", ""},
       KeyValue {"rich2_pixels_smartid", ""}},
      {KeyValue {"AllenRich1PhotonData", ""}, KeyValue {"AllenRich2PhotonData", ""}})
  {}

  std::tuple<std::vector<AllenRichPhotonData>, std::vector<AllenRichPhotonData>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<AllenRichPhoton>& dev_r1_photons,
    const Allen::device_buffer<unsigned>& dev_r1_offsets,
    const Allen::device_buffer<Allen::Rich::HypoData<float>>& dev_r1_signals,
    const Allen::device_buffer<unsigned>& dev_track_offsets,
    const Allen::device_buffer<Allen::Rich::Decoding::SmartID>& dev_r1_pixels,
    const Allen::device_buffer<AllenRichPhoton>& dev_r2_photons,
    const Allen::device_buffer<unsigned>& dev_r2_offsets,
    const Allen::device_buffer<Allen::Rich::HypoData<float>>& dev_r2_signals,
    const Allen::device_buffer<Allen::Rich::Decoding::SmartID>& dev_r2_pixels) const override
  {
    auto h_r1_photons = dev_r1_photons.to_host();
    auto h_r1_offsets = dev_r1_offsets.to_host();
    auto h_r1_signals = dev_r1_signals.to_host();
    auto h_track_offs = dev_track_offsets.to_host();
    auto h_r1_pixels = dev_r1_pixels.to_host();
    auto h_r2_photons = dev_r2_photons.to_host();
    auto h_r2_offsets = dev_r2_offsets.to_host();
    auto h_r2_signals = dev_r2_signals.to_host();
    auto h_r2_pixels = dev_r2_pixels.to_host();

    const unsigned n_events = h_track_offs.size() - 1;

    auto extract_event =
      [&](const auto& h_photons, const auto& h_offsets, const auto& h_signals, const auto& h_pixels, unsigned evt)
      -> AllenRichPhotonData {
      const unsigned t_begin = h_track_offs[evt];
      const unsigned t_end = h_track_offs[evt + 1];
      const unsigned n_trk = t_end - t_begin;

      AllenRichPhotonData out;
      out.n_tracks = n_trk;
      out.offsets.resize(n_trk + 1);

      for (unsigned t = 0; t < n_trk; ++t) {
        const unsigned global_t = t_begin + t;
        const unsigned ph_begin = h_offsets[global_t];
        const unsigned ph_end = h_offsets[global_t + 1];
        out.offsets[t] = ph_begin - h_offsets[t_begin]; // rebase to 0
        for (unsigned p = ph_begin; p < ph_end; ++p) {
          const auto& photon = h_photons[p];
          out.photons.push_back(photon);
          out.pixelSmartIDs.push_back(h_pixels[photon.pixelIdx].key());
          out.pixelSignals.push_back(h_signals[p]);
        }
      }
      out.offsets[n_trk] = out.photons.size(); // total

      return out;
    };

    std::vector<AllenRichPhotonData> r1_out, r2_out;
    r1_out.reserve(n_events);
    r2_out.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {
      r1_out.push_back(extract_event(h_r1_photons, h_r1_offsets, h_r1_signals, h_r1_pixels, evt));
      r2_out.push_back(extract_event(h_r2_photons, h_r2_offsets, h_r2_signals, h_r2_pixels, evt));
    }

    return {std::move(r1_out), std::move(r2_out)};
  }
};

DECLARE_COMPONENT(ConvertAllenRichPhotons)

// ==================================================================
//  Single-event comparison: Allen vs Rec Rich photons
// ==================================================================

class CompareRecAllenRichPhotons final : public LHCb::Algorithm::Consumer<void(
                                           const AllenRichPhotonData&,
                                           const AllenRichPhotonData&,
                                           const Rich::Future::Rec::SIMDCherenkovPhoton::Vector&,
                                           const Rich::Future::Rec::Relations::PhotonToParents::Vector&,
                                           const Rich::Future::Rec::SIMDPhotonSignals::Vector&)> {

public:
  CompareRecAllenRichPhotons(const std::string& name, ISvcLocator* pSvcLocator);

  StatusCode initialize() override;

  void operator()(
    const AllenRichPhotonData& r1_data,
    const AllenRichPhotonData& r2_data,
    const Rich::Future::Rec::SIMDCherenkovPhoton::Vector& recPhotons,
    const Rich::Future::Rec::Relations::PhotonToParents::Vector& photRels,
    const Rich::Future::Rec::SIMDPhotonSignals::Vector& recPhotonSignals) const override;

private:
  void printPhotonAttributes(const std::string& label, float ckTheta, float ckPhi, uint64_t smartIDKey) const
  {
    info() << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) << label << ": "
           << "ckTheta=" << ckTheta << ", "
           << "ckPhi=" << ckPhi << ", "
           << "SID=" << smartIDKey << endmsg;
  }

  template<typename MatchCount, typename AllenNotInRec, typename RecNotInAllen>
  void matchPhotonsForTrack(
    unsigned trackID,
    const AllenRichPhotonData& data,
    const std::vector<RecPhotonIndividual>& recPhotons,
    MatchCount& match_count,
    AllenNotInRec& allen_not_in_rec,
    RecNotInAllen& rec_not_in_allen,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_allen,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_rec,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_rec_allen,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_allen_all,
    Gaudi::Accumulators::Histogram<1>& ckThetaRec_rec_all) const
  {
    const unsigned ph_begin = data.offsets[trackID];
    const unsigned ph_end = data.offsets[trackID + 1];

    std::vector<bool> allen_matched(ph_end - ph_begin, false);
    std::vector<bool> rec_matched(recPhotons.size(), false);

    for (unsigned i = ph_begin; i < ph_end; ++i) {
      const auto smartID = data.pixelSmartIDs[i];
      for (size_t j = 0; j < recPhotons.size(); ++j) {
        if (rec_matched[j]) continue;
        if (smartID == recPhotons[j].smartID) {
          allen_matched[i - ph_begin] = true;
          rec_matched[j] = true;
          ++match_count;
          ++ckThetaRec_allen[data.photons[i].ckTheta];
          ++ckThetaRec_rec[recPhotons[j].ckTheta];
          ++ckThetaRec_rec_allen[recPhotons[j].ckTheta - data.photons[i].ckTheta];
          break;
        }
      }
    }

    for (unsigned i = ph_begin; i < ph_end; ++i) {
      ++ckThetaRec_allen_all[data.photons[i].ckTheta];
      if (!allen_matched[i - ph_begin]) ++allen_not_in_rec;
    }
    for (size_t j = 0; j < recPhotons.size(); ++j) {
      ++ckThetaRec_rec_all[recPhotons[j].ckTheta];
      if (!rec_matched[j]) ++rec_not_in_allen;
    }
  }

  mutable Gaudi::Accumulators::Counter<> m_allen_not_in_rec_r1 {this, "R1 Photons Allen not found in Rec"};
  mutable Gaudi::Accumulators::Counter<> m_rec_not_in_allen_r1 {this, "R1 Photons Rec not found in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_allen_reviewed_r1 {this, "R1 Photons Allen reviewed"};
  mutable Gaudi::Accumulators::Counter<> m_rec_reviewed_r1 {this, "R1 Photons Rec reviewed"};
  mutable Gaudi::Accumulators::Counter<> m_matched_photons_r1 {this, "R1 Photons Matched"};

  mutable Gaudi::Accumulators::Counter<> m_allen_not_in_rec_r2 {this, "R2 Photons Allen not found in Rec"};
  mutable Gaudi::Accumulators::Counter<> m_rec_not_in_allen_r2 {this, "R2 Photons Rec not found in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_allen_reviewed_r2 {this, "R2 Photons Allen reviewed"};
  mutable Gaudi::Accumulators::Counter<> m_rec_reviewed_r2 {this, "R2 Photons Rec reviewed"};
  mutable Gaudi::Accumulators::Counter<> m_matched_photons_r2 {this, "R2 Photons Matched"};

  mutable Gaudi::Accumulators::Counter<> m_tracks_not_in_rec_r1 {this, "R1 Tracks not used in Rec but in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_tracks_not_in_allen_r1 {this, "R1 Tracks not used in Allen but in Rec"};
  mutable Gaudi::Accumulators::Counter<> m_tracks_used_both_r1 {this, "R1 Tracks used in both"};

  mutable Gaudi::Accumulators::Counter<> m_tracks_not_in_rec_r2 {this, "R2 Tracks not used in Rec but in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_tracks_not_in_allen_r2 {this, "R2 Tracks not used in Allen but in Rec"};
  mutable Gaudi::Accumulators::Counter<> m_tracks_used_both_r2 {this, "R2 Tracks used in both"};

  mutable Gaudi::Accumulators::Counter<> m_n_tracks {this, "Tracks"};

  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_allen_all_r1 {this, "R1 ckTheta Allen (all)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_all_r1 {this, "R1 ckTheta Rec (all)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_allen_all_r2 {this, "R2 ckTheta Allen (all)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_all_r2 {this, "R2 ckTheta Rec (all)"};

  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_allen_r1 {this, "R1 ckTheta Allen (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_r1 {this, "R1 ckTheta Rec (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_allen_r1 {this, "R1 ckTheta Rec-Allen (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_allen_r2 {this, "R2 ckTheta Allen (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_r2 {this, "R2 ckTheta Rec (matched)"};
  mutable Gaudi::Accumulators::Histogram<1> m_ckThetaRec_rec_allen_r2 {this, "R2 ckTheta Rec-Allen (matched)"};
};

DECLARE_COMPONENT(CompareRecAllenRichPhotons)

CompareRecAllenRichPhotons::CompareRecAllenRichPhotons(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"AllenRich1PhotonData", ""},
     KeyValue {"AllenRich2PhotonData", ""},
     KeyValue {"CherenkovPhotons", ""},
     KeyValue {"PhotonToParents", ""},
     KeyValue {"RecPhotonSignals", ""}})
{}

StatusCode CompareRecAllenRichPhotons::initialize()
{
  return Consumer::initialize().andThen([&] {
    using Axis1D = Gaudi::Accumulators::Axis<double>;
    m_ckThetaRec_allen_all_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.056, 100)});
    m_ckThetaRec_rec_all_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.056, 100)});
    m_ckThetaRec_allen_all_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.033, 100)});
    m_ckThetaRec_rec_all_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.033, 100)});
    m_ckThetaRec_allen_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.056, 100)});
    m_ckThetaRec_rec_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.056, 100)});
    m_ckThetaRec_rec_allen_r1.setAxis<0>(Axis1D {Gaudi::Histo1DDef(-0.0026, 0.0026, 100)});
    m_ckThetaRec_allen_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.033, 100)});
    m_ckThetaRec_rec_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(0.010, 0.033, 100)});
    m_ckThetaRec_rec_allen_r2.setAxis<0>(Axis1D {Gaudi::Histo1DDef(-0.002, 0.002, 100)});
  });
}

void CompareRecAllenRichPhotons::operator()(
  const AllenRichPhotonData& r1_data,
  const AllenRichPhotonData& r2_data,
  const Rich::Future::Rec::SIMDCherenkovPhoton::Vector& recPhotons,
  const Rich::Future::Rec::Relations::PhotonToParents::Vector& photRels,
  const Rich::Future::Rec::SIMDPhotonSignals::Vector& recPhotonSignals) const
{
  auto allen_not_in_rec_r1 = m_allen_not_in_rec_r1.buffer();
  auto rec_not_in_allen_r1 = m_rec_not_in_allen_r1.buffer();
  auto allen_reviewed_r1 = m_allen_reviewed_r1.buffer();
  auto rec_reviewed_r1 = m_rec_reviewed_r1.buffer();
  auto matched_photons_r1 = m_matched_photons_r1.buffer();
  auto allen_not_in_rec_r2 = m_allen_not_in_rec_r2.buffer();
  auto rec_not_in_allen_r2 = m_rec_not_in_allen_r2.buffer();
  auto allen_reviewed_r2 = m_allen_reviewed_r2.buffer();
  auto rec_reviewed_r2 = m_rec_reviewed_r2.buffer();
  auto matched_photons_r2 = m_matched_photons_r2.buffer();
  auto tracks_not_in_rec_r1 = m_tracks_not_in_rec_r1.buffer();
  auto tracks_not_in_allen_r1 = m_tracks_not_in_allen_r1.buffer();
  auto tracks_used_both_r1 = m_tracks_used_both_r1.buffer();
  auto tracks_not_in_rec_r2 = m_tracks_not_in_rec_r2.buffer();
  auto tracks_not_in_allen_r2 = m_tracks_not_in_allen_r2.buffer();
  auto tracks_used_both_r2 = m_tracks_used_both_r2.buffer();
  auto n_tracks_counter = m_n_tracks.buffer();

  allen_reviewed_r1 += r1_data.photons.size();
  allen_reviewed_r2 += r2_data.photons.size();

  // Unpack Rec SIMD photons
  std::vector<RecPhotonIndividual> recPhotonsFlat;
  for (const auto&& [recPhoton, rels, sigs] : Rich::Ranges::ConstZip(recPhotons, photRels, recPhotonSignals)) {
    for (size_t i = 0; i < recPhoton.CherenkovTheta().size(); ++i) {
      if (recPhoton.validityMask()[i]) {
        auto& ph = recPhotonsFlat.emplace_back(
          rels.trackIndex(),
          recPhoton.smartID()[i].key(),
          recPhoton.CherenkovTheta()[i],
          recPhoton.CherenkovPhi()[i],
          recPhoton.smartID()[i].rich());
        for (const auto id : Rich::particles())
          ph.signals[id] = sigs[id][i];
      }
    }
  }

  // Group Rec photons by trackID and rich
  std::map<unsigned, std::vector<RecPhotonIndividual>> rec_r1, rec_r2;
  for (const auto& p : recPhotonsFlat) {
    if (p.rich == Rich::Rich1) {
      ++rec_reviewed_r1;
      rec_r1[p.trackID].push_back(p);
    }
    else if (p.rich == Rich::Rich2) {
      ++rec_reviewed_r2;
      rec_r2[p.trackID].push_back(p);
    }
  }

  const unsigned n_tracks = r1_data.n_tracks;

  // Track-level stats
  for (unsigned trackID = 0; trackID < n_tracks; ++trackID) {
    bool allen_has_r1 = (r1_data.offsets[trackID + 1] > r1_data.offsets[trackID]);
    bool rec_has_r1 = !rec_r1[trackID].empty();
    if (rec_has_r1 && allen_has_r1)
      ++tracks_used_both_r1;
    else if (rec_has_r1)
      ++tracks_not_in_allen_r1;
    else if (allen_has_r1)
      ++tracks_not_in_rec_r1;

    bool allen_has_r2 = (r2_data.offsets[trackID + 1] > r2_data.offsets[trackID]);
    bool rec_has_r2 = !rec_r2[trackID].empty();
    if (rec_has_r2 && allen_has_r2)
      ++tracks_used_both_r2;
    else if (rec_has_r2)
      ++tracks_not_in_allen_r2;
    else if (allen_has_r2)
      ++tracks_not_in_rec_r2;

    ++n_tracks_counter;
  }

  // Match photons per track
  for (unsigned trackID = 0; trackID < n_tracks; ++trackID) {
    matchPhotonsForTrack(
      trackID,
      r1_data,
      rec_r1[trackID],
      matched_photons_r1,
      allen_not_in_rec_r1,
      rec_not_in_allen_r1,
      m_ckThetaRec_allen_r1,
      m_ckThetaRec_rec_r1,
      m_ckThetaRec_rec_allen_r1,
      m_ckThetaRec_allen_all_r1,
      m_ckThetaRec_rec_all_r1);
    matchPhotonsForTrack(
      trackID,
      r2_data,
      rec_r2[trackID],
      matched_photons_r2,
      allen_not_in_rec_r2,
      rec_not_in_allen_r2,
      m_ckThetaRec_allen_r2,
      m_ckThetaRec_rec_r2,
      m_ckThetaRec_rec_allen_r2,
      m_ckThetaRec_allen_all_r2,
      m_ckThetaRec_rec_all_r2);
  }
}
