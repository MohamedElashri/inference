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
#include "GaudiAlg/Consumer.h"
#include "GaudiAlg/Transformer.h"
#include "Gaudi/Accumulators.h"

// Rec
#include "RichFutureRecEvent/RichRecSIMDPixels.h"

// Allen
#include <RichSmartID.cuh>
#include "AllenBuffer.cuh"
#include "EventTransformer.h"

// std
#include <cstdint>
#include <iomanip>
#include <limits>
#include <unordered_map>

// ------------------------------------------------------------------
//  Converted Allen Rich pixel  (host-side, per-event)
// ------------------------------------------------------------------

struct AllenRichPixel {
  float3 gpos;
  float2 lpos;
  Allen::Rich::Decoding::SmartID smartID;
};

using AllenRichPixels = std::vector<AllenRichPixel>;

// ==================================================================
//  Multi-event converter: raw device buffers → per-event AllenRichPixels
// ==================================================================

class ConvertAllenRichPixels final : public LHCb::Algorithm::ScatterEvent::MultiTransformer<std::tuple<AllenRichPixels>(
                                       const Allen::device_buffer<float3>&,
                                       const Allen::device_buffer<short2>&,
                                       const Allen::device_buffer<Allen::Rich::Decoding::SmartID>&,
                                       const Allen::device_buffer<float3>&,
                                       const Allen::device_buffer<short2>&,
                                       const Allen::device_buffer<Allen::Rich::Decoding::SmartID>&,
                                       const Allen::device_buffer<unsigned>&,
                                       const Allen::device_buffer<unsigned>&)> {

public:
  ConvertAllenRichPixels(const std::string& name, ISvcLocator* pSvcLocator) :
    MultiTransformer(
      name,
      pSvcLocator,
      {KeyValue {"rich1_pixels_gpos", ""},
       KeyValue {"rich1_pixels_lpos", ""},
       KeyValue {"rich1_pixels_smartid", ""},
       KeyValue {"rich2_pixels_gpos", ""},
       KeyValue {"rich2_pixels_lpos", ""},
       KeyValue {"rich2_pixels_smartid", ""},
       KeyValue {"rich1_pixel_offsets", ""},
       KeyValue {"rich2_pixel_offsets", ""}},
      {KeyValue {"AllenRichPixels", ""}})
  {}

  std::tuple<std::vector<AllenRichPixels>> operator()(
    const EventContext& /*ctx*/,
    const Allen::device_buffer<float3>& dev_r1_gpos,
    const Allen::device_buffer<short2>& dev_r1_lpos,
    const Allen::device_buffer<Allen::Rich::Decoding::SmartID>& dev_r1_sid,
    const Allen::device_buffer<float3>& dev_r2_gpos,
    const Allen::device_buffer<short2>& dev_r2_lpos,
    const Allen::device_buffer<Allen::Rich::Decoding::SmartID>& dev_r2_sid,
    const Allen::device_buffer<unsigned>& dev_r1_offsets,
    const Allen::device_buffer<unsigned>& dev_r2_offsets) const override
  {
    // Copy all to host
    auto h_r1_gpos = dev_r1_gpos.to_host();
    auto h_r1_lpos = dev_r1_lpos.to_host();
    auto h_r1_sid = dev_r1_sid.to_host();
    auto h_r2_gpos = dev_r2_gpos.to_host();
    auto h_r2_lpos = dev_r2_lpos.to_host();
    auto h_r2_sid = dev_r2_sid.to_host();
    auto h_r1_off = dev_r1_offsets.to_host();
    auto h_r2_off = dev_r2_offsets.to_host();

    // Offsets layout: 2 panels per detector → (n_events * 2 + 1) entries.
    // Panel 0: offs[0..n_events]; Panel 1: offs[n_events..2*n_events]
    const unsigned n_events = (h_r1_off.size() - 1) / 2;

    std::vector<AllenRichPixels> all_pixels;
    all_pixels.reserve(n_events);

    for (unsigned evt = 0; evt < n_events; ++evt) {

      // Rich1 panel boundaries
      const unsigned r1_p0_begin = h_r1_off[evt];
      const unsigned r1_p0_end = h_r1_off[evt + 1];
      const unsigned r1_p1_begin = h_r1_off[n_events + evt];
      const unsigned r1_p1_end = h_r1_off[n_events + evt + 1];

      // Rich2 panel boundaries
      const unsigned r2_p0_begin = h_r2_off[evt];
      const unsigned r2_p0_end = h_r2_off[evt + 1];
      const unsigned r2_p1_begin = h_r2_off[n_events + evt];
      const unsigned r2_p1_end = h_r2_off[n_events + evt + 1];

      const unsigned n_r1 = (r1_p0_end - r1_p0_begin) + (r1_p1_end - r1_p1_begin);
      const unsigned n_r2 = (r2_p0_end - r2_p0_begin) + (r2_p1_end - r2_p1_begin);

      AllenRichPixels pixels;
      pixels.reserve(n_r1 + n_r2);

      // Helper to append pixels from a panel
      auto append_panel =
        [&](const auto& gpos_host, const auto& lpos_host, const auto& sid_host, unsigned begin, unsigned end) {
          for (unsigned i = begin; i < end; ++i) {
            pixels.push_back(
              {gpos_host[i],
               make_float2(
                 static_cast<float>(lpos_host[i].x) * (750.f / (1 << 15)),
                 static_cast<float>(lpos_host[i].y) * (750.f / (1 << 15))),
               sid_host[i]});
          }
        };

      // Rich1: panel 0 + panel 1
      append_panel(h_r1_gpos, h_r1_lpos, h_r1_sid, r1_p0_begin, r1_p0_end);
      append_panel(h_r1_gpos, h_r1_lpos, h_r1_sid, r1_p1_begin, r1_p1_end);

      // Rich2: panel 0 + panel 1
      append_panel(h_r2_gpos, h_r2_lpos, h_r2_sid, r2_p0_begin, r2_p0_end);
      append_panel(h_r2_gpos, h_r2_lpos, h_r2_sid, r2_p1_begin, r2_p1_end);

      all_pixels.emplace_back(std::move(pixels));
    }

    return std::make_tuple(std::move(all_pixels));
  }
};

DECLARE_COMPONENT(ConvertAllenRichPixels)

// ===================================================================
//  Single-event comparison: AllenRichPixels vs Rec SIMDPixelSummaries
// ===================================================================

namespace {

  enum ReturnState { IS_NULL, NOT_EXISTS, EXISTS };

  // Reference to a single scalar pixel inside a SIMD-packed Rec pixel summary,
  // used to index Rec pixels by SmartID key without flattening/copying them.
  struct RecPixelRef {
    unsigned summaryIdx {};
    unsigned lane {};
  };

} // namespace

class CompareRecAllenRichPixels final
  : public Gaudi::Functional::Consumer<void(const AllenRichPixels&, const Rich::Future::Rec::SIMDPixelSummaries&)> {

public:
  CompareRecAllenRichPixels(const std::string& name, ISvcLocator* pSvcLocator);

  void operator()(const AllenRichPixels& allenPixels, const Rich::Future::Rec::SIMDPixelSummaries& recPixelSummaries)
    const override;

private:
  /// Compare the attributes of an Allen Pixel and a Rec Pixel
  bool matchPixels(const AllenRichPixel& allen, const Rich::Future::Rec::SIMDPixel& recPixelSummary, size_t i) const
  {
    const auto equal = []<typename T>(T a, T b, T tol = T(1e-3)) { return std::abs(a - b) < tol; };

    // Check the cheap, exact, maximally-discriminating SmartID key first: it is
    // required for any real match anyway, and rejects almost all non-matching
    // candidates without paying for five floating point tolerance checks.
    return (
      allen.smartID.key() == recPixelSummary.smartID()[i].key() &&
      equal(allen.gpos.x, recPixelSummary.gloPos().X()[i]) && equal(allen.gpos.y, recPixelSummary.gloPos().Y()[i]) &&
      equal(allen.gpos.z, recPixelSummary.gloPos().Z()[i]) &&
      equal(allen.lpos.x, recPixelSummary.locPos().X()[i], 1e-1f) &&
      equal(allen.lpos.y, recPixelSummary.locPos().Y()[i], 1e-1f));
  }

  /// Report the attributes of a pixel that failed to find a match, via a Gaudi info() message
  template<typename DetectorType, typename Side>
  void printPixelAttributes(
    const std::string& label,
    float gx,
    float gy,
    float gz,
    float lx,
    float ly,
    uint32_t smartIDKey,
    const DetectorType rich,
    const Side side) const
  {
    info() << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10) << label << ": "
           << "GP=(" << gx << "," << gy << "," << gz << "), "
           << "LP=(" << lx << "," << ly << "), "
           << "SID=" << smartIDKey << ", "
           << "R=" << static_cast<int>(rich) << ", "
           << "S=" << static_cast<int>(side) << endmsg;
  }

  mutable Gaudi::Accumulators::Counter<> m_allen_in_rec {this, "Allen Pixels found in HLT2"};
  mutable Gaudi::Accumulators::Counter<> m_allen_not_in_rec {this, "Allen Pixels not found in HLT2"};
  mutable Gaudi::Accumulators::Counter<> m_rec_in_allen {this, "HLT2 Pixels found in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_rec_not_in_allen {this, "HLT2 Pixels not found in Allen"};
  mutable Gaudi::Accumulators::Counter<> m_allen_null {this, "Null Allen Pixels"};
  mutable Gaudi::Accumulators::Counter<> m_rec_null {this, "Null HLT2 Pixels"};
  mutable Gaudi::Accumulators::Counter<> m_allen_reviewed {this, "Allen Pixels reviewed"};
  mutable Gaudi::Accumulators::Counter<> m_rec_reviewed {this, "HLT2 Pixels reviewed"};
};

DECLARE_COMPONENT(CompareRecAllenRichPixels)

CompareRecAllenRichPixels::CompareRecAllenRichPixels(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(name, pSvcLocator, {KeyValue {"AllenRichPixels", ""}, KeyValue {"SIMDPixelSummaries", ""}})
{}

void CompareRecAllenRichPixels::operator()(
  const AllenRichPixels& allenPixels,
  const Rich::Future::Rec::SIMDPixelSummaries& recPixelSummaries) const
{
  auto allen_in_rec = m_allen_in_rec.buffer();
  auto allen_not_in_rec = m_allen_not_in_rec.buffer();
  auto rec_in_allen = m_rec_in_allen.buffer();
  auto rec_not_in_allen = m_rec_not_in_allen.buffer();
  auto allen_null = m_allen_null.buffer();
  auto rec_null = m_rec_null.buffer();
  auto allen_reviewed = m_allen_reviewed.buffer();
  auto rec_reviewed = m_rec_reviewed.buffer();

  // Index both pixel sets by SmartID key so the two match directions below can
  // do O(1)-average lookups instead of linear scans. A vector of candidates
  // preserves correct handling if more than one pixel shares a key.
  std::unordered_map<std::uint64_t, std::vector<unsigned>> allenIndexByKey;
  allenIndexByKey.reserve(allenPixels.size());
  for (unsigned i = 0; i < allenPixels.size(); ++i) {
    allenIndexByKey[allenPixels[i].smartID.key()].push_back(i);
  }

  std::unordered_map<std::uint64_t, std::vector<RecPixelRef>> recIndexByKey;
  for (unsigned summaryIdx = 0; summaryIdx < recPixelSummaries.size(); ++summaryIdx) {
    const auto& summary = recPixelSummaries[summaryIdx];
    for (unsigned lane = 0; lane < summary.gloPos().X().size(); ++lane) {
      if (summary.validMask()[lane]) {
        recIndexByKey[summary.smartID()[lane].key()].push_back({summaryIdx, lane});
      }
    }
  }

  // ----- Allen pixels → check existence in Rec -----
  const auto allenInRec = [&](const AllenRichPixel& a) -> ReturnState {
    ++allen_reviewed;
    const auto it = recIndexByKey.find(a.smartID.key());
    if (it != recIndexByKey.end()) {
      for (const auto& ref : it->second) {
        if (matchPixels(a, recPixelSummaries[ref.summaryIdx], ref.lane)) {
          ++allen_in_rec;
          return ReturnState::EXISTS;
        }
      }
    }
    ++allen_not_in_rec;
    error() << "Allen pixel " << a.smartID.key() << " not found in HLT2" << endmsg;
    return ReturnState::NOT_EXISTS;
  };

  for (const auto& a : allenPixels) {
    const auto state = allenInRec(a);
    if (state == ReturnState::NOT_EXISTS) {
      printPixelAttributes(
        "Allen",
        a.gpos.x,
        a.gpos.y,
        a.gpos.z,
        a.lpos.x,
        a.lpos.y,
        a.smartID.key(),
        static_cast<int>(a.smartID.rich()),
        static_cast<int>(a.smartID.side()));
    }
  }

  // ----- Rec pixels → check existence in Allen -----
  const auto recInAllen = [&](const Rich::Future::Rec::SIMDPixel& rec, size_t i) -> ReturnState {
    ++rec_reviewed;
    if (!rec.validMask()[i]) {
      ++rec_null;
      return ReturnState::IS_NULL;
    }
    const auto it = allenIndexByKey.find(rec.smartID()[i].key());
    if (it != allenIndexByKey.end()) {
      for (const auto allenIdx : it->second) {
        if (matchPixels(allenPixels[allenIdx], rec, i)) {
          ++rec_in_allen;
          return ReturnState::EXISTS;
        }
      }
    }
    ++rec_not_in_allen;
    error() << "HLT2 pixel " << rec.smartID()[i].key() << " not found in Allen" << endmsg;
    return ReturnState::NOT_EXISTS;
  };

  for (const auto& rec : recPixelSummaries) {
    for (size_t i = 0; i < rec.gloPos().X().size(); i++) {
      const auto state = recInAllen(rec, i);
      if (state == ReturnState::NOT_EXISTS) {
        printPixelAttributes(
          "Rec",
          rec.gloPos().X()[i],
          rec.gloPos().Y()[i],
          rec.gloPos().Z()[i],
          rec.locPos().X()[i],
          rec.locPos().Y()[i],
          rec.smartID()[i].key(),
          static_cast<int>(rec.rich()),
          static_cast<int>(rec.side()));
      }
    }
  }

  // verify that all Allen pixels were accounted for
  if ((allen_in_rec.value() + allen_null.value()) != allen_reviewed.value()) {
    error() << "Found " << allen_in_rec.value() << " Allen pixels in HLT2, and " << allen_null.value()
            << " Allen null pixels totalling " << allen_null.value() + allen_in_rec.value() << ". Expected "
            << allen_reviewed.value() << endmsg;
  }
  // verify that all HLT2 pixels were accounted for
  if ((rec_in_allen.value() + rec_null.value()) != rec_reviewed.value()) {
    error() << "Found " << rec_in_allen.value() << " HLT2 pixels in Allen, and " << rec_null.value()
            << " HLT2 null pixels totalling " << rec_null.value() + rec_in_allen.value() << ". Expected "
            << rec_reviewed.value() << endmsg;
  }
}
