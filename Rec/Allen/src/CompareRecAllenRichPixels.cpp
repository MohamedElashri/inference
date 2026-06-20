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
#include "Gaudi/Accumulators.h"
#include <Kernel/EventLocalAllocator.h>

// Rec
#include "RichFutureRecEvent/RichRecSIMDPixels.h"

// Allen
#include <RichSmartID.cuh>

// std
#include <iomanip>
#include <limits>

enum ReturnState { IS_NULL, NOT_EXISTS, EXISTS };

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
  const Side side)
{
  std::cout << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10);
  std::cout << label << ": ";
  std::cout << "GP=(" << gx << "," << gy << "," << gz << "), ";
  std::cout << "LP=(" << lx << "," << ly << "), ";
  std::cout << "SID=" << smartIDKey << ", ";
  std::cout << "R=" << static_cast<int>(rich) << ", ";
  std::cout << "S=" << static_cast<int>(side) << "\n";
}

// This test verifies that all valid HLT2 pixels exist in Allen, and all valid Allen pixels exist in HLT2
// Having the same number of m_allen_found_in_hlt2, and m_hlt2_found_in_allen means success.
class CompareRecAllenRichPixels final
  : public Gaudi::Functional::Consumer<void(
      const std::vector<float3, LHCb::Allocators::EventLocal<float3>>&,
      const std::vector<short2, LHCb::Allocators::EventLocal<short2>>&,
      const std::vector<Allen::Rich::Decoding::SmartID, LHCb::Allocators::EventLocal<Allen::Rich::Decoding::SmartID>>&,
      const std::vector<float3, LHCb::Allocators::EventLocal<float3>>&,
      const std::vector<short2, LHCb::Allocators::EventLocal<short2>>&,
      const std::vector<Allen::Rich::Decoding::SmartID, LHCb::Allocators::EventLocal<Allen::Rich::Decoding::SmartID>>&,
      const Rich::Future::Rec::SIMDPixelSummaries&)> {

public:
  /// Standard constructor
  CompareRecAllenRichPixels(const std::string& name, ISvcLocator* pSvcLocator);

  /// Algorithm execution
  void operator()(
    const std::vector<float3, LHCb::Allocators::EventLocal<float3>>&,
    const std::vector<short2, LHCb::Allocators::EventLocal<short2>>&,
    const std::vector<Allen::Rich::Decoding::SmartID, LHCb::Allocators::EventLocal<Allen::Rich::Decoding::SmartID>>&,
    const std::vector<float3, LHCb::Allocators::EventLocal<float3>>&,
    const std::vector<short2, LHCb::Allocators::EventLocal<short2>>&,
    const std::vector<Allen::Rich::Decoding::SmartID, LHCb::Allocators::EventLocal<Allen::Rich::Decoding::SmartID>>&,
    const Rich::Future::Rec::SIMDPixelSummaries&) const override;

  /// Compare the attributes of an Allen Pixel and a Rec Pixel
  bool matchPixels(
    const float3& allenGpos,
    [[maybe_unused]] const float2& allenLpos,
    const Allen::Rich::Decoding::SmartID& allenID,
    const Rich::Future::Rec::SIMDPixel& recPixelSummary,
    size_t i) const
  {
    auto equal = [](float a, float b, float tol = 1e-3f) { return std::abs(a - b) < tol; };

    return (
      equal(allenGpos.x, recPixelSummary.gloPos().X()[i]) && equal(allenGpos.y, recPixelSummary.gloPos().Y()[i]) &&
      equal(allenGpos.z, recPixelSummary.gloPos().Z()[i]) &&
      equal(allenLpos.x, recPixelSummary.locPos().X()[i], 1e-1f) &&
      equal(allenLpos.y, recPixelSummary.locPos().Y()[i], 1e-1f) &&
      allenID.key() == recPixelSummary.smartID()[i].key());
  }

private:
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
  Consumer(
    name,
    pSvcLocator,
    {KeyValue {"rich1_pixels_gpos", ""},
     KeyValue {"rich1_pixels_lpos", ""},
     KeyValue {"rich1_pixels_smartid", ""},
     KeyValue {"rich2_pixels_gpos", ""},
     KeyValue {"rich2_pixels_lpos", ""},
     KeyValue {"rich2_pixels_smartid", ""},
     KeyValue {"SIMDPixelSummaries", ""}})
{}

// When reading this code, keep in mind that recPixelSummaries contain multiple pixels, while allenRichPixels contain
// individual pixels
void CompareRecAllenRichPixels::operator()(
  const std::vector<float3, LHCb::Allocators::EventLocal<float3>>& allenRich1PixelsGpos,
  const std::vector<short2, LHCb::Allocators::EventLocal<short2>>& allenRich1PixelsLpos,
  const std::vector<Allen::Rich::Decoding::SmartID, LHCb::Allocators::EventLocal<Allen::Rich::Decoding::SmartID>>&
    allenRich1PixelsSmartID,
  const std::vector<float3, LHCb::Allocators::EventLocal<float3>>& allenRich2PixelsGpos,
  const std::vector<short2, LHCb::Allocators::EventLocal<short2>>& allenRich2PixelsLpos,
  const std::vector<Allen::Rich::Decoding::SmartID, LHCb::Allocators::EventLocal<Allen::Rich::Decoding::SmartID>>&
    allenRich2PixelsSmartID,
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

  // Concatenate Allen Pixel vectors
  std::vector<float3> allenPixelsGpos;
  std::vector<float2> allenPixelsLpos;
  std::vector<Allen::Rich::Decoding::SmartID> allenPixelsSmartID;
  allenPixelsGpos.reserve(allenRich1PixelsGpos.size() + allenRich2PixelsGpos.size());
  allenPixelsGpos.insert(allenPixelsGpos.end(), allenRich1PixelsGpos.begin(), allenRich1PixelsGpos.end());
  allenPixelsGpos.insert(allenPixelsGpos.end(), allenRich2PixelsGpos.begin(), allenRich2PixelsGpos.end());

  allenPixelsLpos.reserve(allenRich1PixelsLpos.size() + allenRich2PixelsLpos.size());
  for (short2 lpos16 : allenRich1PixelsLpos) {
    allenPixelsLpos.emplace_back(make_float2(lpos16.x * (750.f / (1 << 15)), lpos16.y * (750.f / (1 << 15))));
  }
  for (short2 lpos16 : allenRich2PixelsLpos) {
    allenPixelsLpos.emplace_back(make_float2(lpos16.x * (750.f / (1 << 15)), lpos16.y * (750.f / (1 << 15))));
  }

  allenPixelsSmartID.reserve(allenRich1PixelsSmartID.size() + allenRich2PixelsSmartID.size());
  allenPixelsSmartID.insert(allenPixelsSmartID.end(), allenRich1PixelsSmartID.begin(), allenRich1PixelsSmartID.end());
  allenPixelsSmartID.insert(allenPixelsSmartID.end(), allenRich2PixelsSmartID.begin(), allenRich2PixelsSmartID.end());

  // functor to check if allen pixels exist in HLT2
  auto allenPixelExistsInRec =
    [&](
      const float3& allenGpos, const float2& allenLpos, const Allen::Rich::Decoding::SmartID& allenID) -> ReturnState {
    ++allen_reviewed;

    // iterate over all pixel summaries
    for (const auto& recPixelSummary : recPixelSummaries) {
      // arbitralilly use any of the vectors in a pixel summary to get the pixel count and iterate that many times.
      for (size_t i = 0; i < recPixelSummary.gloPos().X().size(); i++) {
        // ensure HLT2 pixel validity
        if (recPixelSummary.validMask()[i]) {
          // check for match
          if (matchPixels(allenGpos, allenLpos, allenID, recPixelSummary, i)) {
            ++allen_in_rec;
            return ReturnState::EXISTS;
          }
        } // invalid HLT2 pix
      }   // didn't find pix match
    }     // covered all HLT2 pixels
    ++allen_not_in_rec;
    error() << "Allen pixel " << allenID.key() << " not found in HLT2" << endmsg;
    return ReturnState::NOT_EXISTS;
  };

  // functor to check if HLT2 pixels exist in Allen
  auto recPixelExistsInAllen = [&](const Rich::Future::Rec::SIMDPixel& recPixelSummary) -> std::vector<ReturnState> {
    std::vector<ReturnState> summaryStates;
    bool found_current_pixel = true;

    // arbitralilly use any of the vector in a pixel summary to get the pixel count and iterate that many times.
    for (size_t i = 0; i < recPixelSummary.gloPos().X().size(); i++) {
      ++rec_reviewed;
      if (found_current_pixel) {
        found_current_pixel = false;
        // ensure HLT2 pixel validity
        if (recPixelSummary.validMask()[i]) {
          // iterate over Allen pixels
          for (unsigned j = 0; j < allenPixelsGpos.size(); j++) {
            // check for match
            if (matchPixels(allenPixelsGpos[j], allenPixelsLpos[j], allenPixelsSmartID[j], recPixelSummary, i)) {
              ++rec_in_allen;
              found_current_pixel = true;
              summaryStates.push_back(ReturnState::EXISTS);
              continue;
            } // found a match
          }   // covered all Allen pixels
        }
        else {
          ++rec_null;
          found_current_pixel = true; // assume correctness on invalid pix to ignore it.

          summaryStates.push_back(ReturnState::IS_NULL);
        } // invalid HLT2 pix
      }
      else { // didn't find pix match
        ++rec_not_in_allen;
        summaryStates.push_back(ReturnState::NOT_EXISTS);
        error() << "HLT2 pixel " << recPixelSummary.smartID()[i].key() << " not found in Allen" << endmsg;
      }
    }
    return summaryStates;
  };

  // call allen in hlt2 functor
  for (unsigned j = 0; j < allenPixelsGpos.size(); j++) {
    ReturnState state = allenPixelExistsInRec(allenPixelsGpos[j], allenPixelsLpos[j], allenPixelsSmartID[j]);
    if (state == ReturnState::NOT_EXISTS) {
      printPixelAttributes(
        "Allen",
        allenPixelsGpos[j].x,
        allenPixelsGpos[j].y,
        allenPixelsGpos[j].z,
        allenPixelsLpos[j].x,
        allenPixelsLpos[j].y,
        allenPixelsSmartID[j].key(),
        allenPixelsSmartID[j].rich(),
        allenPixelsSmartID[j].side());
    }
  }

  // call hlt2 in allen functor

  for (auto& recPixelSummary : recPixelSummaries) {
    std::vector<ReturnState> summaryStates = recPixelExistsInAllen(recPixelSummary);
    for (size_t i = 0; i < summaryStates.size(); ++i) {
      if (summaryStates[i] == ReturnState::NOT_EXISTS) {
        printPixelAttributes(
          "Rec",
          recPixelSummary.gloPos().X()[i],
          recPixelSummary.gloPos().Y()[i],
          recPixelSummary.gloPos().Z()[i],
          recPixelSummary.locPos().X()[i],
          recPixelSummary.locPos().Y()[i],
          recPixelSummary.smartID()[i].key(),
          recPixelSummary.rich(),
          recPixelSummary.side());
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
