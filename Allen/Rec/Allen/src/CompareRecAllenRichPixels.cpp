/*****************************************************************************\ * (c) Copyright 2018-2020 CERN for the
 benefit of the LHCb Collaboration      *
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
#include "RichMakePixels.cuh"
#include "RichPixel.cuh"

// std
#include <iomanip>
#include <limits>

using AllenRichPixel = Allen::Rich::PixelReco::Pixel;

enum ReturnState { IS_NULL, NOT_EXISTS, EXISTS };

void printPixelAttributes(
  const std::string& label,
  float gx,
  float gy,
  float gz,
  float lx,
  float ly,
  float lz,
  uint32_t smartIDKey,
  float effArea,
  int rich,
  int side,
  float timeWindow,
  bool isInnerRegion)
{
  std::cout << std::fixed << std::setprecision(std::numeric_limits<float>::max_digits10);
  std::cout << label << ": ";
  std::cout << "GP=(" << gx << "," << gy << "," << gz << "), ";
  std::cout << "LP=(" << lx << "," << ly << "," << lz << "), ";
  std::cout << "SID=" << smartIDKey << ", ";
  std::cout << "EA=" << effArea << ", ";
  std::cout << "R=" << rich << ", ";
  std::cout << "S=" << side << ", ";
  std::cout << "TW=" << timeWindow << ", ";
  std::cout << "IR=" << isInnerRegion << "\n";
}

// This test verifies that all valid HLT2 pixels exist in Allen, and all valid Allen pixels exist in HLT2
// Having the same number of m_allen_found_in_hlt2, and m_hlt2_found_in_allen means success.
class CompareRecAllenRichPixels final
  : public Gaudi::Functional::Consumer<void(
      const std::vector<AllenRichPixel, LHCb::Allocators::EventLocal<AllenRichPixel>>&,
      const std::vector<AllenRichPixel, LHCb::Allocators::EventLocal<AllenRichPixel>>&,
      const Rich::Future::Rec::SIMDPixelSummaries&)> {

public:
  /// Standard constructor
  CompareRecAllenRichPixels(const std::string& name, ISvcLocator* pSvcLocator);

  /// Algorithm execution
  void operator()(
    const std::vector<AllenRichPixel, LHCb::Allocators::EventLocal<AllenRichPixel>>&,
    const std::vector<AllenRichPixel, LHCb::Allocators::EventLocal<AllenRichPixel>>&,
    const Rich::Future::Rec::SIMDPixelSummaries&) const override;

  /// Compare the attributes of an Allen Pixel and a Rec Pixel
  bool matchPixels(const AllenRichPixel& allenPixel, const Rich::Future::Rec::SIMDPixel& recPixelSummary, size_t i)
    const
  {
    auto equal = [](float a, float b, float tol = 1e-3f) { return std::abs(a - b) < tol; };

    return (
      equal(allenPixel.gloPos().x, recPixelSummary.gloPos().X()[i]) &&
      equal(allenPixel.gloPos().y, recPixelSummary.gloPos().Y()[i]) &&
      equal(allenPixel.gloPos().z, recPixelSummary.gloPos().Z()[i]) &&

      equal(allenPixel.locPos().x, recPixelSummary.locPos().X()[i]) &&
      equal(allenPixel.locPos().y, recPixelSummary.locPos().Y()[i]) &&
      equal(allenPixel.locPos().z, recPixelSummary.locPos().Z()[i]) &&

      allenPixel.smartID().key() == recPixelSummary.smartID()[i].key() &&
      equal(allenPixel.effArea(), recPixelSummary.effArea()[i]) &&

      static_cast<int>(allenPixel.rich()) == static_cast<int>(recPixelSummary.rich()) &&
      static_cast<int>(allenPixel.side()) == static_cast<int>(recPixelSummary.side()) &&

      equal(allenPixel.timeWindow(), recPixelSummary.timeWindow()[i]) &&
      allenPixel.isInnerRegion() == recPixelSummary.isInnerRegion()[i]);
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
    {KeyValue {"rich1_pixels", ""}, KeyValue {"rich2_pixels", ""}, KeyValue {"SIMDPixelSummaries", ""}})
{}

// When reading this code, keep in mind that recPixelSummaries contain multiple pixels, while allenRichPixels contain
// individual pixels
void CompareRecAllenRichPixels::operator()(
  const std::vector<AllenRichPixel, LHCb::Allocators::EventLocal<AllenRichPixel>>& allenRich1Pixels,
  const std::vector<AllenRichPixel, LHCb::Allocators::EventLocal<AllenRichPixel>>& allenRich2Pixels,
  const Rich::Future::Rec::SIMDPixelSummaries& recPixelSummaries) const
{
  // Concatenate Allen Pixel vectors
  std::vector<AllenRichPixel> allenPixels;
  allenPixels.reserve(allenRich1Pixels.size() + allenRich2Pixels.size());
  allenPixels.insert(allenPixels.end(), allenRich1Pixels.begin(), allenRich1Pixels.end());
  allenPixels.insert(allenPixels.end(), allenRich2Pixels.begin(), allenRich2Pixels.end());

  // functor to check if allen pixels exist in HLT2
  auto allenPixelExistsInRec = [&](const AllenRichPixel& allenPixel) -> ReturnState {
    ++m_allen_reviewed;
    // Check if pixel is valid
    if (allenPixel.isNull()) {
      ++m_allen_null;
      return ReturnState::IS_NULL;
    }

    // iterate over all pixel summaries
    for (const auto& recPixelSummary : recPixelSummaries) {
      // arbitralilly use any of the vectors in a pixel summary to get the pixel count and iterate that many times.
      for (size_t i = 0; i < recPixelSummary.gloPos().X().size(); i++) {
        // ensure HLT2 pixel validity
        if (recPixelSummary.validMask()[i]) {
          // check for match
          if (matchPixels(allenPixel, recPixelSummary, i)) {
            ++m_allen_in_rec;
            return ReturnState::EXISTS;
          }
        } // invalid HLT2 pix
      }   // didn't find pix match
    }     // covered all HLT2 pixels
    ++m_allen_not_in_rec;
    error() << "Allen pixel " << allenPixel.smartID().key() << " not found in HLT2" << endmsg;
    return ReturnState::NOT_EXISTS;
  };

  // functor to check if HLT2 pixels exist in Allen
  auto recPixelExistsInAllen = [&](const Rich::Future::Rec::SIMDPixel& recPixelSummary) -> std::vector<ReturnState> {
    std::vector<ReturnState> summaryStates;
    bool found_current_pixel = true;

    // arbitralilly use any of the vector in a pixel summary to get the pixel count and iterate that many times.
    for (size_t i = 0; i < recPixelSummary.gloPos().X().size(); i++) {
      ++m_rec_reviewed;
      if (found_current_pixel) {
        found_current_pixel = false;
        // ensure HLT2 pixel validity
        if (recPixelSummary.validMask()[i]) {
          // iterate over Allen pixels
          for (const auto& allenPixel : allenPixels) {
            // check allen pix validity
            if (!allenPixel.isNull()) {
              // check for match
              if (matchPixels(allenPixel, recPixelSummary, i)) {
                ++m_rec_in_allen;
                found_current_pixel = true;
                summaryStates.push_back(ReturnState::EXISTS);
                continue;
              } // found a match
            }   // ignore null Allen pix
          }     // covered all Allen pixels
        }
        else {
          ++m_rec_null;
          found_current_pixel = true; // assume correctness on invalid pix to ignore it.

          summaryStates.push_back(ReturnState::IS_NULL);
        } // invalid HLT2 pix
      }
      else { // didn't find pix match
        ++m_rec_not_in_allen;
        summaryStates.push_back(ReturnState::NOT_EXISTS);
        error() << "HLT2 pixel " << recPixelSummary.smartID()[i].key() << " not found in Allen" << endmsg;
      }
    }
    return summaryStates;
  };

  // call allen in hlt2 functor
  for (auto& allenPixel : allenPixels) {
    ReturnState state = allenPixelExistsInRec(allenPixel);
    if (state == ReturnState::NOT_EXISTS) {
      printPixelAttributes(
        "Allen",
        allenPixel.gloPos().x,
        allenPixel.gloPos().y,
        allenPixel.gloPos().z,
        allenPixel.locPos().x,
        allenPixel.locPos().y,
        allenPixel.locPos().z,
        allenPixel.smartID().key(),
        allenPixel.effArea(),
        static_cast<int>(allenPixel.rich()),
        static_cast<int>(allenPixel.side()),
        allenPixel.timeWindow(),
        allenPixel.isInnerRegion());
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
          recPixelSummary.locPos().Z()[i],
          recPixelSummary.smartID()[i].key(),
          recPixelSummary.effArea()[i],
          static_cast<int>(recPixelSummary.rich()),
          static_cast<int>(recPixelSummary.side()),
          recPixelSummary.timeWindow()[i],
          recPixelSummary.isInnerRegion()[i]);
      }
    }
  }

  // verify that all Allen pixels were accounted for
  if ((m_allen_in_rec.value() + m_allen_null.value()) != m_allen_reviewed.value()) {
    error() << "Found " << m_allen_in_rec.value() << " Allen pixels in HLT2, and " << m_allen_null.value()
            << " Allen null pixels totalling " << m_allen_null.value() + m_allen_in_rec.value() << ". Expected "
            << m_allen_reviewed.value() << endmsg;
  }
  // verify that all HLT2 pixels were accounted for
  if ((m_rec_in_allen.value() + m_rec_null.value()) != m_rec_reviewed.value()) {
    error() << "Found " << m_rec_in_allen.value() << " HLT2 pixels in Allen, and " << m_rec_null.value()
            << " HLT2 null pixels totalling " << m_rec_null.value() + m_rec_in_allen.value() << ". Expected "
            << m_rec_reviewed.value() << endmsg;
  }
}
