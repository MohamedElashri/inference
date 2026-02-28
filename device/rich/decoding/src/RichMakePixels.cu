/*****************************************************************************\
 * (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
 *                                                                             *
 * This software is distributed under the terms of the Apache License          *
 * version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
 \*****************************************************************************/

#include "RichDefinitions.cuh"
#include "Rich.cuh"
#include <RichMakePixels.cuh>
#include <RichPhotonDetector.cuh>
#include <MEPTools.h>

INSTANTIATE_ALGORITHM(rich_make_pixels::rich_make_pixels_t);

/// Create a Pixel object from a given SmartID
template<unsigned rich>
__device__ Allen::Rich::PixelReco::Pixel make_pixel(
  const Allen::Rich::Detector::PDPanel& panel,
  const std::array<float, 12>& g2panel,
  const Allen::Rich::Decoding::SmartID& smartID)
{
  bool smartIDSelected = true;
  if constexpr (Allen::Rich::m_enable4D[rich]) {
    if (smartID.adcTimeIsSet()) {
      const auto hitT = smartID.time();
      if (fabsf(hitT - Allen::Rich::m_avHitTime[rich]) > Allen::Rich::m_timeWindow[rich]) {
        smartIDSelected = false;
      }
    }
  }

  Allen::Rich::PixelReco::Pixel pixel;

  // if time checks pass, and a non-Null PD matches the SmartID, save the pix
  if (smartIDSelected) {
    const auto& pixelPD = panel.dePD(smartID);

    // From panel local coordinates to global coordinates.
    const auto gPos = panel.globalDetectionPoint(smartID);
    // From global coordinates to RICH detector local coordinates.
    const auto lPos = Allen::Rich::transform3DTimesPoint(g2panel, gPos);

    const auto isInner =
      (Allen::Rich::m_overrideRegions[rich] ? (fabsf(lPos.x) < float(Allen::Rich::m_innerPixX[rich]) &&
                                               fabsf(lPos.y) < float(Allen::Rich::m_innerPixY[rich])) :
                                              !smartID.isLargePMT());

    // 4D specific properties
    float timeWindow = 999999;
    if constexpr (Allen::Rich::m_enable4D[rich]) {
      timeWindow = isInner ? float(Allen::Rich::m_innerTimeWindow[rich]) : float(Allen::Rich::m_outerTimeWindow[rich]);
    }
    pixel = Allen::Rich::PixelReco::Pixel(gPos, lPos, smartID, pixelPD.effectivePixelArea(), timeWindow);
    if constexpr (Allen::Rich::m_overrideRegions[rich]) {
      pixel.overrideRegions(isInner);
    }
    pixel.setIsNull(pixelPD.getIsNull());
  }
  return pixel;
}

/// Kernel function to iterate over Events, SmartIDs, and create Pixels
template<unsigned rich>
__global__ void rich_make_pixels_kernel(
  rich_make_pixels::Parameters parameters,
  const Allen::Rich::RichDetector* deRich)
{
  const auto side = blockIdx.y; // 0 or 1 for RICH1 top/bottom or RICH2 left/right
  const auto number_of_events = parameters.dev_event_list.size();
  auto offset = parameters.dev_rich_hit_offsets[side * number_of_events];
  auto pixelsSize = parameters.dev_rich_hit_offsets[(side + 1) * number_of_events] - offset;

  const Allen::Rich::Decoding::SmartID* ids = parameters.dev_smart_ids + offset;
  Allen::Rich::PixelReco::Pixel* pixels = parameters.dev_rich_pixels + offset;

  const Allen::Rich::Detector::PDPanel& panel = deRich->pdPanels()[side];
  const auto g2panel = panel.globalToPDPanel(); // cache global to pd panel matrix in registers

  // pixel rec loop, grid-stride loop
  uint global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (uint i = global_tid; i < pixelsSize; i += stride) {
    pixels[i] = make_pixel<rich>(panel, g2panel, ids[i]);
  }
}

/// Function to resize Allen sequence output data structure
void rich_make_pixels::rich_make_pixels_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_rich_pixels_t>(arguments, size<dev_smart_ids_t>(arguments));
}

void rich_make_pixels::rich_make_pixels_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  // Create RICH object from dump
  const auto richValue = m_current_rich;
  const Allen::Rich::RichDetector* rich =
    richValue == 1 ? constants.dev_rich_1_geometry : constants.dev_rich_2_geometry;
  const auto& kernel = richValue == 1 ? rich_make_pixels_kernel<0> : rich_make_pixels_kernel<1>;

  // dont launch too many blocks and instead rely on the grid stride loop to amortize launch overhead
  global_function(kernel)(dim3(16, 2), m_block_dim, context)(arguments, rich);
}
