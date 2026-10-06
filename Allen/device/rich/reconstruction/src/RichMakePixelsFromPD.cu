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

#include "RichDefinitions.cuh"
#include "Rich.cuh"
#include <RichMakePixelsFromPD.cuh>
#include <RichPhotonDetector.cuh>
#include <BinarySearch.cuh>

INSTANTIATE_ALGORITHM(rich_make_pixels_from_pd::rich_make_pixels_from_pd_t);

/// Kernel function to iterate over Events, SmartIDs, and create Pixels
template<Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_make_pixels_from_pd_k(
  rich_make_pixels_from_pd::Parameters parameters,
  const Allen::Rich::RichDetector<richIdx>* deRich)
{
  const auto side = static_cast<Allen::Rich::Detector::Side>(blockIdx.y);
  const auto number_of_events = parameters.dev_event_list.size();
  auto offset = side * number_of_events * Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;

  const unsigned* pd_offsets = parameters.dev_rich_pd_offsets + offset;
  const uint64_t* pd_pixels = parameters.dev_pd_pixels + offset;

  auto pixelsEnd = pd_offsets[number_of_events * Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel];

  const Allen::Rich::Detector::PDPanel<richIdx>& panel = deRich->pdPanels()[side];
  const auto g2panel = panel.globalToPDPanel(); // cache global to pd panel matrix in registers

  // pixel rec loop, grid-stride loop
  uint global_tid = blockIdx.x * blockDim.x + threadIdx.x + pd_offsets[0];
  uint stride = blockDim.x * gridDim.x;
  for (uint i = global_tid; i < pixelsEnd; i += stride) {
    unsigned pd_id = binary_search_rightmost(
      pd_offsets, number_of_events * Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel + 1, i);

    uint64_t pixels = pd_pixels[pd_id];

    const unsigned denseID = pd_id % Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;
    const auto& pd = panel.pds()[denseID];

    unsigned pixId = i - pd_offsets[pd_id];
    unsigned half_count = __popc((uint32_t) pixels);

    unsigned anode = pixId < half_count ? __fns((uint32_t) pixels, 0, pixId + 1) :
                                          __fns((uint32_t) (pixels >> 32), 0, pixId - half_count + 1) + 32;

    Allen::Rich::Decoding::SmartID hitID {pd.pdSmartID()}; // sets RICH, side, module and PMT type
    hitID.setData(
      anode,
      Allen::Rich::Decoding::SmartID::ShiftPixelCol,
      Allen::Rich::Decoding::SmartID::MaskPixelCol | Allen::Rich::Decoding::SmartID::MaskPixelRow,
      Allen::Rich::Decoding::SmartID::MaskPixelColIsSet | Allen::Rich::Decoding::SmartID::MaskPixelRowIsSet);

    // From panel local coordinates to global coordinates.
    const auto gPos = pd.globalDetectionPoint(hitID);
    // From global coordinates to RICH detector local coordinates.
    const auto lPos = Allen::Rich::transform3DTimesPoint(g2panel, gPos);
    parameters.dev_rich_pixels_gpos[i] = gPos;
    parameters.dev_rich_pixels_lpos[i] =
      make_short2(lPos.x * ((1 << 15) / 750.f) + .5f, lPos.y * ((1 << 15) / 750.f) + .5f);
  }
}

/// Function to resize Allen sequence output data structure
void rich_make_pixels_from_pd::rich_make_pixels_from_pd_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_rich_pixels_lpos_t>(arguments, first<host_rich_total_number_of_hits_t>(arguments));
  set_size<dev_rich_pixels_gpos_t>(arguments, first<host_rich_total_number_of_hits_t>(arguments));
}

void rich_make_pixels_from_pd::rich_make_pixels_from_pd_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  const auto rich = Allen::Rich::Detector::detectorTypeFromNumber(m_current_rich.value());

  // dont launch too many blocks and instead rely on the grid stride loop to amortize launch overhead
  if (rich == Allen::Rich::Detector::Rich1) {
    global_function(rich_make_pixels_from_pd_k<Allen::Rich::Detector::Rich1>)(
      dim3(32, Allen::Rich::NPDPanelsPerRICH), dim3(128), context)(arguments, constants.dev_rich_1_geometry);
  }
  else {
    global_function(rich_make_pixels_from_pd_k<Allen::Rich::Detector::Rich2>)(
      dim3(32, Allen::Rich::NPDPanelsPerRICH), dim3(128), context)(arguments, constants.dev_rich_2_geometry);
  }
}
