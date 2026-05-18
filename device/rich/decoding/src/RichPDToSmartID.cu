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
#include <RichPDToSmartID.cuh>
#include <RichDefinitions.cuh>
#include <Rich.cuh>
#include <RichPhotonDetector.cuh>

/**
 * Simple algorithm to convert photodetectors bitmasks to pixels smartID.
 * Used to compare with HLT2 and mc checking, this shouldn't normally be needed.
 */

INSTANTIATE_ALGORITHM(rich_pd_to_smartid::rich_pd_to_smartid_t)

template<unsigned richIdx>
__global__ void rich_pd_to_smartid_k(
  const Allen::Rich::RichDetector<richIdx>* deRich,
  const unsigned number_of_pds,
  const uint64_t* pd_pixels,
  const unsigned* pd_offsets,
  Allen::Rich::Decoding::SmartID* smartids)
{
  const unsigned threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned stride = blockDim.x * gridDim.x;
  for (unsigned i = threadId; i < number_of_pds; i += stride) {
    uint64_t pixels = pd_pixels[i];
    Allen::Rich::Decoding::SmartID* out = smartids + pd_offsets[i];

    const unsigned side = i / (number_of_pds / 2);
    const unsigned denseID = i % Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;

    const auto& pd = deRich->pdPanels()[side].pds()[denseID];

    unsigned idx = 0;
    while (pixels != 0) {
      unsigned anode = __ffsll(pixels) - 1; // __ffsll returns 1-based index

      Allen::Rich::Decoding::SmartID hitID {pd.pdSmartID()}; // sets RICH, side, module and PMT type
      hitID.setData(
        anode,
        Allen::Rich::Decoding::SmartID::ShiftPixelCol,
        Allen::Rich::Decoding::SmartID::MaskPixelCol | Allen::Rich::Decoding::SmartID::MaskPixelRow,
        Allen::Rich::Decoding::SmartID::MaskPixelColIsSet | Allen::Rich::Decoding::SmartID::MaskPixelRowIsSet);

      out[idx++] = hitID;
      pixels &= ~(1ull << anode); // Clear first set bit
    }
  }
}

void rich_pd_to_smartid::rich_pd_to_smartid_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_smart_ids_t>(arguments, first<host_rich_total_number_of_hits_t>(arguments));
}

void rich_pd_to_smartid::rich_pd_to_smartid_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants& constants,
  const Allen::Context& context) const
{
  if (m_current_rich == 1) {
    global_function(rich_pd_to_smartid_k<0>)(dim3(32), dim3(256), context)(
      constants.dev_rich_1_geometry,
      size<dev_pd_pixels_t>(arguments),
      data<dev_pd_pixels_t>(arguments),
      data<dev_rich_pd_offsets_t>(arguments),
      data<dev_smart_ids_t>(arguments));
  }
  else if (m_current_rich == 2) {
    global_function(rich_pd_to_smartid_k<1>)(dim3(32), dim3(256), context)(
      constants.dev_rich_2_geometry,
      size<dev_pd_pixels_t>(arguments),
      data<dev_pd_pixels_t>(arguments),
      data<dev_rich_pd_offsets_t>(arguments),
      data<dev_smart_ids_t>(arguments));
  }
}
