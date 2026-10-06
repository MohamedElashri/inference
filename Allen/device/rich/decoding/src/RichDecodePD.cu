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
#include <RichDecodePD.cuh>
#include <PrefixSum.cuh>
#include <RichTel40CableMapping.cuh>
#include <RichPDMDBDecodeMapping.cuh>
#include <MEPTools.h>
#include <RichDefinitions.cuh>
#include <Rich.cuh>
#include <RichPhotonDetector.cuh>

#if defined(TARGET_DEVICE_CUDA)
#define RICH_DECODING_BLOCK_DIM_X 32
#define RICH_DECODING_BLOCK_DIM_Y 4
#else
// Default to CPU-like configuration
// Although Allen automatically sets this up for CPU,
// we need a variable for array sizes in compile-time
#define RICH_DECODING_BLOCK_DIM_X 1
#define RICH_DECODING_BLOCK_DIM_Y 1
#endif

INSTANTIATE_ALGORITHM(rich_decode_pd::rich_decode_pd_t)

constexpr unsigned MaxConnectionsPerTel40 = Allen::Rich::Decoding::Tel40CableMapping::MaxConnectionsPerTel40;

template<Allen::Rich::Detector::DetectorType richIdx>
__device__ void rich_decode_bank(
  const Allen::RawBank& bank,
  const Allen::Rich::Decoding::Tel40CableMapping* cable_mapping,
  const Allen::Rich::Decoding::PDMDBDecodeMapping* pdmdb_mapping,
  const Allen::Rich::Detector::PDPanel<richIdx>& panel,
  Allen::Rich::Decoding::PackedFrameSizes::IntType* connSizes,
  uint64_t* pd_pixels)
{
  auto tel40ID = bank.source_id;
  const auto connMeta = cable_mapping->tel40Meta(tel40ID);
  const unsigned nActiveLinks = __popc(connMeta.validLinkMask);
  const unsigned nPackedSizeW = (nActiveLinks + 1) / 2;

  auto threadRowDataW = bank.data;

  for (unsigned i = threadIdx.x; i < nPackedSizeW; i += blockDim.x) {
    const Allen::Rich::Decoding::PackedFrameSizes sizes(threadRowDataW[i]);
    connSizes[2 * i] = sizes.size1();
    connSizes[2 * i + 1] = sizes.size0();
  }
  __syncwarp();

  auto dataW = threadRowDataW;
  dataW += nPackedSizeW;

  const auto& connData = cable_mapping->tel40Data(tel40ID);

  const unsigned nWords = bank.size - nPackedSizeW;
  uint32_t prev_NZS_first_mask = 0;
  uint32_t prev_NZS_second_mask = 0;
  unsigned iLink = 0;
  unsigned sum = 0;
  unsigned linkSize = connSizes[0];
  for (unsigned iW = threadIdx.x; iW < nWords; iW += blockDim.x) {
    const unsigned loop_mask = ((iW + RICH_DECODING_BLOCK_DIM_X) > nWords) ?
                                 ~(0xFFFFFFFF << (nWords - (iW & ~(RICH_DECODING_BLOCK_DIM_X - 1)))) :
                                 0xFFFFFFFF;

    uint8_t word = dataW[iW];
    bool isNZS = (word & 0x80) != 0;

    // find which link we are in and the start of the link word range
    while (sum + linkSize <= iW) {
      sum += linkSize;
      iLink++;
      linkSize = connSizes[iLink];
    }

    // handle first half (6bytes)
    bool is_first_half_head = sum == iW && isNZS && linkSize > 5;
    uint64_t first_half_mask = __ballot_sync(loop_mask, is_first_half_head);
    first_half_mask *= (1 << 6) - 1; // propagate nzs info to the next 6 bits
    first_half_mask |= prev_NZS_first_mask;
    prev_NZS_first_mask = first_half_mask >> RICH_DECODING_BLOCK_DIM_X;

    bool isFirstHalfNZS = first_half_mask & (1 << threadIdx.x);

    // handle second half (5bytes)
    bool is_second_half_head = (sum + linkSize - 5) == iW && !isFirstHalfNZS && isNZS;
    uint64_t second_half_mask = __ballot_sync(loop_mask, is_second_half_head);
    second_half_mask *= (1 << 5) - 1; // propagate nzs info to the next 5 bits
    second_half_mask |= prev_NZS_second_mask;
    prev_NZS_second_mask = second_half_mask >> RICH_DECODING_BLOCK_DIM_X;

    // merge propagated masks and extract true isNZS
    uint32_t NZS_mask = first_half_mask | second_half_mask;
    isNZS = NZS_mask & (1 << threadIdx.x);

    // get frame mask
    const auto& cData = connData[__fns(connMeta.validLinkMask, 0, iLink + 1)];
    const auto& frameMask = pdmdb_mapping->getFrameValidMask(cData);
    const auto& frameData = pdmdb_mapping->getFrameData(cData);

    // decode the word
    if (!isNZS) { // ZS decoding
      if (static_cast<unsigned>(word) < frameData.size()) {
        auto b = word + (word >= 39 ? 1 : 0);
        if (frameMask[b / 8] & (1 << (b % 8))) {
          const auto& aData = frameData[word];

          // make a smart ID
          auto hitID = Allen::Rich::Decoding::SmartID {cData.smartID}; // sets RICH, side, module and PMT type
          const auto nInMod = (aData.ec * (0 != ((hitID.key() >> 27) & 0x1) ? 1 : 4)) + aData.pmtInEC;
          hitID.setData(
            nInMod, Allen::Rich::Decoding::SmartID::ShiftPDNumInMod, Allen::Rich::Decoding::SmartID::MaskPDNumInMod);

          const auto denseID = Allen::Rich::Detector::PDPanel<richIdx>::smartID2denseID(hitID);
          if (!panel.pds()[denseID].getIsNull()) {
            atomicOr(&pd_pixels[denseID], 1ULL << aData.anode);
          }
        }
      }
    }
    else { // NZS decoding
      const unsigned iNZS = isFirstHalfNZS ? (iW - sum) : (iW - (sum + linkSize - 5));
      const unsigned nNZSwords = isFirstHalfNZS ? 6 : 5;
      const unsigned halfBitOffset = isFirstHalfNZS ? 39 : 0;
      const unsigned bitOffset = halfBitOffset + (8 * (nNZSwords - 1 - iNZS));

      auto b = bitOffset + (bitOffset >= 39 ? 1 : 0);
      word &= frameMask[b / 8]; // mask invalid hits

      // Use find first set intrinsic to quickly iterate over valid hits
      while (word != 0) {
        unsigned iLB = __ffs(word) - 1; // __ffs returns 1-based index
        const auto bit = iLB + bitOffset;

        if (bit < frameData.size()) {
          const auto& aData = frameData[bit];
          // make a smart ID
          auto hitID = Allen::Rich::Decoding::SmartID {cData.smartID}; // sets RICH, side, module and PMT type
          const auto nInMod = (aData.ec * (0 != ((hitID.key() >> 27) & 0x1) ? 1 : 4)) + aData.pmtInEC;
          hitID.setData(
            nInMod, Allen::Rich::Decoding::SmartID::ShiftPDNumInMod, Allen::Rich::Decoding::SmartID::MaskPDNumInMod);

          const auto denseID = Allen::Rich::Detector::PDPanel<richIdx>::smartID2denseID(hitID);
          if (!panel.pds()[denseID].getIsNull()) {
            atomicOr(&pd_pixels[denseID], 1ULL << aData.anode);
          }
        }
        word &= ~(1u << iLB); // Clear first set bit
      }
    }
  }
  __syncwarp();
}

template<bool mep_layout, Allen::Rich::Detector::DetectorType richIdx>
__global__ void rich_make_pd_pixels_k(
  rich_decode_pd::Parameters parameters,
  const unsigned event_start,
  const Allen::Rich::Decoding::Tel40CableMapping* cable_mapping,
  const Allen::Rich::Decoding::PDMDBDecodeMapping* pdmdb_mapping,
  const Allen::Rich::RichDetector<richIdx>* rich)
{
  const auto event_number = parameters.dev_event_list[blockIdx.x];
  const auto number_of_events = parameters.dev_event_list.size();

  const auto raw_event = Allen::RawEvent<mep_layout> {
    parameters.dev_rich_raw_input,
    parameters.dev_rich_raw_input_offsets,
    parameters.dev_rich_raw_input_sizes,
    parameters.dev_rich_raw_input_types,
    event_number + event_start};

  __shared__ Allen::Rich::Decoding::PackedFrameSizes::IntType
    connSizes_shared[MaxConnectionsPerTel40 * RICH_DECODING_BLOCK_DIM_Y];
  Allen::Rich::Decoding::PackedFrameSizes::IntType* connSizes = connSizes_shared + threadIdx.y * MaxConnectionsPerTel40;

  for (unsigned bank_number = threadIdx.y; bank_number < raw_event.number_of_raw_banks; bank_number += blockDim.y) {
    const auto bank = raw_event.raw_bank(bank_number);
    const auto side = static_cast<Allen::Rich::Detector::Side>((bank.source_id >> 10) & 0x1);
    const Allen::Rich::Detector::PDPanel<richIdx>& panel = rich->pdPanels()[side];
    uint64_t* pd_pixels = parameters.dev_pd_pixels + (number_of_events * side + event_number) *
                                                       Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;

    rich_decode_bank<richIdx>(bank, cable_mapping, pdmdb_mapping, panel, connSizes, pd_pixels);
  }
}

template<bool mep_layout, Allen::Rich::Detector::DetectorType richIdx, typename word_t>
__global__ void rich_make_pd_pixels_StreamIDs_k(
  rich_decode_pd::rich_decode_pd_t::Parameters parameters,
  const unsigned event_start)
{
  const auto event_number = parameters.dev_event_list[blockIdx.x];
  const auto number_of_events = parameters.dev_event_list.size();

  const auto raw_event = Allen::RawEvent<mep_layout> {
    parameters.dev_rich_raw_input,
    parameters.dev_rich_raw_input_offsets,
    parameters.dev_rich_raw_input_sizes,
    parameters.dev_rich_raw_input_types,
    event_number + event_start};

  for (unsigned bank_number = threadIdx.y; bank_number < raw_event.number_of_raw_banks; bank_number += blockDim.y) {
    const auto bank = raw_event.raw_bank(bank_number);
    const auto side = static_cast<Allen::Rich::Detector::Side>((bank.source_id >> 10) & 0x1);
    uint64_t* pd_pixels = parameters.dev_pd_pixels + (number_of_events * side + event_number) *
                                                       Allen::Rich::Detector::PDPanel<richIdx>::PDsPerPanel;
    const word_t* bank_data = reinterpret_cast<const word_t*>(bank.data);

    for (unsigned iW = threadIdx.x; iW < bank.size / sizeof(word_t); iW += blockDim.x) {
      auto hitID = Allen::Rich::Decoding::SmartID {bank_data[iW]};
      if (hitID.rich() == Allen::Rich::Detector::Rich2) {
        const auto pdMod = hitID.pdMod() + (hitID.panel() == Allen::Rich::Detector::firstSide ? 6 : 18);
        hitID.setData(pdMod, Allen::Rich::Decoding::SmartID::ShiftPDMod, Allen::Rich::Decoding::SmartID::MaskPDMod);
      }
      const auto denseID = Allen::Rich::Detector::PDPanel<richIdx>::smartID2denseID(hitID);
      atomicOr(&pd_pixels[denseID], 1ULL << hitID.anodeIndex());
    }
  }
}

__global__ void rich_count_pixels_k(const unsigned number_of_pds, const uint64_t* pd_pixels, unsigned* counts)
{
  const unsigned threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned stride = blockDim.x * gridDim.x;
  for (unsigned i = threadId; i < number_of_pds; i += stride) {
    counts[i] = __popcll(pd_pixels[i]);
  }
}

__global__ void rich_copy_offsets_k(
  const unsigned number_of_panels,
  const unsigned pds_per_panel,
  const unsigned* pd_offsets,
  unsigned* hits_offsets)
{
  const unsigned threadId = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned stride = blockDim.x * gridDim.x;
  for (unsigned i = threadId; i < number_of_panels; i += stride) {
    hits_offsets[i] = pd_offsets[i * pds_per_panel];
  }
}

void rich_decode_pd::rich_decode_pd_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  const unsigned number_of_events = size<dev_event_list_t>(arguments);
  const auto rich = Allen::Rich::Detector::detectorTypeFromNumber(m_current_rich.value());
  const unsigned n_pds = rich == Allen::Rich::Detector::Rich1 ?
                           Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich1>::PDsPerRich :
                           Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich2>::PDsPerRich;
  set_size<dev_rich_pd_offsets_t>(arguments, number_of_events * n_pds + 1);
  set_size<dev_pd_pixels_t>(arguments, number_of_events * n_pds);
  set_size<host_rich_total_number_of_hits_t>(arguments, 1);
  set_size<dev_rich_hit_offsets_t>(arguments, Allen::Rich::NPDPanelsPerRICH * number_of_events + 1);
}

void rich_decode_pd::rich_decode_pd_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_rich_pd_offsets_t>(arguments, 0, context);
  Allen::memset_async<dev_pd_pixels_t>(arguments, 0, context);
  Allen::synchronize(context);

  const auto rich = Allen::Rich::Detector::detectorTypeFromNumber(m_current_rich.value());
  const auto bank_version = first<host_raw_bank_version_t>(arguments);
  if (bank_version == 10) { /// 'Production' PMT version  (Used for real data)
    Allen::Rich::Decoding::Tel40CableMapping* cable_mapping = constants.dev_rich_cable_mapping;
    Allen::Rich::Decoding::PDMDBDecodeMapping* pdmdb_mapping = constants.dev_rich_pdmdb_mapping;

    // Decode rawbanks into dense pixel bitmasks
    if (rich == Allen::Rich::Detector::Rich1) {
      auto kernel = runtime_options.mep_layout ? rich_make_pd_pixels_k<true, Allen::Rich::Detector::Rich1> :
                                                 rich_make_pd_pixels_k<false, Allen::Rich::Detector::Rich1>;
      global_function(kernel)(
        dim3(size<dev_event_list_t>(arguments)), dim3(RICH_DECODING_BLOCK_DIM_X, RICH_DECODING_BLOCK_DIM_Y), context)(
        arguments,
        std::get<0>(runtime_options.event_interval),
        cable_mapping,
        pdmdb_mapping,
        constants.dev_rich_1_geometry);
    }
    else {
      auto kernel = runtime_options.mep_layout ? rich_make_pd_pixels_k<true, Allen::Rich::Detector::Rich2> :
                                                 rich_make_pd_pixels_k<false, Allen::Rich::Detector::Rich2>;
      global_function(kernel)(
        dim3(size<dev_event_list_t>(arguments)), dim3(RICH_DECODING_BLOCK_DIM_X, RICH_DECODING_BLOCK_DIM_Y), context)(
        arguments,
        std::get<0>(runtime_options.event_interval),
        cable_mapping,
        pdmdb_mapping,
        constants.dev_rich_2_geometry);
    }
  }
  else if (bank_version == 3) { /// Simple version. Just streams SmartIDs, excluding the higher time bits. (Used for MC
                                /// data)
    auto kernel = runtime_options.mep_layout ?
                    (rich == Allen::Rich::Detector::Rich1 ?
                       rich_make_pd_pixels_StreamIDs_k<true, Allen::Rich::Detector::Rich1, uint32_t> :
                       rich_make_pd_pixels_StreamIDs_k<true, Allen::Rich::Detector::Rich2, uint32_t>) :
                    (rich == Allen::Rich::Detector::Rich1 ?
                       rich_make_pd_pixels_StreamIDs_k<false, Allen::Rich::Detector::Rich1, uint32_t> :
                       rich_make_pd_pixels_StreamIDs_k<false, Allen::Rich::Detector::Rich2, uint32_t>);
    global_function(kernel)(
      dim3(size<dev_event_list_t>(arguments)), dim3(RICH_DECODING_BLOCK_DIM_X, RICH_DECODING_BLOCK_DIM_Y), context)(
      arguments, std::get<0>(runtime_options.event_interval));
  }
  else if (bank_version == 4) { /// Updated streaming for 64 bit SmartIDs, with adc time information (Used for MC data)
    auto kernel = runtime_options.mep_layout ?
                    (rich == Allen::Rich::Detector::Rich1 ?
                       rich_make_pd_pixels_StreamIDs_k<true, Allen::Rich::Detector::Rich1, uint64_t> :
                       rich_make_pd_pixels_StreamIDs_k<true, Allen::Rich::Detector::Rich2, uint64_t>) :
                    (rich == Allen::Rich::Detector::Rich1 ?
                       rich_make_pd_pixels_StreamIDs_k<false, Allen::Rich::Detector::Rich1, uint64_t> :
                       rich_make_pd_pixels_StreamIDs_k<false, Allen::Rich::Detector::Rich2, uint64_t>);
    global_function(kernel)(
      dim3(size<dev_event_list_t>(arguments)), dim3(RICH_DECODING_BLOCK_DIM_X, RICH_DECODING_BLOCK_DIM_Y), context)(
      arguments, std::get<0>(runtime_options.event_interval));
  }
  else {
    throw StrException("Rich bank version not supported (" + std::to_string(bank_version) + ")");
  }

  // Count pixels and prefix sum
  global_function(rich_count_pixels_k)(dim3(32), dim3(256), context)(
    size<dev_pd_pixels_t>(arguments), data<dev_pd_pixels_t>(arguments), data<dev_rich_pd_offsets_t>(arguments));
  PrefixSum::prefix_sum<dev_rich_pd_offsets_t, host_rich_total_number_of_hits_t>(*this, arguments, context);

  const unsigned pds_per_panel = rich == Allen::Rich::Detector::Rich1 ?
                                   Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich1>::PDsPerPanel :
                                   Allen::Rich::Detector::PDPanel<Allen::Rich::Detector::Rich2>::PDsPerPanel;
  global_function(rich_copy_offsets_k)(dim3(1), dim3(256), context)(
    size<dev_rich_hit_offsets_t>(arguments),
    pds_per_panel,
    data<dev_rich_pd_offsets_t>(arguments),
    data<dev_rich_hit_offsets_t>(arguments));
}
