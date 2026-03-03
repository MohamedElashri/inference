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
#include <RichDecoding.cuh>
#include <PrefixSum.cuh>
#include <RichTel40CableMapping.cuh>
#include <RichPDMDBDecodeMapping.cuh>
#include <MEPTools.h>
#include <WarpIntrinsicsTools.cuh>

INSTANTIATE_ALGORITHM(rich_decoding::rich_decoding_t)

constexpr unsigned MaxConnectionsPerTel40 = Allen::Rich::Decoding::Tel40CableMapping::MaxConnectionsPerTel40;

__device__ unsigned rich_calculate_number_of_hits_in_raw_bank(
  const Allen::RawBank& bank,
  const Allen::Rich::Decoding::Tel40CableMapping* cable_mapping,
  const Allen::Rich::Decoding::PDMDBDecodeMapping* pdmdb_mapping,
  Allen::Rich::Decoding::PackedFrameSizes::IntType* connSizes)
{
  auto tel40ID = bank.source_id;
  const auto connMeta = cable_mapping->tel40Meta(tel40ID);
  const unsigned nActiveLinks = __popc(connMeta.validLinkMask);
  const unsigned nPackedSizeW = (nActiveLinks + 1) / 2;

  auto threadRowDataW = bank.data;
  auto threadRowBankEnd = threadRowDataW + bank.size;

  // Extract sizes using improved memory coalescing pattern
  for (unsigned i = threadIdx.x; i < nPackedSizeW && threadRowDataW + i < threadRowBankEnd; i += blockDim.x) {
    const Allen::Rich::Decoding::PackedFrameSizes sizes(threadRowDataW[i]);
    connSizes[2 * i] = sizes.size1();
    connSizes[2 * i + 1] = sizes.size0();
  }
  __syncwarp();

  auto dataW = threadRowDataW + nPackedSizeW;

  const auto& connData = cable_mapping->tel40Data(tel40ID);

  unsigned localHits = 0; // Hits for this specific section of the bank (thread row)

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

    // decode the word
    if (!isNZS) { // ZS decoding
      if (static_cast<unsigned>(word) < 86) {
        auto b = word + (word >= 39 ? 1 : 0);
        if (frameMask[b / 8] & (1 << (b % 8))) {
          localHits++;
        }
      }
    }
    else { // NZS decoding
      const unsigned iNZS = isFirstHalfNZS ? (iW - sum) : (iW - (sum + linkSize - 5));
      const unsigned nNZSwords = isFirstHalfNZS ? 6 : 5;
      const unsigned halfBitOffset = isFirstHalfNZS ? 40 : 0;
      const unsigned bit = halfBitOffset + (8 * (nNZSwords - 1 - iNZS));
      localHits += __popc(word & frameMask[bit / 8]);
    }
  }

  return Allen::warp::reduce_add_warp_sync(localHits); // Return the total hit count for this bank
}

template<bool mep_layout>
__global__ void rich_calculate_number_of_hits(
  rich_decoding::Parameters parameters,
  const unsigned event_start,
  const Allen::Rich::Decoding::Tel40CableMapping* cable_mapping,
  const Allen::Rich::Decoding::PDMDBDecodeMapping* pdmdb_mapping)
{
  const auto event_number = parameters.dev_event_list[blockIdx.x];
  const auto number_of_events = parameters.dev_event_list.size();

  const auto raw_event = Allen::RawEvent<mep_layout> {parameters.dev_rich_raw_input,
                                                      parameters.dev_rich_raw_input_offsets,
                                                      parameters.dev_rich_raw_input_sizes,
                                                      parameters.dev_rich_raw_input_types,
                                                      event_number + event_start};

  __shared__ Allen::Rich::Decoding::PackedFrameSizes::IntType
    connSizes_shared[MaxConnectionsPerTel40 * RICH_DECODING_BLOCK_DIM_Y];
  Allen::Rich::Decoding::PackedFrameSizes::IntType* connSizes = connSizes_shared + threadIdx.y * MaxConnectionsPerTel40;

  unsigned hitCountPannel0 = 0;
  unsigned hitCountPannel1 = 0;
  for (unsigned bank_number = threadIdx.y; bank_number < raw_event.number_of_raw_banks; bank_number += blockDim.y) {
    const auto bank = raw_event.raw_bank(bank_number);
    const auto hitCount = rich_calculate_number_of_hits_in_raw_bank(bank, cable_mapping, pdmdb_mapping, connSizes);
    const auto side = (bank.source_id >> 10) & 0x1;
    if (side == 0) {
      hitCountPannel0 += hitCount;
    }
    else {
      hitCountPannel1 += hitCount;
    }
  }
  if (threadIdx.x == 0) {
    atomicAdd(parameters.dev_rich_hit_offsets + event_number, hitCountPannel0);
    atomicAdd(parameters.dev_rich_hit_offsets + event_number + number_of_events, hitCountPannel1);
  }
}

template<bool mep_layout, typename word_t>
__global__ void rich_calculate_number_of_hits_StreamIDs(
  rich_decoding::Parameters parameters,
  const unsigned event_start)
{
  const auto event_number = parameters.dev_event_list[blockIdx.x];
  const auto number_of_events = parameters.dev_event_list.size();

  const auto raw_event = Allen::RawEvent<mep_layout> {parameters.dev_rich_raw_input,
                                                      parameters.dev_rich_raw_input_offsets,
                                                      parameters.dev_rich_raw_input_sizes,
                                                      parameters.dev_rich_raw_input_types,
                                                      event_number + event_start};

  for (unsigned bank_number = threadIdx.y * blockDim.x + threadIdx.x; bank_number < raw_event.number_of_raw_banks;
       bank_number += blockDim.x * blockDim.y) {
    const auto bank = raw_event.raw_bank(bank_number);
    const auto hitCount = bank.size / sizeof(word_t);
    const auto side = (bank.source_id >> 10) & 0x1;
    if (side == 0) {
      atomicAdd(parameters.dev_rich_hit_offsets + event_number, hitCount);
    }
    else {
      atomicAdd(parameters.dev_rich_hit_offsets + event_number + number_of_events, hitCount);
    }
  }
}

// ----------------------------------------------------------------------

__device__ void rich_decode_bank(
  const Allen::RawBank& bank,
  const Allen::Rich::Decoding::Tel40CableMapping* cable_mapping,
  const Allen::Rich::Decoding::PDMDBDecodeMapping* pdmdb_mapping,
  Allen::Rich::Decoding::PackedFrameSizes::IntType* connSizes,
  unsigned* event_inserted_hits,
  Allen::Rich::Decoding::SmartID* event_smart_ids)
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

          // Add the PMT and pixel info
          const auto nInMod = (aData.ec * (0 != ((hitID.key() >> 27) & 0x1) ? 1 : 4)) + aData.pmtInEC;
          hitID.setData(
            nInMod, Allen::Rich::Decoding::SmartID::ShiftPDNumInMod, Allen::Rich::Decoding::SmartID::MaskPDNumInMod);
          hitID.setData(
            aData.anode,
            Allen::Rich::Decoding::SmartID::ShiftPixelCol,
            Allen::Rich::Decoding::SmartID::MaskPixelCol | Allen::Rich::Decoding::SmartID::MaskPixelRow,
            Allen::Rich::Decoding::SmartID::MaskPixelColIsSet | Allen::Rich::Decoding::SmartID::MaskPixelRowIsSet);

          const auto insert_index = atomicAdd(event_inserted_hits, 1);
          event_smart_ids[insert_index] = hitID;
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

      // Reserve enough space to push hits
      auto insert_index = atomicAdd(event_inserted_hits, __popc(word));

      // Use find first set intrinsic to quickly iterate over valid hits
      while (word != 0) {
        unsigned iLB = __ffs(word) - 1; // __ffs returns 1-based index
        const auto bit = iLB + bitOffset;

        if (bit < frameData.size()) {
          const auto& aData = frameData[bit];
          // make a smart ID
          auto hitID = Allen::Rich::Decoding::SmartID {cData.smartID}; // sets RICH, side, module and PMT type

          // Add the PMT and pixel info
          const auto nInMod = (aData.ec * (0 != ((hitID.key() >> 27) & 0x1) ? 1 : 4)) + aData.pmtInEC;
          hitID.setData(
            nInMod, Allen::Rich::Decoding::SmartID::ShiftPDNumInMod, Allen::Rich::Decoding::SmartID::MaskPDNumInMod);
          hitID.setData(
            aData.anode,
            Allen::Rich::Decoding::SmartID::ShiftPixelCol,
            Allen::Rich::Decoding::SmartID::MaskPixelCol | Allen::Rich::Decoding::SmartID::MaskPixelRow,
            Allen::Rich::Decoding::SmartID::MaskPixelColIsSet | Allen::Rich::Decoding::SmartID::MaskPixelRowIsSet);

          event_smart_ids[insert_index++] = hitID;
        }
        word &= ~(1u << iLB); // Clear first set bit
      }
    }
  }
  __syncwarp();
}

template<bool mep_layout>
__global__ void rich_decoding_kernel(
  rich_decoding::Parameters parameters,
  const unsigned event_start,
  const Allen::Rich::Decoding::Tel40CableMapping* cable_mapping,
  const Allen::Rich::Decoding::PDMDBDecodeMapping* pdmdb_mapping)
{
  const auto event_number = parameters.dev_event_list[blockIdx.x];
  const auto number_of_events = parameters.dev_event_list.size();

  // Read raw event
  const auto raw_event = Allen::RawEvent<mep_layout> {parameters.dev_rich_raw_input,
                                                      parameters.dev_rich_raw_input_offsets,
                                                      parameters.dev_rich_raw_input_sizes,
                                                      parameters.dev_rich_raw_input_types,
                                                      event_number + event_start};

  __shared__ Allen::Rich::Decoding::PackedFrameSizes::IntType
    connSizes_shared[MaxConnectionsPerTel40 * RICH_DECODING_BLOCK_DIM_Y];
  Allen::Rich::Decoding::PackedFrameSizes::IntType* connSizes = connSizes_shared + threadIdx.y * MaxConnectionsPerTel40;

  auto event_smart_ids0 = parameters.dev_smart_ids + parameters.dev_rich_hit_offsets[event_number];
  auto event_smart_ids1 = parameters.dev_smart_ids + parameters.dev_rich_hit_offsets[event_number + number_of_events];

  __shared__ unsigned event_inserted_hits[2];
  if (threadIdx.y == 0) {
    for (unsigned i = threadIdx.x; i < 2; i += blockDim.x) {
      event_inserted_hits[i] = 0;
    }
  }
  __syncthreads();

  for (unsigned bank_number = threadIdx.y; bank_number < raw_event.number_of_raw_banks; bank_number += blockDim.y) {
    const auto bank = raw_event.raw_bank(bank_number);
    const auto side = (bank.source_id >> 10) & 0x1;
    rich_decode_bank(
      bank,
      cable_mapping,
      pdmdb_mapping,
      connSizes,
      &event_inserted_hits[side],
      side == 0 ? event_smart_ids0 : event_smart_ids1);
  }
}

template<bool mep_layout, typename word_t>
__global__ void rich_decoding_kernel_StreamIDs(rich_decoding::Parameters parameters, const unsigned event_start)
{
  const auto event_number = parameters.dev_event_list[blockIdx.x];
  const auto number_of_events = parameters.dev_event_list.size();

  const auto raw_event = Allen::RawEvent<mep_layout> {parameters.dev_rich_raw_input,
                                                      parameters.dev_rich_raw_input_offsets,
                                                      parameters.dev_rich_raw_input_sizes,
                                                      parameters.dev_rich_raw_input_types,
                                                      event_number + event_start};

  auto event_smart_ids0 = parameters.dev_smart_ids + parameters.dev_rich_hit_offsets[event_number];
  auto event_smart_ids1 = parameters.dev_smart_ids + parameters.dev_rich_hit_offsets[event_number + number_of_events];

  __shared__ unsigned event_inserted_hits[2];
  if (threadIdx.y == 0) {
    for (unsigned i = threadIdx.x; i < 2; i += blockDim.x) {
      event_inserted_hits[i] = 0;
    }
  }
  __syncthreads();

  for (unsigned bank_number = threadIdx.y; bank_number < raw_event.number_of_raw_banks; bank_number += blockDim.y) {
    const auto bank = raw_event.raw_bank(bank_number);
    const auto side = (bank.source_id >> 10) & 0x1;
    const word_t* bank_data = reinterpret_cast<const word_t*>(bank.data);

    unsigned* bank_inserted_hits = &event_inserted_hits[side];
    Allen::Rich::Decoding::SmartID* event_smart_ids = side == 0 ? event_smart_ids0 : event_smart_ids1;

    for (unsigned iW = threadIdx.x; iW < bank.size / sizeof(word_t); iW += blockDim.x) {
      auto hitID = Allen::Rich::Decoding::SmartID {bank_data[iW]};
#ifdef USE_DD4HEP
      if (hitID.rich() == 1) {
        const auto pdMod = hitID.pdMod() + (hitID.panel() == 0 ? 6 : 18);
        hitID.setData(pdMod, Allen::Rich::Decoding::SmartID::ShiftPDMod, Allen::Rich::Decoding::SmartID::MaskPDMod);
      }
#endif
      const auto insert_index = atomicAdd(bank_inserted_hits, 1);
      event_smart_ids[insert_index] = hitID;
    }
  }
}

void rich_decoding::rich_decoding_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_rich_hit_offsets_t>(arguments, size<dev_event_list_t>(arguments) * 2 + 1);
  set_size<host_rich_total_number_of_hits_t>(arguments, 1);
}

void rich_decoding::rich_decoding_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions& runtime_options,
  const Constants& constants,
  const Allen::Context& context) const
{
  Allen::memset_async<dev_rich_hit_offsets_t>(arguments, 0, context);

  const auto bank_version = first<host_raw_bank_version_t>(arguments);
  if (bank_version == 10) { /// 'Production' PMT version  (Used for real data)
    Allen::Rich::Decoding::Tel40CableMapping* cable_mapping = constants.dev_rich_cable_mapping;
    Allen::Rich::Decoding::PDMDBDecodeMapping* pdmdb_mapping = constants.dev_rich_pdmdb_mapping;

    // Calculate number of hits into dev_rich_hit_offsets_t
    global_function(
      runtime_options.mep_layout ? rich_calculate_number_of_hits<true> : rich_calculate_number_of_hits<false>)(
      dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
      arguments, std::get<0>(runtime_options.event_interval), cable_mapping, pdmdb_mapping);
    PrefixSum::prefix_sum<dev_rich_hit_offsets_t, host_rich_total_number_of_hits_t>(*this, arguments, context);

    // Decode RICH hits
    resize<dev_smart_ids_t>(arguments, first<host_rich_total_number_of_hits_t>(arguments));

    global_function(runtime_options.mep_layout ? rich_decoding_kernel<true> : rich_decoding_kernel<false>)(
      dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
      arguments, std::get<0>(runtime_options.event_interval), cable_mapping, pdmdb_mapping);
  }
  else if (bank_version == 3) { /// Simple version. Just streams SmartIDs, excluding the higher time bits. (Used for MC
                                /// data)
    global_function(
      runtime_options.mep_layout ? rich_calculate_number_of_hits_StreamIDs<true, uint32_t> :
                                   rich_calculate_number_of_hits_StreamIDs<false, uint32_t>)(
      dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
      arguments, std::get<0>(runtime_options.event_interval));
    PrefixSum::prefix_sum<dev_rich_hit_offsets_t, host_rich_total_number_of_hits_t>(*this, arguments, context);
    resize<dev_smart_ids_t>(arguments, first<host_rich_total_number_of_hits_t>(arguments));

    global_function(
      runtime_options.mep_layout ?
        rich_decoding_kernel_StreamIDs<true, uint32_t> :
        rich_decoding_kernel_StreamIDs<false, uint32_t>)(dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
      arguments, std::get<0>(runtime_options.event_interval));
  }
  else if (bank_version == 4) { /// Updated streaming for 64 bit SmartIDs, with adc time information (Used for MC data)
    global_function(
      runtime_options.mep_layout ? rich_calculate_number_of_hits_StreamIDs<true, uint64_t> :
                                   rich_calculate_number_of_hits_StreamIDs<false, uint64_t>)(
      dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
      arguments, std::get<0>(runtime_options.event_interval));
    PrefixSum::prefix_sum<dev_rich_hit_offsets_t, host_rich_total_number_of_hits_t>(*this, arguments, context);
    resize<dev_smart_ids_t>(arguments, first<host_rich_total_number_of_hits_t>(arguments));

    global_function(
      runtime_options.mep_layout ?
        rich_decoding_kernel_StreamIDs<true, uint64_t> :
        rich_decoding_kernel_StreamIDs<false, uint64_t>)(dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
      arguments, std::get<0>(runtime_options.event_interval));
  }
  else {
    throw StrException("Rich bank version not supported (" + std::to_string(bank_version) + ")");
  }

  if (m_verbosity >= logger::debug) {
    // Print output
    print<dev_rich_hit_offsets_t>(arguments);
    print<dev_smart_ids_t>(arguments);
  }
}
