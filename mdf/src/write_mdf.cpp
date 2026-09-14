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
#include <cstring>

#include "Event/RawBank.h"
#include "write_mdf.hpp"

LHCb::MDFHeader*
Allen::add_mdf_header(std::span<char> event_span, unsigned const run_number, std::span<unsigned const> routing_bits)
{
  auto const header_size = LHCb::MDFHeader::sizeOf(Allen::mdf_header_version);

  // Add the header
  auto* header = reinterpret_cast<LHCb::MDFHeader*>(event_span.data());
  // Set header version first so the subsequent call to setSize can
  // use it
  header->setHeaderVersion(Allen::mdf_header_version);
  // MDFHeader::setSize adds the header size internally, so pass
  // only the payload size here
  header->setSize(event_span.size() - header_size);

  // No compression here, handled at write time
  header->setCompression(0);
  header->setSubheaderLength(header_size - sizeof(LHCb::MDFHeader));
  header->setDataType(LHCb::MDFHeader::BODY_TYPE_BANKS);
  header->setSpare(0);

  // Put the routing bits into the trigger mask
  std::array<uint32_t, 4> trigger_mask {~0u, ~0u, ~0u, ~0u};
  std::memcpy(&trigger_mask[0], routing_bits.data(), routing_bits.size_bytes());
  header->subHeader().H1->setTriggerMask(trigger_mask.data());
  // Set run number
  // FIXME: get orbit and bunch number from ODIN
  // The batch is offset by start_event with respect to the slice, so we add start_event
  header->subHeader().H1->setRunNumber(run_number);

  return header;
}

size_t Allen::add_raw_bank(
  unsigned char const type,
  unsigned char const version,
  short const sourceID,
  std::span<char const> fragment,
  char* buffer)
{
  auto* bank = reinterpret_cast<LHCb::RawBank*>(buffer);
  bank->setMagic();
  bank->setSize(fragment.size());
  bank->setType(static_cast<LHCb::RawBank::BankType>(type));
  bank->setVersion(version);
  bank->setSourceID(sourceID);
  std::memcpy(bank->begin<char>(), fragment.data(), fragment.size());

  // pad to a multiple of 4 bytes
  auto const padded_size = padded_bank_size(fragment.size());
  std::memset(bank->begin<char>() + fragment.size(), 0, padded_size - fragment.size());
  if (static_cast<LHCb::RawBank::BankType>(type) < LHCb::RawBank::BankType::DaqErrorFragmentThrottled)
    assert(static_cast<unsigned long>(bank->totalSize()) == bank->hdrSize() + padded_size);

  return bank->totalSize();
}
