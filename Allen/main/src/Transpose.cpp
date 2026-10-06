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
#include <Transpose.h>

namespace {
  std::unordered_set<LHCb::RawBank::BankType> dont_count = {
    LHCb::RawBank::BankType::DAQ,
    LHCb::RawBank::BankType::TAEHeader,
    LHCb::RawBank::BankType::HltDecReports,
    LHCb::RawBank::BankType::HltSelReports,
    LHCb::RawBank::BankType::HltRoutingBits,
    LHCb::RawBank::BankType::HltLumiSummary};
}

std::array<int, LHCb::NBankTypes> Allen::bank_ids()
{
  // Cache the mapping of LHCb::RawBank::BankType to Allen::BankType
  std::array<int, LHCb::NBankTypes> ids;
  for (auto bt : LHCb::RawBank::types()) {
    auto it = Allen::bank_mapping.find(bt);
    if (it != Allen::bank_mapping.end()) {
      for (auto allen_bt : it->second) {
        ids[(uint8_t) bt] = static_cast<int>(allen_bt);
      }
    }
    else {
      ids[(uint8_t) bt] = -1;
    }
  }
  return ids;
}

/**
 * @brief      Check if any of the soruce IDs have a non-zero value
 *             in the 5 most-significant bits
 *
 * @param      span with banks in MDF layout
 *
 * @return     true if any of the sourceIDs has a non-zero value in
 *             its 5 most-significant bits
 */
bool check_sourceIDs(std::span<char const> bank_data)
{

  auto const* bank = bank_data.data();

  // Loop over all the banks and check if any of the sourceIDs has
  // the most-significant bits set. In MC data they are not set.
  size_t n_banks = 0;
  size_t has_top5 = 0;
  while (bank < bank_data.data() + bank_data.size()) {

    const auto* b = reinterpret_cast<const LHCb::RawBank*>(bank);
    if (!dont_count.count(b->type())) {
      has_top5 += (SourceId_sys(static_cast<short>(b->sourceID())) != 0);
      ++n_banks;
    }

    // Increment overall bank pointer
    bank += b->totalSize();
  }

  // In real data or simulation with all the 5 most significant bits
  // correctly set, there is only a single bank with those set to 0:
  // ODIN.
  return (n_banks - has_top5) != 1;
}

/**
 * @brief      Get the (Allen) subdetector from the bank type
 *
 * @param      raw bank
 *
 * @return     Allen subdetector
 */
BankTypes sd_from_bank_type(LHCb::RawBank const* raw_bank)
{
  static auto const bank_ids = Allen::bank_ids();
  auto const bid = bank_ids[(uint8_t) raw_bank->type()];
  auto const bt = bid == -1 ? BankTypes::Unknown : static_cast<BankTypes>(bid);
  if (bt == BankTypes::Rich1) { // Some banks can only be distinguished by sourceID
    const auto nbt = sd_from_sourceID(raw_bank);
    // For very old MC samples, the subdetector type cannot always be resolved
    // from the raw bank information. In such cases we fall back to the old
    // behavior. This should be fine, as RICH is not expected to be used with
    // such old data (for example 2018 MC).
    return (nbt == BankTypes::Rich1 || nbt == BankTypes::Rich2) ? nbt : bt;
  }
  return bt;
}

/**
 * @brief      Get the (Allen) subdetector from the 5
 *             most-significant bits of a source ID
 *
 * @param      raw bank
 *
 * @return     Allen subdetector
 */
BankTypes sd_from_sourceID(LHCb::RawBank const* raw_bank)
{
  auto sd = SourceId_sys(raw_bank->sourceID());
  auto it = Allen::subdetectors.find(static_cast<SourceIdSys>(sd));
  auto source_type = (it == Allen::subdetectors.end()) ? BankTypes::Unknown : it->second;
  if (dont_count.count(raw_bank->type())) {
    return BankTypes::Unknown;
  }
  else {
    return source_type;
  }
}
