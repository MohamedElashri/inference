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
#pragma once

#include <unordered_map>
#include <Event/RawBankType.h>

#include "sourceid.h"
#include "BankTypes.h"

namespace Allen {
  const std::unordered_map<LHCb::Event::Enum::RawBank::BankType, std::unordered_set<BankTypes>> bank_mapping = {
    {LHCb::Event::Enum::RawBank::BankType::VP, {BankTypes::VP}},
    {LHCb::Event::Enum::RawBank::BankType::VPRetinaCluster, {BankTypes::VP}},
    {LHCb::Event::Enum::RawBank::BankType::UT, {BankTypes::UT}},
    {LHCb::Event::Enum::RawBank::BankType::UTError, {BankTypes::UT}},
    {LHCb::Event::Enum::RawBank::BankType::FTCluster, {BankTypes::FT}},
    {LHCb::Event::Enum::RawBank::BankType::Muon, {BankTypes::MUON}},
    {LHCb::Event::Enum::RawBank::BankType::MuonError, {BankTypes::MUON}},
    {LHCb::Event::Enum::RawBank::BankType::ODIN, {BankTypes::ODIN}},
    {LHCb::Event::Enum::RawBank::BankType::HcalPacked, {BankTypes::HCal}},
    {LHCb::Event::Enum::RawBank::BankType::EcalPacked, {BankTypes::ECal}},
    {LHCb::Event::Enum::RawBank::BankType::Calo, {BankTypes::ECal, BankTypes::HCal}},
    {LHCb::Event::Enum::RawBank::BankType::Rich, {BankTypes::Rich1, BankTypes::Rich2}},
    {LHCb::Event::Enum::RawBank::BankType::OTError, {BankTypes::MCVertices}}, // used for PV MC info
    {LHCb::Event::Enum::RawBank::BankType::OTRaw, {BankTypes::MCTracks}},
    {LHCb::Event::Enum::RawBank::BankType::OTError, {BankTypes::Gen}},  // used for beam crossing angles Gen info
    {LHCb::Event::Enum::RawBank::BankType::Plume, {BankTypes::Plume}}}; // used for track MC info

  const std::unordered_map<SourceIdSys, BankTypes> subdetectors = {
    {SourceIdSys::SourceIdSys_ODIN, BankTypes::ODIN},
    {SourceIdSys::SourceIdSys_VELO_A, BankTypes::VP},
    {SourceIdSys::SourceIdSys_VELO_C, BankTypes::VP},
    {SourceIdSys::SourceIdSys_UT_A, BankTypes::UT},
    {SourceIdSys::SourceIdSys_UT_C, BankTypes::UT},
    {SourceIdSys::SourceIdSys_SCIFI_A, BankTypes::FT},
    {SourceIdSys::SourceIdSys_SCIFI_C, BankTypes::FT},
    {SourceIdSys::SourceIdSys_RICH_1, BankTypes::Rich1},
    {SourceIdSys::SourceIdSys_RICH_2, BankTypes::Rich2},
    {SourceIdSys::SourceIdSys_MUON_A, BankTypes::MUON},
    {SourceIdSys::SourceIdSys_MUON_C, BankTypes::MUON},
    {SourceIdSys::SourceIdSys_HCAL, BankTypes::HCal},
    {SourceIdSys::SourceIdSys_ECAL, BankTypes::ECal},
    {SourceIdSys::SourceIdSys_PLUME, BankTypes::Plume}};

  const unsigned NSourceIdSys = to_integral(SourceIdSys::SourceIdSys_TDET) + 1;
} // namespace Allen

#ifndef ALLEN_STANDALONE
#include <cstring>
#include <fstream>
#include <string>
#include <Gaudi/Parsers/Factory.h>
#include <Event/RawBank.h>
#include <Event/RawEvent.h>
#include <GaudiKernel/GaudiException.h>

inline std::string toString(BankTypes e) { return bank_name(e); }
inline std::ostream& toStream(BankTypes e, std::ostream& os) { return os << std::quoted(toString(e), '\''); }
inline std::ostream& operator<<(std::ostream& s, BankTypes e) { return toStream(e, s); }

inline StatusCode parse(BankTypes& bt, const std::string& in)
{
  auto s = std::string_view {in};
  if (!s.empty() && s.front() == s.back() && (s.front() == '\'' || s.front() == '\"')) {
    s.remove_prefix(1);
    s.remove_suffix(1);
  }
  // Use BankSizes here because it has all he BankTypes as keys.
  auto i = std::find_if(BankSizes.begin(), BankSizes.end(), [s](auto e) { return s == bank_name(std::get<0>(e)); });
  if (i == BankSizes.end()) return StatusCode::FAILURE;
  bt = i->first;
  return StatusCode::SUCCESS;
}

namespace Gaudi::Parsers {
  inline StatusCode parse(std::set<BankTypes>& s, const std::string& in)
  {
    s.clear();
    using Gaudi::Parsers::parse;
    std::set<std::string> ss;
    return parse(ss, in).andThen([&]() -> StatusCode {
      try {
        std::transform(begin(ss), end(ss), std::inserter(s, begin(s)), [](const std::string& str) {
          BankTypes t {};
          parse(t, str).orThrow("Bad Parse", "");
          return t;
        });
        return StatusCode::SUCCESS;
      } catch (const GaudiException& e) {
        return e.code();
      }
    });
  }
} // namespace Gaudi::Parsers

#endif
