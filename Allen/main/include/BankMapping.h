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

  const std::unordered_map<SourceIdSys, BankTypes> subdetectors = {{SourceIdSys::SourceIdSys_ODIN, BankTypes::ODIN},
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
