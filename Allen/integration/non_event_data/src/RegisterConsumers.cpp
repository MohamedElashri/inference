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
#include <RegisterConsumers.h>
#include <Common.h>
#include <Updater.h>
#include <InputReader.h>

#include "Beamline.h"
#include "EcalGeometry.h"
#include "MagneticField.h"
#include "MagneticFieldPolarity.h"
#include "MuonGeometry.h"
#include "MuonLookupTables.h"
#include "RichGeometry.h"
#include "RichCableMapping.h"
#include "RichPDMDBMapping.h"
#include "SciFiGeometry.h"
#include "UTBoards.h"
#include "UTGeometry.h"
#include "UTLookupTables.h"
#include "VPGeometry.h"

void load_geometry(
  Allen::NonEventData::IUpdater* updater,
  const std::unordered_set<BankTypes> requested_banks,
  std::map<std::string, std::string> const& options)
{
  auto get_option = [&options](const std::string& f) -> std::string {
    auto it = options.find(f);
    if (it == options.end()) {
      StrException {std::string {"Unknown option flag: "} + f};
    }
    else {
      return it->second;
    }
    return "";
  };

  auto folder_detector_configuration = get_option("g");
  GeometryReader reader {folder_detector_configuration};

  const auto consumers = std::make_tuple(
    std::make_tuple(std::type_identity<Allen::Conditions::UTBoards> {}, BankTypes::UT),
    std::make_tuple(std::type_identity<Allen::Conditions::UTLookupTables> {}, BankTypes::UT),
    std::make_tuple(std::type_identity<Allen::Conditions::UTGeometry> {}, BankTypes::UT),
    std::make_tuple(std::type_identity<Allen::Conditions::SciFiGeometry> {}, BankTypes::FT),
    std::make_tuple(std::type_identity<Allen::Conditions::Beamline> {}, BankTypes::VP),
    std::make_tuple(std::type_identity<Allen::Conditions::VPGeometry> {}, BankTypes::VP),
    std::make_tuple(std::type_identity<Allen::Conditions::EcalGeometry> {}, BankTypes::ECal),
    std::make_tuple(std::type_identity<Allen::Conditions::MuonGeometry> {}, BankTypes::MUON),
    std::make_tuple(std::type_identity<Allen::Conditions::MuonLookupTables> {}, BankTypes::MUON),
    std::make_tuple(std::type_identity<Allen::Conditions::RichPDMDBMapping> {}, BankTypes::Rich1),
    std::make_tuple(std::type_identity<Allen::Conditions::RichCableMapping> {}, BankTypes::Rich1),
    std::make_tuple(std::type_identity<Allen::Conditions::Rich1Geometry> {}, BankTypes::Rich1),
    std::make_tuple(std::type_identity<Allen::Conditions::RichPDMDBMapping> {}, BankTypes::Rich2),
    std::make_tuple(std::type_identity<Allen::Conditions::RichCableMapping> {}, BankTypes::Rich2),
    std::make_tuple(std::type_identity<Allen::Conditions::Rich2Geometry> {}, BankTypes::Rich2));

  for_each(consumers, [&](const auto& c) {
    const auto& [tid, bt] = c;
    using T = typename decltype(tid)::type;
    if (requested_banks.count(bt)) {
      updater->update_constants(T {reader.read_geometry(T::filename)});
    }
  });

  using UnconditionalConsumers = std::tuple<Allen::Conditions::MagneticField, Allen::Conditions::MagneticFieldPolarity>;

  [&]<typename... Ts>(std::tuple<Ts...>*) {
    (updater->update_constants(Ts {reader.read_geometry(Ts::filename)}), ...);
  }(static_cast<UnconditionalConsumers*>(nullptr));
}
