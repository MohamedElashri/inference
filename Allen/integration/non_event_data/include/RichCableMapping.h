/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <vector>
#include "Constants.cuh"
#include "RichTel40CableMapping.cuh"

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
#include "RichFutureKernel/RichAlgBase.h"
#include "RichFutureUtils/RichDecodedData.h"
#include "RichUtils/RichException.h"
#include "RichUtils/RichHashMap.h"
#include "RichUtils/RichMap.h"
#include "RichUtils/RichSmartIDSorter.h"

// RICH DAQ
#include "RichFutureDAQ/RichPackedFrameSizes.h"
#include "RichFutureDAQ/RichTel40CableMapping.h"
#endif

namespace Allen::Conditions {
  struct RichCableMapping {
    inline static std::string const id = "RichCableMapping";
    inline static std::string const filename = "rich_tel40maps.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-rich-cable-mapping";
#else
    inline static std::string const DefaultLocation = "AllenConditions-rich-cable-mapping";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      ::Rich::Future::DAQ::Tel40CableMapping::addConditionDerivation(
        parent, ::Rich::Future::DAQ::Tel40CableMapping::DefaultConditionKey);

      return parent->addConditionDerivation(
        {::Rich::Future::DAQ::Tel40CableMapping::DefaultConditionKey},
        RichCableMapping::DefaultLocation,
        [](const ::Rich::Future::DAQ::Tel40CableMapping& det) { return Allen::Conditions::RichCableMapping {det}; });
    }

    RichCableMapping(const ::Rich::Future::DAQ::Tel40CableMapping& tel40Maps)
    {
      Allen::Rich::Decoding::Tel40CableMapping allenTel40Maps;

      allenTel40Maps.m_isInitialised = tel40Maps.isInitialised();
      allenTel40Maps.m_mappingVer = tel40Maps.version();

      if (tel40Maps.isInitialised()) {
        for (unsigned int i = 0; i < tel40Maps.tel40ModuleData().size(); ++i) {
          for (unsigned int j = 0; j < tel40Maps.tel40ModuleData()[i].size(); ++j) {
            for (unsigned int k = 0; k < tel40Maps.tel40ModuleData()[i][j].size(); ++k) {
              auto& allenData = allenTel40Maps.m_tel40ModuleData[i][j][k];
              const auto& data = tel40Maps.tel40ModuleData()[i][j][k];

              allenData.smartID = LHCb::RichSmartID::KeyType {data.smartID};
              allenData.moduleNum = data.moduleNum.data();
              allenData.sourceID = data.sourceID.data();
              allenData.connector = data.connector.data();
              allenData.pdmdbNum = data.pdmdbNum.data();
              allenData.linkNum = data.linkNum.data();
              allenData.isHType = data.isHType;
              allenData.isActive = data.isActive;
            }
          }
        }

        for (unsigned int i = 0; i < tel40Maps.tel40ConnData().size(); ++i) {
          for (unsigned int j = 0; j < tel40Maps.tel40ConnData()[i].size(); ++j) {
            for (unsigned int k = 0; k < tel40Maps.tel40ConnData()[i][j].size(); ++k) {
              const auto& link_data = tel40Maps.tel40ConnData()[i][j][k];
              allenTel40Maps.m_tel40ConnMeta[i][j][k] = Allen::Rich::Decoding::Tel40CableMapping::Tel40MetaData {
                static_cast<uint32_t>(link_data.nActiveLinks), link_data.hasInactiveLinks};
              for (unsigned int l = 0; l < link_data.size(); ++l) {
                auto& allenData = allenTel40Maps.m_tel40ConnData[i][j][k][l];
                const auto& data = link_data[l];

                allenData.smartID = LHCb::RichSmartID::KeyType {data.smartID};
                allenData.moduleNum = data.moduleNum.data();
                allenData.sourceID = data.sourceID.data();
                allenData.connector = data.connector.data();
                allenData.pdmdbNum = data.pdmdbNum.data();
                allenData.linkNum = data.linkNum.data();
                allenData.isHType = data.isHType;
                allenData.isActive = data.isActive;
              }
            }
          }
        }
      }

      DumpUtils::Writer output {};
      output.write(allenTel40Maps);
      m_data = output.buffer();
    }
#endif

    RichCableMapping(const std::vector<char>& data) { m_data = data; }

    void update_constants(Constants& constants) const
    {
      auto& dev_rich_cable_mapping = constants.dev_rich_cable_mapping;
      if (dev_rich_cable_mapping == nullptr) {
        Allen::malloc((void**) &constants.dev_rich_cable_mapping, sizeof(Allen::Rich::Decoding::Tel40CableMapping));
      }
      auto& host_rich_cable_mapping = constants.host_rich_cable_mapping;
      host_rich_cable_mapping = m_data;

      // patch metadata to hold valid mask, TODO: do this properly in the dumpers
      Allen::Rich::Decoding::Tel40CableMapping* richCableMapping =
        reinterpret_cast<Allen::Rich::Decoding::Tel40CableMapping*>(host_rich_cable_mapping.data());
      auto& metadata = richCableMapping->m_tel40ConnMeta;
      auto& sources = richCableMapping->m_tel40ConnData;
      for (unsigned i = 0; i < metadata.size(); ++i) {
        for (unsigned j = 0; j < metadata[i].size(); ++j) {
          for (unsigned k = 0; k < metadata[i][j].size(); ++k) {
            uint32_t mask = 0;
            for (unsigned l = 0; l < sources[i][j][k].size(); ++l) {
              mask |= (sources[i][j][k][l].isActive ? (1 << l) : 0);

              sources[i][j][k][l].smartID.setData(
                sources[i][j][k][l].moduleNum,
                Allen::Rich::Decoding::SmartID::ShiftPDMod,
                Allen::Rich::Decoding::SmartID::MaskPDMod,
                Allen::Rich::Decoding::SmartID::MaskPDIsSet);
            }
            metadata[i][j][k].validLinkMask = mask;
          }
        }
      }

      Allen::memcpy(
        dev_rich_cable_mapping,
        host_rich_cable_mapping.data(),
        sizeof(Allen::Rich::Decoding::Tel40CableMapping),
        Allen::memcpyHostToDevice);
    }

    auto const& data() const { return m_data; }

    std::vector<char> m_data;
  };
} // namespace Allen::Conditions
