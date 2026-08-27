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
#include "HostDeviceCondition.h"
#include "RichPDMDBDecodeMapping.cuh"

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
// Rich Utils
#include "RichFutureUtils/RichDecodedData.h"
#include "RichUtils/RichException.h"
#include "RichUtils/RichHashMap.h"
#include "RichUtils/RichMap.h"
#include "RichUtils/RichSmartIDSorter.h"

// RICH DAQ
#include "RichFutureDAQ/RichPDMDBDecodeMapping.h"
#include "RichFutureDAQ/RichPackedFrameSizes.h"
#endif

namespace Allen::Conditions {
  struct RichPDMDBMapping : HostDeviceCondition<Allen::Rich::Decoding::PDMDBDecodeMapping> {
    inline static std::string const id = "RichPDMDBMapping";
    inline static std::string const filename = "rich_pdmdbmaps.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-rich-PDMDB-mapping";
#else
    inline static std::string const DefaultLocation = "AllenConditions-rich-PDMDB-mapping";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      ::Rich::Future::DAQ::PDMDBDecodeMapping::addConditionDerivation(
        parent, ::Rich::Future::DAQ::PDMDBDecodeMapping::DefaultConditionKey);

      return parent->addConditionDerivation(
        {::Rich::Future::DAQ::PDMDBDecodeMapping::DefaultConditionKey},
        RichPDMDBMapping::DefaultLocation,
        [](const ::Rich::Future::DAQ::PDMDBDecodeMapping& det) { return Allen::Conditions::RichPDMDBMapping {det}; });
    }

    RichPDMDBMapping(const ::Rich::Future::DAQ::PDMDBDecodeMapping& mapping) :
      HostDeviceCondition<Allen::Rich::Decoding::PDMDBDecodeMapping>([&mapping]() {
        Allen::Rich::Decoding::PDMDBDecodeMapping allenPDMDBMapping;

        allenPDMDBMapping.m_pdmDataR =
          std::bit_cast<Allen::Rich::Decoding::PDMDBDecodeMapping::RTypeRichData>(mapping.pdmDataR());
        allenPDMDBMapping.m_pdmDataH =
          std::bit_cast<Allen::Rich::Decoding::PDMDBDecodeMapping::PDMData>(mapping.pdmDataH());
        allenPDMDBMapping.m_isInitialised = mapping.isInitialised();
        allenPDMDBMapping.m_mappingVer = mapping.version();

        // Patch anode (reverse column) and compute valid masks
        auto& pdmDataR = allenPDMDBMapping.m_pdmDataR;
        auto& pdmDataH = allenPDMDBMapping.m_pdmDataH;
        auto& pdmMaskR = allenPDMDBMapping.m_pdmMaskR;
        auto& pdmMaskH = allenPDMDBMapping.m_pdmMaskH;

        for (const auto rich : Allen::Rich::Detector::detectors()) {
          for (unsigned pdmdb = 0; pdmdb < Allen::Rich::Decoding::PDMDBDecodeMapping::PDMDBPerModule; pdmdb++) {
            for (unsigned link = 0; link < Allen::Rich::Decoding::PDMDBDecodeMapping::FramesPerPDMDB; link++) {
              for (unsigned b = 0; b < Allen::Rich::Decoding::PDMDBDecodeMapping::BytesPerFrameBitmask; b++) {
                pdmMaskR[rich][pdmdb][link][b] = 0;
                if (rich == Allen::Rich::Detector::Rich1) pdmMaskH[pdmdb][link][b] = 0;
              }
              for (unsigned bit = 0; bit < Allen::Rich::Decoding::PDMDBDecodeMapping::BitsPerFrame; bit++) {
                auto b = bit + (bit >= 39 ? 1 : 0);
                auto& dataR = pdmDataR[rich][pdmdb][link][bit];
                auto& dataH = pdmDataH[pdmdb][link][bit];

                unsigned validR = dataR.ec != -1 && dataR.pmtInEC != -1 && dataR.anode != -1;
                unsigned validH = dataH.ec != -1 && dataH.pmtInEC != -1 && dataH.anode != -1;

                pdmMaskR[rich][pdmdb][link][b / 8] |= validR << (b % 8);
                if (rich == Allen::Rich::Detector::Rich1) pdmMaskH[pdmdb][link][b / 8] |= validH << (b % 8);

                // Patch anode (reverse column)
                auto row = dataR.anode / 8;
                auto col = 7 - (dataR.anode % 8);
                dataR.anode = col + 8 * row;
                if (rich == Allen::Rich::Detector::Rich1) {
                  row = dataH.anode / 8;
                  col = 7 - (dataH.anode % 8);
                  dataH.anode = col + 8 * row;
                }
              }
            }
          }
        }
        return allenPDMDBMapping;
      }())
    {}
#endif

    using HostDeviceCondition<Allen::Rich::Decoding::PDMDBDecodeMapping>::HostDeviceCondition;

    void update_constants(Constants& constants) const
    {
      constants.dev_rich_pdmdb_mapping = device();
      constants.host_rich_pdmdb_mapping = host();
    }
  };
} // namespace Allen::Conditions
