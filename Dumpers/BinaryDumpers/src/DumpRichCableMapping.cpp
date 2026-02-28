/*****************************************************************************\
* (c) Copyright 2000-2019 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

// Gaudi Array properties ( must be first ...)
#include "Gaudi/Parsers/Factory.h"
#include "GaudiKernel/StdArrayAsProperty.h"

// Rich Kernel
#include "RichFutureKernel/RichAlgBase.h"

// Gaudi Functional
#include "LHCbAlgs/Transformer.h"

// Rich Utils
#include "RichFutureUtils/RichDecodedData.h"
#include "RichUtils/RichException.h"
#include "RichUtils/RichHashMap.h"
#include "RichUtils/RichMap.h"
#include "RichUtils/RichSmartIDSorter.h"

// RICH DAQ
#include "RichFutureDAQ/RichPackedFrameSizes.h"
#include "RichFutureDAQ/RichTel40CableMapping.h"

// Dumper
#include "Dumper.h"
#include <Dumpers/Utils.h>
#include "RichTel40CableMapping.cuh"

namespace {
  struct RichCableMapping {
    RichCableMapping() = default;
    RichCableMapping(std::vector<char>& data, const Rich::Future::DAQ::Tel40CableMapping& tel40Maps)
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
      data = output.buffer();
    }
  };
} // namespace

/**
 * @brief Dump cable mapping for the RICH detector.
 */
class DumpRichCableMapping final
  : public Allen::Dumpers::
      Dumper<void(RichCableMapping const&), LHCb::Algorithm::Traits::usesConditions<RichCableMapping>> {
public:
  DumpRichCableMapping(const std::string& name, ISvcLocator* svcLoc);

  void operator()(const RichCableMapping&) const override;

  StatusCode initialize() override;

private:
  std::vector<char> m_data;
};

DECLARE_COMPONENT(DumpRichCableMapping)

DumpRichCableMapping::DumpRichCableMapping(const std::string& name, ISvcLocator* svcLoc) :
  Dumper(name, svcLoc, {KeyValue {"RichCableMappingLocation", location(name, "cablemapping")}})
{}

StatusCode DumpRichCableMapping::initialize()
{
  return Dumper::initialize().andThen([&] {
    register_producer(Allen::NonEventData::RichCableMapping::id, "rich_tel40maps", m_data);

    Rich::Future::DAQ::Tel40CableMapping::addConditionDerivation(
      this, Rich::Future::DAQ::Tel40CableMapping::DefaultConditionKey);

    addConditionDerivation(
      {Rich::Future::DAQ::Tel40CableMapping::DefaultConditionKey},
      inputLocation<RichCableMapping>(),
      [&](const Rich::Future::DAQ::Tel40CableMapping& det) {
        auto cableMapping = RichCableMapping {m_data, det};
        dump();
        return cableMapping;
      });
  });
}

void DumpRichCableMapping::operator()(const RichCableMapping&) const {}
