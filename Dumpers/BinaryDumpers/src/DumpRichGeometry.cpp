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
#include "GaudiKernel/StdArrayAsProperty.h"

// Rich Kernel
#include "RichFutureKernel/RichAlgBase.h"
#include "Kernel/RichDetectorType.h"

// Gaudi Functional
#include "LHCbAlgs/Transformer.h"

// Rich Utils
#include "RichFutureUtils/RichDecodedData.h"
#include "RichUtils/RichException.h"
#include "RichUtils/RichHashMap.h"
#include "RichUtils/RichMap.h"
#include "RichUtils/RichSmartIDSorter.h"

// LHCb
#include <RichDetectors/Rich1.h>
#include <RichDetectors/Rich2.h>

// Dumper
#include "Dumper.h"
#include <Dumpers/Utils.h>
#include "Rich.cuh"

namespace {
  template<typename RichT, Allen::Rich::Detector::Type RichID>
  struct RichGeometry_t {
    RichGeometry_t() = default;
    RichGeometry_t(std::vector<char>& data, const RichT& rich)
    {
      Allen::Rich::RichDetector allenRich;
      allenRich.m_type = RichID;
      for (size_t k = 0; k < rich.pdPanels().size(); k++) {
        auto& panel = rich.pdPanels()[k];
        auto& allenPanel = allenRich.m_panels[k];

        allenPanel.m_modNumOffset = panel.modNumOffset();
        allenPanel.m_panelID = {panel.rich(), panel.side(), 0};
        allenPanel.m_rich = (Allen::Rich::Detector::Type) panel.rich();
        allenPanel.m_side = (Allen::Rich::Detector::Side) panel.side();
        panel.globalToPDPanel().GetComponents(allenPanel.m_gloToPDPanelM.begin());
        for (size_t i = 0; i < panel.pdModules().size(); i++) {
          for (size_t j = 0; j < panel.pdModules()[i].size(); j++) {
            if (panel.pdModules()[i][j] != nullptr) {
              auto& allenPD = allenPanel.m_PDs[i][j];
              auto& pd = panel.pdModules()[i][j];

              allenPD.m_pdSmartID = pd->pdSmartID();
              allenPD.m_effPixelArea = pd->effectivePixelArea();
              allenPD.m_numPixels = pd->effectiveNumActivePixels();
              allenPD.m_isHType = pd->isHType();
              allenPD.m_localZcoord = Rich::Detector::scalar(pd->localZCoord());
              allenPD.m_numPixColFrac = Rich::Detector::scalar(pd->getNumPixColFrac());
              allenPD.m_numPixRowFrac = Rich::Detector::scalar(pd->getNumPixRowFrac());
              allenPD.m_effectivePixelXSize = Rich::Detector::scalar(pd->getEffectivePixelXSize());
              allenPD.m_effectivePixelYSize = Rich::Detector::scalar(pd->getEffectivePixelYSize());
              pd->localToGlobal().GetComponents(allenPD.m_locToGloM.begin(), allenPD.m_locToGloM.end());
              pd->centrePointPanel().GetCoordinates(allenPD.m_zeroInPanelFrame.begin());
            }
            else {
              // Create null PD where HLT2 has nullptr
              allenPanel.m_PDs[i][j].setIsNull(true);
            }
          }
        }
      }

      DumpUtils::Writer output {};
      output.write(allenRich);
      data = output.buffer();
    }
  };

  using Rich1Geometry_t = RichGeometry_t<Rich::Detector::Rich1, Allen::Rich::Detector::Type::Rich1>;
  using Rich2Geometry_t = RichGeometry_t<Rich::Detector::Rich2, Allen::Rich::Detector::Type::Rich2>;
} // namespace

/**
 * @brief Dump RICH detector object information.
 */
class DumpRichGeometry final : public Allen::Dumpers::Dumper<
                                 void(Rich1Geometry_t const&, Rich2Geometry_t const&),
                                 LHCb::DetDesc::usesConditions<Rich1Geometry_t, Rich2Geometry_t>> {
public:
  DumpRichGeometry(const std::string& name, ISvcLocator* svcLoc);
  void operator()(const Rich1Geometry_t& Rich1Geo, const Rich2Geometry_t& Rich2Geo) const override;
  StatusCode initialize() override;

private:
  std::vector<char> m_Rich1Data;
  std::vector<char> m_Rich2Data;
};

DECLARE_COMPONENT(DumpRichGeometry)

DumpRichGeometry::DumpRichGeometry(const std::string& name, ISvcLocator* svcLoc) :
  Dumper(
    name,
    svcLoc,
    {KeyValue {"Rich1GeometryLocation", location(name, "rich1geometry")},
     KeyValue {"Rich2GeometryLocation", location(name, "rich2Geometry")}})
{}

StatusCode DumpRichGeometry::initialize()
{
  return Dumper::initialize().andThen([&, this] {
    register_producer(Allen::NonEventData::Rich1Geometry::id, "rich_1_geometry", m_Rich1Data);
    Rich::Detector::Rich1::addConditionDerivation(this);
    addConditionDerivation(
      {Rich::Detector::Rich1::DefaultConditionKey},
      inputLocation<Rich1Geometry_t>(),
      [&](const Rich::Detector::Rich1& det) {
        Rich1Geometry_t Rich1Geo {m_Rich1Data, det};
        dump();
        return Rich1Geo;
      });

    register_producer(Allen::NonEventData::Rich2Geometry::id, "rich_2_geometry", m_Rich2Data);
    Rich::Detector::Rich2::addConditionDerivation(this);
    addConditionDerivation(
      {Rich::Detector::Rich2::DefaultConditionKey},
      inputLocation<Rich2Geometry_t>(),
      [&](const Rich::Detector::Rich2& det) {
        Rich2Geometry_t Rich2Geo {m_Rich2Data, det};
        dump();
        return Rich2Geo;
      });
  });
}

void DumpRichGeometry::operator()(const Rich1Geometry_t&, const Rich2Geometry_t&) const {}
