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
#include "GaudiKernel/PhysicalConstants.h"
#include "LHCbAlgs/Transformer.h"

// LHCb
#include <RichDetectors/Rich1.h>
#include <RichDetectors/Rich2.h>
#include "RichFutureRecEvent/RichRecPhotonYields.h"

// Dumper
#include "Dumper.h"
#include <Dumpers/Utils.h>
#include "Rich.cuh"

namespace {
  // Returns the side for a given mirror and Rich Detector
  template<typename MIRROR>
  inline Rich::Side side(
    const MIRROR* mirror, //
    const Rich::DetectorType rich) noexcept
  {
    return (
      Rich::Rich1 == rich ? mirror->mirrorCentre().y() > 0.0 ? Rich::top : Rich::bottom :
                            mirror->mirrorCentre().x() > 0.0 ? Rich::left : Rich::right);
  }

  template<typename RichT, Allen::Rich::Detector::Type RichID>
  struct RichGeometry_t {
    RichGeometry_t() = default;
    RichGeometry_t(std::vector<char>& data, const RichT& rich, const Rich::Detector::Rich1& sellParams)
    {
      Allen::Rich::RichDetector<RichID> allenRich;
      for (size_t k = 0; k < rich.pdPanels().size(); k++) {
        auto& panel = rich.pdPanels()[k];
        auto& allenPanel = allenRich.m_panels[k];
        auto& panelPlane = panel.detectionPlaneSIMD();

        allenPanel.m_modNumOffset = panel.modNumOffset();
        allenPanel.m_panelID = {panel.rich(), panel.side(), 0};
        allenPanel.m_rich = (Allen::Rich::Detector::Type) panel.rich();
        allenPanel.m_side = (Allen::Rich::Detector::Side) panel.side();
        panel.globalToPDPanel().GetComponents(allenPanel.m_gloToPDPanelM.begin());

        allenPanel.m_detectionPlane[0] = panelPlane.A()[0];
        allenPanel.m_detectionPlane[1] = panelPlane.B()[0];
        allenPanel.m_detectionPlane[2] = panelPlane.C()[0];
        allenPanel.m_detectionPlane[3] = panelPlane.D()[0];

        for (size_t i = 0; i < allenPanel.m_PDs.size(); i++) {
          for (size_t j = 0; j < allenPanel.m_PDs[i].size(); j++) {
            allenPanel.m_PDs[i][j].setIsNull(true);
          }
          for (size_t j = 0; j < panel.pdModules()[i].size(); j++) {
            if (panel.pdModules()[i][j] != nullptr) {
              auto& pd = panel.pdModules()[i][j];

              const auto ec = pd->pdSmartID().elementaryCell();
              const auto pdInEC = pd->pdSmartID().pdNumInEC();

              // dispatch EC / PDInEC:
              const auto tj = ec * Allen::Rich::Decoding::SmartID::MaxPDsPerEC + pdInEC;
              auto& allenPD = allenPanel.m_PDs[i][tj];

              allenPD.setIsNull(false);
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
          }
        }
      }

      // nominal mirror geometry
      allenRich.m_sphMirrorRadius = (rich.sphMirrorRadius());

      // nominal CoCs for both sides
      for (int side = 0; side < 2; ++side) {
        const auto& coc = rich.nominalCentreOfCurvature((Allen::Rich::Detector::Side)(side));
        allenRich.m_nominalCentresOfCurvature[side].x = Rich::Detector::scalar(coc.X());
        allenRich.m_nominalCentresOfCurvature[side].y = Rich::Detector::scalar(coc.Y());
        allenRich.m_nominalCentresOfCurvature[side].z = Rich::Detector::scalar(coc.Z());

        // nominal planes
        const auto& plane = rich.nominalPlane((Allen::Rich::Detector::Side)(side));
        allenRich.m_nominalPlanes[side][0] = Rich::Detector::scalar(plane.A());
        allenRich.m_nominalPlanes[side][1] = Rich::Detector::scalar(plane.B());
        allenRich.m_nominalPlanes[side][2] = Rich::Detector::scalar(plane.C());
        allenRich.m_nominalPlanes[side][3] = Rich::Detector::scalar(plane.D());
      }

      // mirror segments
      {
        unsigned i_side0 = 0;
        unsigned i_side1 = 0;
        for (const auto& m : rich.primaryMirrors()) {
          auto s = side(m.get(), rich.rich()); // we need to get the side this way for detdesc compatibility.
          assert((s == 0 ? i_side0 : i_side1) < allenRich.m_primary_finder[s].m_mirrors.size());
          auto& allenMirror = allenRich.m_primary_finder[s].m_mirrors[s == 0 ? (i_side0++) : (i_side1++)];
          allenMirror.centreOfCurvature.x = m->centreOfCurvature().x();
          allenMirror.centreOfCurvature.y = m->centreOfCurvature().y();
          allenMirror.centreOfCurvature.z = m->centreOfCurvature().z();
          allenMirror.radiusOfCurvature = m->radius();
          allenMirror.mirrorCentre.x = m->mirrorCentre().x();
          allenMirror.mirrorCentre.y = m->mirrorCentre().y();
        }
        allenRich.m_primary_finder[0].init();
        allenRich.m_primary_finder[1].init();
      }

      {
        unsigned i_side0 = 0;
        unsigned i_side1 = 0;
        for (const auto& m : rich.secondaryMirrors()) {
          auto s = side(m.get(), rich.rich()); // we need to get the side this way for detdesc compatibility.
          assert((s == 0 ? i_side0 : i_side1) < allenRich.m_secondary_finder[s].m_mirrors.size());
          auto& allenMirror = allenRich.m_secondary_finder[s].m_mirrors[s == 0 ? (i_side0++) : (i_side1++)];
          allenMirror.centreOfCurvature.x = m->centreOfCurvature().x();
          allenMirror.centreOfCurvature.y = m->centreOfCurvature().y();
          allenMirror.centreOfCurvature.z = m->centreOfCurvature().z();
          allenMirror.radiusOfCurvature = m->radius();
          allenMirror.mirrorCentre.x = m->mirrorCentre().x();
          allenMirror.mirrorCentre.y = m->mirrorCentre().y();
        }
        allenRich.m_secondary_finder[0].init();
        allenRich.m_secondary_finder[1].init();
      }

      // Radiator
      const Gaudi::XYZPoint position {0, 0, 0};
      const Gaudi::XYZVector direction {0, 0, 1};
      Gaudi::XYZPoint entryPoint;
      Gaudi::XYZPoint exitPoint;

      std::ignore = rich.radiator().get()->intersectionPoints(position, direction, entryPoint, exitPoint);

      allenRich.m_radZEntry = entryPoint.z();
      allenRich.m_radZExit = exitPoint.z();

      // Beampipe
      const auto& bp = rich.beampipe();
      double zmin = bp.zmin();
      double zmax = bp.zmax();
      double rmin = bp.rmin();
      double rmax = bp.rmax();

      allenRich.m_zmin = zmin;
      allenRich.m_zmax = zmax;
      allenRich.m_r2min = rmin * rmin;
      allenRich.m_r2max = rmax * rmax;
      allenRich.m_a = (rmin - rmax) / (zmin - zmax);
      allenRich.m_b = (rmin) - (static_cast<double>(allenRich.m_a) * zmin);

      // Sellmeier parameters, always read from Rich1 when using DetDesc
      allenRich.m_selF1 = RichID == Allen::Rich::Detector::Rich1 ?
                            sellParams.template param<double>("SellC4F10F1Param") :
                            sellParams.template param<double>("SellCF4F1Param");
      allenRich.m_selF2 = RichID == Allen::Rich::Detector::Rich1 ?
                            sellParams.template param<double>("SellC4F10F2Param") :
                            sellParams.template param<double>("SellCF4F2Param");
      allenRich.m_selE1 = RichID == Allen::Rich::Detector::Rich1 ?
                            sellParams.template param<double>("SellC4F10E1Param") :
                            sellParams.template param<double>("SellCF4E1Param");
      allenRich.m_selE2 = RichID == Allen::Rich::Detector::Rich1 ?
                            sellParams.template param<double>("SellC4F10E2Param") :
                            sellParams.template param<double>("SellCF4E2Param");
      allenRich.m_molW = RichID == Allen::Rich::Detector::Rich1 ?
                           sellParams.template param<double>("GasMolWeightC4F10Param") :
                           sellParams.template param<double>("GasMolWeightCF4Param");
      allenRich.m_rho = RichID == Allen::Rich::Detector::Rich1 ?
                          sellParams.template param<double>("RhoEffectiveSellC4F10Param") :
                          sellParams.template param<double>("RhoEffectiveSellCF4Param");
      allenRich.m_selLorGasFac = sellParams.template param<double>("SellLorGasFacParam");

      // Average spectra efficiency
#ifdef USE_DD4HEP
      // PMT Eff.
      const auto pdEff = rich.template param<double>("PMTSiHitDetectionEff");
#else
      const auto pdEff = sellParams.template param<double>("HPDQuartzWindowEff") *
                         sellParams.template param<double>("PMTPedestalDigiEff");
#endif

      // Quartz window params
      const double qWinZSize = RichID == Allen::Rich::Detector::Rich1 ?
                                 rich.template param<double>("Rich1GasQuartzWindowThickness") :
                                 rich.template param<double>("Rich2GasQuartzWindowThickness");

      const double min_photon_E = static_cast<double>(Allen::Rich::MinPhotonEnergy);
      const double max_photon_E = static_cast<double>(Allen::Rich::MaxPhotonEnergy);
      const unsigned nbins = allenRich.m_spectraEffs.size();
      const double binw = (max_photon_E - min_photon_E) / nbins;
      for (unsigned iEnBin = 0; iEnBin < nbins; iEnBin++) {
        const auto binEn = min_photon_E + ((double) iEnBin + 0.5) * binw;
        double eff = pdEff;
        // bin energy ( in eV )
        const auto energy = binEn * Gaudi::Units::eV;
        // Get weighted average PD Q.E. ( scale from % to fraction )
        eff *= 0.01 * (*(rich.nominalPDQuantumEff()))[energy];
        // primary mirror reflectivity
        eff *= (*(rich.nominalSphMirrorRefl()))[energy];
        // secondary mirror reflectivity
        eff *= (*(rich.nominalSecMirrorRefl()))[energy];
        // The Quartz window efficiency
        eff *= std::exp(-qWinZSize / (*(rich.gasWinAbsLength()))[energy]);
        allenRich.m_spectraEffs[iEnBin] = eff;
        allenRich.m_refIndexE[iEnBin] = rich.radiator().refractiveIndex(binEn);
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
        Rich1Geometry_t Rich1Geo {m_Rich1Data, det, det}; // unused extra det required for rich2 in detdesc
        dump();
        return Rich1Geo;
      });

    register_producer(Allen::NonEventData::Rich2Geometry::id, "rich_2_geometry", m_Rich2Data);
    Rich::Detector::Rich2::addConditionDerivation(this);
    addConditionDerivation(
      {
        Rich::Detector::Rich1::DefaultConditionKey,
        Rich::Detector::Rich2::DefaultConditionKey,
      },
      inputLocation<Rich2Geometry_t>(),
      [&](const Rich::Detector::Rich1& det1, const Rich::Detector::Rich2& det2) {
        Rich2Geometry_t Rich2Geo {m_Rich2Data, det2, det1}; // det1 for sellmeier params in detdesc
        dump();
        return Rich2Geo;
      });
  });
}

void DumpRichGeometry::operator()(const Rich1Geometry_t&, const Rich2Geometry_t&) const {}
