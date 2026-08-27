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
#include <Dumpers/Utils.h>
#include "Constants.cuh"
#include "HostDeviceCondition.h"
#include "Rich.cuh"

#ifndef ALLEN_STANDALONE
// Rich Kernel
#include "RichFutureKernel/RichAlgBase.h"
#include "Kernel/RichDetectorType.h"

// Rich Utils
#include "RichFutureUtils/RichDecodedData.h"
#include "RichUtils/RichException.h"
#include "RichUtils/RichHashMap.h"
#include "RichUtils/RichMap.h"
#include "RichUtils/RichSmartIDSorter.h"

// LHCb
#include <RichDetectors/Rich1.h>
#include <RichDetectors/Rich2.h>
#endif

#ifndef ALLEN_STANDALONE
namespace {
  // Returns the side for a given mirror and Rich Detector
  template<typename MIRROR>
  inline Rich::Side side(
    const MIRROR* mirror, //
    const Rich::DetectorType rich) noexcept
  {
    return (
      Rich::Rich1 == rich              ? mirror->mirrorCentre().y() > 0.0 ? Rich::top : Rich::bottom :
      mirror->mirrorCentre().x() > 0.0 ? Rich::left :
                                         Rich::right);
  }
} // namespace
#endif

namespace Allen::Conditions {
  template<Allen::Rich::Detector::DetectorType RichID>
  struct RichGeometry : HostDeviceCondition<Allen::Rich::RichDetector<RichID>> {
    inline static std::string const id = RichID == Allen::Rich::Detector::Rich1 ? "Rich1Geometry" : "Rich2Geometry";
    inline static std::string const filename =
      RichID == Allen::Rich::Detector::Rich1 ? "rich_1_geometry.bin" : "rich_2_geometry.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = RichID == Allen::Rich::Detector::Rich1 ?
                                                        "/world:AllenConditions-rich1-geometry" :
                                                        "/world:AllenConditions-rich2-geometry";
#else
    inline static std::string const DefaultLocation =
      RichID == Allen::Rich::Detector::Rich1 ? "AllenConditions-rich1-geometry" : "AllenConditions-rich2-geometry";
#endif

#ifndef ALLEN_STANDALONE
    using RichT =
      std::conditional_t<RichID == Allen::Rich::Detector::Rich1, ::Rich::Detector::Rich1, ::Rich::Detector::Rich2>;

    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      RichT::addConditionDerivation(parent);

      return parent->addConditionDerivation(
        {RichT::DefaultConditionKey, ::Rich::Detector::Rich1::DefaultConditionKey},
        Allen::Conditions::RichGeometry<RichID>::DefaultLocation,
        [](const RichT& det, const ::Rich::Detector::Rich1& sellParams) {
          return Allen::Conditions::RichGeometry<RichID> {det, sellParams};
        });
    }

    RichGeometry(const RichT& rich, const ::Rich::Detector::Rich1& sellParams) :
      HostDeviceCondition<Allen::Rich::RichDetector<RichID>>([&rich, &sellParams]() {
        Allen::Rich::RichDetector<RichID> allenRich;
        for (auto& panel : rich.pdPanels()) {
          const auto panelSide = static_cast<Allen::Rich::Detector::Side>(panel.side());
          auto& allenPanel = allenRich.m_panels[panelSide];
          auto& panelPlane = panel.detectionPlaneSIMD();

          allenPanel.m_modNumOffset = panel.modNumOffset();
          allenPanel.m_panelID = {panel.rich(), panel.side(), 0};
          allenPanel.m_rich = static_cast<Allen::Rich::Detector::DetectorType>(panel.rich());
          allenPanel.m_side = panelSide;
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
                allenPD.m_localZcoord = ::Rich::Detector::scalar(pd->localZCoord());
                allenPD.m_numPixColFrac = ::Rich::Detector::scalar(pd->getNumPixColFrac());
                allenPD.m_numPixRowFrac = ::Rich::Detector::scalar(pd->getNumPixRowFrac());
                allenPD.m_effectivePixelXSize = ::Rich::Detector::scalar(pd->getEffectivePixelXSize());
                allenPD.m_effectivePixelYSize = ::Rich::Detector::scalar(pd->getEffectivePixelYSize());
                pd->localToGlobal().GetComponents(allenPD.m_locToGloM.begin(), allenPD.m_locToGloM.end());
                pd->centrePointPanel().GetCoordinates(allenPD.m_zeroInPanelFrame.begin());
              }
            }
          }
        }

        // nominal mirror geometry
        allenRich.m_sphMirrorRadius = (rich.sphMirrorRadius());

        // nominal CoCs for both sides
        for (const auto side : Allen::Rich::Detector::sides()) {
          const auto& coc = rich.nominalCentreOfCurvature(side);
          allenRich.m_nominalCentresOfCurvature[side].x = ::Rich::Detector::scalar(coc.X());
          allenRich.m_nominalCentresOfCurvature[side].y = ::Rich::Detector::scalar(coc.Y());
          allenRich.m_nominalCentresOfCurvature[side].z = ::Rich::Detector::scalar(coc.Z());

          // nominal planes
          const auto& plane = rich.nominalPlane(side);
          allenRich.m_nominalPlanes[side][0] = ::Rich::Detector::scalar(plane.A());
          allenRich.m_nominalPlanes[side][1] = ::Rich::Detector::scalar(plane.B());
          allenRich.m_nominalPlanes[side][2] = ::Rich::Detector::scalar(plane.C());
          allenRich.m_nominalPlanes[side][3] = ::Rich::Detector::scalar(plane.D());
        }

        // mirror segments
        {
          unsigned i_side0 = 0;
          unsigned i_side1 = 0;
          for (const auto& m : rich.primaryMirrors()) {
            const auto s = static_cast<Allen::Rich::Detector::Side>(
              side(m.get(), rich.rich())); // we need to get the side this way for detdesc compatibility.
            assert(
              (s == Allen::Rich::Detector::firstSide ? i_side0 : i_side1) <
              allenRich.m_primary_finder[s].m_mirrors.size());
            auto& allenMirror = allenRich.m_primary_finder[s]
                                  .m_mirrors[s == Allen::Rich::Detector::firstSide ? (i_side0++) : (i_side1++)];
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
            const auto s = static_cast<Allen::Rich::Detector::Side>(
              side(m.get(), rich.rich())); // we need to get the side this way for detdesc compatibility.
            assert(
              (s == Allen::Rich::Detector::firstSide ? i_side0 : i_side1) <
              allenRich.m_secondary_finder[s].m_mirrors.size());
            auto& allenMirror = allenRich.m_secondary_finder[s]
                                  .m_mirrors[s == Allen::Rich::Detector::firstSide ? (i_side0++) : (i_side1++)];
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

        return allenRich;
      }())
    {}
#endif

    using HostDeviceCondition<Allen::Rich::RichDetector<RichID>>::HostDeviceCondition;

    void update_constants(Constants& constants) const
    {
      if constexpr (RichID == Allen::Rich::Detector::Rich1) {
        constants.dev_rich_1_geometry = this->device();
      }
      else {
        constants.dev_rich_2_geometry = this->device();
      }

      if constexpr (RichID == Allen::Rich::Detector::Rich1) {
        constants.host_rich_1_geometry = this->host();
      }
      else {
        constants.host_rich_2_geometry = this->host();
      }
    }
  };

  using Rich1Geometry = RichGeometry<Allen::Rich::Detector::DetectorType::Rich1>;
  using Rich2Geometry = RichGeometry<Allen::Rich::Detector::DetectorType::Rich2>;
} // namespace Allen::Conditions
