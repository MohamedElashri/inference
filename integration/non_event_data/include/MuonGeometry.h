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
#include "BackendCommon.h"
#include "Logger.h"
#include "MuonDefinitions.cuh"
#include "MuonGeometry.cuh"
#include "MuonTables.cuh"

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
#include <MuonDet/DeMuonDetector.h>
#include <MuonDet/MuonNamespace.h>
#include <MuonDet/MuonStationCabling.h>
#include <MuonDet/MuonTell40Board.h>
#include <MuonDet/MuonNODEBoard.h>
#include <MuonDet/MuonTell40PCI.h>
#include "Detector/Muon/TileID.h"
#endif

namespace Allen::Conditions {
  struct MuonGeometry {
    inline static std::string const id = "MuonGeometry";
    inline static std::string const filename = "muon_geometry.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-muon-geometry";
#else
    inline static std::string const DefaultLocation = "AllenConditions-muon-geometry";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      return parent->addConditionDerivation(
        {DeMuonLocation::Default}, MuonGeometry::DefaultLocation, [](DeMuonDetector const& det) {
          return Allen::Conditions::MuonGeometry {det};
        });
    }

    MuonGeometry(const DeMuonDetector& det)
    {
      using namespace ::Muon::Constants;

      DumpUtils::Writer output {};
      const int nStations = det.stations();
      assert(nStations == 4);
      unsigned int geom_version = det.upgradeReadout() ? 3 : 2;
      output.write(geom_version);

      if (geom_version == 3) {
        auto daqHelper = det.getUpgradeDAQInfo();

        // containers initialization
        std::array<unsigned int, maxTell40Number> whichStationIsTell40 {};
        std::array<std::array<unsigned int, maxTell40PCINumber>, maxTell40Number> tell40PCINumberOfActiveLink {};
        std::array<std::array<std::array<unsigned int, maxNumberLinks>, maxTell40PCINumber>, maxTell40Number>
          mapRegionOfLink {};
        std::array<std::array<std::array<unsigned int, maxNumberLinks>, maxTell40PCINumber>, maxTell40Number>
          mapQuarterOfLink {};
        std::array<
          std::array<std::array<unsigned int, maxNumberLinks * ODEFrameSize>, maxTell40PCINumber>,
          maxTell40Number>
          mapTileInTell40 {};

        for (int station = 0; station < nStations; station++) {
          for (unsigned int iTell = 0; iTell < daqHelper->getNumberOfTell40Boards(station); iTell++) {
            whichStationIsTell40[iTell] = daqHelper->whichstationTell40(iTell);
          }
        }

        for (unsigned int itell = 0; itell < maxTell40Number; itell++) {
          for (unsigned int ipci = 0; ipci < maxTell40PCINumber; ipci++) {
            unsigned int active_link_per_PCI = 0;
            for (unsigned int ilink = 0; ilink < maxNumberLinks; ilink++) {
              unsigned int node = daqHelper->getODENumberNoHole(itell, ipci, ilink);
              unsigned int frame = daqHelper->getODEFrameNumberNoHole(itell, ipci, ilink);
              if (node > 0) {
                active_link_per_PCI++;
                tell40PCINumberOfActiveLink[itell][ipci] = active_link_per_PCI;

                for (unsigned int ich = 0; ich < ODEFrameSize; ich++) {
                  auto tileID = daqHelper->getTileIDInNODE(node - 1, frame * ODEFrameSize + ich);
                  if (tileID.isValid()) {
                    mapRegionOfLink[itell][ipci][ilink] = tileID.region();
                    mapQuarterOfLink[itell][ipci][ilink] = tileID.quarter();
                  } // valid tileID
                  mapTileInTell40[itell][ipci][(ilink) *ODEFrameSize + ich] = int(tileID);
                } // loop on channels
              }   // valid node
            }     // loop on links
          }       // loop on pci boards
        }         // loop on tell40s

        output.write(
          whichStationIsTell40, tell40PCINumberOfActiveLink, mapRegionOfLink, mapQuarterOfLink, mapTileInTell40);
      }
      else {
        auto daqHelper = det.getDAQInfo();

        std::vector<unsigned int> nTiles(daqHelper->TotTellNumber(), 0);
        for (auto tell1 = 0u; tell1 < daqHelper->TotTellNumber(); ++tell1) {
          nTiles[tell1] = daqHelper->getADDInTell1(tell1).size();
        }

        output.write(nTiles.size());
        for (auto tell1 = 0u; tell1 < daqHelper->TotTellNumber(); ++tell1) {
          auto const& tiles = daqHelper->getADDInTell1(tell1);
          output.write(tiles.size());
          for (auto const& tile : tiles) {
            output.write(static_cast<unsigned int>(tile));
          }
        }
      } // Run2 cabling
      m_data = output.buffer();
    }
#endif

    MuonGeometry(const std::vector<char>& data) { m_data = data; }

    void update_constants(Constants& constants) const
    {
      auto& dev_geometry_raw = constants.dev_muon_geometry_raw;
      auto& host_geometry_raw = constants.host_muon_geometry_raw;
      auto& dev_muon_geometry = constants.dev_muon_geometry;

      const char* raw_input = m_data.data();
      int version;
      std::copy_n((uint*) raw_input, 1, &version);
      raw_input += sizeof(uint);

      if (version == 2) {

        size_t nTilesSize;
        std::copy_n((size_t*) raw_input, 1, &nTilesSize);
        assert(nTilesSize == Muon::MuonGeometry::m_tiles_size);
        size_t sizes[Muon::MuonGeometry::m_tiles_size];
        unsigned int* tiles[Muon::MuonGeometry::m_tiles_size];
        size_t tilesOffset[Muon::MuonGeometry::m_tiles_size];
        raw_input += sizeof(size_t);
        for (size_t i = 0; i < nTilesSize; i++) {
          size_t tilesSize;
          std::copy_n((size_t*) raw_input, 1, &tilesSize);
          sizes[i] = tilesSize;

          raw_input += sizeof(size_t);
          tilesOffset[i] = ((unsigned*) raw_input) - ((unsigned*) m_data.data());
          raw_input += sizeof(unsigned) * tilesSize;
        }

        if (!dev_muon_geometry) {
          Allen::malloc((void**) &dev_geometry_raw, m_data.size());
          Allen::malloc((void**) &dev_muon_geometry, sizeof(Muon::MuonGeometry));
        }
        else if (host_geometry_raw.size() != m_data.size()) {
          throw StrException {
            std::string {"[muon geom] sizes don't match: "} + std::to_string(host_geometry_raw.size()) + " " +
            std::to_string(m_data.size())};
        }
        host_geometry_raw = m_data;
        Allen::memcpy(dev_geometry_raw, host_geometry_raw.data(), host_geometry_raw.size(), Allen::memcpyHostToDevice);

        for (size_t i = 0; i < nTilesSize; i++) {
          tiles[i] = ((unsigned*) dev_geometry_raw) + tilesOffset[i];
        }

        Muon::MuonGeometry host_muon_geometry {sizes, tiles};
        host_muon_geometry.setVersion(2);

        Allen::memcpy(dev_muon_geometry, &host_muon_geometry, sizeof(Muon::MuonGeometry), Allen::memcpyHostToDevice);
      }
      else if (version == 3) {

        unsigned int stationsTell40[Muon::Constants::maxTell40Number];
        unsigned int activeLink[Muon::Constants::maxTell40Number][Muon::Constants::maxTell40PCINumber];
        unsigned int regionOfLink[Muon::Constants::maxTell40Number][Muon::Constants::maxTell40PCINumber]
                                 [Muon::Constants::maxNumberLinks];
        unsigned int quarterOfLink[Muon::Constants::maxTell40Number][Muon::Constants::maxTell40PCINumber]
                                  [Muon::Constants::maxNumberLinks];
        unsigned int tileinTell40[Muon::Constants::maxTell40Number][Muon::Constants::maxTell40PCINumber]
                                 [Muon::Constants::maxNumberLinks * Muon::Constants::ODEFrameSize];

        for (size_t itell = 0; itell < Muon::Constants::maxTell40Number; itell++) {
          unsigned int station;
          std::copy_n((unsigned int*) raw_input, 1, &station);
          raw_input += sizeof(unsigned int);
          stationsTell40[itell] = station;
        }

        for (unsigned int itell = 0; itell < Muon::Constants::maxTell40Number; itell++) {
          for (unsigned int ipci = 0; ipci < Muon::Constants::maxTell40PCINumber; ipci++) {
            unsigned int activelink;
            std::copy_n((unsigned int*) raw_input, 1, &activelink);
            raw_input += sizeof(unsigned int);
            activeLink[itell][ipci] = activelink;
          }
        }

        for (unsigned int itell = 0; itell < Muon::Constants::maxTell40Number; itell++) {
          for (unsigned int ipci = 0; ipci < Muon::Constants::maxTell40PCINumber; ipci++) {
            for (unsigned int ilink = 0; ilink < Muon::Constants::maxNumberLinks; ilink++) {
              unsigned int linkregion;
              std::copy_n((unsigned int*) raw_input, 1, &linkregion);
              raw_input += sizeof(unsigned int);
              regionOfLink[itell][ipci][ilink] = linkregion;
            }
          }
        }

        for (unsigned int itell = 0; itell < Muon::Constants::maxTell40Number; itell++) {
          for (unsigned int ipci = 0; ipci < Muon::Constants::maxTell40PCINumber; ipci++) {
            for (unsigned int ilink = 0; ilink < Muon::Constants::maxNumberLinks; ilink++) {
              unsigned int linkquarter;
              std::copy_n((unsigned int*) raw_input, 1, &linkquarter);
              raw_input += sizeof(unsigned int);
              quarterOfLink[itell][ipci][ilink] = linkquarter;
            }
          }
        }

        for (unsigned int itell = 0; itell < Muon::Constants::maxTell40Number; itell++) {
          for (unsigned int ipci = 0; ipci < Muon::Constants::maxTell40PCINumber; ipci++) {
            for (unsigned int ilink = 0; ilink < Muon::Constants::maxNumberLinks; ilink++) {
              for (unsigned int ich = 0; ich < Muon::Constants::ODEFrameSize; ich++) {
                unsigned int tileID;
                std::copy_n((unsigned int*) raw_input, 1, &tileID);
                raw_input += sizeof(unsigned int);
                tileinTell40[itell][ipci][ilink * Muon::Constants::ODEFrameSize + ich] = tileID;
              }
            }
          }
        }

        if (!dev_muon_geometry) {
          Allen::malloc((void**) &dev_geometry_raw, m_data.size());
          Allen::malloc((void**) &dev_muon_geometry, sizeof(Muon::MuonGeometry));
        }
        else if (host_geometry_raw.size() != m_data.size()) {
          throw StrException {
            std::string {"[muon geom] sizes don't match: "} + std::to_string(host_geometry_raw.size()) + " " +
            std::to_string(m_data.size())};
        }
        host_geometry_raw = m_data;
        Allen::memcpy(dev_geometry_raw, host_geometry_raw.data(), host_geometry_raw.size(), Allen::memcpyHostToDevice);

        Muon::MuonGeometry host_muon_geometry {stationsTell40, activeLink, regionOfLink, quarterOfLink, tileinTell40};
        host_muon_geometry.setVersion(3);
        Allen::memcpy(dev_muon_geometry, &host_muon_geometry, sizeof(Muon::MuonGeometry), Allen::memcpyHostToDevice);
      }
      else {
        error_cout << "unrecognized muon geometry version" << std::endl;
      }
    }

    auto const& data() const { return m_data; }
    std::vector<char> m_data;
  };
} // namespace Allen::Conditions
