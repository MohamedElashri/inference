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

#include <array>
#include <cstring>
#include <tuple>
#include <vector>
#include "Constants.cuh"
#include "BackendCommon.h"
#include "Logger.h"
#include "MuonDefinitions.cuh"
#include "MuonGeometry.cuh"
#include "MuonTables.cuh"

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
#include <Detector/Muon/Layout.h>
#include <Detector/Muon/TileID.h>
#include <MuonDet/DeMuonDetector.h>
#include <MuonDet/MuonNamespace.h>

#include <range/v3/algorithm/any_of.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/core.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/cartesian_product.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>

#if RANGE_V3_VERSION < 900
namespace ranges::views {
  using namespace ranges::view;
}
#endif

#include <boost/format.hpp>
#include <boost/numeric/conversion/cast.hpp>
#endif

namespace Allen::Conditions {
  struct MuonLookupTables {
    inline static std::string const id = "MuonLookupTables";
    inline static std::string const filename = "muon_tables.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-muon-lookup-tables";
#else
    inline static std::string const DefaultLocation = "AllenConditions-muon-lookup-tables";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      return parent->addConditionDerivation(
        {DeMuonLocation::Default}, MuonLookupTables::DefaultLocation, [](DeMuonDetector const& det) {
          return Allen::Conditions::MuonLookupTables {det};
        });
    }

    MuonLookupTables(const DeMuonDetector& det)
    {
      using namespace ranges;

      DumpUtils::Writer output {};
      const int nStations = det.stations();
      assert(nStations == 4);
      const int nRegions = det.regions() / nStations;
      assert(nRegions == 4);

      std::array<int, 16> padGridX {48, 48, 48, 48, 48, 48, 48, 48, 12, 12, 12, 12, 12, 12, 12, 12};
      std::array<int, 16> stripXGridX {48, 48, 48, 48, 48, 48, 48, 48, 12, 12, 12, 12, 12, 12, 12, 12};
      std::vector<float> padSizeX {}, stripXSizeX {}, stripYSizeX {}, padSizeY {}, stripXSizeY {}, stripYSizeY {};
      std::array<unsigned int, 16> padOffset {}, stripXOffset {}, stripYOffset {};
      std::array<int, 16> padGridY {}, stripXGridY {}, stripYGridX {}, stripYGridY {};
      std::array<std::vector<std::array<float, 3>>, 4> padTable {}, stripXTable {}, stripYTable {};

      auto nChannels = [](size_t s, const auto& gridX, const auto& gridY) {
        return [s, &gridX, &gridY](auto tot, const auto r) { return tot + gridX[4 * s + r] * gridY[4 * s + r]; };
      };

      unsigned int geom_version = det.upgradeReadout() ? 3 : 2;
      output.write(geom_version);

      if (geom_version == 3) {
        padGridY = {8, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
        stripXGridY = {1, 2, 8, 8, 1, 2, 2, 2, 8, 2, 2, 2, 8, 2, 2, 8};
        stripYGridX = {8, 4, 48, 48, 8, 4, 2, 2, 12, 4, 2, 2, 12, 4, 2, 12};
        stripYGridY = {8, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
      }
      else {
        padGridY = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
        stripXGridY = {1, 2, 2, 2, 1, 2, 2, 2, 8, 2, 2, 2, 8, 2, 2, 2};
        stripYGridX = {8, 4, 2, 2, 8, 4, 2, 2, 12, 4, 2, 2, 12, 4, 2, 2};
        stripYGridY = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};
      }

      for (int s = 0; s < nStations; ++s) {
        const auto pad_resize = geom_version == 3 ?
                                  12 * accumulate(views::ints(0, 4), 0, nChannels(s, padGridX, padGridY)) :
                                  Muon::Constants::ODEFrameSize * padGridX[s] * padGridY[s];
        padTable[s].resize(pad_resize);
        stripXTable[s].resize(12 * accumulate(views::ints(0, 4), 0, nChannels(s, stripXGridX, stripXGridY)));
        stripYTable[s].resize(12 * accumulate(views::ints(0, 4), 0, nChannels(s, stripYGridX, stripYGridY)));
      }

      for (auto& [sizeX, sizeY, gridY] :
           {std::make_tuple(std::ref(padSizeX), std::ref(padSizeY), std::ref(padGridY)),
            std::make_tuple(std::ref(stripXSizeX), std::ref(stripXSizeY), std::ref(stripXGridY)),
            std::make_tuple(std::ref(stripYSizeX), std::ref(stripYSizeY), std::ref(stripYGridY))}) {
        sizeX.resize(16);
        sizeY.resize(16);
      }

      std::string padType {"pad"}, stripXType {"stripX"}, stripYType {"stripY"};
      // Pads
      auto pad = std::tie(padType, padGridX, padGridY, padSizeX, padSizeY, padOffset, padTable);
      // X strips
      auto stripX = std::tie(stripXType, stripXGridX, stripXGridY, stripXSizeX, stripXSizeY, stripXOffset, stripXTable);
      // Y strips
      auto stripY = std::tie(stripYType, stripYGridX, stripYGridY, stripYSizeX, stripYSizeY, stripYOffset, stripYTable);

      for (auto& [t, gridX, gridY, sizeX, sizeY, offset, table] : {pad, stripX, stripY}) {
        for (auto station : views::ints(0, nStations)) {
          size_t index = 0;
          for (auto region : views::ints(0, nRegions)) {
            size_t gidx = station * 4 + region;
            offset[gidx] = index;

            auto yxRange = views::concat(
              views::cartesian_product(views::ints(0, gridY[gidx]), views::ints(gridX[gidx], 2 * gridX[gidx])),
              views::cartesian_product(views::ints(gridY[gidx], 2 * gridY[gidx]), views::ints(0, 2 * gridX[gidx])));
            // loop over quarters
            for (auto [quarter, yx] : views::cartesian_product(views::ints(0, 4), yxRange)) {
              auto [y, x] = yx;
              LHCb::Detector::Muon::TileID tile {
                station,
                LHCb::Detector::Muon::Layout {
                  static_cast<unsigned int>(gridX[gidx]), static_cast<unsigned int>(gridY[gidx])},
                region,
                quarter,
                x,
                y};
              auto pos = det.position(tile);
              if (!pos) {
                std::stringstream e;
                e << t << " " << station << " " << region << " " << quarter << " " << gridX[gidx] << " " << gridY[gidx]
                  << " " << x << " " << y << "\n";
                throw GaudiException {e.str(), __FILE__, StatusCode::FAILURE};
              }
              else {
                auto sizeIdx = 4 * tile.station() + tile.region();

                // positions are always indexed by station
                table[station][index++] = {
                  boost::numeric_cast<float>(pos->x()),
                  boost::numeric_cast<float>(pos->y()),
                  boost::numeric_cast<float>(pos->z())};

                // sizes are specially indexed
                if (y < gridY[gidx]) {
                  if ((t == "pad") || (float) pos->dX() > sizeX[sizeIdx]) sizeX[sizeIdx] = pos->dX();
                  if ((t == "pad") || (float) pos->dY() > sizeY[sizeIdx]) sizeY[sizeIdx] = pos->dY();
                }
              }
            }
          }
        }

        output.write(
          gridX.size(),
          gridX,
          gridY.size(),
          gridY,
          sizeX.size(),
          sizeX,
          sizeY.size(),
          sizeY,
          offset.size(),
          offset,
          table.size());
        for (const auto& station : table) {
          output.write(station.size());
          for (const auto& point : station) {
            output.write(point);
          }
        }
      }
      m_data = output.buffer();
    }
#endif

    MuonLookupTables(const std::vector<char>& data) { m_data = data; }

    void update_constants(Constants& constants) const
    {
      const char* raw_input = m_data.data();

      // raw_input is walked forward by variable, not-necessarily-aligned byte
      // increments below, so reads must use memcpy rather than a pointer-cast
      // dereference (std::copy_n((T*) raw_input, ...)): the latter requires
      // raw_input to already be aligned as T*, which isn't guaranteed here and
      // is undefined behaviour when it isn't (caught by UBSan, see Allen#624).
      int version;
      std::memcpy(&version, raw_input, sizeof(uint));
      raw_input += sizeof(uint);

      if (version != 2 && version != 3) {
        error_cout << "Unrecognized MuonTable version!" << std::endl;
      }

      // 3 stations, each has gridX, gridY, sizeX, sizeY, offset, and 4 blocks of coordinates (1 block per station)
      static constexpr size_t n_data_blocks = 27;

      size_t allOffsets[n_data_blocks];
      int gridY[Muon::Constants::n_stations * Muon::Constants::n_regions * Muon::MuonTables::n_tables];
      size_t currentAllOffsetsIndex = 0;

      for (size_t tableNumber = 0; tableNumber < Muon::MuonTables::n_tables; tableNumber++) {
        size_t gridXSize;
        std::memcpy(&gridXSize, raw_input, sizeof(size_t));
        assert(gridXSize == Muon::Constants::n_stations * Muon::Constants::n_regions);
        raw_input += sizeof(size_t);
        allOffsets[currentAllOffsetsIndex++] = raw_input - m_data.data();
        raw_input += sizeof(int) * gridXSize;

        size_t gridYSize;
        std::memcpy(&gridYSize, raw_input, sizeof(size_t));
        raw_input += sizeof(size_t);
        std::memcpy(
          gridY + Muon::MuonTables::tableStationRegionOffset[tableNumber], raw_input, gridYSize * sizeof(int));
        allOffsets[currentAllOffsetsIndex++] = raw_input - m_data.data();
        raw_input += sizeof(int) * gridYSize;

        size_t sizeXSize;
        std::memcpy(&sizeXSize, raw_input, sizeof(size_t));
        raw_input += sizeof(size_t);
        allOffsets[currentAllOffsetsIndex++] = raw_input - m_data.data();
        raw_input += sizeof(float) * sizeXSize;

        size_t sizeYSize;
        std::memcpy(&sizeYSize, raw_input, sizeof(size_t));
        raw_input += sizeof(size_t);
        allOffsets[currentAllOffsetsIndex++] = raw_input - m_data.data();
        raw_input += sizeof(float) * sizeYSize;

        size_t offsetSize;
        std::memcpy(&offsetSize, raw_input, sizeof(size_t));
        raw_input += sizeof(size_t);
        allOffsets[currentAllOffsetsIndex++] = raw_input - m_data.data();
        raw_input += sizeof(unsigned int) * offsetSize;

        size_t tableSize;
        std::memcpy(&tableSize, raw_input, sizeof(size_t));
        raw_input += sizeof(size_t);
        assert(tableSize == Muon::Constants::n_stations);
        for (size_t i = 0; i < tableSize; i++) {
          size_t stationTableSize;
          std::memcpy(&stationTableSize, raw_input, sizeof(size_t));
          raw_input += sizeof(size_t);
          allOffsets[currentAllOffsetsIndex++] = raw_input - m_data.data();
          raw_input += sizeof(float) * Muon::MuonTables::n_dimensions * stationTableSize;
        }
      }
      assert(currentAllOffsetsIndex == n_data_blocks);

      auto& dev_muon_tables_raw = constants.dev_muon_lookup_tables_raw;
      auto& host_muon_tables_raw = constants.host_muon_lookup_tables_raw;
      if (!constants.dev_muon_tables) {
        Allen::malloc((void**) &dev_muon_tables_raw, m_data.size());
        Allen::malloc((void**) &constants.dev_muon_tables, sizeof(Muon::MuonTables));
      }
      else if (host_muon_tables_raw.size() != m_data.size()) {
        throw StrException {
          std::string {"[muon lut] sizes don't match: "} + std::to_string(host_muon_tables_raw.size()) + " " +
          std::to_string(m_data.size())};
      }
      host_muon_tables_raw = m_data;
      Allen::memcpy(
        dev_muon_tables_raw, host_muon_tables_raw.data(), host_muon_tables_raw.size(), Allen::memcpyHostToDevice);
      Muon::MuonTables host_muon_tables {allOffsets, dev_muon_tables_raw, version};
      Allen::memcpy(constants.dev_muon_tables, &host_muon_tables, sizeof(Muon::MuonTables), Allen::memcpyHostToDevice);
    }

    auto const& data() const { return m_data; }
    std::vector<char> m_data;
  };
} // namespace Allen::Conditions
