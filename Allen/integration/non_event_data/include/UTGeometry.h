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
#include "UTDefinitions.cuh"
#include "UTUniqueID.cuh"

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
#include <range/v3/view/repeat_n.hpp>
#include "range/v3/range/conversion.hpp"
#include <Kernel/IUTReadoutTool.h>
#include <Kernel/UTTell1Board.h> //v4
#include <Kernel/UTDAQBoard.h>   //v5
#include <UTDet/DeUTDetector.h>
#endif

namespace {
#ifndef ALLEN_STANDALONE
  /// get allen ut sector index
  inline int get_allen_ut_sector_index(const DeUTSector& ut_sector)
  {
    const auto ch = ut_sector.elementID();
    const auto side = ch.side();
    const auto layer = ch.layer();
    const auto stave = ch.stave();
    const auto face = ch.face();
    const auto module = ch.module();
    const auto sector = ch.sector();
    return sector_unique_id(side, layer, stave, face, module, sector);
  }
#endif

  struct AverageCounter {
    float mean = 0.0;
    unsigned count = 0;

    float value() const { return count == 0 ? 0.f : mean; }

    void add(float value)
    {
      ++count;
      mean += (value - mean) / count;
    }

    AverageCounter& operator+=(float value)
    {
      ++count;
      mean += (value - mean) / count;
      return *this;
    }
  };

  // UT sector numbering
  const unsigned sector_layer_offsets[] = {0, 248, 248 + 248, 248 + 248 + 276, 248 + 248 + 276 + 276};

  // Hardcoded dxdy
  const float hardcoded_layer_dxdy[] = {0, 0.08748866, -0.08748866, 0};

  unsigned map_sector_to_group(unsigned layer, unsigned sector)
  {
    static const unsigned r0_map[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    static const unsigned r1_map[] = {0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 13};
    static const unsigned r2_map[] = {0, 1, 2, 3, 4, 5, 5, 14, 14, 15, 15, 16, 16, 17, 17, 8, 8, 9, 10, 11, 12, 13};

    const auto group_offset = layer * UT::Constants::n_groups_in_layer;
    const auto station = layer / 2;
    const auto num_large_columns = 6 + station;

    const unsigned offsets[] = {
      0,
      14 * num_large_columns,
      14 * num_large_columns + 18,
      14 * num_large_columns + 18 + 22 * 2,
      14 * num_large_columns + 18 * 2 + 22 * 2,
      14 * num_large_columns * 2 + 18 * 2 + 22 * 2};
    std::array<const unsigned*, 5> maps = {r0_map, r1_map, r2_map, r1_map, r0_map};
    const unsigned maps_size[5] = {14, 18, 22, 18, 14};
    for (unsigned region = 0; region < 5; region++) {
      if (sector >= offsets[region] && sector < offsets[region + 1]) {
        const auto map = maps[region];
        const auto size = maps_size[region];
        const auto idx = (sector - offsets[region]) % size;
        return group_offset + map[idx];
      }
    }
    return 0;
  }

  template<typename T>
  void alloc_and_copy(std::vector<T>& host_data, std::span<T>& device_data) // FIXME: span by & is weird
  {
    T* p;
    Allen::malloc((void**) &p, host_data.size() * sizeof(T));
    device_data = {p, host_data.size()};
    Allen::memcpy(device_data.data(), host_data.data(), host_data.size() * sizeof(T), Allen::memcpyHostToDevice);
  }

  template<typename T>
  void alloc_and_copy(T*& host_data, T*& device_data)
  {
    Allen::malloc((void**) &device_data, sizeof(T));
    Allen::memcpy(device_data, host_data, sizeof(T), Allen::memcpyHostToDevice);
  }
} // namespace

namespace Allen::Conditions {
  struct UTGeometry {
    inline static std::string const id = "UTGeometry";
    inline static std::string const filename = "ut_geometry.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-ut-geometry";
#else
    inline static std::string const DefaultLocation = "AllenConditions-ut-geometry";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      return parent->addConditionDerivation(
        {DeUTDetLocation::location()}, UTGeometry::DefaultLocation, [](DeUTDetector const& det) {
          return Allen::Conditions::UTGeometry {det};
        });
    }

    UTGeometry(const DeUTDetector& det)
    {
      DumpUtils::Writer output {};

      // To ensure backward compatibility, we use first 16 bits to store UT geometry version
      // and the last 16 bits to store the number of sectors:
      // For old UT geometry which uses 32 bits to store number of sectors, the first 16 bits
      // are always empty, so the version is always zero
      uint32_t number_of_sectors = det.nSectors();
      uint32_t version = 1u; // 0 -> hardcoded dxdy, 1 -> per-serctor dxdy
      uint32_t metadata = number_of_sectors | (version << 16);

      // first strip is always 1
      std::vector<uint32_t> firstStrip =
        ranges::views::repeat_n(1, number_of_sectors) | ranges::to<std::vector<uint32_t>>();
      std::vector<float> pitch;
      std::vector<float> cos;
      std::vector<float> dy;
      std::vector<float> dp0diX;
      std::vector<float> dp0diY;
      std::vector<float> dp0diZ;
      std::vector<float> p0X;
      std::vector<float> p0Y;
      std::vector<float> p0Z;
      std::vector<float> dxDy;

      pitch.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());
      cos.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());
      dy.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());
      dp0diX.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());
      dp0diY.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());
      dp0diZ.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());
      p0X.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());
      p0Y.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());
      p0Z.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());
      dxDy.resize(number_of_sectors, std::numeric_limits<float>::quiet_NaN());

      det.applyToAllSectors([&](DeUTSector const& sector) {
        const auto idx = get_allen_ut_sector_index(sector);
        pitch[idx] = sector.pitch();
        cos[idx] = (sector.cosAngle());
        dy[idx] = (sector.get_dy());
        const auto dp0di = sector.get_dp0di();
        dp0diX[idx] = (dp0di.x());
        dp0diY[idx] = (dp0di.y());
        dp0diZ[idx] = (dp0di.z());
        const auto p0 = sector.get_p0();
        p0X[idx] = (p0.x());
        p0Y[idx] = (p0.y());
        // hack: since p0z is always positive, we can use the signbit to encode whether or not to "stripflip"
        p0Z[idx] = (((sector.xInverted() && sector.getStripflip()) ? -1 : 1) * p0.z());
        // this hack will be used in UTClusterAndPreDecode.cu
        dxDy[idx] = sector.get_dxdy();
      });

      // cross check
      assert(std::all_of(pitch.begin(), pitch.end(), [](auto i) { return !std::isnan(i); }));
      assert(std::all_of(cos.begin(), cos.end(), [](auto i) { return !std::isnan(i); }));
      assert(std::all_of(dy.begin(), dy.end(), [](auto i) { return !std::isnan(i); }));
      assert(std::all_of(dp0diX.begin(), dp0diX.end(), [](auto i) { return !std::isnan(i); }));
      assert(std::all_of(dp0diY.begin(), dp0diY.end(), [](auto i) { return !std::isnan(i); }));
      assert(std::all_of(dp0diZ.begin(), dp0diZ.end(), [](auto i) { return !std::isnan(i); }));
      assert(std::all_of(p0X.begin(), p0X.end(), [](auto i) { return !std::isnan(i); }));
      assert(std::all_of(p0Y.begin(), p0Y.end(), [](auto i) { return !std::isnan(i); }));
      assert(std::all_of(p0Z.begin(), p0Z.end(), [](auto i) { return !std::isnan(i); }));
      assert(std::all_of(dxDy.begin(), dxDy.end(), [](auto i) { return !std::isnan(i); }));

      output.write(metadata, firstStrip, pitch, dy, dp0diX, dp0diY, dp0diZ, p0X, p0Y, p0Z, cos, dxDy);

      m_data = output.buffer();
    }
#endif

    UTGeometry(const std::vector<char>& data) { m_data = data; }

    void initialize(Constants& constants) const
    {
      // Allocate space for geometry
      auto& dev_ut_geometry = constants.dev_ut_geometry;
      using span_type = typename std::remove_reference_t<decltype(dev_ut_geometry)>::value_type;
      char* g = nullptr;
      Allen::malloc((void**) &g, m_data.size());
      dev_ut_geometry = {g, static_cast<span_size_t<span_type>>(m_data.size())};
      const ::UTGeometry geometry {m_data};

      // Fill sector to group map
      auto& host_ut_sector_to_group_map = constants.host_ut_sector_to_group_map;
      host_ut_sector_to_group_map.resize(UT::Constants::n_sectors);
      auto& dev_ut_sector_to_group_map = constants.dev_ut_sector_to_group_map;
      for (unsigned layer = 0; layer < UT::Constants::n_layers; layer++) {
        const auto sector_offset = sector_layer_offsets[layer];
        const auto sector_size = sector_layer_offsets[layer + 1] - sector_offset;
        for (unsigned sector = 0; sector < sector_size; sector++) {
          host_ut_sector_to_group_map[sector_offset + sector] = map_sector_to_group(layer, sector);
        }
      }

      // Fill layer geometry
      auto& host_ut_layer_geometry = constants.host_ut_layer_geometry;
      auto& dev_ut_layer_geometry = constants.dev_ut_layer_geometry;
      host_ut_layer_geometry = new UT::Constants::UTLayerGeometry;

      for (unsigned layer = 0; layer < 4; layer++) {
        const auto inf = std::numeric_limits<float>::infinity();
        float min_z = inf, max_z = -inf, min_dxdy = inf, max_dxdy = -inf;
        AverageCounter mean_z, mean_dxdy, inner_ymin, inner_ymax, outer_ymin, outer_ymax, mean_dy[2];

        // Sector range
        const auto sector_offset = sector_layer_offsets[layer];
        const auto sector_size = sector_layer_offsets[layer + 1] - sector_offset;

        for (unsigned sector = 0; sector < sector_size; sector++) {
          const auto group = map_sector_to_group(layer, sector) % UT::Constants::n_groups_in_layer;

          // Get values
          float sector_dxdy = (geometry.dxDy) ? geometry.dxDy[sector_offset + sector] : hardcoded_layer_dxdy[layer];
          float sector_z = fabsf(geometry.p0Z[sector_offset + sector]);
          float sector_y0 = geometry.p0Y[sector_offset + sector];
          float sector_dy = geometry.dy[sector_offset + sector];

          // Fill Min and Max
          min_z = std::min(min_z, sector_z);
          max_z = std::max(max_z, sector_z);
          min_dxdy = std::min(min_dxdy, sector_dxdy);
          max_dxdy = std::max(max_dxdy, sector_dxdy);

          // Fill mean
          mean_z += sector_z;
          mean_dxdy += sector_dxdy;
          if (group == 0) {
            outer_ymin += sector_y0;
          }
          if (group == 13) {
            outer_ymax += sector_y0;
          }
          if (group == 14) {
            inner_ymin += sector_y0;
          }
          if (group == 17) {
            inner_ymax += sector_y0;
          }
          mean_dy[(group < 14)] += sector_dy;
        }

        // Compute step
        float outer_dy = (outer_ymax.value() - outer_ymin.value()) / (UT::Constants::n_groups_outer_in_layer - 1);
        float inner_dy = (inner_ymax.value() - inner_ymin.value()) / (UT::Constants::n_groups_inner_in_layer - 1);

        // Fill layer info
        host_ut_layer_geometry->mean_z[layer] = mean_z.value();
        host_ut_layer_geometry->mean_dxdy[layer] = mean_dxdy.value();
        host_ut_layer_geometry->error_z[layer] = max_z - min_z;
        host_ut_layer_geometry->error_dxdy[layer] = max_dxdy - min_dxdy;
        host_ut_layer_geometry->two_dy[layer][0] = mean_dy[0].value();
        host_ut_layer_geometry->two_dy[layer][1] = mean_dy[1].value();
        host_ut_layer_geometry->outer_1_over_dy[layer] = 1.f / outer_dy;
        host_ut_layer_geometry->outer_y0_over_dy[layer] = outer_ymin.value() / outer_dy;
        host_ut_layer_geometry->inner_1_over_dy[layer] = 1.f / inner_dy;
        host_ut_layer_geometry->inner_y0_over_dy[layer] = inner_ymin.value() / inner_dy;
      }

      alloc_and_copy(host_ut_sector_to_group_map, dev_ut_sector_to_group_map);
      alloc_and_copy(host_ut_layer_geometry, dev_ut_layer_geometry);
    }

    void update_constants(Constants& constants) const
    {
      auto& dev_ut_geometry = constants.dev_ut_geometry;
      if (dev_ut_geometry.empty()) {
        initialize(constants);
      }
      else if (static_cast<size_t>(dev_ut_geometry.size()) != m_data.size()) {
        throw StrException {
          std::string {"[ut geom] sizes don't match: "} + std::to_string(dev_ut_geometry.size()) + " " +
          std::to_string(m_data.size())};
      }

      auto& host_ut_geometry = constants.host_ut_geometry;
      host_ut_geometry = m_data;
      Allen::memcpy(
        dev_ut_geometry.data(), host_ut_geometry.data(), host_ut_geometry.size(), Allen::memcpyHostToDevice);
    }

    auto const& data() const { return m_data; }
    std::vector<char> m_data;
  };
} // namespace Allen::Conditions
