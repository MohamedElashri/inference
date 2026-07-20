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
#include "UTMagnetToolDefinitions.h"

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
#include <PrKernel/IPrUTMagnetTool.h>
#include <Magnet/DeMagnet.h>
#endif

namespace Allen::Conditions {
  struct UTLookupTables {
    inline static std::string const id = "UTLookupTables";
    inline static std::string const filename = "ut_tables.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-ut-lookup-tables";
#else
    inline static std::string const DefaultLocation = "AllenConditions-ut-lookup-tables";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      static ToolHandle<IPrUTMagnetTool> magnetTool {parent, "PrUTMagnetTool", "PrUTMagnetTool"};

      return parent->addConditionDerivation(
        {magnetTool->cacheLocation()}, UTLookupTables::DefaultLocation, [](IPrUTMagnetTool::Cache const& cache) {
          return Allen::Conditions::UTLookupTables {cache};
        });
    }

    UTLookupTables(const IPrUTMagnetTool::Cache& cache)
    {
      DumpUtils::Writer output {};

      std::tuple tables {*cache.lutDxLay, *cache.lutBdl};
      for_each(tables, [&output](auto const& t) {
        auto const& table = t.table();
        output.write(t.nVar);
        for (int i = 0; i < t.nVar; ++i)
          output.write(t.nBins(i));
        output.write(table.size(), table);
      });

      m_data = output.buffer();
    }
#endif

    UTLookupTables(const std::vector<char>& data) { m_data = data; }

    void update_constants(Constants& constants) const
    {
      auto& dev_ut_magnet_tool = constants.dev_ut_magnet_tool;
      char const* p = m_data.data();
      int const* layout = reinterpret_cast<int const*>(p);
      p += sizeof(int);
      int nVar = layout[0];
      assert(nVar == 2);
      std::vector<int> nBins(nVar);
      std::copy_n(layout + 1, nVar, nBins.begin());
      assert(nBins[0] == 3);
      assert(nBins[1] == 30);
      p += nVar * sizeof(int);

      size_t table_size;
      std::memcpy(&table_size, p, sizeof(size_t));
      assert(table_size == 124);
      p += sizeof(size_t);
      float const* deflection = reinterpret_cast<float const*>(p);
      p += table_size * sizeof(float);

      layout = reinterpret_cast<int const*>(p);
      p += sizeof(int);
      nVar = layout[0];
      assert(nVar == 3);
      nBins.resize(nVar);
      std::copy_n(layout + 1, nVar, nBins.begin());
      assert(nBins[0] == 30);
      assert(nBins[1] == 10);
      assert(nBins[2] == 10);
      p += nVar * sizeof(int);

      std::memcpy(&table_size, p, sizeof(size_t));
      assert(table_size == 3751);
      p += sizeof(size_t);

      float const* bdl = reinterpret_cast<float const*>(p);
      p += table_size * sizeof(float);

      if (!dev_ut_magnet_tool) {
        Allen::malloc((void**) &dev_ut_magnet_tool, sizeof(UTMagnetTool));
      }
      if (sizeof(UTMagnetTool) != (m_data.size() - 7 * sizeof(int) - 2 * sizeof(size_t))) {
        throw StrException {
          std::string {"[ut lut] sizes don't match: "} + std::to_string(sizeof(UTMagnetTool)) + " " +
          std::to_string(m_data.size())};
      }

      UTMagnetTool host_tool {deflection, bdl};

      // deflection table
      Allen::memcpy(dev_ut_magnet_tool, &host_tool, sizeof(UTMagnetTool), Allen::memcpyHostToDevice);
    }

    auto const& data() const { return m_data; }
    std::vector<char> m_data;
  };
} // namespace Allen::Conditions
