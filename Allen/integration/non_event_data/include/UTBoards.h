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
#include <Kernel/IUTReadoutTool.h>
#include <Kernel/UTTell1Board.h> //v4
#include <Kernel/UTDAQBoard.h>   //v5
#include <UTDet/DeUTDetector.h>
#endif

namespace Allen::Conditions {
  struct UTBoards {
    inline static std::string const id = "UTBoards";
    inline static std::string const filename = "ut_boards.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-ut-boards";
    inline static std::string const ReadoutLocation = "/world/BeforeMagnetRegion/UT:ReadoutMap";
#else
    inline static std::string const DefaultLocation = "AllenConditions-ut-boards";
    inline static std::string const ReadoutLocation = "/dd/Conditions/ReadoutConf/UT/ReadoutMap";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      static ToolHandle<IUTReadoutTool> readoutTool {parent, "UTReadoutTool", "UTReadoutTool"};

      return parent->addConditionDerivation(
        {readoutTool->getReadoutInfoKey(), ReadoutLocation},
        UTBoards::DefaultLocation,
        [](IUTReadoutTool::ReadoutInfo const& roInfo, nlohmann::json const& readoutMap) {
          return Allen::Conditions::UTBoards {&roInfo, readoutMap};
        });
    }

    UTBoards(IUTReadoutTool::ReadoutInfo const* roInfo, nlohmann::json const& readoutMap)
    {
      DumpUtils::Writer output {};

      std::vector<uint32_t> stripsPerHybrids;
      std::vector<uint32_t> sectors;
      std::vector<uint32_t> modules;
      std::vector<uint32_t> faces;
      std::vector<uint32_t> staves;
      std::vector<uint32_t> layers;
      std::vector<uint32_t> sides;
      std::vector<uint32_t> types;
      std::vector<uint32_t> chanIDs;

      UTDAQ::version UT_version; // Kernel/UTDAQDefinitions.h
      constexpr uint32_t n_lanes_max = 6;
      // mstahl: this is the condition for the new UT geometry. we might want a version field in the readout map
      if (readoutMap.contains("nTell40InUT"))
        UT_version = UTDAQ::version::v5;
      else if (readoutMap.contains("hybridsPerBoard"))
        UT_version = UTDAQ::version::v4;
      else
        throw GaudiException {
          "Cannot parse UT geometry version from ReadoutMap.", "DumpUTGeometry::Boards", StatusCode::FAILURE};
      // things that (might) depend on the decoding version
      const bool geometry_v5 = UT_version == UTDAQ::version::v5;
      const auto stripsPerHybrid = geometry_v5 ? UTDAQ::nStripsPerBoard / n_lanes_max :
                                                 UTDAQ::nStripsPerBoard / readoutMap["hybridsPerBoard"].get<int>();

      uint32_t currentBoardID = 0, cbID = 0;
      for (; cbID < roInfo->nBoards; ++cbID) {
        if (geometry_v5) {
          const auto b = roInfo->daqBoards[cbID].get(); // readout.findByDAQOrder(cbID, roInfo); // UTDAQ::Board
          const auto sector_ids = b->sectorIDs();
          stripsPerHybrids.push_back(stripsPerHybrid);
          const auto n_lanes_in_this_sector = sector_ids.size();
          for (typename std::decay<decltype(n_lanes_in_this_sector)>::type lane = 0; lane < n_lanes_in_this_sector;
               ++lane) {                     // old lingo: sectors, new lingo: lanes
            const auto s = sector_ids[lane]; // LHCb::UTChannelID
            sectors.push_back(s.sector());
            modules.push_back(s.module());
            faces.push_back(s.face());
            staves.push_back(s.stave());
            layers.push_back(s.layer());
            sides.push_back(s.side());
            types.push_back(s.type());
            chanIDs.push_back(s.channelID());
          }
          // If the number of lanes is less than 6, fill the remaining ones up to 6 with zeros
          // this is necessary to be compatible with the Allen UT boards layout
          for (uint32_t dummy_lane = n_lanes_in_this_sector; dummy_lane < n_lanes_max; ++dummy_lane) {
            sectors.push_back(0);
            modules.push_back(0);
            faces.push_back(0);
            staves.push_back(0);
            layers.push_back(0);
            sides.push_back(0);
            types.push_back(0);
            chanIDs.push_back(0);
          }
          ++currentBoardID;
        }
        else {
          const auto b = roInfo->boards[cbID].get(); // readout.findByOrder(cbID, roInfo); // UTTell1Board
          const auto boardID = b->boardID().id();
          // Insert empty boards if there is a gap between the last boardID and the
          // current one
          for (; boardID != 0 && currentBoardID < boardID; ++currentBoardID) {
            stripsPerHybrids.push_back(0);
            for (auto i = 0u; i < n_lanes_max; ++i) {
              sectors.push_back(0);
              modules.push_back(0);
              faces.push_back(0);
              staves.push_back(0);
              layers.push_back(0);
              sides.push_back(0);
              types.push_back(0);
              chanIDs.push_back(0);
            }
          }

          stripsPerHybrids.push_back(stripsPerHybrid);

          for (auto is = 0u; is < b->nSectors(); ++is) {
            auto s = std::get<0>(b->DAQToOfflineFull(
              0, UT_version, is * stripsPerHybrid)); // UTTell1Board::ExpandedChannelID (Kernel/UTTell1Board.h)
            sectors.push_back(s.sector);
            modules.push_back(s.module);
            faces.push_back(s.face);
            staves.push_back(s.stave);
            layers.push_back(s.layer);
            sides.push_back(s.side);
            types.push_back(s.type);
            chanIDs.push_back(s.chanID);
          }
          // If the number of sectors is less than 6, fill the remaining ones up to 6 with zeros
          // this is necessary to be compatible with the Allen UT boards layout
          for (auto is = b->nSectors(); is < n_lanes_max; ++is) {
            sectors.push_back(0);
            modules.push_back(0);
            faces.push_back(0);
            staves.push_back(0);
            layers.push_back(0);
            sides.push_back(0);
            types.push_back(0);
            chanIDs.push_back(0);
          }
          ++currentBoardID;
        } // geometry version
      }   // end loop boards

      output.write(
        currentBoardID,
        static_cast<uint32_t>(UT_version),
        stripsPerHybrids,
        sectors,
        modules,
        faces,
        staves,
        layers,
        sides,
        types,
        chanIDs);

      m_data = output.buffer();
    }
#endif

    UTBoards(const std::vector<char>& data) { m_data = data; }

    void initialize(Constants& constants) const
    {
      const ::UTBoards boards {m_data};

      // UT board -> geometry
      auto& host_ut_board_geometry_map = constants.host_ut_board_geometry_map;
      auto& dev_ut_board_geometry_map = constants.dev_ut_board_geometry_map;

      // UT board -> sector group
      auto& host_ut_board_to_sector_group_map = constants.host_ut_board_to_sector_group_map;
      auto& dev_ut_board_to_sector_group_map = constants.dev_ut_board_to_sector_group_map;

      // Fill maps
      for (uint32_t fullChanIndex = 0; fullChanIndex < boards.number_of_channels; fullChanIndex++) {
        const uint32_t side = boards.sides[fullChanIndex];
        const uint32_t layer = boards.layers[fullChanIndex];
        const uint32_t stave = boards.staves[fullChanIndex];
        const uint32_t face = boards.faces[fullChanIndex];
        const uint32_t module = boards.modules[fullChanIndex];
        const uint32_t sector = boards.sectors[fullChanIndex];
        int sec = sector_unique_id(side, layer, stave, face, module, sector);
        host_ut_board_geometry_map.emplace_back(sec);
        unsigned sg = get_sector_group_id(stave, face, module) + layer * UT::Constants::n_groups_in_layer;
        host_ut_board_to_sector_group_map.emplace_back(sg);
      }

      // Allocate and copy
      auto alloc_and_copy = [](auto const& host_data, auto& device_data) {
        using value_type = typename std::remove_reference_t<decltype(host_data)>::value_type;
        using span_type = typename std::remove_reference_t<decltype(device_data)>::value_type;
        value_type* p = nullptr;
        Allen::malloc((void**) &p, host_data.size() * sizeof(value_type));
        device_data = std::span {p, static_cast<span_size_t<span_type>>(host_data.size())};
        Allen::memcpy(
          device_data.data(), host_data.data(), host_data.size() * sizeof(value_type), Allen::memcpyHostToDevice);
      };
      alloc_and_copy(host_ut_board_geometry_map, dev_ut_board_geometry_map);
      alloc_and_copy(host_ut_board_to_sector_group_map, dev_ut_board_to_sector_group_map);
    }

    void update_constants(Constants& constants) const
    {
      auto& host_ut_boards = constants.host_ut_boards;
      auto& dev_ut_boards = constants.dev_ut_boards;
      if (host_ut_boards.empty()) {
        initialize(constants);
        char* p = nullptr;
        Allen::malloc((void**) &p, m_data.size());
        dev_ut_boards = {p, m_data.size()};
      }
      else if (host_ut_boards.size() != m_data.size()) {
        throw StrException {
          std::string {"[ut boards] sizes don't match: "} + std::to_string(host_ut_boards.size()) + " " +
          std::to_string(m_data.size())};
      }
      host_ut_boards = m_data;
      Allen::memcpy(dev_ut_boards.data(), host_ut_boards.data(), host_ut_boards.size(), Allen::memcpyHostToDevice);
    }

    auto const& data() const { return m_data; }
    std::vector<char> m_data;
  };
} // namespace Allen::Conditions
