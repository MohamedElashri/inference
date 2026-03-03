/*****************************************************************************\
 * * (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
 * *                                                                             *
 * * This software is distributed under the terms of the Apache License          *
 * * version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
 * *                                                                             *
 * * In applying this licence, CERN does not waive the privileges and immunities *
 * * granted to it by virtue of its status as an Intergovernmental Organization  *
 * * or submit itself to any jurisdiction.                                       *
 * \*****************************************************************************/
#include <string>

#include <BackendCommon.h>
#include <Common.h>
#include <Consumers.h>
#include "RichPDMDBDecodeMapping.cuh"
#include "RichTel40CableMapping.cuh"
#include "Rich.cuh"

namespace {
  using std::string;
  using std::to_string;
  using std::vector;
} // namespace

// PDMDB mapping
Consumers::RichPDMDBDecodeMapping::RichPDMDBDecodeMapping(Constants& constants) : m_constants {constants} {}

void Consumers::RichPDMDBDecodeMapping::initialize(vector<char> const&)
{
  Allen::malloc((void**) &m_constants.get().dev_rich_pdmdb_mapping, sizeof(Allen::Rich::Decoding::PDMDBDecodeMapping));
}

void Consumers::RichPDMDBDecodeMapping::consume(vector<char> const& data)
{
  auto& dev_rich_pdmdb_mapping = m_constants.get().dev_rich_pdmdb_mapping;
  if (dev_rich_pdmdb_mapping == nullptr) {
    initialize(data);
  }
  auto& host_rich_pdmdb_mapping = m_constants.get().host_rich_pdmdb_mapping;
  host_rich_pdmdb_mapping = data;
  host_rich_pdmdb_mapping.resize(sizeof(Allen::Rich::Decoding::PDMDBDecodeMapping));

  // Patch anode (reverse column) and compute valid masks
  Allen::Rich::Decoding::PDMDBDecodeMapping* richPDMDBMapping =
    reinterpret_cast<Allen::Rich::Decoding::PDMDBDecodeMapping*>(host_rich_pdmdb_mapping.data());
  auto& pdmDataR = richPDMDBMapping->m_pdmDataR;
  auto& pdmDataH = richPDMDBMapping->m_pdmDataH;
  auto& pdmMaskR = richPDMDBMapping->m_pdmMaskR;
  auto& pdmMaskH = richPDMDBMapping->m_pdmMaskH;

  for (unsigned rich = 0; rich < 2; rich++) {
    for (unsigned pdmdb = 0; pdmdb < 2; pdmdb++) {
      for (unsigned link = 0; link < 6; link++) {
        for (unsigned b = 0; b < 11; b++) {
          pdmMaskR[rich][pdmdb][link][b] = 0;
          if (rich == 0) pdmMaskH[pdmdb][link][b] = 0;
        }
        for (unsigned bit = 0; bit < 86; bit++) {
          auto b = bit + (bit >= 39 ? 1 : 0);
          auto& dataR = pdmDataR[rich][pdmdb][link][bit];
          auto& dataH = pdmDataH[pdmdb][link][bit];

          unsigned validR = dataR.ec != -1 && dataR.pmtInEC != -1 && dataR.anode != -1;
          unsigned validH = dataH.ec != -1 && dataH.pmtInEC != -1 && dataH.anode != -1;

          pdmMaskR[rich][pdmdb][link][b / 8] |= validR << (b % 8);
          if (rich == 0) pdmMaskH[pdmdb][link][b / 8] |= validH << (b % 8);

          // Patch anode (reverse column)
          auto row = dataR.anode / 8;
          auto col = 7 - (dataR.anode % 8);
          dataR.anode = col + 8 * row;
          if (rich == 0) {
            row = dataH.anode / 8;
            col = 7 - (dataH.anode % 8);
            dataH.anode = col + 8 * row;
          }
        }
      }
    }
  }

  Allen::memcpy(
    dev_rich_pdmdb_mapping,
    host_rich_pdmdb_mapping.data(),
    sizeof(Allen::Rich::Decoding::PDMDBDecodeMapping),
    Allen::memcpyHostToDevice);
}

// Tel40 cable mapping
Consumers::RichTel40CableMapping::RichTel40CableMapping(Constants& constants) : m_constants {constants} {}

void Consumers::RichTel40CableMapping::initialize(vector<char> const&)
{
  Allen::malloc((void**) &m_constants.get().dev_rich_cable_mapping, sizeof(Allen::Rich::Decoding::Tel40CableMapping));
}

void Consumers::RichTel40CableMapping::consume(vector<char> const& data)
{
  auto& dev_rich_cable_mapping = m_constants.get().dev_rich_cable_mapping;
  if (dev_rich_cable_mapping == nullptr) {
    initialize(data);
  }
  auto& host_rich_cable_mapping = m_constants.get().host_rich_cable_mapping;
  host_rich_cable_mapping = data;

  // patch metadata to hold valid mask, TODO: do this properly in the dumpers
  Allen::Rich::Decoding::Tel40CableMapping* richCableMapping =
    reinterpret_cast<Allen::Rich::Decoding::Tel40CableMapping*>(host_rich_cable_mapping.data());
  auto& metadata = richCableMapping->m_tel40ConnMeta;
  auto& sources = richCableMapping->m_tel40ConnData;
  for (unsigned i = 0; i < metadata.size(); ++i) {
    for (unsigned j = 0; j < metadata[i].size(); ++j) {
      for (unsigned k = 0; k < metadata[i][j].size(); ++k) {
        uint32_t mask = 0;
        for (unsigned l = 0; l < sources[i][j][k].size(); ++l) {
          mask |= (sources[i][j][k][l].isActive ? (1 << l) : 0);

          sources[i][j][k][l].smartID.setData(
            sources[i][j][k][l].moduleNum,
            Allen::Rich::Decoding::SmartID::ShiftPDMod,
            Allen::Rich::Decoding::SmartID::MaskPDMod,
            Allen::Rich::Decoding::SmartID::MaskPDIsSet);
        }
        metadata[i][j][k].validLinkMask = mask;
      }
    }
  }

  Allen::memcpy(
    dev_rich_cable_mapping,
    host_rich_cable_mapping.data(),
    sizeof(Allen::Rich::Decoding::Tel40CableMapping),
    Allen::memcpyHostToDevice);
}

// RICH 1 geometry
Consumers::Rich1Geometry::Rich1Geometry(Constants& constants) : m_constants {constants} {}

void Consumers::Rich1Geometry::initialize(vector<char> const&)
{
  Allen::malloc((void**) &m_constants.get().dev_rich_1_geometry, sizeof(Allen::Rich::RichDetector));
}

void Consumers::Rich1Geometry::consume(vector<char> const& data)
{
  auto& dev_rich_1_geometry = m_constants.get().dev_rich_1_geometry;
  if (dev_rich_1_geometry == nullptr) {
    initialize(data);
  }
  auto& host_rich_1_geometry = m_constants.get().host_rich_1_geometry;
  host_rich_1_geometry = data;
  Allen::memcpy(
    dev_rich_1_geometry, host_rich_1_geometry.data(), sizeof(Allen::Rich::RichDetector), Allen::memcpyHostToDevice);
}

// RICH 2 geometry
Consumers::Rich2Geometry::Rich2Geometry(Constants& constants) : m_constants {constants} {}

void Consumers::Rich2Geometry::initialize(vector<char> const&)
{
  Allen::malloc((void**) &m_constants.get().dev_rich_2_geometry, sizeof(Allen::Rich::RichDetector));
}

void Consumers::Rich2Geometry::consume(vector<char> const& data)
{
  auto& dev_rich_2_geometry = m_constants.get().dev_rich_2_geometry;
  if (dev_rich_2_geometry == nullptr) {
    initialize(data);
  }
  auto& host_rich_2_geometry = m_constants.get().host_rich_2_geometry;
  host_rich_2_geometry = data;
  Allen::memcpy(
    dev_rich_2_geometry, host_rich_2_geometry.data(), sizeof(Allen::Rich::RichDetector), Allen::memcpyHostToDevice);
}
