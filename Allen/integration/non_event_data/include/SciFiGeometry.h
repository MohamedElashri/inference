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
#include "SciFiDefinitions.cuh"

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
#include <Detector/FT/FTChannelID.h>
#include <Detector/FT/FTConstants.h>
#include <FTDet/DeFTDetector.h>
#include <FTDAQ/FTReadoutMap.h>
#endif

namespace Allen::Conditions {
  struct SciFiGeometry {
    inline static std::string const id = "SciFiGeometry";
    inline static std::string const filename = "scifi_geometry.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-scifi-geometry";
    inline static std::string const FTReadoutMapLocation = "/world:AllenConditions-FTReadoutMap";
#else
    inline static std::string const DefaultLocation = "AllenConditions-scifi-geometry";
    inline static std::string const FTReadoutMapLocation = "AllenConditions-FTReadoutMap";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      auto dataSvc = parent->template service<IDataProviderSvc>("DetectorDataSvc", true);
      FTReadoutMap::addConditionDerivation(parent, &*dataSvc, SciFiGeometry::FTReadoutMapLocation);

      return parent->addConditionDerivation(
        {DeFTDetectorLocation::Default, SciFiGeometry::FTReadoutMapLocation},
        SciFiGeometry::DefaultLocation,
        [](DeFT const& det, FTReadoutMap const& map) {
          return Allen::Conditions::SciFiGeometry {det, map};
        });
    }

    SciFiGeometry(const DeFT& det, const FTReadoutMap& readoutMap)
    {
      // Detector and mat geometry
      uint32_t const number_of_stations = LHCb::Detector::FT::nStations;
      uint32_t const number_of_layers_per_station = LHCb::Detector::FT::nLayers;
      uint32_t const number_of_layers = number_of_stations * number_of_layers_per_station;
      uint32_t const number_of_quarters_per_layer = LHCb::Detector::FT::nQuarters;
      uint32_t const number_of_quarters = number_of_quarters_per_layer * number_of_layers;
      std::vector<uint32_t> number_of_modules(number_of_quarters);
      uint32_t number_of_mats = 0;

      std::vector<float> mirrorPointX;
      std::vector<float> mirrorPointY;
      std::vector<float> mirrorPointZ;
      std::vector<float> ddxX;
      std::vector<float> ddxY;
      std::vector<float> ddxZ;
      std::vector<float> uBegin;
      std::vector<float> halfChannelPitch;
      std::vector<float> dieGap;
      std::vector<float> sipmPitch;
      std::vector<float> dxdy;
      std::vector<float> dzdy;
      std::vector<float> globaldy;
      std::vector<float> average_z;
      std::vector<float> average_dxdy;
      std::vector<std::array<float, 128 * 4>> matEndCalibrationVector; // fix me

      // First uniqueMat is 512, save space by subtracting
      const uint32_t uniqueMatOffset = 512; // FIXME -- hardcoded
      // PrStoreFTHit.h uses hardcoded 2<<11, which is too much.
      uint32_t max_uniqueMat = (1 << 11) - uniqueMatOffset; // FIXME -- hardcoded
      mirrorPointX.resize(max_uniqueMat);
      mirrorPointY.resize(max_uniqueMat);
      mirrorPointZ.resize(max_uniqueMat);
      ddxX.resize(max_uniqueMat);
      ddxY.resize(max_uniqueMat);
      ddxZ.resize(max_uniqueMat);
      uBegin.resize(max_uniqueMat);
      halfChannelPitch.resize(max_uniqueMat);
      dieGap.resize(max_uniqueMat);
      sipmPitch.resize(max_uniqueMat);
      dxdy.resize(max_uniqueMat);
      dzdy.resize(max_uniqueMat);
      globaldy.resize(max_uniqueMat);
      matEndCalibrationVector.resize(max_uniqueMat);
      average_z.resize(number_of_layers);
      average_dxdy.resize(number_of_layers);

      std::array<unsigned, number_of_stations> stations = {1, 2, 3};

      for (auto i_station : stations) {
        LHCb::Detector::FTChannelID::StationID station_id {i_station};

        for (unsigned i_layer = 0; i_layer < number_of_layers_per_station; ++i_layer) {
          unsigned n_measurements = 0;
          unsigned index_layer = (i_station - 1) * number_of_layers_per_station + i_layer;
          LHCb::Detector::FTChannelID::LayerID layer_id {i_layer};
          auto const& layer = det.findLayer(LHCb::Detector::FTChannelID {
            station_id,
            layer_id,
            LHCb::Detector::FTChannelID::QuarterID {0u},
            LHCb::Detector::FTChannelID::ModuleID {0u},
            LHCb::Detector::FTChannelID::MatID {0u},
            0u,
            0u});
          for (unsigned i_quarter = 0; i_quarter < number_of_quarters_per_layer; ++i_quarter) {
            LHCb::Detector::FTChannelID::QuarterID quarter_id {i_quarter};
            LHCb::Detector::FTChannelID quarterChanID(
              station_id,
              layer_id,
              quarter_id,
              LHCb::Detector::FTChannelID::ModuleID {0u},
              LHCb::Detector::FTChannelID::MatID {0u},
              0,
              0);
            auto const& quarter = layer->findQuarter(quarterChanID);
            auto const n_modules = quarter->modules().size();
            number_of_modules[quarterChanID.globalQuarterIdx()] = n_modules;
            for (unsigned i_module = 0; i_module < n_modules; ++i_module) {
              LHCb::Detector::FTChannelID::ModuleID module_id {i_module};
              auto const& mod = quarter->findModule(LHCb::Detector::FTChannelID {
                station_id, layer_id, quarter_id, module_id, LHCb::Detector::FTChannelID::MatID {0u}, 0u, 0u});
              number_of_mats += LHCb::Detector::FT::nMats;
              for (unsigned i_mat = 0; i_mat < LHCb::Detector::FT::nMats; ++i_mat) {
                LHCb::Detector::FTChannelID::MatID mat_id {i_mat};
                const auto& mat =
                  mod->findMat(LHCb::Detector::FTChannelID {station_id, layer_id, quarter_id, module_id, mat_id, 0, 0});
                auto index = mat->elementID().globalMatID_shift();
                const auto& mirrorPoint = mat->mirrorPoint();
                const auto& ddx = mat->ddx();
                mirrorPointX[index] = mirrorPoint.x();
                mirrorPointY[index] = mirrorPoint.y();
                mirrorPointZ[index] = mirrorPoint.z();
                ddxX[index] = ddx.x();
                ddxY[index] = ddx.y();
                ddxZ[index] = ddx.z();
                uBegin[index] = mat->uBegin();
                halfChannelPitch[index] = mat->halfChannelPitch();
                dieGap[index] = mat->dieGap();
                sipmPitch[index] = mat->sipmPitch();
                dxdy[index] = mat->dxdy();
                dzdy[index] = mat->dzdy();
                globaldy[index] = mat->globaldy();
                std::copy(
                  mat->getmatContractionParameterVector().begin(),
                  mat->getmatContractionParameterVector().end(),
                  matEndCalibrationVector[index].begin()); // copy the vector
                average_z[index_layer] += (mirrorPoint.z() + mat->dzdy() * mirrorPoint.y());
                average_dxdy[index_layer] += mat->dxdy();
                n_measurements += 1;
              }
            }
          }
          average_z[index_layer] = average_z[index_layer] / n_measurements;
          average_dxdy[index_layer] = average_dxdy[index_layer] / n_measurements;
        }
      }

      DumpUtils::Writer output {};

      // Data from Condition
      auto comp = readoutMap.compatibleVersions();
      auto number_of_banks = readoutMap.nBanks();
      if (comp.count(4)) {
        // Decoding v6
        std::vector<uint32_t> bank_first_channel;
        bank_first_channel.reserve(number_of_banks);
        for (unsigned int i = 0; i < number_of_banks; i++) {
          bank_first_channel.push_back(readoutMap.channelIDShift(i));
        }
        output.write(
          number_of_stations,
          number_of_layers_per_station,
          number_of_layers,
          number_of_quarters_per_layer,
          number_of_quarters,
          number_of_modules,
          LHCb::Detector::FT::nMats,
          number_of_mats,
          number_of_banks,
          2, // v0 hardcoded, v2 read-in geometry (decoding v4,5,6)
          bank_first_channel,
          LHCb::Detector::FT::nMatsMax,
          mirrorPointX,
          mirrorPointY,
          mirrorPointZ,
          ddxX,
          ddxY,
          ddxZ,
          uBegin,
          halfChannelPitch,
          dieGap,
          sipmPitch,
          dxdy,
          dzdy,
          globaldy,
          average_z,
          average_dxdy,
          matEndCalibrationVector);
      }
      else if (comp.count(7)) {
        constexpr uint32_t nLinksPerBank = 24; // FIXME: change to centralised number
        std::vector<uint32_t> source_ids;
        source_ids.reserve(number_of_banks);
        for (unsigned int i = 0; i < number_of_banks; i++)
          source_ids.push_back(readoutMap.sourceID(i).sourceID());
        std::vector<uint32_t> linkMap;
        linkMap.reserve(number_of_banks * nLinksPerBank);
        for (unsigned int i = 0; i < number_of_banks; i++)
          for (unsigned int j = 0; j < nLinksPerBank; j++)
            linkMap.push_back(readoutMap.getGlobalSiPMIDFromIndex(i, j));
        output.write(
          number_of_stations,
          number_of_layers_per_station,
          number_of_layers,
          number_of_quarters_per_layer,
          number_of_quarters,
          number_of_modules,
          LHCb::Detector::FT::nMats,
          number_of_mats,
          number_of_banks,
          3, // v1 hardcoded, v3 read-in geometry (decoding v7,8)
          source_ids,
          linkMap,
          max_uniqueMat,
          mirrorPointX,
          mirrorPointY,
          mirrorPointZ,
          ddxX,
          ddxY,
          ddxZ,
          uBegin,
          halfChannelPitch,
          dieGap,
          sipmPitch,
          dxdy,
          dzdy,
          globaldy,
          average_z,
          average_dxdy,
          matEndCalibrationVector);
      }
      else {
        std::stringstream s;
        Gaudi::Utils::toStream(comp, s);
        throw GaudiException {"Unsupported conditions compatible with " + s.str(), __FILE__, StatusCode::FAILURE};
      }
      m_data = output.buffer();
    }
#endif

    SciFiGeometry(const std::vector<char>& data) { m_data = data; }

    void initialize(Constants& constants) const
    {
      auto& dev_scifi_geometry = constants.dev_scifi_geometry;
      Allen::malloc((void**) &dev_scifi_geometry, m_data.size());
    }

    void update_constants(Constants& constants) const
    {
      auto updated_data = m_data;
      size_t old_size = m_data.size();
      updated_data.resize(
        m_data.size() + 2 * SciFi::Constants::n_layers * sizeof(float)); // add 12 slots for z and 12 for dxdy
      SciFi::SciFiGeometry g {updated_data};
      uint32_t version = g.version;
      if (version == 0 || version == 1) {
        // hardcoded geometry version
        // v0 is decoding 4,5,6
        // v1 is decoding 7,8
        // hardcoded values taken from /device/event_model/SciFi/include/LookingForwardConstants.cuh (old Upgrade
        // minbias 2018 MC sample)
        std::vector<float> z_values = {
          7826.f, 7896.f, 7966.f, 8036.f, 8508.f, 8578.f, 8648.f, 8718.f, 9193.f, 9263.f, 9333.f, 9403.f};
        std::vector<float> dxdy_values = {
          0.f, 0.0874892f, -0.0874892f, 0.f, 0.f, 0.0874892f, -0.0874892f, 0.f, 0.f, 0.0874892f, -0.0874892f, 0.f};
        std::copy(z_values.begin(), z_values.end(), g.average_z);
        std::copy(dxdy_values.begin(), dxdy_values.end(), g.average_dxdy);
      }
      else {
        // extra space was not needed
        updated_data.resize(old_size);
      }

      auto& dev_scifi_geometry = constants.dev_scifi_geometry;
      if (dev_scifi_geometry == nullptr) {
        initialize(constants);
      }

      auto& host_scifi_geometry = constants.host_scifi_geometry;
      host_scifi_geometry = updated_data;
      Allen::memcpy(
        dev_scifi_geometry, host_scifi_geometry.data(), host_scifi_geometry.size(), Allen::memcpyHostToDevice);
    }

    auto const& data() const { return m_data; }
    std::vector<char> m_data;
  };
} // namespace Allen::Conditions
