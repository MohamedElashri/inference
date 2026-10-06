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
#include "HostDeviceCondition.h"
#include "Constants.cuh"
#include "VeloDefinitions.cuh"
#include "ClusteringDefinitions.cuh"

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
#include "Detector/VP/VPChannelID.h"
#include <boost/numeric/conversion/cast.hpp>
#include <VPDet/DeVP.h>

namespace {
  uint64_t reverse_bits(uint64_t x)
  {
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x & 0x00FF00FF00FF00FFULL) << 8);
    x = ((x >> 16) & 0x0000FFFF0000FFFFULL) | ((x & 0x0000FFFF0000FFFFULL) << 16);
    x = (x >> 32) | (x << 32);
    return x;
  }

  uint32_t make_module_pairs_bitmask(uint64_t missing_modules)
  {
    // Magic numbers for deinterleaving
    uint64_t even = missing_modules & 0x5555555555555555; // Extract even bits (0,2,4,...)
    uint64_t odd = missing_modules & 0xAAAAAAAAAAAAAAAA;  // Extract odd bits (1,3,5,...)

    // Compact the bits by shifting and ORing
    even = (even | (even >> 1)) & 0x3333333333333333;
    even = (even | (even >> 2)) & 0x0F0F0F0F0F0F0F0F;
    even = (even | (even >> 4)) & 0x00FF00FF00FF00FF;
    even = (even | (even >> 8)) & 0x0000FFFF0000FFFF;
    even = (even | (even >> 16)) & 0x00000000FFFFFFFF;

    odd = odd >> 1; // Adjust for odd bit positions

    odd = (odd | (odd >> 1)) & 0x3333333333333333;
    odd = (odd | (odd >> 2)) & 0x0F0F0F0F0F0F0F0F;
    odd = (odd | (odd >> 4)) & 0x00FF00FF00FF00FF;
    odd = (odd | (odd >> 8)) & 0x0000FFFF0000FFFF;
    odd = (odd | (odd >> 16)) & 0x00000000FFFFFFFF;

    return even | odd;
  }
} // namespace
#endif

namespace Allen::Conditions {
  struct VPGeometry : HostDeviceCondition<VeloGeometry> {
    inline static std::string const id = "VeloGeometry";
    inline static std::string const filename = "velo_geometry.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-vp-geometry";
#else
    inline static std::string const DefaultLocation = "AllenConditions-vp-geometry";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      return parent->addConditionDerivation({DeVPLocation::Default}, VPGeometry::DefaultLocation, [](DeVP const& det) {
        return Allen::Conditions::VPGeometry {det};
      });
    }

    VPGeometry(const DeVP& det) :
      HostDeviceCondition<VeloGeometry>([&det]() {
        VeloGeometry geom {};

        const size_t sensorPerModule = 4;
        det.runOnAllSensors([&geom](const DeVPSensor& sensor) {
          geom.module_zs[sensor.module()] += boost::numeric_cast<float>(sensor.z() / sensorPerModule);
        });

        for (unsigned int i = 0; i < ::VP::NSensorColumns; i++)
          geom.local_x[i] = det.local_x(i);

        for (unsigned int i = 0; i < ::VP::NSensorColumns; i++)
          geom.x_pitch[i] = det.x_pitch(i);

        for (unsigned int i = 0; i < ::VP::NSensors; i++) {
          for (unsigned int j = 0; j < 12; j++) {
            geom.ltg[i * 12 + j] = det.ltg(LHCb::Detector::VPChannelID::SensorID {i})[j];
          }
        }

        uint64_t missing_modules_hlt1 = reverse_bits(det.missingModulesHlt1() << (64 - ::VP::NModules));
        geom.missing_module_pairs_hlt1 = make_module_pairs_bitmask(missing_modules_hlt1);
        return geom;
      }())
    {}
#endif

    VPGeometry(const std::vector<char>& data) :
      HostDeviceCondition<VeloGeometry>([&data]() {
        VeloGeometry geom {};
        std::memcpy(&geom, data.data(), data.size());

        // the -sizeof(uint32_t) is to account for padding in the struct
        assert(
          data.size() == sizeof(VeloGeometry) - sizeof(uint32_t) ||
          data.size() + sizeof(uint32_t) == sizeof(VeloGeometry) - sizeof(uint32_t));
        assert(geom.n_ltg == Velo::Constants::n_sensors);
        assert(geom.n_trans == 12);
        return geom;
      }())
    {}

    void update_constants(Constants& constants) const
    {
      constants.dev_velo_geometry = device();
      constants.host_velo_geometry = host();
    }
  };
} // namespace Allen::Conditions
