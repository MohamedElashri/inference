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

#include <tuple>
#include <vector>

// LHCb
// #include <DetDesc/Condition.h>
// #include <DetDesc/ConditionAccessorHolder.h>
// #include "DetDesc/IConditionDerivationMgr.h"
#include "Detector/VP/VPChannelID.h"
#include <boost/numeric/conversion/cast.hpp>
#include <VPDet/DeVP.h>

#include <DetDesc/GenericConditionAccessorHolder.h>

// Gaudi
#include "GaudiAlg/Transformer.h"

// Allen
#include <Dumpers/Identifiers.h>
#include <Dumpers/Utils.h>
#include "Dumper.h"

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

/** @class DumpVPGeometry
 *  Dump Velo Geometry.
 *
 *  @author Nabil Garroum
 *  @date   2022-04-15
 *  This Class dumps geometry for Velo using DD4HEP and Gaudi Algorithm
 *  This Class uses a detector description
 *  This Class is basically an instation of a Gaudi algorithm with specific inputs and outputs:
 *  The role of this class is to get data from TES to Allen for the Velo Geometry
 */

namespace Dumpers {
  struct VP {

    VP() = default;
    VP(std::vector<char>& data, const DeVP& det)
    {
      DumpUtils::Writer output {};
      const size_t sensorPerModule = 4;
      std::vector<float> zs(det.numberSensors() / sensorPerModule, 0.f);
      det.runOnAllSensors([&zs](const DeVPSensor& sensor) {
        zs[sensor.module()] += boost::numeric_cast<float>(sensor.z() / sensorPerModule);
      });

      output.write(zs.size(), zs, size_t {::VP::NSensorColumns});
      for (unsigned int i = 0; i < ::VP::NSensorColumns; i++)
        output.write(det.local_x(i));
      output.write(size_t {::VP::NSensorColumns});
      for (unsigned int i = 0; i < ::VP::NSensorColumns; i++)
        output.write(det.x_pitch(i));
      output.write(size_t {::VP::NSensors}, size_t {12});
      for (unsigned int i = 0; i < ::VP::NSensors; i++)
        output.write(det.ltg(LHCb::Detector::VPChannelID::SensorID {i}));

      uint64_t missing_modules_hlt1 = reverse_bits(det.missingModulesHlt1() << (64 - ::VP::NModules));
      output.write(make_module_pairs_bitmask(missing_modules_hlt1));

      data = output.buffer();
    }
  };
} // namespace Dumpers

class DumpVPGeometry final
  : public Allen::Dumpers::Dumper<void(Dumpers::VP const&), LHCb::Algorithm::Traits::usesConditions<Dumpers::VP>> {
public:
  DumpVPGeometry(const std::string& name, ISvcLocator* svcLoc);

  void operator()(const Dumpers::VP& VP) const override;

  StatusCode initialize() override;

private:
  std::vector<char> m_data;
};

DECLARE_COMPONENT(DumpVPGeometry)

// Add the multitransformer call

DumpVPGeometry::DumpVPGeometry(const std::string& name, ISvcLocator* svcLoc) :
  Dumper(name, svcLoc, {KeyValue {"VPLocation", location(name, "geometry")}})
{}

StatusCode DumpVPGeometry::initialize()
{
  return Dumper::initialize().andThen([&] {
    register_producer(Allen::NonEventData::VeloGeometry::id, "velo_geometry", m_data);
    addConditionDerivation({DeVPLocation::Default}, inputLocation<Dumpers::VP>(), [&](DeVP const& det) {
      auto geo = Dumpers::VP {m_data, det};
      dump();
      return geo;
    });
  });
}

void DumpVPGeometry::operator()(const Dumpers::VP&) const {}
