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

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
#include <LHCbDet/InteractionRegion.h>
#include <LHCbDet/LHCInfo.h>
#endif

namespace Allen::Conditions {
  struct Beamline {
    inline static std::string const id = "Beamline";
    inline static std::string const filename = "beamline.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-beamline";
    inline static std::string const InteractionRegionLocation = "/world:AllenConditions-InteractionRegion";
    inline static std::string const LHCInfoLocation = "/world:AllenConditions-LHCInfo";
#else
    inline static std::string const DefaultLocation = "AllenConditions-beamline";
    inline static std::string const InteractionRegionLocation = "AllenConditions-InteractionRegion";
    inline static std::string const LHCInfoLocation = "AllenConditions-LHCInfo";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      // First register a derivation on the interaction region
      LHCb::Conditions::InteractionRegion::addConditionDerivation(parent, Beamline::InteractionRegionLocation);
      LHCb::Conditions::LHCInfo::addConditionDerivation(parent, Beamline::LHCInfoLocation);

      auto updater = parent->template service<AllenUpdater>("AllenUpdater", true);
      if (!updater) {
        parent->error() << "Failed to retrieve AllenUpdater" << endmsg;
#ifdef USE_DD4HEP
        return false;
#else
        return static_cast<std::size_t>(-1); // NoDerivation
#endif
      }

      // Then derived the interaction region to create the host representation and copy it in the device constant memory
      return parent->addConditionDerivation(
        {Beamline::InteractionRegionLocation, Beamline::LHCInfoLocation},
        Beamline::DefaultLocation,
        [updater](LHCb::Conditions::InteractionRegion const& ir, std::optional<LHCb::Detector::LHCInfo> const& li) {
          return Allen::Conditions::Beamline {ir, li, updater->getBeamlineOffset()};
        });
    }

    Beamline(
      LHCb::Conditions::InteractionRegion const& region,
      std::optional<LHCb::Detector::LHCInfo> const& LHC_info,
      std::array<float, 2> offset)
    {
      std::vector<double> pos(3);
      region.avgPosition.GetCoordinates(pos.begin(), pos.end());
      pos[0] = pos[0] + static_cast<double>(offset[0]);
      pos[1] = pos[1] + static_cast<double>(offset[1]);

      std::vector<double> sprd(region.spread.begin(), region.spread.end());
      auto as_float = [](auto const& vd) {
        std::vector<float> vf(vd.size());
        std::transform(vd.begin(), vd.end(), vf.begin(), [](double v) { return static_cast<float>(v); });
        return vf;
      };

      double xangleh = 0.;
      double xanglev = 0.;
      if (LHC_info.has_value()) {
        xangleh = static_cast<double>(LHC_info.value().xangleh);
        xanglev = static_cast<double>(LHC_info.value().xanglev);
      }
      std::vector<double> cross_angles(2);
      cross_angles[0] = xangleh;
      cross_angles[1] = xanglev;

      DumpUtils::Writer output;
      // version 2, position, spread, effectie crossing angles
      output.write(2u, as_float(pos), as_float(sprd), as_float(cross_angles));
      m_data = output.buffer();
    }
#endif

    Beamline(const std::vector<char>& data) { m_data = data; }

    void update_constants(Constants& constants) const
    {
      // Version "0" of the beamline has floats in it: the x and y
      // position. A version number was absent, so we use the fact
      // that we got 8 bytes. This is ugly but at this point it's better
      // to stay backwards compatible.

      // Version 1 has a version number (unsigned int) followed by 3
      // floats for (x, y, z) position and 6 floats representing a
      // triangular covariance matrix. We don't copy the version number to
      // stay backwards compatible.

      // Version 2 has a version number (unsigned int) followed by 3
      // floats for (x, y, z) position, 6 floats representing a
      // triangular covariance matrix and 2 floats representing effective crossing angles. We don't copy the version
      // number to stay backwards compatible. We are not copy angles to host_beamline to be backward compatible
      assert(m_data.size() >= 8u);
      auto const version = m_data.size() == 8u ? 0u : reinterpret_cast<unsigned const*>(m_data.data())[0];
      auto const data_size = version == 0u ? m_data.size() : m_data.size() - sizeof(unsigned);
      char const* data_start = version == 0u ? m_data.data() : m_data.data() + sizeof(unsigned);

      auto& host_beamline = constants.host_beamline;
      auto host_beamline_ptr = reinterpret_cast<float const*>(data_start);

      std::vector<float> host_beamline_local(
        host_beamline_ptr, host_beamline_ptr + static_cast<span_size_t<char>>(data_size / sizeof(float)));
      host_beamline = host_beamline_local;
    }

    auto const& data() const { return m_data; }
    std::vector<char> m_data;
  };
} // namespace Allen::Conditions
