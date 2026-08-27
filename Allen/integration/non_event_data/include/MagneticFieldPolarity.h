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

#ifndef ALLEN_STANDALONE
#include <Dumpers/AllenUpdater.h>
#include <Magnet/DeMagnet.h>
#endif

namespace Allen::Conditions {
  struct MagneticFieldPolarity {
    inline static std::string const id = "MagneticFieldPolarity";
    inline static std::string const filename = "polarity.bin";
#ifdef USE_DD4HEP
    inline static std::string const DefaultLocation = "/world:AllenConditions-magnetic-field-polarity";
#else
    inline static std::string const DefaultLocation = "AllenConditions-magnetic-field-polarity";
#endif

#ifndef ALLEN_STANDALONE
    template<typename PARENT>
    static auto addConditionDerivation(PARENT* parent)
    {
      return parent->addConditionDerivation(
        {LHCb::Det::Magnet::det_path}, MagneticFieldPolarity::DefaultLocation, [](DeMagnet const& deMagnet) {
          return Allen::Conditions::MagneticFieldPolarity {deMagnet};
        });
    }

    MagneticFieldPolarity(const DeMagnet& magField) : m_polarity {magField.isDown() ? -1.f : 1.f} {}
#endif

    MagneticFieldPolarity(const std::vector<char>& data)
    {
      assert(data.size() == sizeof(float));
      m_polarity = *reinterpret_cast<const float*>(data.data());
    }

    void update_constants(Constants& constants) const { constants.magnet_polarity = m_polarity; }

    std::span<const char> data() const { return {reinterpret_cast<const char*>(&m_polarity), sizeof(float)}; }

    float m_polarity {};
  };
} // namespace Allen::Conditions
