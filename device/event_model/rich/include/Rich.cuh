/*****************************************************************************\
 * (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           *
 *                                                                             *
 * This software is distributed under the terms of the Apache License          *
 * version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
 \*****************************************************************************/
#pragma once
#include "RichDefinitions.cuh"
#include "RichPhotonDetectorPanel.cuh"

namespace Allen::Rich {
  struct RichDetector {
    __host__ __device__ inline auto rich() const { return m_type; }

    __host__ __device__ inline const auto& pdPanels() const { return m_panels; }

    /// Access PD Panel for a given side
    __host__ __device__ inline const auto& pdPanel(const Detector::Side side) const noexcept
    {
      return pdPanels()[side];
    }

    std::array<Detector::PDPanel, 2> m_panels {};
    Detector::Type m_type {Detector::Type::InvalidDetector};
  };
} // namespace Allen::Rich
