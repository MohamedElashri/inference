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
#include <RichDefinitions.cuh>
#include <RichPhotonDetector.cuh>
#include <iostream>

namespace Allen::Rich::Detector {
  struct PDPanel {
    using PDArray = std::array<PhotonDetector, Decoding::SmartID::MaPMT::MaxPDsPerModule>;
    using ModuleArray = std::array<PDArray, Decoding::SmartID::MaPMT::MaxModulesPerPanel>;

    /// Access the RICH detector type
    __host__ __device__ inline auto rich() const noexcept { return m_rich; }

    /// Access the PD panel side
    __host__ __device__ inline auto side() const noexcept { return m_side; }

    __host__ __device__ PhotonDetector dePD(const Decoding::SmartID smartID) const
    {
      PhotonDetector result;
      result.setIsNull(true);

      if (!smartID.pdIsSet()) return result;

      const auto localModNum = smartID.pdMod() - m_modNumOffset;

      const bool mod_in_bounds = (localModNum < pdModules().size());
      const auto& mod =
        mod_in_bounds ? pdModules()[localModNum] : pdModules()[0]; // Fallback to first (dummy) module if OOB

      const bool pd_in_bounds = (smartID.pdNumInMod() < mod.size());
      result = pd_in_bounds ? mod[smartID.pdNumInMod()] : result; // Keep default if OOB

      return result;
    }

    /// Access Panel ID
    const std::array<int8_t, 3>& getPanelID() const { return m_panelID; }

    /// Compute the global detection point for a given RichSmartID
    __host__ __device__ inline auto globalDetectionPoint(const Decoding::SmartID id) const noexcept
    {
      const auto pd = dePD(id);
      return (!pd.getIsNull() ? pd.globalDetectionPoint(id) : Point {0, 0, 0});
    }

    /// Access the global to local transform
    __host__ __device__ inline const auto& globalToPDPanel() const noexcept { return m_gloToPDPanelM; }

    /// Access all owned PD Modules
    __host__ __device__ const ModuleArray& pdModules() const noexcept { return m_PDs; }

    std::string toString() const
    {
      std::stringstream ss;
      for (size_t i = 0; i < m_PDs.size(); ++i) {
        ss << "RICH " << (int) m_rich << ", Side " << (int) m_side << ", Module No. " << i << ", PDs: ";
        for (size_t j = 0; j < m_PDs[i].size(); ++j) {
          if (!(m_PDs[i][j].getIsNull())) {
            auto pd = m_PDs[i][j];
            ss << pd.toString() << '\n';
          }
          else {
            ss << " 0\n";
          }
        }
        ss << "\n";
      }
      return ss.str();
    }

    ModuleArray m_PDs {};
    std::array<float, 12> m_gloToPDPanelM {}; // 3D Transform
    uint32_t m_modNumOffset {0};
    std::array<int8_t, 3> m_panelID {Type::InvalidDetector, Side::InvalidSide, -1};
    Type m_rich {Type::InvalidDetector};
    Side m_side {Side::InvalidSide};
  };
} // namespace Allen::Rich::Detector
