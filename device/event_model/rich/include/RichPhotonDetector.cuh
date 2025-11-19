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
#include <RichSmartID.cuh>
#include <array>

namespace Allen::Rich::Detector {
  struct PhotonDetector {
    __host__ __device__ inline auto pdSmartID() const { return m_pdSmartID; }

    __host__ __device__ inline auto panelDetectionPoint(const Decoding::SmartID id) const noexcept
    {
      const auto fPixCol = id.pixelCol();
      const auto fPixRow = id.pixelRow();
      const auto xh = (fPixCol - m_numPixColFrac) * m_effectivePixelXSize;
      const auto yh = (fPixRow - m_numPixRowFrac) * m_effectivePixelYSize;
      return Point {xh, yh, m_localZcoord};
    }

    __host__ __device__ inline auto globalDetectionPoint(const Decoding::SmartID id) const noexcept
    {
      return transform3DTimesPoint(localToGlobal(), panelDetectionPoint(id));
    }

    __host__ __device__ inline float getEffectivePixelXSize() const noexcept { return m_effectivePixelXSize; }
    __host__ __device__ inline float getEffectivePixelYSize() const noexcept { return m_effectivePixelYSize; }
    __host__ __device__ inline float getLocalZcoord() const noexcept { return m_localZcoord; }
    __host__ __device__ inline float getNumPixRowFrac() const { return m_numPixRowFrac; }
    __host__ __device__ inline float getNumPixColFrac() const noexcept { return m_numPixColFrac; }

    /// Effective pixel area
    __host__ __device__ inline auto effectivePixelArea() const noexcept { return m_effPixelArea; }

    /// Access the local to global transform
    __host__ __device__ inline const std::array<float, 12>& localToGlobal() const noexcept { return m_locToGloM; }

    std::string toString() const
    {
      std::ostringstream oss;
      oss << "PD { "
          << "m_locToGloM: " << m_locToGloM[0] << " " << m_locToGloM[1] << " " << m_locToGloM[2] << " "
          << m_locToGloM[3] << " " << m_locToGloM[4] << " " << m_locToGloM[5] << " " << m_locToGloM[6] << " "
          << m_locToGloM[7] << " " << m_locToGloM[8] << " " << m_locToGloM[9] << " " << m_locToGloM[10] << " "
          << m_locToGloM[11] << " "
          << ", "
          << "m_zeroInPanelFrame: " << m_zeroInPanelFrame[0] << m_zeroInPanelFrame[1] << m_zeroInPanelFrame[2] << ", "
          << "effPixelArea: " << m_effPixelArea << ", "
          << "numPixels: " << m_numPixels << ", "
          << "localZcoord: " << m_localZcoord << ", "
          << "NumPixColFrac: " << m_numPixColFrac << ", "
          << "NumPixRowFrac: " << m_numPixRowFrac << ", "
          << "EffectivePixelXSize: " << m_effectivePixelXSize << ", "
          << "EffectivePixelYSize: " << m_effectivePixelYSize << ", "
          << "pdSmartID: " << m_pdSmartID << ", "
          << "isHType: " << (m_isHType ? "true" : "false") << "isNull: " << (m_isNull ? "true" : "false") << " }";
      return oss.str();
    }

    __host__ __device__ bool getIsNull() const { return m_isNull; }

    __host__ __device__ void setIsNull(bool value) { m_isNull = value; }

    std::array<float, 12> m_locToGloM {};
    std::array<float, 3> m_zeroInPanelFrame {};
    std::uint64_t m_pdSmartID {0};
    float m_effPixelArea {0.f};
    float m_numPixels {0.f};
    float m_localZcoord {0.f};
    float m_numPixColFrac {0.f};
    float m_numPixRowFrac {0.f};
    float m_effectivePixelXSize {0.f};
    float m_effectivePixelYSize {0.f};
    bool m_isHType {false};
    bool m_isNull {false};
  };
} // namespace Allen::Rich::Detector
