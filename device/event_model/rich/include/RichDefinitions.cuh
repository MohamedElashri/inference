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

#include <cstdint>
#include <string>
#include <ostream>
#include "BackendCommon.h"

namespace Allen::Rich {
  using FP = float;
  using Point = float3;
  using DataType = std::uint32_t;
  using Transform3D = std::array<float, 12>;
  using KeyType = std::uint64_t;
  using BitPackType = KeyType;
  using ADCTimeType = std::uint16_t;

  //// Constants related to pixel reconstruction, value definition based on those in HLT2's
  /// Rec/Rich/RichFutureRecPixelAlgorithms/src/RichSIMDSummaryPixels.cpp

  /// Enabled 4D reconstruction
  [[maybe_unused]] __device__ constexpr bool m_enable4D[2] = {false, false};

  /// Average expected hit time for signal in each RICH in ns
  [[maybe_unused]] __device__ constexpr float m_avHitTime[2] = {13.03, 52.94};

  /// Course (pixel) Time window for each RICH in ns
  [[maybe_unused]] __device__ constexpr float m_timeWindow[2] = {3.0, 3.0};

  /// Enable the override of inner and out regions
  [[maybe_unused]] __device__ constexpr bool m_overrideRegions[2] = {false, false};

  /// Size in X defining the inner pixels for each RICH
  [[maybe_unused]] __device__ constexpr double m_innerPixX[2] = {250.0, 99999.9};

  /// Size in Y defining the inner pixels for each RICH
  [[maybe_unused]] __device__ constexpr double m_innerPixY[2] = {300.0, 300.0};

  /// Time resolution for inner regions in ns
  [[maybe_unused]] __device__ constexpr float m_innerTimeWindow[2] = {0.15, 0.15};

  /// Time resolution for outer regions in ns
  [[maybe_unused]] __device__ constexpr float m_outerTimeWindow[2] = {0.3, 0.3};

  /// Matrix-point transform3D, adapted from ROOT::Math
  __host__ __device__ inline Point transform3DTimesPoint(Transform3D fM, Point point)
  {
    return Point {fM[0] * point.x + fM[1] * point.y + fM[2] * point.z + fM[3],
                  fM[4] * point.x + fM[5] * point.y + fM[6] * point.z + fM[7],
                  fM[8] * point.x + fM[9] * point.y + fM[10] * point.z + fM[11]};
  }
} // namespace Allen::Rich

namespace Allen::Rich::Detector {
  enum Type : std::int8_t {
    InvalidDetector = -1, //< Unspecified Detector
    Rich1 = 0,            //< RICH1 detector
    Rich2 = 1,            //< RICH2 detector
    Rich = 1              //< Single RICH detector
  };

  // Detector side enum
  enum Side : std::int8_t {
    InvalidSide = -1, //< Invalid side
    // RICH1
    top = 0,    //< Upper panel in RICH1
    bottom = 1, //< Lower panel in RICH1
    // RICH2
    left = 0,  //< Left panel in RICH2
    right = 1, //< Right panel in RICH2
    aside = 0, //< A-Side panel in RICH2
    cside = 1, //< C-Side panel in RICH2
    // Generic
    firstSide = 0, //< Upper panel in RICH1 or Left panel in RICH2
    secondSide = 1 //< Lower panel in RICH1 or Right panel in RICH2
  };
} // namespace Allen::Rich::Detector

namespace Allen::Rich::Decoding {
  // Helper class for RichDecoding
  class PackedFrameSizes final {
  public:
    // Packed type
    using IntType = std::uint8_t;

  private:
    // Bits for each Size
    static const IntType Bits0 = 4;
    static const IntType Bits1 = 4;
    // shifts
    static const IntType Shift0 = 0;
    static const IntType Shift1 = Shift0 + Bits0;
    // masks
    static const IntType Mask0 = (IntType)((1 << Bits0) - 1) << Shift0;
    static const IntType Mask1 = (IntType)((1 << Bits1) - 1) << Shift1;
    // max values
    static const IntType Max0 = (1 << Bits0) - 1;
    static const IntType Max1 = (1 << Bits1) - 1;

  public:
    // Contructor from a single word
    __host__ __device__ explicit PackedFrameSizes(const IntType d) : m_data(d) {}

    // Get the overall data
    __host__ __device__ inline IntType data() const noexcept { return m_data; }

    // Get first size word
    __host__ __device__ inline IntType size0() const noexcept { return ((data() & Mask0) >> Shift0); }

    // Get second size word
    __host__ __device__ inline IntType size1() const noexcept { return ((data() & Mask1) >> Shift1); }

    // Get the total size
    __host__ __device__ inline auto totalSize() const noexcept { return size0() + size1(); }

  private:
    // The data word
    IntType m_data {0};
  };
} // namespace Allen::Rich::Decoding
