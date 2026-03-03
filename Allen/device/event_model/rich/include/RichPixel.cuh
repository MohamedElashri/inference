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

namespace Allen::Rich::PixelReco {
  class Pixel {
  public:
    // Default constructor
    __host__ __device__ Pixel() {}

    // 3D
    __host__ __device__ Pixel(const Point& g, const Point& l, const Decoding::SmartID& sid, float area) :
      gPos(g), lPos(l), m_smartID(sid), m_effArea(area), m_timeWindow(0.0f)
    {
      m_flags.setIsNull(false);
      m_flags.setIsInnerRegion(!sid.isLargePMT());
    }

    // 4D constructor
    __host__ __device__ Pixel(const Point& g, const Point& l, const Decoding::SmartID& sid, float area, float time) :
      gPos(g), lPos(l), m_smartID(sid), m_effArea(area), m_timeWindow(time)
    {
      m_flags.setIsNull(false);
      m_flags.setIsInnerRegion(!sid.isLargePMT());
    }

    __host__ __device__ bool isNull() const { return m_flags.isNull(); }
    __host__ __device__ bool isInnerRegion() const { return m_flags.isInnerRegion(); }
    __host__ __device__ void setIsNull(bool v) { m_flags.setIsNull(v); }
    __host__ __device__ void overrideRegions(bool inner) { m_flags.setIsInnerRegion(inner); }

    __host__ __device__ inline auto rich() const { return m_smartID.rich(); }
    __host__ __device__ inline auto side() const { return m_smartID.side(); }

    __host__ __device__ auto smartID() const { return m_smartID; }
    __host__ __device__ auto gloPos() const { return gPos; }
    __host__ __device__ auto locPos() const { return lPos; }
    __host__ __device__ auto effArea() const { return m_effArea; }
    __host__ __device__ auto timeWindow() const { return m_timeWindow; }

    __host__ std::string toString()
    {
      std::stringstream ss;
      ss << m_smartID.key() << ',' << gPos.x << ',' << gPos.y << ',' << gPos.z << ',' << lPos.x << ',' << lPos.y << ','
         << lPos.z << ',' << m_effArea << ',' << (int) smartID().rich() << ',' << (int) smartID().side() << ','
         << m_timeWindow << ',' << isInnerRegion() << ',' << isNull() << '\n';
      return ss.str();
    }

  private:
    Point gPos {0.0f, 0.0f, 0.0f};
    Point lPos {0.0f, 0.0f, 0.0f};
    Decoding::SmartID m_smartID {};
    float m_effArea {0.0f};
    float m_timeWindow {0.0f};

    struct Flags {
      uint8_t value = 3; // all bits set to 1 by default (isInnerRegion=true, isNull=true)
      static constexpr uint8_t IsInnerRegionMask = 1 << 0;
      static constexpr uint8_t IsNullMask = 1 << 1;

      __host__ __device__ bool isInnerRegion() const { return value & IsInnerRegionMask; }

      __host__ __device__ bool isNull() const { return value & IsNullMask; }

      __host__ __device__ void setIsInnerRegion(bool v)
      {
        if (v)
          value |= IsInnerRegionMask;
        else
          value &= ~IsInnerRegionMask;
      }

      __host__ __device__ void setIsNull(bool v)
      {
        if (v)
          value |= IsNullMask;
        else
          value &= ~IsNullMask;
      }
    } m_flags;
  };
} // namespace Allen::Rich::PixelReco
