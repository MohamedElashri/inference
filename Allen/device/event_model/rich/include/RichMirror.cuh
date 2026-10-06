/*****************************************************************************\
 * (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
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
#include <algorithm>

namespace Allen::Rich {
  struct Mirror {
    // Mirror CoC
    Point centreOfCurvature {};
    // Mirror RoC
    float radiusOfCurvature {};
    // Mirror centre point
    float2 mirrorCentre {};
  };

  struct TwoSegmentXFinder {
    __host__ void init()
    {
      if (m_mirrors[0].mirrorCentre.x > m_mirrors[1].mirrorCentre.x) {
        std::swap(m_mirrors[0], m_mirrors[1]);
      }
    }

    __device__ const auto& find(float2 P) const { return P.x < 0 ? m_mirrors[0] : m_mirrors[1]; }

    std::array<Mirror, 2> m_mirrors {};
  };

  template<unsigned n_mirrors, unsigned n_binsX, unsigned n_binsY, float eSize>
  struct LookupTableMirrorFinder {
    __host__ void init()
    {
      // sort mirrors by distance from beam line (proxy for most used)
      std::sort(
        m_mirrors.begin(),
        m_mirrors.end(), //
        [](const auto& m1, const auto& m2) {
          return (
            (std::pow(m1.mirrorCentre.x, 2) +  //
             std::pow(m1.mirrorCentre.y, 2)) < //
            (std::pow(m2.mirrorCentre.x, 2) +  //
             std::pow(m2.mirrorCentre.y, 2)));
        });

      // Loop over the mirrors to find the (x,y) range
      for (const auto& m : m_mirrors) {
        const auto& mC = m.mirrorCentre;
        if (mC.x > m_maxX) {
          m_maxX = mC.x;
        }
        if (mC.x < m_minX) {
          m_minX = mC.x;
        }
        if (mC.y > m_maxY) {
          m_maxY = mC.y;
        }
        if (mC.y < m_minY) {
          m_minY = mC.y;
        }
      }

      // Add/Subtract a little to take the extent of the mirrors into account
      m_maxX += eSize;
      m_maxY += eSize;
      m_minX -= eSize;
      m_minY -= eSize;

      // 1 / X and Y increments ( inverse, for speed in lookup )
      m_incX = (float) n_binsX / (m_maxX - m_minX);
      m_incY = (float) n_binsY / (m_maxY - m_minY);

      // initialise the look up table
      for (auto iY = 0u; iY < n_binsY; ++iY) {
        for (auto iX = 0u; iX < n_binsX; ++iX) {
          // get (x,y) coords
          const float x = m_minX + (((float) iX + 0.5f) / m_incX);
          const float y = m_minY + (((float) iY + 0.5f) / m_incY);
          // find the mirror for this bin
          float min_d = 9e30;
          unsigned min_i = 0;
          for (unsigned i = 0; i < m_mirrors.size(); i++) {
            const auto& m = m_mirrors[i];
            const float dx = x - m.mirrorCentre.x;
            const float dy = y - m.mirrorCentre.y;
            const float d = dx * dx + dy * dy;
            if (min_d > d) {
              min_d = d;
              min_i = i;
            }
          }
          m_lookupTable[iY][iX] = min_i;
        }
      }
    }

    __device__ const auto& find(float2 P) const { return m_mirrors[m_lookupTable[yindex(P.y)][xindex(P.x)]]; }

    __device__ unsigned xindex(float x) const
    {
      if (std::isnan(x) || x <= m_minX) return 0;
      if (x >= m_maxX) return n_binsX - 1;
      const auto index = static_cast<unsigned>((x - m_minX) * m_incX);
      return index < n_binsX ? index : n_binsX - 1;
    }

    __device__ unsigned yindex(float y) const
    {
      if (std::isnan(y) || y <= m_minY) return 0;
      if (y >= m_maxY) return n_binsY - 1;
      const auto index = static_cast<unsigned>((y - m_minY) * m_incY);
      return index < n_binsY ? index : n_binsY - 1;
    }

    float m_minX {9e9};  ///< Minimum X
    float m_maxX {-9e9}; ///< Maximum X
    float m_minY {9e9};  ///< Minimum Y
    float m_maxY {-9e9}; ///< Maximum Y
    float m_incX {0};    ///< 1 / Increment in X
    float m_incY {0};    ///< 1 / Increment in Y

    std::array<Mirror, n_mirrors> m_mirrors {};
    std::array<std::array<uint8_t, n_binsX>, n_binsY> m_lookupTable {};
  };
} // namespace Allen::Rich
