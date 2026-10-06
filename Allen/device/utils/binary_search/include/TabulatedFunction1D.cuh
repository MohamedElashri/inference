/*****************************************************************************\
* (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <BinarySearch.cuh>
#include <limits>

namespace Allen {
  struct TabulatedFunction1D {
    __device__ TabulatedFunction1D(const float* x, const float* y, unsigned nbins) : m_x(x), m_y(y), m_nbins(nbins) {}

    __device__ float value(float x) const
    {
      if (m_nbins == 0) return std::numeric_limits<float>::quiet_NaN();
      if (m_nbins == 1 || x <= m_x[0]) return m_y[0];
      if (x >= m_x[m_nbins - 1]) return m_y[m_nbins - 1];
      if (std::isnan(x)) return x;

      const unsigned upper_bin = binary_search_leftmost(m_x, m_nbins, x);
      const unsigned lower_bin = upper_bin - 1;
      const float r = (x - m_x[lower_bin]) / (m_x[upper_bin] - m_x[lower_bin]);
      return m_y[lower_bin] + r * (m_y[upper_bin] - m_y[lower_bin]);
    }

  private:
    const float *m_x, *m_y;
    const unsigned m_nbins;
  };
} // namespace Allen
