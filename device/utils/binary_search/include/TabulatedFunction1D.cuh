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

namespace Allen {
  struct TabulatedFunction1D {
    __device__ TabulatedFunction1D(const float* x, const float* y, unsigned nbins) : m_x(x), m_y(y), m_nbins(nbins) {}

    __device__ float value(float x) const
    {
      unsigned bin = binary_search_leftmost(m_x, m_nbins, x);
      // TODO: check bounds
      float r = (x - m_x[bin]) / (m_x[bin + 1] - m_x[bin]);
      return m_y[bin] + r * (m_y[bin + 1] - m_y[bin]);
    }

  private:
    const float *m_x, *m_y;
    const unsigned m_nbins;
  };
} // namespace Allen
