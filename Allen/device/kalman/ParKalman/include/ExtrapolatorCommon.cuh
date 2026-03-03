/*****************************************************************************\
* (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include <MagneticField.cuh>

namespace Extrapolators {
  constexpr float c_light = 2.99792458e+8 * 1000. / 1.e+9; // 299.792458 (float=299.792449951171875)
  constexpr float eplus = 1.;

  struct State {
    float x {0}, y {0}, z {0}, tx {0}, ty {0}, qop {0};

    struct Derivative {
      float dx, dy, dz, dtx, dty;
      __device__ friend Derivative operator*(const Derivative& a, float b)
      {
        return {a.dx * b, a.dy * b, a.dz * b, a.dtx * b, a.dty * b};
      }
    };

    struct Error {
      float x {0}, y {0}, tx {0}, ty {0};
      __device__ void clear()
      {
        x = 0;
        y = 0;
        tx = 0;
        ty = 0;
      }
      __device__ friend Error operator+(const Error& a, const Derivative& b)
      {
        return {a.x + b.dx, a.y + b.dy, a.tx + b.dtx, a.ty + b.dty};
      }
    };

    // dState / Dz
    __device__ friend Derivative derivative(const State& v, const float3& B)
    {
      const auto tx2 = v.tx * v.tx;
      const auto ty2 = v.ty * v.ty;
      const auto norm = sqrtf(1.f + tx2 + ty2);
      const auto ax = norm * (v.ty * (v.tx * B.x + B.z) - (1.f + tx2) * B.y);
      const auto ay = norm * (-v.tx * (v.ty * B.y + B.z) + (1.f + ty2) * B.x);
      return {v.tx, v.ty, 1.f, v.qop * ax, v.qop * ay};
    }

    __device__ friend State operator+(const State& a, const Derivative& b)
    {
      return {a.x + b.dx, a.y + b.dy, a.z + b.dz, a.tx + b.dtx, a.ty + b.dty, a.qop};
    }
  };
} // namespace Extrapolators
