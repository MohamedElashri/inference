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
#include <States.cuh>

namespace Extrapolators {
  constexpr float c_light = 2.99792458e+8 * 1000. / 1.e+9; // 299.792458 (float=299.792449951171875)
  constexpr float eplus = 1.;

  struct State {
    float x {0}, y {0}, z {0}, tx {0}, ty {0}, qop {0};

    __device__ State() = default;

    __device__ State(float _x, float _y, float _z, float _tx, float _ty, float _qop) :
      x(_x), y(_y), z(_z), tx(_tx), ty(_ty), qop(_qop)
    {}

    __device__ State(const SimpleKalmanState& s) : x(s.x), y(s.y), z(s.z), tx(s.tx), ty(s.ty), qop(s.qop) {}

    __device__ float3 pos() const { return {x, y, z}; }

    __device__ float3 dir() const
    {
      float l = sqrtf(tx * tx + ty * ty + 1.f);
      return {tx / l, ty / l, 1.f / l};
    }

    __device__ float px() const { return (tx / fabsf(qop)) / sqrtf(1.0f + tx * tx + ty * ty); }

    __device__ float py() const { return (ty / fabsf(qop)) / sqrtf(1.0f + tx * tx + ty * ty); }

    __device__ float pz() const { return (1.0f / fabsf(qop)) / sqrtf(1.0f + tx * tx + ty * ty); }

    __device__ float pt() const
    {
      const float sumt2 = tx * tx + ty * ty;
      return (sqrtf(sumt2) / fabsf(qop)) / sqrtf(1.0f + sumt2);
    }

    __device__ float p() const { return 1.0f / fabsf(qop); }

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
      const auto ax =
        norm * (v.ty * (v.tx * B.x + B.z) - (1.f + tx2) * B.y) * Extrapolators::c_light * Extrapolators::eplus;
      const auto ay =
        norm * (-v.tx * (v.ty * B.y + B.z) + (1.f + ty2) * B.x) * Extrapolators::c_light * Extrapolators::eplus;
      return {v.tx, v.ty, 1.f, v.qop * ax, v.qop * ay};
    }

    __device__ friend State operator+(const State& a, const Derivative& b)
    {
      return {a.x + b.dx, a.y + b.dy, a.z + b.dz, a.tx + b.dtx, a.ty + b.dty, a.qop};
    }

    __host__ friend inline std::ostream& operator<<(std::ostream& os, const State& s)
    {
      os << "{ x: " << s.x << " y: " << s.y << " z: " << s.z << " tx: " << s.tx << " ty: " << s.ty << " qop: " << s.qop
         << " }";
      return os;
    }
  };
} // namespace Extrapolators
