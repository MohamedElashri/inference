/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "TrackTrajectory.cuh"
#include "LinearAlgebra.cuh"
#include "SystemOfUnits.h"
#include "FloatOperations.cuh"

// This algorithm is a simplified version of the LHCb TrajPoca:
// https://gitlab.cern.ch/lhcb/LHCb/-/blob/master/Kernel/LHCbAlgs/src/TrajPoca.cpp

namespace Trajectories {

  namespace Constants = TTracks::Constants;

  struct POCA {
    float3 pos;
    float doca2 = std::numeric_limits<float>::infinity();
    float muA, muB;

    __device__ operator bool() const
    {
      return std::isfinite(doca2) && std::abs(pos.x) < 10000.f && std::abs(pos.y) < 10000.f &&
             std::abs(pos.z) < 20000.f;
    }
  };

  struct Step {
    float muA = std::numeric_limits<float>::quiet_NaN(), muB = std::numeric_limits<float>::quiet_NaN();
    float doca2 = std::numeric_limits<float>::infinity();
    bool finished = false;

    __device__ operator bool() const { return std::isfinite(muA) && std::isfinite(muB) && std::isfinite(doca2); }

    __device__ bool inRange(float min_mu, float max_mu) const
    {
      return (fminf(muA, muB) > min_mu) && (fmaxf(muA, muB) < max_mu);
    }
  };

  template<uint8_t maxIterations = 5>
  class TrajectoryPoca {
  public:
    __device__ TrajectoryPoca(
      float maxDist = 100000000.f,
      float trajTolerance = 1.f * Allen::Units::mm,
      float trajMaxDistToErr = 1000.f * Allen::Units::mm) :
      m_maxDist(maxDist),
      m_trajTolerance(trajTolerance), m_trajMaxDistToErr(trajMaxDistToErr)
    {}

    template<typename SplineA, typename SplineB>
    __device__ POCA minimize(SplineA& proxyA, SplineB& proxyB, float mu1, float mu2, float min_mu, float max_mu) const
    {

      TrajectoryExpansion exp1, exp2;

      Step prevstep {.muA = mu1, .muB = mu2}, step;
      float prevprevdelta2 = std::numeric_limits<float>::infinity();

      for (uint8_t istep = 0; istep < maxIterations; ++istep) {
        exp1 = proxyA.expansion(prevstep.muA, m_trajTolerance, m_trajMaxDistToErr);
        exp2 = proxyB.expansion(prevstep.muB, m_trajTolerance, m_trajMaxDistToErr);
        step = stepTowardPoca(exp1, exp2, prevstep.muA, prevstep.muB);

        if (
          !step || (!step.inRange(min_mu, max_mu) && !prevstep.inRange(min_mu, max_mu)) ||
          (step.doca2 >= prevstep.doca2 && prevstep.doca2 >= prevprevdelta2)) {
          step.doca2 = std::numeric_limits<float>::infinity();
          break;
        }
        if ((step.finished && prevstep.finished)) break;

        // we have to catch some problematic cases
        if (!step.finished && istep > 2 && step.doca2 > prevstep.doca2 && prevstep.doca2 < prevprevdelta2) {
          // we might be oscillating, but we could also just have
          // stepped over the minimum. choose a solution `in
          // between'.
          step.muA = 0.5f * (prevstep.muA + step.muA);
          step.muB = 0.5f * (prevstep.muB + step.muB);
          step.doca2 = mag2(proxyA.position(step.muA) - proxyB.position(step.muB));
        }

        prevprevdelta2 = prevstep.doca2;
        prevstep = step;
      }

      return POCA {
        .pos = (proxyA.position(step.muA) + proxyB.position(step.muB)) * 0.5f,
        .doca2 = step.doca2,
        .muA = step.muA,
        .muB = step.muB,
      };
    }

  private:
    __device__ Step
    stepTowardPoca(const TrajectoryExpansion& exp1, const TrajectoryExpansion& exp2, float mu1, float mu2) const
    {

      const auto d = exp1.p - exp2.p;
      // if the distance between points is below 1e-4 mm, we consider the
      // minimisation to be converged
      constexpr float tolerance_threshold = 1e-4f * Allen::Units::mm;
      if (mag2(d) < tolerance_threshold * tolerance_threshold)
        return Step {
          .muA = mu1,
          .muB = mu2,
          .doca2 = mag2(d),
          .finished = true,
        };
      const float mat[3] = {
        // keep all terms up to order mu1, mu2, mu1 * mu2
        mag2(exp1.dpdmu) + dot(d, exp1.d2pdmu2),
        -dot(exp1.dpdmu, exp2.dpdmu),
        mag2(exp2.dpdmu) - dot(d, exp2.d2pdmu2),
      };
      const float rhs[2] = {-dot(d, exp1.dpdmu), dot(d, exp2.dpdmu)};

      float dmu[2];
      if (!LinAlg::SolveCholesky<2>(mat, rhs, dmu)) return Step {};

      constexpr float smudge = 1.01f; // Factor to push just over border of piecewise traj (essential!)
      const auto distToErr1 = smudge * exp1.distTo2ndError;
      const auto distToErr2 = smudge * exp2.distTo2ndError;

      // Factor to push just over border of piecewise traj (essential!)
      if (0 < distToErr1 && distToErr1 < fabsf(dmu[0])) {
        // choose solution for which dmu[0] steps just over border
        dmu[0] = distToErr1 * sign(dmu[0]);
        // now recalculate dmu[1], given dmu[0]:
        dmu[1] = (rhs[1] - dmu[0] * mat[1]) / mat[2];
      }

      if (0 < distToErr2 && distToErr2 < fabsf(dmu[1])) {
        // choose solution for which dmu[1] steps just over border
        dmu[1] = distToErr2 * sign(dmu[1]);
        // now recalculate dmu[0], given dmu[1]:
        dmu[0] = (rhs[0] - dmu[1] * mat[1]) / mat[0];
        // if still not okay,
        if (0 < distToErr1 && distToErr1 < fabsf(dmu[0])) {
          dmu[0] = distToErr1 * sign(dmu[0]);
        }
      }

      mu1 += dmu[0], mu2 += dmu[1];

      // another check for parallel trajectories
      if (min(fabsf(mu1), fabsf(mu2)) > m_maxDist) return Step {};

      // printf()

      return Step {
        .muA = mu1,
        .muB = mu2,
        .doca2 = mag2(
          exp1.p + exp1.dpdmu * dmu[0] + exp1.d2pdmu2 * dmu[0] * dmu[0] -
          (exp2.p + exp2.dpdmu * dmu[1] + exp2.d2pdmu2 * dmu[1] * dmu[1])),
        .finished = (fabsf(dmu[0]) < exp1.distTo1stError) && (fabsf(dmu[1]) < exp2.distTo1stError),
      };
    }

    __host__ __device__ inline float dot(const float3& a, const float3& b) const
    {
      return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ inline float mag2(const float3& v) const { return dot(v, v); }

    template<typename T>
    __host__ __device__ inline T sign(const T& v) const
    {
      return (v < T(0)) ? T(-1) : T(1);
    }

    float m_maxDist = 100000000.f;
    float m_trajTolerance = 1.f * Allen::Units::mm;
    float m_trajMaxDistToErr = 1000.f * Allen::Units::mm;
  };
} // namespace Trajectories
