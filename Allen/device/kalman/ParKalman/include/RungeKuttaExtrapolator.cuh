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

#include <BackendCommon.h>
#include <ButcherTableau.cuh>
#include <ExtrapolatorCommon.cuh>
#include <MagneticField.cuh>
#include <FloatOperations.cuh>

namespace Extrapolators {
  template<typename ftype = float, typename Table = ButcherTableau::CashKarp<ftype>>
  struct RungeKuttaExtrapolator {
    // Implementation taken from:
    // https://gitlab.cern.ch/lhcb/Rec/-/blob/master/Tr/TrackExtrapolators/src/TrackRungeKuttaExtrapolator.cpp

    __device__ static void propagate(State& state, State::Error& err, ftype dz, const MagneticField::Magfield& field)
    {
      State::Derivative k[Table::N_stages];
      UNROLL(10)
      for (int stage = 0; stage < Table::N_stages; stage++) {
        State s = state;
        UNROLL(10)
        for (int i = 0; i < stage - 1; i++) {
          s = s + k[i] * Table::a(stage, i);
        }
        float3 B = field.fieldVectorLinearInterpolation(make_float3(s.x, s.y, s.z));
        k[stage] = derivative(s, B) * dz;
      }

      err.clear();
      UNROLL(10)
      for (int i = 0; i < Table::N_stages; i++) {
        err = err + k[i] * (Table::b(i) - Table::b_star(i));
        state = state + k[i] * Table::b(i);
      }
    }
  };

  /**
   * RungeKuttaNystromExtrapolator
   *
   * Numerical extrapolator for second-order ordinary differential equations
   * of the form x'' = f(x, x', t).
   *
   * Compared to generic Runge–Kutta methods, the Nyström formulation avoids
   * reducing the system to first order, resulting in fewer function
   * evaluations and significantly better efficiency. This makes it well suited
   * for trajectory and track extrapolation in detector magnetic fields.
   */
  struct RungeKuttaNystromExtrapolator {

  private:
    // Compute the derivative of the state vector
    __device__ inline static float2 gamma(const float2 t, const float3 B)
    {
      const float norm = sqrtf(1.f + t.x * t.x + t.y * t.y);
      return make_float2(
        norm * (t.x * t.y * B.x - (1.f + t.x * t.x) * B.y + t.y * B.z),
        norm * ((1.f + t.y * t.y) * B.x - t.x * t.y * B.y - t.x * B.z));
    }

    __device__ inline static void
    fillStages(float2 k[], const float2 t, const float qop, const float3 B, const float dz)
    {
      constexpr float c[4] = {0.f, 0.5f, 0.5f, 1.f};
      k[0] = gamma(t, B) * qop;
      UNROLL(3)
      for (int stage = 1; stage < 4; stage++) {
        k[stage] = gamma(t + k[stage - 1] * (dz * c[stage]), B) * qop;
      }
    }

    /*
    Update the Jacobian matrix with the contribution from the current step
    of the Runge-Kutta-Nystrom extrapolator.
    */
    template<typename Matrix_t>
    __device__ inline static void incrementJacobian(
      Matrix_t& jacobian,
      const short stage,
      const float dz,
      const float2 tn,
      const float qop,
      const float3 B)
    {

      const float norm = sqrtf(1.f + tn.x * tn.x + tn.y * tn.y);
      const float factorY = tn.x * tn.y * B.x - (1.f + tn.x * tn.x) * B.y + tn.y * B.z;
      const float factorX = (1.f + tn.y * tn.y) * B.x - tn.x * tn.y * B.y - tn.x * B.z;

      const float a11 = qop * (norm * (tn.y * B.x - 2 * tn.x * B.y) + tn.x / norm * factorY);
      const float a12 = qop * (norm * (tn.x * B.x + B.z) + tn.y / norm * factorY);
      const float a21 = qop * (-norm * (tn.y * B.y + B.z) + tn.x / norm * factorX);
      const float a22 = qop * (norm * (2 * tn.y * B.x - tn.x * B.y) + tn.y / norm * factorX);
      const float e1 = norm * factorY;
      const float e2 = norm * factorX;

      // Not needed for the coordinate components
      if (stage != 3) {
        jacobian(0, 2) += (dz * dz / 6.f) * a11;
        jacobian(0, 3) += (dz * dz / 6.f) * a12;
        jacobian(0, 4) += (dz * dz / 6.f) * e1;
        jacobian(1, 2) += (dz * dz / 6.f) * a21;
        jacobian(1, 3) += (dz * dz / 6.f) * a22;
        jacobian(1, 4) += (dz * dz / 6.f) * e2;
      }

      const auto double_counting = (stage == 1) || (stage == 2);
      jacobian(2, 2) += (dz / 6.f) * a11 * (1.f + double_counting);
      jacobian(2, 3) += (dz / 6.f) * a12 * (1.f + double_counting);
      jacobian(2, 4) += (dz / 6.f) * e1 * (1.f + double_counting);
      jacobian(3, 2) += (dz / 6.f) * a21 * (1.f + double_counting);
      jacobian(3, 3) += (dz / 6.f) * a22 * (1.f + double_counting);
      jacobian(3, 4) += (dz / 6.f) * e2 * (1.f + double_counting);
    }

  public:
    /*
    Make a fast Runge-Kutta-Nystrom step of size dz. The magnetic field is
    evaluated only once, at the midpoint of the step. This reduces the number of
    accesses to the TEX memory
    */
    __device__ static inline void make_fast_step(State& state, float dz, const MagneticField::Magfield& field)
    {
      const float3 B = field.fieldVectorLinearInterpolation(
        make_float3(state.x + 0.5f * state.tx * dz, state.y + 0.5f * state.ty * dz, state.z + 0.5f * dz));

      const float2 tn = make_float2(state.tx, state.ty);

      float2 k[4];
      fillStages(k, tn, state.qop, B, dz);

      const float2 dRn = tn * dz + (k[0] + k[1] + k[2]) * (dz * dz * (1.f / 6.f));
      const float2 dTn = (k[0] + k[1] * 2.f + k[2] * 2.f + k[3]) * (dz * (1.f / 6.f));

      state.x += dRn.x;
      state.y += dRn.y;
      state.z += dz;
      state.tx += dTn.x;
      state.ty += dTn.y;
    }

    /*
    Make a fast Runge-Kutta-Nystrom step of size dz and evaluate the Jacobian
    matrix. The magnetic field is evaluated only once, at the midpoint of the
    step. This reduces the number of accesses to the TEX memory
    */
    template<typename Matrix_t>
    __device__ static void make_fast_step_and_evaluate_jacobian(
      State& state,
      Matrix_t& jacobian,
      float dz,
      const MagneticField::Magfield& field) requires requires(Matrix_t m, unsigned i, unsigned j)
    {
      {
        m(i, j)
      }
      ->std::convertible_to<float>;
    }
    {
      const float3 B = field.fieldVectorLinearInterpolation(
        make_float3(state.x + 0.5f * state.tx * dz, state.y + 0.5f * state.ty * dz, state.z + 0.5f * dz));

      const float2 tn = make_float2(state.tx, state.ty);

      float2 k[4];
      fillStages(k, tn, state.qop, B, dz);

      // Update state vector
      {
        const float2 dRn = tn * dz + (k[0] + k[1] + k[2]) * (dz * dz * (1.f / 6.f));
        const float2 dTn = (k[0] + k[1] * 2.f + k[2] * 2.f + k[3]) * (dz * (1.f / 6.f));
        state.x += dRn.x;
        state.y += dRn.y;
        state.z += dz;
        state.tx += dTn.x;
        state.ty += dTn.y;
      }

      // Update Jacobian matrix
      {
        jacobian(0, 0) = 1.f;
        jacobian(0, 1) = 0.f;
        jacobian(1, 0) = 0.f;
        jacobian(1, 1) = 1.f;
        jacobian(2, 0) = 0.f;
        jacobian(2, 1) = 0.f;
        jacobian(3, 0) = 0.f;
        jacobian(3, 1) = 0.f;
        jacobian(4, 0) = 0.f;
        jacobian(4, 1) = 0.f;

        jacobian(0, 2) = dz;
        jacobian(0, 3) = 0.f;
        jacobian(1, 2) = 0.f;
        jacobian(1, 3) = dz;
        jacobian(2, 2) = 1.f;
        jacobian(2, 3) = 0.f;
        jacobian(3, 2) = 0.f;
        jacobian(3, 3) = 1.f;
        jacobian(4, 2) = 0.f;
        jacobian(4, 3) = 0.f;

        jacobian(0, 4) = 0.f;
        jacobian(1, 4) = 0.f;
        jacobian(2, 4) = 0.f;
        jacobian(3, 4) = 0.f;
        jacobian(4, 4) = 1.f;

        incrementJacobian(jacobian, 0, dz, tn, state.qop, B);
        for (short stage = 1; stage < 4; stage++) {
          constexpr float c[4] = {0.f, 0.5f, 0.5f, 1.f};
          incrementJacobian(jacobian, stage, dz, tn + k[stage - 1] * c[stage] * dz, state.qop, B);
        }
      }
    }

    /*
      Propagate the state to the target z using fixed-size Runge-Kutta-Nystrom steps.
    */
    __device__ static void propagate(
      State& state,
      float target_z,
      const MagneticField::Magfield& field,
      const float step_size = 500.f,
      const int max_steps = 100)
    {
      const float dir = (target_z > state.z) ? 1.f : -1.f;

      for (int i_iter = 0; i_iter < max_steps; i_iter++) {
        const auto absstep = min(step_size, fabsf(target_z - state.z));
        make_fast_step(state, dir * absstep, field);

        if (fabsf(state.z - target_z) < 1.f) {
          break;
        }
      }
    };
  };

} // namespace Extrapolators
