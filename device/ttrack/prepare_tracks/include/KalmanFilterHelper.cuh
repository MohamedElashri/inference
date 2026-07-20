/*****************************************************************************\
* (c) Copyright 2025 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "LinearAlgebra.cuh"
#include "RungeKuttaExtrapolator.cuh"

namespace TTracks {

  namespace KalmanFilter {

    using RK = Extrapolators::RungeKuttaNystromExtrapolator;

    template<unsigned N, typename T = float>
    struct SymmMatrix {
      T m_data[N * (N + 1) / 2];

      __device__ inline T& operator()(short i, short j)
      {
        if (i > j) std::swap(i, j);
        return m_data[j * (j + 1) / 2 + i];
      }
    };

    template<unsigned N, typename T = float>
    struct SquareMatrix {
      T m_data[N * N];
      __device__ inline T& operator()(short i, short j) { return m_data[i * N + j]; }
    };

    template<unsigned N, typename T = float>
    struct Vector {
      T m_data[N];
      __device__ inline T& operator()(short i) { return m_data[i]; }
    };

    /*
        Propagate the covariance matrix using the Jacobian: Cnew = J * Cold * J^T
    */
    __device__ inline void propagateCovMatrix(SymmMatrix<5>& cov, SquareMatrix<5>& jacobian)
    {
      SymmMatrix<5> new_cov;

      UNROLL(5)
      for (short i = 0; i < 5; i++) {
        UNROLL(5)
        for (short j = 0; j <= i; j++) {
          float sum = 0.f;
          UNROLL(5)
          for (short k = 0; k < 5; k++) {
            UNROLL(5)
            for (short l = 0; l < 5; l++) {
              sum += jacobian(i, k) * cov(k, l) * jacobian(j, l);
            }
          }
          new_cov(i, j) = sum;
        }
      }

      cov = new_cov;
    }

    // C = (I - K * H) * Cprop, where H = [1 dxdy 0 0 0], K is a gain vector and Cprop is the propagated covariance
    // matrix (Cprop = J * Cold * J^T)
    __device__ inline void updateCovMatrix(Vector<5>& gain, SymmMatrix<5>& cov, float hit_dxdy)
    {
      Vector<5> transform = {
        cov(0, 0) - cov(0, 1) * hit_dxdy,
        cov(0, 1) - cov(1, 1) * hit_dxdy,
        cov(0, 2) - cov(1, 2) * hit_dxdy,
        cov(0, 3) - cov(1, 3) * hit_dxdy,
        cov(0, 4) - cov(1, 4) * hit_dxdy};

      UNROLL(5)
      for (short i = 0; i < 5; i++) {
        UNROLL(5)
        for (short j = i; j < 5; j++) {
          cov(i, j) -= transform(j) * gain(i);
        }
      }
    }

    /*
        Make a Kalman filter step to update the state and covariance matrix with a hit measurement at (hit_x0, hit_z0)
       with uncertainty hit_x0_cov.
    */
    __device__ void makeStep(
      Extrapolators::State& state,
      SymmMatrix<5>& cov,
      float hit_x0,
      float hit_x0_cov,
      float hit_z0,
      float hit_dxdy,
      const MagneticField::Magfield& field)
    {

      // Predict the next state and covariance matrix using the Runge-Kutta-Nystrom extrapolator
      SquareMatrix<5> jacobian;
      RK::make_fast_step_and_evaluate_jacobian(state, jacobian, hit_z0 - state.z, field);
      propagateCovMatrix(cov, jacobian);

      // Compare the predicted state with the measurement
      const float dx = hit_x0 - (state.x - hit_dxdy * state.y);

      // Evaluate the Gain vector
      const float gain_denominator =
        1.f / (cov(0, 0) - 2.f * cov(0, 1) * hit_dxdy + cov(1, 1) * hit_dxdy * hit_dxdy + hit_x0_cov);
      Vector<5> gain {
        (cov(0, 0) - cov(0, 1) * hit_dxdy) * gain_denominator,
        (cov(0, 1) - cov(1, 1) * hit_dxdy) * gain_denominator,
        (cov(0, 2) - cov(1, 2) * hit_dxdy) * gain_denominator,
        (cov(0, 3) - cov(1, 3) * hit_dxdy) * gain_denominator,
        (cov(0, 4) - cov(1, 4) * hit_dxdy) * gain_denominator};

      // Update the covariance matrix
      updateCovMatrix(gain, cov, hit_dxdy);

      // Update the state
      state.x += gain(0) * dx;
      state.y += gain(1) * dx;
      state.tx += gain(2) * dx;
      state.ty += gain(3) * dx;
      state.qop += gain(4) * dx;
    }
  } // namespace KalmanFilter
} // namespace TTracks
