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

#include "BackendCommon.h"
#include "States.cuh"
#include "ParKalmanMath.cuh"

namespace ParKalmanFilter {

  typedef Vector<10> Vector10;
  typedef Vector<5> Vector5;
  typedef Vector<2> Vector2;
  typedef SquareMatrix<true, 5> SymMatrix5x5;
  typedef SquareMatrix<true, 4> SymMatrix4x4;
  typedef SquareMatrix<true, 2> SymMatrix2x2;
  typedef SquareMatrix<false, 5> Matrix5x5;

  // Max number of bins for the UT <-> SciFi extrapolation.
  constexpr int nBinXMax = 60;
  constexpr int nBinYMax = 50;

  // Size of the StandardCoefs
  constexpr int DEGx1 = 7;
  constexpr int DEGx2 = 9;
  constexpr int DEGy1 = 5;
  constexpr int DEGy2 = 7;

  // Number of velo parameters.
  constexpr int nParsV = 6;
  constexpr int nSetsV = 2;

  // Number of velo-UT parameters.
  constexpr int nParsVUT = 15;
  constexpr int nSetsVUT = 1;

  // Number of UT parameters.
  constexpr int nParsUT = 12;
  constexpr int nSetsUT = 6;

  // Number of UTTF parameters.
  constexpr int nParsUTTF = 20;
  constexpr int nSetsUTTF = 1;

  // Number of UTTMETA parameters.
  constexpr int nParsUTTMETA = 19;
  constexpr int nSetsUTTMETA = 1;

  // Number of TFT parameters.
  constexpr int nParsTFT = 4;
  constexpr int nSetsTFT = 1;

  // Number of T parameters.
  constexpr int nParsT = 12;
  constexpr int nSetsT = 22;

  // Number of TLayer parameters.
  constexpr int nParsTLayer = 12;
  constexpr int nSetsTLayer = 4;

  // Number of UTLayer parameters.
  constexpr int nParsUTLayer = 4;
  constexpr int nSetsUTLayer = 1;

  // SciFi Geo constants
  constexpr float Approx_dy = 2424.0f;         // TODO simplified SCIFI GEO
  constexpr float Approx_BeamHole_dy = 116.5f; // TODO simplified SCIFI GEO

  // Parameters for beamline propagation and VELO-only Kalman Filter.
  static constexpr float rffoilscatter = 0.6;

  static constexpr float scatterSensorParameter_VPHit2VPHit_cms = 1.48;
  static constexpr float scatterSensorParameter_VPHit2VPHit_etaxx = 0.643;
  static constexpr float scatterSensorParameter_VPHit2VPHit_etaxtx = 0.526;
  static constexpr float scatterSensorParameter_VPHit2VPHit_Eloss = 0.592;

  static constexpr float scatterSensorParameter_VPHit2ClosestToBeam_cms = 2.91;
  static constexpr float scatterSensorParameter_VPHit2ClosestToBeam_etaxx = 0.808;
  static constexpr float scatterSensorParameter_VPHit2ClosestToBeam_etaxtx = 0.793;
  static constexpr float scatterSensorParameter_VPHit2ClosestToBeam_Eloss = 1.29;

  // Some hardcoded parameters for the Rich prototype
  // TODO: move definitions to ParamFiles
  constexpr KalmanFloat RICH1_F_zTo = (KalmanFloat) 990.f;
  constexpr KalmanFloat RICH1_B_zTo = (KalmanFloat) 2165.f;
  [[maybe_unused]] __constant__ constexpr float PAR_RICH1_F[15] = {
    -0.27461,
    0.02130,
    0.13020,
    0.63412,
    0.00629,
    0.66392,
    -0.57832,
    -0.12243,
    0.11905,
    0.67957,
    -0.04346,
    41.12807,
    -0.36773,
    0.01405,
    0.13020};
  [[maybe_unused]] __constant__ constexpr float PAR_RICH1_B[15] = {
    -1.79468,
    0.09097,
    0.13020,
    0.92482,
    0.01176,
    2.25478,
    -0.59956,
    -1.26963,
    0.14044,
    1.41178,
    0.00099,
    27.52532,
    -2.78371,
    0.03879,
    0.13020};
  constexpr KalmanFloat RICH2_F_zTo = (KalmanFloat) 9450.f;
  constexpr KalmanFloat RICH2_B_zTo = (KalmanFloat) 11900.f;
  [[maybe_unused]] __constant__ constexpr float PAR_RICH2_F[1 * 12] =
    {0.51594, 0.92742, 0.50147, 0.90342, -0.23370, -4.42730, 0.05454, -0.62828, 1.27656, 0.87818, 1.24467, 0.88083};
  [[maybe_unused]] __constant__ constexpr float PAR_RICH2_B[1 * 12] =
    {0.32917, 1.10458, 0.00001, 0.44354, -0.09239, -1.10418, 0.02300, -2.05242, 1.20123, 0.87818, 0.71955, 0.88083};
} // namespace ParKalmanFilter
