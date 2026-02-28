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
} // namespace ParKalmanFilter
