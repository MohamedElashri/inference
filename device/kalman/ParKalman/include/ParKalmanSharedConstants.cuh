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

#include "ParKalmanDefinitions.cuh"
#include "BackendCommon.h"

struct Constants;

namespace parkalman_shared {
  using namespace ParKalmanFilter;

  // Shared constant memory arrays for Kalman filter parametrizations.
  // Used by kalman_filter; downstream_kalman_filter uses a subset (excludes V and VUT).
  // Defined in ParKalmanSharedConstants.cu
  extern __constant__ float dev_UT_lay[nSetsUTLayer * nParsUTLayer];
  extern __constant__ float dev_T_lay[nSetsTLayer * nParsTLayer];
  extern __constant__ float dev_UTT_META[nSetsUTTMETA * nParsUTTMETA];
  extern __constant__ float dev_V_pars[nSetsV * nParsV];
  extern __constant__ float dev_VUT_pars[nSetsVUT * nParsVUT];
  extern __constant__ float dev_UT_pars[nSetsUT * nParsUT];
  extern __constant__ float dev_UTTF_pars[nSetsUTTF * nParsUTTF];
  extern __constant__ float dev_T_pars[nSetsT * nParsT];
  extern __constant__ float dev_TFT_pars[nSetsTFT * nParsTFT];

  // Update function to copy constants from host to device.
  void update_shared_constants(const Constants& constants);

} // namespace parkalman_shared
