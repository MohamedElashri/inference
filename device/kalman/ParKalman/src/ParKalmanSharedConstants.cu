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
#include "ParKalmanSharedConstants.cuh"
#include "AlgorithmTypes.cuh"

namespace parkalman_shared {
  using namespace ParKalmanFilter;

  // Define shared constant memory arrays
  __constant__ float dev_UT_lay[nSetsUTLayer * nParsUTLayer];
  __constant__ float dev_T_lay[nSetsTLayer * nParsTLayer];
  __constant__ float dev_UTT_META[nSetsUTTMETA * nParsUTTMETA];
  __constant__ float dev_V_pars[nSetsV * nParsV];
  __constant__ float dev_VUT_pars[nSetsVUT * nParsVUT];
  __constant__ float dev_UT_pars[nSetsUT * nParsUT];
  __constant__ float dev_UTTF_pars[nSetsUTTF * nParsUTTF];
  __constant__ float dev_T_pars[nSetsT * nParsT];
  __constant__ float dev_TFT_pars[nSetsTFT * nParsTFT];

  void update_shared_constants(const Constants& constants)
  {
    if (constants.host_UT_Layers == nullptr) {
      throw std::runtime_error("host_UT_Layers is empty and thus likely all constant ParKF parameters are empty");
    }
    Allen::memcpyToSymbol(dev_UT_lay, constants.host_UT_Layers, (nSetsUTLayer * nParsUTLayer) * sizeof(float));
    Allen::memcpyToSymbol(dev_T_lay, constants.host_T_Layers, (nSetsTLayer * nParsTLayer) * sizeof(float));
    Allen::memcpyToSymbol(dev_UTT_META, constants.host_UTT_META, (nSetsUTTMETA * nParsUTTMETA) * sizeof(float));
    Allen::memcpyToSymbol(dev_V_pars, constants.host_VP_pars, (nSetsV * nParsV) * sizeof(float));
    Allen::memcpyToSymbol(dev_VUT_pars, constants.host_VPUT_pars, (nSetsVUT * nParsVUT) * sizeof(float));
    Allen::memcpyToSymbol(dev_UT_pars, constants.host_UT_pars, (nSetsUT * nParsUT) * sizeof(float));
    Allen::memcpyToSymbol(dev_UTTF_pars, constants.host_UTTF_pars, (nSetsUTTF * nParsUTTF) * sizeof(float));
    Allen::memcpyToSymbol(dev_T_pars, constants.host_T_pars, (nSetsT * nParsT) * sizeof(float));
    Allen::memcpyToSymbol(dev_TFT_pars, constants.host_TFT_pars, (nSetsTFT * nParsTFT) * sizeof(float));
  }

} // namespace parkalman_shared
