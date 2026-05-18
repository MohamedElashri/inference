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
#pragma once

#include <RichDefinitions.cuh>
#include <RichSmartID.cuh>

namespace Allen::Rich::PhotonReco {
  struct Photon {
    float ckTheta {};
    float ckPhi {};
    unsigned pixelIdx {};
  };
} // namespace Allen::Rich::PhotonReco
