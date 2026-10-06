/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
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
#include "CuDNNCheck.h"

namespace Allen::CuDNN {

  // The cuDNN handle of a CUDA stream, bound to it: one per stream, shared by
  // every algorithm on that stream, created on first use and kept for the
  // process lifetime (Allen may reset the device at teardown, so handles are
  // not destroyed from static destructors). Thread safe; lock-free after a
  // thread's first call for its stream.
  cudnnHandle_t handle(cudaStream_t stream);

  // The handle of an algorithm's stream, in operator().
  inline cudnnHandle_t handle(const Allen::Context& context) { return handle(context.stream()); }

} // namespace Allen::CuDNN
