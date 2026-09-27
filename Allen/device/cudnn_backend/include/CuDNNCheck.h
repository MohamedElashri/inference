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

#include "Common.h"
#include <cudnn.h>
#include <string>

// Throws StrException with the statement, cuDNN's message and the location
// when a cuDNN call does not succeed.
#define ALLEN_CUDNN_CHECK(stmt)                                                                                      \
  do {                                                                                                               \
    const cudnnStatus_t allen_cudnn_status_ = (stmt);                                                                \
    if (allen_cudnn_status_ != CUDNN_STATUS_SUCCESS) {                                                               \
      throw StrException(                                                                                            \
        std::string("cuDNN: ") + #stmt + ": " + cudnnGetErrorString(allen_cudnn_status_) + " at " + __FILE__ + ":" + \
        std::to_string(__LINE__));                                                                                   \
    }                                                                                                                \
  } while (0)
