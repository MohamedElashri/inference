// (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration
// Apache-2.0 License
#pragma once

#ifdef ALLEN_WITH_CUDNN
#include <cudnn.h>
#include <stdexcept>
#include <string>
#include <cstdio>

#define ALLEN_CUDNN_CHECK(stmt)                                                          \
  do {                                                                                   \
    cudnnStatus_t _allen_cudnn_err = (stmt);                                             \
    if (_allen_cudnn_err != CUDNN_STATUS_SUCCESS) {                                      \
      fprintf(stderr,                                                                     \
              "Failed to run %s\n%s (%d) at %s: %d\n",                                  \
              #stmt,                                                                      \
              cudnnGetErrorString(_allen_cudnn_err),                                     \
              static_cast<int>(_allen_cudnn_err),                                        \
              __FILE__,                                                                   \
              __LINE__);                                                                  \
      throw std::invalid_argument("ALLEN_CUDNN_CHECK failed");                           \
    }                                                                                    \
  } while (0)

#endif // ALLEN_WITH_CUDNN
