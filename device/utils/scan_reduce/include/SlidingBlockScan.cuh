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

#include "CUDABackend.h"
#include <cuda/std/tuple>
#include "ScanReduce.cuh"

namespace ScanReduce {
  namespace SlidingBlockScan {
    template<BinaryOperator OPERATOR, Numeric T>
    struct BlockPrefixCallback {
      __device__ BlockPrefixCallback() = default;

      __inline__ __device__ cuda::std::tuple<T, T> operator()(const T block_aggregate, OPERATOR function, const T init)
      {

        // exclusive_prefix
        T carry_in = init;
        // inclusive_prefix
        T carry_out = function(block_aggregate, carry_in);

        return cuda::std::make_tuple(carry_in, carry_out);
      }
    }; // struct BlockPrefixCallback
  }    // namespace SlidingBlockScan
} // namespace ScanReduce
