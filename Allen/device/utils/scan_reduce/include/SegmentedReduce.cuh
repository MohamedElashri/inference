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

#include "ScanReduce.cuh"

namespace ScanReduce {
  namespace Segmented {
    template<Numeric T, BinaryOperator OPERATOR>
    __inline__ __device__ T warp_reduce_to_tail(T value, int tail_flag, OPERATOR function)
    {

      const unsigned mask = __ballot_sync(FULL_MASK, tail_flag);
      // result of __ffs is the least significant bit set to zero (i + 1), which is equivalent to width
      // the least significant lane with tail flag is also included in the reduce
      // if none of the threads have tail_flag, mask is 0, so we reduce the entire warp
      const int width = (mask) ? __ffs(mask) : warp_size;

      const int predicate = threadIdx.x < width;
      const unsigned shuffle_mask = __ballot_sync(FULL_MASK, predicate);

      T reduce = value;
      if (predicate) {
        // printf("BlockIdx.x: %u, ThreadIdx.x: %u, value: %u, tail_flag: %d\n", blockIdx.x, threadIdx.x, value,
        // tail_flag);
        for (int delta = 1; delta < width; delta <<= 1) {
          const T temp = __shfl_down_sync(shuffle_mask, reduce, delta);
          if ((delta + threadIdx.x) < width) reduce = function(temp, reduce);
        }
      }
      return reduce;
    }
  } // namespace Segmented
} // namespace ScanReduce
