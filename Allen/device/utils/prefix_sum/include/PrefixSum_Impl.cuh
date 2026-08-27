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

#ifndef TARGET_DEVICE_CPU

#include "SlidingBlockScan.cuh"
#include "SinglePassScan.cuh"
#include "Scan.cuh"

namespace PrefixSum {
  // Place template here so we can test different settings
  template<unsigned THREADS_PER_BLOCK, unsigned VALUES_PER_THREAD>
  __inline__ __device__ void sliding_block_scan(unsigned* dev_array, const unsigned array_size, unsigned* host_total)
  {

    using BinaryOp = ScanReduce::Plus<unsigned>;
    using CallbackOp = ScanReduce::SlidingBlockScan::BlockPrefixCallback<BinaryOp, unsigned>;
    BinaryOp function;
    CallbackOp block_prefix_callback;

    unsigned carry = 0;
    const unsigned values_per_block = THREADS_PER_BLOCK * VALUES_PER_THREAD;
    const unsigned iterations = (array_size + values_per_block - 1) / values_per_block;

    for (unsigned i = 0; i < iterations; i++) {
      const unsigned elements_to_process = min(values_per_block, array_size - i * values_per_block);
      unsigned* array_ptr = dev_array + i * values_per_block;
      carry = block_scan<ScanReduce::ScanType::Exclusive, THREADS_PER_BLOCK, VALUES_PER_THREAD>(
        array_ptr, elements_to_process, array_ptr, function, carry, block_prefix_callback);
    }

    if (threadIdx.x == 0) {
      dev_array[array_size] = carry;
      if (host_total != nullptr) *host_total = carry;
    }
  }

  template<unsigned THREADS_PER_BLOCK, unsigned VALUES_PER_THREAD>
  __global__ void prefix_sum_single_block(unsigned* dev_array, const unsigned array_size, unsigned* host_total)
  {
    sliding_block_scan<THREADS_PER_BLOCK, VALUES_PER_THREAD>(dev_array, array_size, host_total);
  }

  template<unsigned THREADS_PER_BLOCK, unsigned VALUES_PER_THREAD>
  __inline__ __device__ void
  chain_scan(unsigned* dev_array, const unsigned array_size, TileStoreType* tile_state_data, unsigned* host_total)
  {

    using BinaryOp = ScanReduce::Plus<unsigned>;
    using CallbackOp = ScanReduce::SinglePassScan::BlockPrefixCallback<BinaryOp, unsigned>;
    BinaryOp function;
    CallbackOp block_prefix_callback(tile_state_data);

    const unsigned values_per_block = THREADS_PER_BLOCK * VALUES_PER_THREAD;
    unsigned* array_ptr = dev_array + blockIdx.x * values_per_block;
    const unsigned elements_to_process = min(values_per_block, array_size - blockIdx.x * values_per_block);
    const unsigned inclusive_sum =
      ScanReduce::block_scan<ScanReduce::ScanType::Exclusive, THREADS_PER_BLOCK, VALUES_PER_THREAD>(
        array_ptr, elements_to_process, array_ptr, function, 0u, block_prefix_callback);

    if ((blockIdx.x == (gridDim.x - 1)) && threadIdx.x == 0) {
      dev_array[array_size] = inclusive_sum;
      if (host_total != nullptr) *host_total = inclusive_sum;
    }
  }

} // namespace PrefixSum
#endif
