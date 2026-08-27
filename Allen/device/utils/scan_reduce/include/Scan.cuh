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

#include "SinglePassScan.cuh"
#include "CUDABackend.h"

namespace ScanReduce {

  template<unsigned VALUES_PER_THREAD, BinaryOperator OPERATOR, Numeric T>
  __device__ T linear_scan(T (&values)[VALUES_PER_THREAD], OPERATOR function)
  {
#pragma unroll
    for (int i = 1; i < VALUES_PER_THREAD; i++) {
      values[i] = function(values[i], values[i - 1]);
    }
    return values[VALUES_PER_THREAD - 1];
  }

  template<BinaryOperator OPERATOR, Numeric T>
  __device__ T warp_scan(const T value, OPERATOR function, const unsigned mask = FULL_MASK, const int width = 32)
  {
    T scanned_value(value);
    for (int delta = 1; delta < width; delta *= 2) {
      T shuffled = __shfl_up_sync(mask, scanned_value, delta);
      if (laneid() >= delta) scanned_value = function(scanned_value, shuffled);
    }
    return scanned_value;
  }

  template<ScanType TYPE, unsigned VALUES_PER_THREAD, BinaryOperator OPERATOR, Numeric T>
  __device__ void block_downsweep(const T downsweep, OPERATOR function, T (&values)[VALUES_PER_THREAD])
  {
    // For exclusive scan, the current values in the registers are inclusive scan
    // Shift the values rightwards to get exclusive scan
    if constexpr (TYPE == ScanType::Exclusive) {
#pragma unroll
      for (int i = VALUES_PER_THREAD - 1; i > 0; i--) {
        values[i] = values[i - 1];
      }
      values[0] = OPERATOR::Identity;
    }

#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      values[i] = function(values[i], downsweep);
    }
  }

  template<unsigned BLOCK_SIZE, BinaryOperator OPERATOR, Numeric T>
    requires BlockSize<BLOCK_SIZE>
  __device__ cuda::std::tuple<T, T> multi_warp_block_upsweep(const T warp_inclusive_scan, OPERATOR function)
  {
    constexpr int warp_spine_size = (BLOCK_SIZE + warp_size - 1) / warp_size;
    __shared__ T shared_values[warp_spine_size]; // shared storage
    const int warp_id = threadIdx.x / warp_size;

    const bool last_thread_in_warp = laneid() == warp_size - 1;
    if (last_thread_in_warp)
      // warp_inclusive_scan of last thread is the reduce of the warp
      shared_values[warp_id] = warp_inclusive_scan;
    __syncthreads();

    // Scan across the partial sums of each warp
    const bool predicate = threadIdx.x < warp_spine_size;
    const unsigned mask = __ballot_sync(FULL_MASK, predicate);
    if (predicate) {
      T block_spine = shared_values[threadIdx.x];
      block_spine = warp_scan(block_spine, function, mask, warp_spine_size);
      shared_values[threadIdx.x] = block_spine;
    }

    // Calculate per-thread partial
    // First take exclusive scan results
    T thread_partial = __shfl_up_sync(FULL_MASK, warp_inclusive_scan, 1);
    if (laneid() == 0) thread_partial = OPERATOR::Identity;

    // Then sum the block upsweep to the thread partial
    __syncthreads();
    const T upsweep = (warp_id > 0) ? shared_values[warp_id - 1] : OPERATOR::Identity;
    const T block_aggregate = shared_values[warp_spine_size - 1];

    // Synchronize after reading from shared memory
    //   to avoid shared memory from being updated
    //   by other threads before we read its value
    __syncthreads();

    thread_partial = function(thread_partial, upsweep);

    return cuda::std::make_tuple(block_aggregate, thread_partial);
  }

  template<unsigned BLOCK_SIZE, BinaryOperator OPERATOR, Numeric T>
    requires BlockSize<BLOCK_SIZE>
  __device__ cuda::std::tuple<T, T> single_warp_block_upsweep(const T warp_inclusive_scan, OPERATOR function)
  {

    // Calculate per-thread partial
    // First take exclusive scan results
    T thread_partial = __shfl_up_sync(FULL_MASK, warp_inclusive_scan, 1);
    if (laneid() == 0) thread_partial = OPERATOR::Identity;

    const T block_aggregate = __shfl_sync(FULL_MASK, warp_inclusive_scan, warp_size - 1);

    return cuda::std::make_tuple(block_aggregate, thread_partial);
  }

  template<unsigned BLOCK_SIZE, typename... ARGS>
    requires BlockSize<BLOCK_SIZE>
  __device__ auto block_upsweep(ARGS&&... args)
  {
    constexpr int number_of_warps = BLOCK_SIZE / warp_size;
    if (number_of_warps > 1) {
      return multi_warp_block_upsweep<BLOCK_SIZE>(std::forward<ARGS>(args)...);
    }
    else {
      return single_warp_block_upsweep<BLOCK_SIZE>(std::forward<ARGS>(args)...);
    }
  }

  template<
    ScanType TYPE,
    unsigned BLOCK_SIZE,
    unsigned VALUES_PER_THREAD,
    BinaryOperator FUNCTION,
    Numeric T,
    typename BLOCK_PREFIX_CALLBACK>
    requires BlockSize<BLOCK_SIZE>
  __inline__ __device__ T block_scan(
    const T* input,
    const unsigned array_size,
    T* output,
    FUNCTION function,
    const T init,
    BLOCK_PREFIX_CALLBACK block_prefix_callback)
  {

    T values[VALUES_PER_THREAD]; // registers storage
    memory_to_registers(array_size, input, function, values);

    const T thread_reduce = linear_scan(values, function);
    const T warp_inclusive_scan = warp_scan(thread_reduce, function); // warp-wide inclusive scan

    auto [block_aggregate, thread_partial] = block_upsweep<BLOCK_SIZE>(warp_inclusive_scan, function);
    const auto& [exclusive_prefix, inclusive_prefix] = block_prefix_callback(block_aggregate, function, init);

    thread_partial = function(thread_partial, exclusive_prefix);
    block_downsweep<TYPE>(thread_partial, function, values);

    // Transfer the values back to memory
    registers_to_memory(array_size, output, values);

    return inclusive_prefix;
  }
} // namespace ScanReduce
