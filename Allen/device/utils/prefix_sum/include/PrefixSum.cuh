/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "AlgorithmTypes.cuh"
#include "BackendCommon.h"

#ifndef TARGET_DEVICE_CPU
#include "SinglePassScan.cuh"
#endif

namespace PrefixSum {

#ifndef TARGET_DEVICE_CPU
  __global__ void prefix_sum_reduce_x1(unsigned* dev_array, unsigned* dev_auxiliary_array, const unsigned array_size);

  __global__ void prefix_sum_reduce(unsigned* dev_array, unsigned* dev_auxiliary_array, const unsigned array_size);

  __global__ void prefix_sum_single_block(unsigned* dev_array, const unsigned array_size);

  __global__ void prefix_sum_scan(unsigned* dev_array, unsigned* dev_auxiliary_array, const unsigned array_size);

  __global__ void prefix_sum_single_warp(unsigned* dev_array, const unsigned array_size);

  __global__ void prefix_sum_single_warp_x4(unsigned* dev_array, const unsigned array_size, unsigned* host_total);

  // Chain scan
  using TileStoreType = ScanReduce::SinglePassScan::TileState<unsigned>::StoreType;
  constexpr unsigned values_per_thread = 8;
  constexpr unsigned threads_per_block = 128;
  __global__ void initialize_tile_state(unsigned tile_state_size, TileStoreType* tile_state_data);
  __global__ void prefix_sum_single_pass(
    unsigned* dev_array,
    const unsigned array_size,
    TileStoreType* tile_state_data,
    unsigned* host_total);

  // Different block sizes for input of different sizes
  __global__ void prefix_sum_single_block_32x8(unsigned* dev_array, const unsigned array_size, unsigned* host_total);
  __global__ void prefix_sum_single_block_64x8(unsigned* dev_array, const unsigned array_size, unsigned* host_total);
  __global__ void prefix_sum_single_block_128x8(unsigned* dev_array, const unsigned array_size, unsigned* host_total);
#endif

  template<typename Alg, typename Args>
  void prefix_sum(
    [[maybe_unused]] const Alg& alg,
    [[maybe_unused]] const Args& arguments,
    [[maybe_unused]] const Allen::Context& context,
    unsigned* dev_array,
    const unsigned array_size,
    unsigned* host_total = nullptr)
  {
#if defined(TARGET_DEVICE_CPU)
    unsigned sum = 0;
    for (unsigned i = 0; i < array_size + 1; i++) {
      unsigned val = dev_array[i];
      dev_array[i] = sum;
      sum += val;
    }
    if (host_total != nullptr) *host_total = dev_array[array_size];
#else
    if (array_size < 512) {
      alg.global_function(prefix_sum_single_block_32x8)(dim3(1), dim3(32), context)(dev_array, array_size, host_total);
    }
    else if (array_size > 512 && array_size <= 2048) {
      alg.global_function(prefix_sum_single_block_64x8)(dim3(1), dim3(64), context)(dev_array, array_size, host_total);
    }
    else if (array_size > 2048 && array_size <= 4096) {
      alg.global_function(prefix_sum_single_block_128x8)(dim3(1), dim3(128), context)(
        dev_array, array_size, host_total);
    }
    else {
      constexpr unsigned values_per_block = threads_per_block * values_per_thread;
      auto tile_state_size = ScanReduce::SinglePassScan::get_tile_state_size(array_size, values_per_block);
      auto tile_state_data =
        arguments.template make_buffer<Allen::Store::Scope::Device, TileStoreType>(tile_state_size);

      const unsigned initialize_grid_size = (tile_state_size + threads_per_block - 1) / threads_per_block;
      alg.global_function(initialize_tile_state)(dim3(initialize_grid_size), dim3(threads_per_block), context)(
        tile_state_size, tile_state_data.data());

      const unsigned prefix_sum_grid_size = (array_size + values_per_block - 1) / values_per_block;
      alg.global_function(prefix_sum_single_pass)(dim3(prefix_sum_grid_size), dim3(threads_per_block), context)(
        dev_array, array_size, tile_state_data.data(), host_total);
    }

    // Here we use unified memory to write to host a buffer from the device, in order to overlap the transfer and
    // compute but we still need to synchronize the stream to make sure the result is visible on the host
    if (host_total != nullptr) Allen::synchronize(context);
#endif
  }

  // Convenience functions:

  template<typename OffsetArray, typename Alg, typename Args>
  void prefix_sum(
    [[maybe_unused]] const Alg& alg,
    [[maybe_unused]] const Args& arguments,
    [[maybe_unused]] const Allen::Context& context)
  {
    prefix_sum(
      alg,
      arguments,
      context,
      alg.template data<OffsetArray>(arguments),
      alg.template size<OffsetArray>(arguments) - 1);
  }

  template<typename OffsetArray, typename HostTotal, typename Alg, typename Args>
  void prefix_sum(
    [[maybe_unused]] const Alg& alg,
    [[maybe_unused]] const Args& arguments,
    [[maybe_unused]] const Allen::Context& context)
  {
    prefix_sum(
      alg,
      arguments,
      context,
      alg.template data<OffsetArray>(arguments),
      alg.template size<OffsetArray>(arguments) - 1,
      alg.template data<HostTotal>(arguments));
  }

  // Device functions:

#ifndef TARGET_DEVICE_CPU
  inline __device__ void device_prefix_sum_single_warp(unsigned* dev_array, const unsigned array_size)
  {
    unsigned sum = 0;
    for (unsigned i = 0; i < (array_size + 128) / 128; i++) {
      unsigned index = i * 32 + threadIdx.x;

      uint4 elem1;
      if ((index * 4 + 3) < array_size) {
        elem1 = ((uint4*) dev_array)[index];
      }
      else {
        elem1.x = ((index * 4) < array_size) ? dev_array[index * 4] : 0;
        elem1.y = ((index * 4 + 1) < array_size) ? dev_array[index * 4 + 1] : 0;
        elem1.z = ((index * 4 + 2) < array_size) ? dev_array[index * 4 + 2] : 0;
        elem1.w = 0;
      }

      elem1.y += elem1.x;
      elem1.z += elem1.y;
      elem1.w += elem1.z;

      unsigned val = elem1.w;

#pragma unroll
      for (unsigned d = 1; d < 32; d *= 2) {
        unsigned other_val = __shfl_up_sync(0xFFFFFFFF, val, d);
        if (threadIdx.x >= d) val += other_val;
      }

      val += sum;
      sum = __shfl_sync(0xFFFFFFFF, val, 31);
      val -= elem1.w;

      elem1.w = val + elem1.z;
      elem1.z = val + elem1.y;
      elem1.y = val + elem1.x;
      elem1.x = val;

      if ((index * 4 + 3) <= array_size) {
        ((uint4*) dev_array)[index] = elem1;
      }
      else {
        if ((index * 4) <= array_size) dev_array[index * 4] = elem1.x;
        if ((index * 4 + 1) <= array_size) dev_array[index * 4 + 1] = elem1.y;
        if ((index * 4 + 2) <= array_size) dev_array[index * 4 + 2] = elem1.z;
      }
    }
  }
#else
  inline __device__ void device_prefix_sum_single_warp(unsigned* dev_array, const unsigned array_size)
  {
    unsigned sum = 0;
    for (unsigned i = 0; i <= array_size; i++) {
      unsigned val = dev_array[i];
      dev_array[i] = sum;
      sum += val;
    }
  }
#endif

} // namespace PrefixSum
