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

#include <algorithm>
#include <regsort_functions.cuh>

namespace SegSort {
#ifndef TARGET_DEVICE_CPU
  template<typename KeyType>
  __device__ void global_sort_warp(unsigned start, unsigned size, const KeyType* keys, unsigned* permutations)
  {
    const unsigned tid = threadIdx.x % warp_size;
    for (unsigned i = tid; i < size; i += warp_size) {
      const auto key0 = keys[start + i];

      unsigned position = 0;
      for (unsigned j = 0; j < size; ++j) {
        if (i == j) continue;

        const auto key1 = keys[start + j];
        position += key0 > key1 || (key0 == key1 && i < j);
      }

      permutations[start + position] = start + i;
    }
  }

  inline __global__ void
  segsort_bin(const unsigned* seg_offsets, unsigned n_segments, unsigned* bins, unsigned* bins_offsets)
  {
    const auto get_bin = [](unsigned size) {
      if (size <= 1) return 0;
      if (size > 256) return 9;
      return 32 - __clz(size - 1);
    };

    if (threadIdx.x < 10) {
      bins_offsets[threadIdx.x] = 0;
    }

    __syncthreads();
    unsigned bins_reg[10];

    UNROLL(10)
    for (unsigned i = 0; i < 10; i++)
      bins_reg[i] = 0;

    for (unsigned i = threadIdx.x; i < n_segments; i += blockDim.x) {
      const auto seg_size = seg_offsets[i + 1] - seg_offsets[i];
      if (seg_size > 0) {
        if (seg_size <= 1)
          bins_reg[0]++;
        else if (seg_size <= 2)
          bins_reg[1]++;
        else if (seg_size <= 4)
          bins_reg[2]++;
        else if (seg_size <= 8)
          bins_reg[3]++;
        else if (seg_size <= 16)
          bins_reg[4]++;
        else if (seg_size <= 32)
          bins_reg[5]++;
        else if (seg_size <= 64)
          bins_reg[6]++;
        else if (seg_size <= 128)
          bins_reg[7]++;
        else if (seg_size <= 256)
          bins_reg[8]++;
        else
          bins_reg[9]++;
      }
    }

    UNROLL(10)
    for (unsigned i = 0; i < 10; i++)
      atomicAdd(&bins_offsets[i], bins_reg[i]);

    __syncthreads();
    // Exclusive prefix sum
    if (threadIdx.x == 0) {
      unsigned sum = 0;

      UNROLL(10)
      for (int bin = 0; bin < 10; bin++) {
        unsigned val = bins_offsets[bin];
        bins_offsets[bin] = sum;
        sum += val;
        // printf("segment size <= %d : %d (+%d)\n", 1<<bin, bins_offsets[bin], val);
      }
      bins_offsets[10] = sum;
    }

    __syncthreads();
    // Dispatch segments to bins
    for (unsigned i = threadIdx.x; i < n_segments; i += blockDim.x) {
      const auto seg_size = seg_offsets[i + 1] - seg_offsets[i];
      if (seg_size > 0) {
        unsigned index = atomicAdd(&bins_offsets[get_bin(seg_size)], 1);
        bins[index] = i;
      }
    }
  }

  template<typename KeyType>
  __global__ void segsort_warpdispatch(
    const unsigned* seg_offsets,
    const unsigned* bins,
    const unsigned* bins_offsets,
    const KeyType* keys,
    unsigned* permutations)
  {
    // bin              0 1 2 3  4  5  6   7   8    9
    // size             1 2 4 8 16 32 64 128 256 >256
    // threads per seg: 1 2 2 4  4  8 32  32  32   32
    // /!\ important: warp can work on different start/size as long as they are from same bin
    constexpr unsigned thread_per_seg[10] = {1, 2, 2, 4, 4, 8, 32, 32, 32, 32};

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < bins_offsets[10] * warp_size;
         i += blockDim.x * gridDim.x) {
      unsigned prev_offset = 0;
      unsigned bin = 0;
      unsigned seg_id = 0;
      unsigned threads_before = 0;

      UNROLL(10)
      for (unsigned j = 0; j < 10; j++) {
        unsigned n_segments_in_bin = bins_offsets[j] - prev_offset; // how many segments in this bin

        unsigned threads_in_bin = n_segments_in_bin * thread_per_seg[j]; // how many threads needed to cover this bin
        threads_in_bin = (threads_in_bin + warp_size - 1) & ~(warp_size - 1); // round up thread count to warp size

        if (i < threads_before + threads_in_bin) {
          // This is the bin
          seg_id = prev_offset + (i - threads_before) / thread_per_seg[j];
          bin = j;
          break;
        }
        prev_offset = bins_offsets[j];
        threads_before += threads_in_bin;
      }

      unsigned start = 0;
      unsigned size = 0;
      if (seg_id < bins_offsets[bin]) {
        const unsigned seg = bins[seg_id];
        start = seg_offsets[seg];
        size = seg_offsets[seg + 1] - start;
      }

      if (bin == 0) { // Bin 0 (size == 1)
        if (size > 0) {
          permutations[start] = start;
        }
      }
      else if (bin == 1) { // Bin 1  (1 < size <= 2)
        regsort_2_2t_1ept_f<KeyType>(start, size, keys, permutations);
      }
      else if (bin == 2) { // Bin 2  (size <= 4)
        regsort_4_2t_2ept_f<KeyType>(start, size, keys, permutations);
      }
      else if (bin == 3) { // Bin 3  (size <= 8)
        regsort_8_4t_2ept_f<KeyType>(start, size, keys, permutations);
      }
      else if (bin == 4) { // Bin 4  (size <= 16)
        regsort_16_4t_4ept_f<KeyType>(start, size, keys, permutations);
      }
      else if (bin == 5) { // Bin 5  (size <= 32)
        regsort_32_8t_4ept_f<KeyType>(start, size, keys, permutations);
      }
      else if (bin == 6) { // Bin 6 (size <= 64)
        regsort_64_32t_2ept_f<KeyType>(start, size, keys, permutations);
      }
      else if (bin == 7) { // Bin 7 (size <= 128)
        regsort_128_32t_4ept_f<KeyType>(start, size, keys, permutations);
      }
      else if (bin == 8) { // Bin 8 (size <= 256)
        regsort_256_32t_8ept_f<KeyType>(start, size, keys, permutations);
      }
      else { // Bin 9 (size > 256)
        global_sort_warp<KeyType>(start, size, keys, permutations);
      }
    }
  }
#endif

  template<typename KeyType, typename Alg, typename Args>
  void segsort(
    [[maybe_unused]] const Alg& alg,
    [[maybe_unused]] const Args& arguments,
    [[maybe_unused]] const Allen::Context& context,
    const KeyType* keys,
    const unsigned* seg_offsets,
    unsigned n_segments,
    unsigned* permutations)
  {
#if defined(TARGET_DEVICE_CPU)
    unsigned n_keys = seg_offsets[n_segments];
    for (unsigned i = 0; i < n_keys; i++) {
      permutations[i] = i;
    }
    for (unsigned s = 0; s < n_segments; s++) {
      unsigned start = seg_offsets[s];
      unsigned end = seg_offsets[s + 1];
      std::sort(
        permutations + start, permutations + end, [&](const auto i, const auto j) { return keys[i] < keys[j]; });
    }
#else
    auto dev_bins = arguments.template make_buffer<Allen::Store::Scope::Device, unsigned>(n_segments);
    auto dev_bins_offsets = arguments.template make_buffer<Allen::Store::Scope::Device, unsigned>(11);
    alg.global_function(segsort_bin)(dim3(1), dim3(256), context)(
      seg_offsets, n_segments, dev_bins.data(), dev_bins_offsets.data());
    alg.global_function(segsort_warpdispatch<KeyType>)(dim3(n_segments), dim3(32), context)(
      seg_offsets, dev_bins.data(), dev_bins_offsets.data(), keys, permutations);
#endif
  }

#ifndef TARGET_DEVICE_CPU
  template<typename T>
  inline __device__ void swap(T& a, T& b)
  {
    T c = a;
    a = b;
    b = c;
  }

  template<typename KeyType>
  __device__ void global_sort_keys_warp(unsigned start, unsigned size, KeyType* keys)
  {
    const unsigned tid = threadIdx.x % warp_size;
    bool isSorted = false;
    while (!isSorted) {
      isSorted = true;

      // Even phase
      for (unsigned i = tid; i * 2 + 1 < size; i += warp_size) {
        if (keys[start + i * 2] > keys[start + i * 2 + 1]) {
          swap(keys[start + i * 2], keys[start + i * 2 + 1]);
          isSorted = false;
        }
      }
      __syncwarp(0xFFFFFFFF);

      // Odd phase
      for (unsigned i = tid; i * 2 + 2 < size; i += warp_size) {
        if (keys[start + i * 2 + 1] > keys[start + i * 2 + 2]) {
          swap(keys[start + i * 2 + 1], keys[start + i * 2 + 2]);
          isSorted = false;
        }
      }

      isSorted = __all_sync(0xFFFFFFFF, isSorted);
    }
  }

  template<typename KeyType>
  __global__ void segsort_keys_warpdispatch(
    const unsigned* seg_offsets,
    const unsigned* bins,
    const unsigned* bins_offsets,
    KeyType* keys)
  {
    // /!\ important: warp can work on different start/size as long as they are from same bin
    constexpr unsigned thread_per_seg[10] = {1, 2, 2, 4, 4, 8, 32, 32, 32, 32};

    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < bins_offsets[10] * warp_size;
         i += blockDim.x * gridDim.x) {

      unsigned prev_offset = 0;
      unsigned bin = 0;
      unsigned seg_id = 0;
      unsigned threads_before = 0;

      UNROLL(10)
      for (unsigned j = 0; j < 10; j++) {
        unsigned n_segments_in_bin = bins_offsets[j] - prev_offset; // how many segments in this bin

        unsigned threads_in_bin = n_segments_in_bin * thread_per_seg[j]; // how many threads needed to cover this bin
        threads_in_bin = (threads_in_bin + warp_size - 1) & ~(warp_size - 1); // round up thread count to warp size

        if (i < threads_before + threads_in_bin) {
          // This is the bin
          seg_id = prev_offset + (i - threads_before) / thread_per_seg[j];
          bin = j;
          break;
        }
        prev_offset = bins_offsets[j];
        threads_before += threads_in_bin;
      }

      unsigned start = 0;
      unsigned size = 0;
      if (seg_id < bins_offsets[bin]) {
        const unsigned seg = bins[seg_id];
        start = seg_offsets[seg];
        size = seg_offsets[seg + 1] - start;
      }

      if (bin == 1) { // Bin 1  (1 < size <= 2)
        regsort_2_2t_1ept_f<KeyType>(start, size, keys);
      }
      else if (bin == 2) { // Bin 2  (size <= 4)
        regsort_4_2t_2ept_f<KeyType>(start, size, keys);
      }
      else if (bin == 3) { // Bin 3  (size <= 8)
        regsort_8_4t_2ept_f<KeyType>(start, size, keys);
      }
      else if (bin == 4) { // Bin 4  (size <= 16)
        regsort_16_4t_4ept_f<KeyType>(start, size, keys);
      }
      else if (bin == 5) { // Bin 5  (size <= 32)
        regsort_32_8t_4ept_f<KeyType>(start, size, keys);
      }
      else if (bin == 6) { // Bin 6 (size <= 64)
        regsort_64_32t_2ept_f<KeyType>(start, size, keys);
      }
      else if (bin == 7) { // Bin 7 (size <= 128)
        regsort_128_32t_4ept_f<KeyType>(start, size, keys);
      }
      else if (bin == 8) { // Bin 8 (size <= 256)
        regsort_256_32t_8ept_f<KeyType>(start, size, keys);
      }
      else { // Bin 9 (size > 256)
        global_sort_keys_warp<KeyType>(start, size, keys);
      }
    }
  }

#endif

  template<typename KeyType>
  __global__ void check_keys_sorted(const unsigned* seg_offsets, unsigned n_segments, const KeyType* keys)
  {
    for (unsigned seg = 0; seg < n_segments; seg++) {
      bool sorted = true;

      unsigned start = seg_offsets[seg];
      unsigned size = seg_offsets[seg + 1] - start;

      for (unsigned i = 1; i < size; i++) {
        if (keys[start + i - 1] > keys[start + i]) {
          sorted = false;
        }
      }
      if (!sorted) {
        printf("Segment %d (start=%d, size=%d) not sorted\n", seg, start, size);
      }
    }
  }

  template<typename KeyType, typename Alg, typename Args>
  void segsort_keys(
    [[maybe_unused]] const Alg& alg,
    [[maybe_unused]] const Args& arguments,
    [[maybe_unused]] const Allen::Context& context,
    KeyType* keys,
    const unsigned* seg_offsets,
    unsigned n_segments)
  {
#if defined(TARGET_DEVICE_CPU)
    for (unsigned s = 0; s < n_segments; s++) {
      unsigned start = seg_offsets[s];
      unsigned end = seg_offsets[s + 1];
      std::sort(keys + start, keys + end);
    }
#else
    auto dev_bins = arguments.template make_buffer<Allen::Store::Scope::Device, unsigned>(n_segments);
    auto dev_bins_offsets = arguments.template make_buffer<Allen::Store::Scope::Device, unsigned>(11);
    alg.global_function(segsort_bin)(dim3(1), dim3(256), context)(
      seg_offsets, n_segments, dev_bins.data(), dev_bins_offsets.data());
    alg.global_function(segsort_keys_warpdispatch<KeyType>)(dim3(n_segments), dim3(32), context)(
      seg_offsets, dev_bins.data(), dev_bins_offsets.data(), keys);
#endif
  }
} // namespace SegSort
