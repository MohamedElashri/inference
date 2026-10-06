/*****************************************************************************\
* (c) Copyright 2018-2024 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include <iostream>
#include <iomanip>
#include <chrono>
#include "PrefixSum.cuh"
#include "PrefixSum_Impl.cuh"
#include "BackendCommon.h"

template<typename F>
bool verify_prefix_sum(unsigned size, F&& prefix_sum)
{
  unsigned* array_h = nullptr;
  unsigned* array_d = nullptr;

  Allen::malloc_host((void**) &array_h, sizeof(unsigned) * (size + 2));
  Allen::malloc((void**) &array_d, sizeof(unsigned) * (size + 2));

  // Allocate 2 scratch buffers device and host, should cover the need of all implementations
  unsigned* scratch_h = nullptr;
  unsigned* scratch_d = nullptr;
  Allen::malloc_host((void**) &scratch_h, sizeof(unsigned) * (size + 2));
  Allen::malloc((void**) &scratch_d, sizeof(unsigned) * (size + 2));

  for (unsigned i = 0; i < size; i++) {
    array_h[i] = 1;
  }
  array_h[size] = 0;
  array_h[size + 1] = 0xbebebebe; // guard for out of bound checks
  Allen::memcpy((void*) array_d, (void*) array_h, sizeof(unsigned) * (size + 2), Allen::memcpyHostToDevice);

  prefix_sum(array_d, size, scratch_h, scratch_d);

  Allen::memcpy((void*) array_h, (void*) array_d, sizeof(unsigned) * (size + 2), Allen::memcpyDeviceToHost);
  Allen::free(array_d);

  if (size == 100) {
    for (unsigned i = 0; i <= size; i++) {
      printf("%d ", array_h[i]);
    }
    printf("\n");
  }

  Allen::free_host(scratch_h);
  Allen::free(scratch_d);

  for (unsigned i = 0; i <= size; i++) {
    if (array_h[i] != i) {
      Allen::free_host(array_h);
      return false;
    }
  }
  if (array_h[size + 1] != 0xbebebebe) {
    std::cout << "Guard check failed" << std::endl;
    Allen::free_host(array_h);
    return false;
  }
  Allen::free_host(array_h);
  return true;
}

template<bool BENCHMARK_GPU, typename F>
float benchmark_prefix_sum_impl(unsigned size, F&& prefix_sum, unsigned n_repetitions = 1000)
{
  unsigned* array_d = nullptr;
  Allen::malloc((void**) &array_d, sizeof(unsigned) * (size + 1));

  unsigned* scratch_h = nullptr;
  unsigned* scratch_d = nullptr;
  Allen::malloc_host((void**) &scratch_h, sizeof(unsigned) * (size + 1));
  Allen::malloc((void**) &scratch_d, sizeof(unsigned) * (size + 1));

  float timing;

  if constexpr (BENCHMARK_GPU) {
#ifndef TARGET_DEVICE_CPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    for (unsigned i = 0; i < n_repetitions; i++) {
      prefix_sum(array_d, size, scratch_h, scratch_d);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    timing = milliseconds / n_repetitions / 1000.;
#endif
  }
  // CPU implementation
  else {
    const auto start = std::chrono::high_resolution_clock::now();

    for (unsigned i = 0; i < n_repetitions; i++) {
      prefix_sum(array_d, size, scratch_h, scratch_d);
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float> diff = end - start;
    timing = diff.count() / n_repetitions;
  }

  Allen::free(array_d);
  Allen::free_host(scratch_h);
  Allen::free(scratch_d);

  return timing;
}

template<typename... ARGS>
float benchmark_prefix_sum_cpu(ARGS&&... args)
{
  return benchmark_prefix_sum_impl<false>(std::forward<ARGS>(args)...);
}

#ifndef TARGET_DEVICE_CPU
template<typename... ARGS>
float benchmark_prefix_sum_gpu(ARGS&&... args)
{
  return benchmark_prefix_sum_impl<true>(std::forward<ARGS>(args)...);
}
#endif

void prefix_sum_cpu1(
  unsigned* array_d,
  unsigned array_size,
  [[maybe_unused]] unsigned* scratch_h,
  [[maybe_unused]] unsigned* scratch_d)
{
  unsigned* array_h = scratch_h;
  Allen::memcpy((void*) array_h, (void*) array_d, sizeof(unsigned) * (array_size + 1), Allen::memcpyDeviceToHost);

  unsigned sum = 0;
  for (unsigned i = 0; i < array_size + 1; i++) {
    unsigned val = array_h[i];
    array_h[i] = sum;
    sum += val;
  }

  Allen::memcpy((void*) array_d, (void*) array_h, sizeof(unsigned) * (array_size + 1), Allen::memcpyHostToDevice);
}

#ifndef TARGET_DEVICE_CPU
void prefix_sum_cuda1(
  unsigned* array_d,
  unsigned array_size,
  [[maybe_unused]] unsigned* scratch_h,
  [[maybe_unused]] unsigned* scratch_d)
{
  constexpr unsigned block_size = 256;
  constexpr unsigned elements_per_thread = 1;
  unsigned n_blocks = (array_size + (block_size * elements_per_thread) - 1) / (block_size * elements_per_thread);

  unsigned* prefix_sum_aux_array = scratch_d;

  PrefixSum::prefix_sum_reduce_x1<<<dim3(n_blocks), dim3(block_size / 2)>>>(array_d, prefix_sum_aux_array, array_size);

  PrefixSum::prefix_sum_single_block<<<dim3(1), dim3(block_size / 2)>>>(prefix_sum_aux_array, n_blocks);

  PrefixSum::prefix_sum_scan<<<dim3(n_blocks), dim3(block_size * elements_per_thread)>>>(
    array_d, prefix_sum_aux_array, array_size + 1);
}

void prefix_sum_cuda2(
  unsigned* array_d,
  unsigned array_size,
  [[maybe_unused]] unsigned* scratch_h,
  [[maybe_unused]] unsigned* scratch_d)
{
  constexpr unsigned block_size = 256;
  constexpr unsigned elements_per_thread = 4;
  unsigned n_blocks = (array_size + (block_size * elements_per_thread) - 1) / (block_size * elements_per_thread);

  unsigned* prefix_sum_aux_array = scratch_d;

  PrefixSum::prefix_sum_reduce<<<dim3(n_blocks), dim3(block_size / 2)>>>(array_d, prefix_sum_aux_array, array_size);

  PrefixSum::prefix_sum_single_block<<<dim3(1), dim3(block_size / 2)>>>(prefix_sum_aux_array, n_blocks);

  PrefixSum::prefix_sum_scan<<<dim3(n_blocks), dim3(block_size * elements_per_thread)>>>(
    array_d, prefix_sum_aux_array, array_size + 1);
}

void prefix_sum_cuda3(
  unsigned* array_d,
  unsigned array_size,
  [[maybe_unused]] unsigned* scratch_h,
  [[maybe_unused]] unsigned* scratch_d)
{
  constexpr unsigned block_size = 256;

  PrefixSum::prefix_sum_single_block<<<dim3(1), dim3(block_size / 2)>>>(array_d, array_size);
}

void prefix_sum_cuda4(
  unsigned* array_d,
  unsigned array_size,
  [[maybe_unused]] unsigned* scratch_h,
  [[maybe_unused]] unsigned* scratch_d)
{
  PrefixSum::prefix_sum_single_warp<<<dim3(1), dim3(32)>>>(array_d, array_size);
}

void prefix_sum_cuda5(
  unsigned* array_d,
  unsigned array_size,
  [[maybe_unused]] unsigned* scratch_h,
  [[maybe_unused]] unsigned* scratch_d)
{
  PrefixSum::prefix_sum_single_warp_x4<<<dim3(1), dim3(32)>>>(array_d, array_size, nullptr);
}

void prefix_sum_cuda6(
  unsigned* array_d,
  unsigned array_size,
  [[maybe_unused]] unsigned* scratch_h,
  [[maybe_unused]] unsigned* scratch_d)
{
  constexpr unsigned values_per_block = PrefixSum::threads_per_block * PrefixSum::values_per_thread;
  PrefixSum::TileStoreType* tile_state_data = reinterpret_cast<PrefixSum::TileStoreType*>(scratch_d);
  auto tile_state_size = ScanReduce::SinglePassScan::get_tile_state_size(array_size, values_per_block);
  const unsigned initialize_grid_size =
    (tile_state_size + PrefixSum::threads_per_block - 1) / PrefixSum::threads_per_block;
  PrefixSum::initialize_tile_state<<<dim3(initialize_grid_size), dim3(PrefixSum::threads_per_block)>>>(
    tile_state_size, tile_state_data);

  const unsigned scan_grid_size = (array_size + values_per_block - 1) / values_per_block;
  PrefixSum::prefix_sum_single_pass<<<dim3(scan_grid_size), dim3(PrefixSum::threads_per_block)>>>(
    array_d, array_size, tile_state_data, nullptr);
}

void prefix_sum_cuda7(
  unsigned* array_d,
  unsigned array_size,
  [[maybe_unused]] unsigned* scratch_h,
  [[maybe_unused]] unsigned* scratch_d)
{
  PrefixSum::prefix_sum_single_block<PrefixSum::threads_per_block, PrefixSum::values_per_thread>
    <<<dim3(1), dim3(PrefixSum::threads_per_block)>>>(array_d, array_size, nullptr);
}

template<unsigned THREADS_PER_BLOCK, unsigned VALUES_PER_THREAD>
void sliding_block_cuda(
  unsigned* array_d,
  unsigned array_size,
  [[maybe_unused]] unsigned* scratch_h,
  [[maybe_unused]] unsigned* scratch_d)
{
  PrefixSum::prefix_sum_single_block<THREADS_PER_BLOCK, VALUES_PER_THREAD>
    <<<dim3(1), dim3(THREADS_PER_BLOCK)>>>(array_d, array_size, nullptr);
}
#endif

#define TEST(x)                                                      \
  if (verify_prefix_sum(size, x)) {                                  \
    std::cout << #x << " SUCCESSFUL for size " << size << std::endl; \
  }                                                                  \
  else {                                                             \
    std::cout << #x << " FAILED for size " << size << std::endl;     \
  }

int main()
{
  for (unsigned size = 10; size <= 1000000; size = size * 10) {
    TEST(prefix_sum_cpu1)
#ifndef TARGET_DEVICE_CPU
    TEST(prefix_sum_cuda1)
    TEST(prefix_sum_cuda2)
    TEST(prefix_sum_cuda3)
    TEST(prefix_sum_cuda4)
    TEST(prefix_sum_cuda5)
    TEST(prefix_sum_cuda6)
    TEST(prefix_sum_cuda7)
#endif
    std::cout << std::endl;
  }
  for (unsigned size = 1; size <= 1024; size = size * 2) {
    TEST(prefix_sum_cpu1)
#ifndef TARGET_DEVICE_CPU
    TEST(prefix_sum_cuda1)
    TEST(prefix_sum_cuda2)
    TEST(prefix_sum_cuda3)
    TEST(prefix_sum_cuda4)
    TEST(prefix_sum_cuda5)
    TEST(prefix_sum_cuda6)
    TEST(prefix_sum_cuda7)
#endif
    std::cout << std::endl;
  }

  std::cout << "\nsize      32x4        64x4        128x4        256x4" << std::endl;
  for (unsigned size = 10; size <= 20000; size = size * 2) {
    std::cout << std::setw(9) << size << " " << std::scientific;
#ifndef TARGET_DEVICE_CPU
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<32, 4>) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<64, 4>) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<128, 4>) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<256, 4>) << " ";
#endif
    std::cout << std::endl;
  }

  std::cout << "\nsize      32x8        64x8        128x8        256x8" << std::endl;
  for (unsigned size = 10; size <= 20000; size = size * 2) {
    std::cout << std::setw(9) << size << " " << std::scientific;
#ifndef TARGET_DEVICE_CPU
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<32, 8>) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<64, 8>) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<128, 8>) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<256, 8>) << " ";
#endif
    std::cout << std::endl;
  }

  std::cout << "\nsize      32x16        64x16        128x16        256x16" << std::endl;
  for (unsigned size = 10; size <= 20000; size = size * 2) {
    std::cout << std::setw(9) << size << " " << std::scientific;
#ifndef TARGET_DEVICE_CPU
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<32, 16>) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<64, 16>) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<128, 16>) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, sliding_block_cuda<256, 16>) << " ";
#endif
    std::cout << std::endl;
  }

  std::cout
    << "\nsize      cpu1         cuda1        cuda2        cuda3        cuda4        cuda5        cuda6        cuda7"
    << std::endl;
  for (unsigned size = 10; size <= 10000000; size = size * 2) {
    std::cout << std::setw(9) << size << " " << std::scientific;
    std::cout << benchmark_prefix_sum_cpu(size, prefix_sum_cpu1) << " ";
#ifndef TARGET_DEVICE_CPU
    std::cout << benchmark_prefix_sum_gpu(size, prefix_sum_cuda1) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, prefix_sum_cuda2) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, prefix_sum_cuda3) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, prefix_sum_cuda4) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, prefix_sum_cuda5) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, prefix_sum_cuda6) << " ";
    std::cout << benchmark_prefix_sum_gpu(size, prefix_sum_cuda7) << " ";
#endif
    std::cout << std::endl;
  }

  return 0;
}
