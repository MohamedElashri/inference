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

namespace ScanReduce {
  constexpr unsigned FULL_MASK = 0xFFFFFFFF;
  enum ScanType { Exclusive, Inclusive };

  template<typename T>
  concept Numeric = (std::integral<T> ||
                     std::floating_point<T>) &&(sizeof(T) == 8 || sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1) &&
                    !std::same_as<T, bool>;

  // Ensures that block sizes are divisible by warp size
  // And the block spine fits into registers of a single warp
  template<unsigned BLOCK_SIZE>
  concept BlockSize = (BLOCK_SIZE % warp_size == 0) && (BLOCK_SIZE / warp_size <= warp_size);

  template<typename OPERATOR>
  concept BinaryOperator = requires(OPERATOR op, typename OPERATOR::Type a, typename OPERATOR::Type b) {
    {
      op(a, b)
    } -> std::same_as<typename OPERATOR::Type>;
    {
      OPERATOR::Identity
    } -> std::convertible_to<typename OPERATOR::Type>;
  };

  __inline__ __device__ int laneid() { return threadIdx.x % warp_size; }

  template<Numeric T>
  struct Plus {
    static constexpr T Identity = 0;
    using Type = T;
    __device__ __host__ T operator()(const T a, const T b) { return a + b; }
  };

  template<Numeric T>
  struct Multiply {
    static constexpr T Identity = 1;
    using Type = T;
    __device__ __host__ T operator()(const T a, const T b) { return a * b; }
  };

  template<Numeric T>
  struct Max {
    static constexpr T Identity = std::numeric_limits<T>::lowest();
    using Type = T;
    __device__ __host__ T operator()(const T a, const T b) { return a > b ? a : b; }
  };

  template<Numeric T>
  struct Min {
    static constexpr T Identity = std::numeric_limits<T>::max();
    using Type = T;
    __device__ __host__ T operator()(const T a, const T b) { return a < b ? a : b; }
  };

  template<unsigned VALUES_PER_THREAD, BinaryOperator OPERATOR, Numeric T>
  __device__ void
  memory_to_registers(const unsigned total_size, const T* memory, const OPERATOR&, T (&values)[VALUES_PER_THREAD])
  {
#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      const unsigned index = threadIdx.x * VALUES_PER_THREAD + i;
      values[i] = (index < total_size) ? memory[index] : OPERATOR::Identity;
    }
  }

  template<unsigned VALUES_PER_THREAD, Numeric T>
  __device__ void registers_to_memory(const unsigned total_size, T* memory, T (&values)[VALUES_PER_THREAD])
  {
#pragma unroll
    for (int i = 0; i < VALUES_PER_THREAD; i++) {
      const unsigned index = threadIdx.x * VALUES_PER_THREAD + i;
      if (index < total_size) memory[index] = values[i];
    }
  }
} // namespace ScanReduce
