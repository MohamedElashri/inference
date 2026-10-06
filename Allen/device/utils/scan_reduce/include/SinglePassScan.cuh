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
#include "SegmentedReduce.cuh"

#include <cub/detail/strong_load.cuh>
#include <cub/detail/strong_store.cuh>

namespace ScanReduce {
  namespace SinglePassScan {

    namespace Constants {
      static constexpr unsigned int L2WriteLantency = 450;
      static constexpr unsigned int DelayInterval = 350;
      static constexpr int Padding = warp_size - 1;
    } // namespace Constants

    template<unsigned GRID_THRESHOLD = 500>
    __device__ void delay(unsigned nanoseconds)
    {
      if (gridDim.x < GRID_THRESHOLD) {
        __threadfence_block();
      }
      else {
        __nanosleep(nanoseconds);
      }
    }

    // template TileState for different types
    template<Numeric T>
    struct TileState {

      using InputType = T;
      using StatusType = std::conditional_t<
        (sizeof(T) == 8),
        unsigned long long,
        std::conditional_t<(sizeof(T) == 4), unsigned int, unsigned short>>;
      using StoreType =
        std::conditional_t<(sizeof(T) == 8), ulonglong2, std::conditional_t<(sizeof(T) == 4), uint2, unsigned int>>;

      enum Flag : StatusType {
        OutOfBounds = 9,
        Invalid = 99,
        Partial,
        Inclusive,
      };

      StatusType status;
      InputType value;

      static __inline__ __device__ StoreType dump_state(StatusType status, InputType value)
      {
        StoreType store_data;
        TileState tile_state {status, value};
        *reinterpret_cast<TileState*>(&store_data) = tile_state;
        return store_data;
      }

      static __inline__ __device__ cuda::std::tuple<StatusType, InputType> parse_state(StoreType store_data)
      {
        TileState tile_state;
        tile_state = *reinterpret_cast<TileState*>(&store_data);
        return cuda::std::make_tuple(tile_state.status, tile_state.value);
      }

      static __inline__ __device__ void store_inclusive(StoreType* tile_state_data, const T inclusive_prefix)
      {

        if (threadIdx.x == 0) {
          const auto store_data = dump_state(Flag::Inclusive, inclusive_prefix);
          auto* address = tile_state_data + Constants::Padding + blockIdx.x;
          cub::detail::store_relaxed(address, store_data);
        }
      }

      static __inline__ __device__ void store_partial(StoreType* tile_state_data, const T block_aggregate)
      {

        if (threadIdx.x == 0) {
          const auto store_data = dump_state(Flag::Partial, block_aggregate);
          auto* address = tile_state_data + Constants::Padding + blockIdx.x;
          cub::detail::store_relaxed(address, store_data);
        }
      }

      static __inline__ __device__ cuda::std::tuple<StatusType, T> load_state(
        const int predecessor_index,
        StoreType* tile_state_data)
      {
        const auto* address = tile_state_data + Constants::Padding + predecessor_index;
        return parse_state(cub::detail::load_relaxed(address));
      }

      // For initializing the prefix sum
      static __inline__ __device__ void set_initial_tile_state(const int index, StoreType* tile_state_data)
      {
        const Flag status = (index < Constants::Padding) ? Flag::OutOfBounds : Flag::Invalid;
        const StoreType store_data = dump_state(status, 0); // 0 is a dummy value
        auto* address = tile_state_data + index;
        tile_state_data[index] = store_data;
      }
    }; // struct TileState

    // Non-templated function, implementation is separated
    static __host__ __inline__ unsigned get_tile_state_size(unsigned scan_size, unsigned values_per_block)
    {
      unsigned number_of_execution_blocks = (scan_size + values_per_block - 1) / values_per_block;
      unsigned scan_state_size = Constants::Padding + number_of_execution_blocks;

      return scan_state_size;
    }

    template<BinaryOperator OPERATOR, Numeric T>
    struct BlockPrefixCallback {

      using StatusType = TileState<T>::StatusType;
      using StoreType = TileState<T>::StoreType;

      StoreType* tile_state_data;

      __device__ BlockPrefixCallback(StoreType* _tile_state_data) : tile_state_data(_tile_state_data) {};

      __inline__ __device__ cuda::std::tuple<StatusType, T> process_window(
        const int predecessor_index,
        OPERATOR function)
      {

        auto [predecessor_status, value] = TileState<T>::load_state(predecessor_index, tile_state_data);

        while (__any_sync(0xffffffff, (predecessor_status == TileState<T>::Flag::Invalid))) {
          delay(Constants::DelayInterval);
          const auto& [new_status, new_value] = TileState<T>::load_state(predecessor_index, tile_state_data);
          predecessor_status = new_status;
          value = new_value;
        }

        int tail_flag = (predecessor_status == TileState<T>::Flag::Inclusive);
        const T window_aggregate = Segmented::warp_reduce_to_tail(value, tail_flag, function);

        return cuda::std::make_tuple(predecessor_status, window_aggregate);
      }

      __inline__ __device__ T scan_predecessors(OPERATOR function)
      {

        T exclusive_prefix;
        __shared__ T shared_exclusive_prefix;
        const auto warp_index = threadIdx.x / warp_size;

        if (warp_index == 0) {
          // Wait for L2 latency or __threadfence_block() so previous thread blocks can update their values
          delay(Constants::L2WriteLantency);

          int predecessor_index = blockIdx.x - threadIdx.x - 1;

          auto [predecessor_state, first_aggregate] = process_window(predecessor_index, function);
          exclusive_prefix = first_aggregate;

          // Keep sliding the window back until we come across a tile whose inclusive
          // prefix is known
          while (__all_sync(FULL_MASK, (predecessor_state != TileState<T>::Flag::Inclusive))) {
            predecessor_index -= warp_size;

            const auto& [next_state, window_aggregate] = process_window(predecessor_index, function);
            predecessor_state = next_state;
            exclusive_prefix = function(window_aggregate, exclusive_prefix);
          }
        }
        // Compute the inclusive tile prefix and update the status for this tile
        if (threadIdx.x == 0) {
          shared_exclusive_prefix = exclusive_prefix;
        }

        __syncthreads();
        exclusive_prefix = shared_exclusive_prefix;
        __syncthreads();

        return exclusive_prefix;
      }

      __inline__ __device__ cuda::std::tuple<T, T> operator()(const T block_aggregate, OPERATOR function, const T init)
      {

        T exclusive_prefix;
        T inclusive_prefix;

        // First block stores block_aggregate as inclusive state
        //   and returns init as exclusive prefix
        if (blockIdx.x == 0) {
          inclusive_prefix = function(block_aggregate, init);
          exclusive_prefix = init;
          // First block has different store logic
          TileState<T>::store_inclusive(tile_state_data, inclusive_prefix);
        }
        else {
          // First store block partial result
          TileState<T>::store_partial(tile_state_data, block_aggregate);

          exclusive_prefix = scan_predecessors(function);
          inclusive_prefix = function(block_aggregate, exclusive_prefix);

          // Now store our inclusive values
          TileState<T>::store_inclusive(tile_state_data, inclusive_prefix);
        }

        return cuda::std::make_tuple(exclusive_prefix, inclusive_prefix);
      }
    }; // struct BlockPrefixCallback
  }    // namespace SinglePassScan
} // namespace ScanReduce
