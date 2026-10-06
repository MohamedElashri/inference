#pragma once

#include "PVFinderFCAggregation.cuh"

namespace pvfinder_fc_aggregation::gpu_work_list {
  constexpr unsigned CHUNK = 64u;
  constexpr unsigned THREADS = 256u;

  // Preserve compact row order and split-slot partial order. The arrival
  // buffer is scratch here: [slots] partial offsets, then [CHUNK+1] bucket
  // cursors. It is reset after emit_items, before the FC consumes it.
  // slot_row's three trailing words contain row, item and partial counts.
  __global__ void make_layout(Parameters p, unsigned n_slots, bool write_histogram)
  {
    __shared__ unsigned histogram[CHUNK + 1];
    __shared__ unsigned long long warp_prefix[THREADS / 32];
    __shared__ unsigned long long carry;
    const unsigned tid = threadIdx.x, lane = tid % 32u, warp = tid / 32u;
    if (tid <= CHUNK) histogram[tid] = 0u;
    if (tid == 0) carry = 0ull;
    __syncthreads();

    for (unsigned base = 0; base < n_slots; base += THREADS) {
      const unsigned slot = base + tid;
      unsigned n = 0u;
      if (slot < n_slots) {
        const int* csr = p.dev_pvfinder_interval_start + (slot / N_INTERVALS) * CSR_STRIDE;
        const unsigned iv = slot % N_INTERVALS;
        n = static_cast<unsigned>(csr[iv + 1] - csr[iv]);
      }
      const unsigned chunks = n > CHUNK ? (n + CHUNK - 1u) / CHUNK : 1u;
      const unsigned partials = n > CHUNK ? chunks : 0u;
      const unsigned occupied = n > 0u ? 1u : 0u;
      // Two independent scans packed into 64 bits; each count is bounded by
      // the already allocated buffers and fits in 32 bits.
      unsigned long long prefix = (static_cast<unsigned long long>(partials) << 32) | occupied;
      for (unsigned offset = 1u; offset < 32u; offset <<= 1) {
        const auto previous = __shfl_up_sync(0xffffffffu, prefix, offset);
        if (lane >= offset) prefix += previous;
      }
      if (lane == 31u) warp_prefix[warp] = prefix;
      __syncthreads();
      if (warp == 0u) {
        unsigned long long sum = lane < THREADS / 32u ? warp_prefix[lane] : 0ull;
        for (unsigned offset = 1u; offset < THREADS / 32u; offset <<= 1) {
          const auto previous = __shfl_up_sync(0xffffffffu, sum, offset);
          if (lane >= offset) sum += previous;
        }
        if (lane < THREADS / 32u) warp_prefix[lane] = sum;
      }
      __syncthreads();
      if (warp > 0u) prefix += warp_prefix[warp - 1u];
      prefix += carry;
      if (slot < n_slots) {
        p.dev_pvfinder_slot_row[slot] = occupied ? static_cast<int>(static_cast<unsigned>(prefix) - 1u) : -1;
        p.dev_pvfinder_fc_arrive[slot] = static_cast<unsigned>(prefix >> 32) - partials;
        if (n > 0u) {
          if (n <= CHUNK) atomicAdd(histogram + n, 1u);
          else {
            atomicAdd(histogram + CHUNK, n / CHUNK);
            if (n % CHUNK) atomicAdd(histogram + n % CHUNK, 1u);
          }
        }
        else if (write_histogram) atomicAdd(histogram, 1u);
      }
      __syncthreads();
      if (tid == THREADS - 1u) carry = prefix;
      __syncthreads();
    }
    if (tid == 0u) {
      unsigned items = 0u;
      for (int size = CHUNK; size >= 0; --size) {
        p.dev_pvfinder_fc_arrive[n_slots + size] = items;
        items += histogram[size];
      }
      p.dev_pvfinder_slot_row[n_slots] = static_cast<int>(static_cast<unsigned>(carry));
      p.dev_pvfinder_slot_row[n_slots + 1u] = static_cast<int>(items);
      p.dev_pvfinder_slot_row[n_slots + 2u] = static_cast<int>(carry >> 32);
    }
  }

  // Descending chunk size, with arbitrary order within each size bucket.
  // Each split slot still reduces its partial sums in canonical chunk order.
  __global__ void emit_items(Parameters p, unsigned n_slots, bool write_histogram)
  {
    auto* items = reinterpret_cast<FCWorkItem*>(static_cast<unsigned*>(p.dev_pvfinder_slot_order));
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < n_slots; slot += gridDim.x * blockDim.x) {
      const unsigned iv = slot % N_INTERVALS;
      const int* csr = p.dev_pvfinder_interval_start + (slot / N_INTERVALS) * CSR_STRIDE;
      const unsigned first = static_cast<unsigned>(csr[iv]);
      const unsigned n = static_cast<unsigned>(csr[iv + 1] - csr[iv]);
      if (n == 0u && !write_histogram) continue;
      const unsigned chunks = n > CHUNK ? (n + CHUNK - 1u) / CHUNK : 1u;
      const unsigned partial = chunks > 1u ? p.dev_pvfinder_fc_arrive[slot] : 0u;
      for (unsigned chunk = 0u; chunk < chunks; ++chunk) {
        const unsigned entries = chunks > 1u ? min(CHUNK, n - chunk * CHUNK) : n;
        const unsigned position = atomicAdd(p.dev_pvfinder_fc_arrive + n_slots + entries, 1u);
        items[position] = FCWorkItem {
          slot, first + chunk * CHUNK, entries, p.dev_pvfinder_slot_row[slot], partial, chunk, chunks, 0u};
      }
    }
  }
} // namespace pvfinder_fc_aggregation::gpu_work_list
