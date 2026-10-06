/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

#include "BackendCommon.h"
#include "PV_Definitions.cuh"
#include "PVFinderConstants.cuh"

// Peak finding on one event's PVFinder KDE (PVFinderConstants::KDE::n_bins
// bins), as pv-finder's pv_locations_updated: a peak is a run of consecutive
// bins at or above `threshold`, split in two where the KDE rises again after
// a drop of more than Peak::split_min_drop and Peak::split_min_ratio between
// two bins; it is kept when it has at least `min_width` bins and their sum is
// at least `integral_threshold`. Its z is the KDE-weighted mean of its bin
// centres.
namespace PVFinderPeakFinding {

  struct Cuts {
    float threshold = PVFinderConstants::Peak::threshold;
    float integral_threshold = PVFinderConstants::Peak::integral_threshold;
    unsigned min_width = PVFinderConstants::Peak::min_width;
    bool split_peaks = true;
  };

  // Largest blockDim.x of find_peaks (size of its per-thread shared array).
  static constexpr unsigned max_block_dim = 512;

  // Scans the peaks of the run of bins at or above threshold that starts at
  // bin `begin`, in increasing z, and calls emit(z) for each accepted one.
  // The bin before the first is taken as 0 (pv-finder reads the last bin
  // there, a wrap-around of its Python indexing).
  template<typename Emit>
  __host__ __device__ void scan_run(const float* kde, const unsigned begin, const Cuts& cuts, Emit&& emit)
  {
    using namespace PVFinderConstants;
    unsigned width = 0;
    float integral = 0.f;
    float weighted_sum = 0.f;
    bool peak_passed = false;
    for (unsigned i = begin; i < KDE::n_bins; ++i) {
      const float value = kde[i];
      const float previous = i > 0 ? kde[i - 1] : 0.f;
      const bool on = value >= cuts.threshold;
      if (on) {
        ++width;
        integral += value;
        weighted_sum += static_cast<float>(i) * value;
        if (previous > value + Peak::split_min_drop && previous > Peak::split_min_ratio * value) {
          peak_passed = true;
        }
      }
      const bool last = i == KDE::n_bins - 1;
      const bool split = cuts.split_peaks && peak_passed && previous < value;
      if ((!on || last || split) && width > 0) {
        if (width >= cuts.min_width && integral >= cuts.integral_threshold) {
          // Weighted mean bin index, +0.5 for the bin centre.
          emit(KDE::z_min + KDE::bin_width * (weighted_sum / integral + 0.5f));
        }
        width = 0;
        integral = 0.f;
        weighted_sum = 0.f;
        peak_passed = false;
      }
      if (!on || last) return;
    }
  }

  __host__ __device__ inline bool starts_run(const float* kde, const unsigned i, const float threshold)
  {
    return kde[i] >= threshold && (i == 0 || kde[i - 1] < threshold);
  }

  // The peaks of one event, one bin after the other: the reference for
  // find_peaks. Writes at most PV::max_number_vertices seeds (the lowest in z)
  // and returns the number of peaks found, which can be larger.
  __host__ __device__ inline unsigned find_peaks_serial(const float* kde, float* zpeaks, const Cuts& cuts)
  {
    unsigned n = 0;
    for (unsigned i = 0; i < PVFinderConstants::KDE::n_bins; ++i) {
      if (starts_run(kde, i, cuts.threshold)) {
        scan_run(kde, i, cuts, [&](float z) {
          if (n < PV::max_number_vertices) zpeaks[n] = z;
          ++n;
        });
      }
    }
    return n;
  }

  // The peaks of one event with a block of threads (blockDim.x <=
  // max_block_dim): the seeds of find_peaks_serial, in the same order. Each
  // thread owns the runs that start in its contiguous share of the bins; a
  // first pass counts their peaks, the second writes them after the peaks of
  // the threads before it. event_kde: global memory. Writes at most
  // PV::max_number_vertices seeds (the lowest in z) and returns the number of
  // peaks found, which can be larger, to every thread; zpeaks is complete
  // after the call.
  __device__ inline unsigned find_peaks(const float* event_kde, float* zpeaks, const Cuts& cuts)
  {
    using namespace PVFinderConstants;
    __shared__ unsigned peak_offset[max_block_dim];

    // The KDE in shared memory (16 KB), loaded coalesced: each thread scans
    // its bins, and runs past them, twice.
    __shared__ float kde[KDE::n_bins];
    for (unsigned i = threadIdx.x; i < KDE::n_bins; i += blockDim.x) {
      kde[i] = event_kde[i];
    }
    __syncthreads();

    const unsigned bins_per_thread = (KDE::n_bins + blockDim.x - 1) / blockDim.x;
    const unsigned first_bin = threadIdx.x * bins_per_thread;
    const unsigned end_bin = min(first_bin + bins_per_thread, KDE::n_bins);

    auto for_each_run = [&](auto&& emit) {
      for (unsigned i = first_bin; i < end_bin; ++i) {
        if (starts_run(kde, i, cuts.threshold)) scan_run(kde, i, cuts, emit);
      }
    };

    unsigned number_of_peaks = 0;
    for_each_run([&](float) { ++number_of_peaks; });

    // Inclusive prefix sum of the threads' peak counts (log-step scan).
    peak_offset[threadIdx.x] = number_of_peaks;
    __syncthreads();
    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
      const unsigned add = threadIdx.x >= stride ? peak_offset[threadIdx.x - stride] : 0u;
      __syncthreads();
      peak_offset[threadIdx.x] += add;
      __syncthreads();
    }

    unsigned index = peak_offset[threadIdx.x] - number_of_peaks;
    for_each_run([&](float z) {
      if (index < PV::max_number_vertices) zpeaks[index] = z;
      ++index;
    });
    const unsigned total = peak_offset[blockDim.x - 1];
    __syncthreads();
    return total;
  }

} // namespace PVFinderPeakFinding
