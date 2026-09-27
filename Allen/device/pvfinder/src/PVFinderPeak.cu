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
#include "PVFinderPeak.cuh"

#include <fstream>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_peak::pvfinder_peak_t)

void pvfinder_peak::pvfinder_peak_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  set_size<dev_zpeaks_t>(arguments, first<host_number_of_events_t>(arguments) * PV::max_number_vertices);
  set_size<dev_number_of_zpeaks_t>(arguments, first<host_number_of_events_t>(arguments));
}

void pvfinder_peak::pvfinder_peak_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  if (m_block_dim.value().x > max_block_dim) {
    throw StrException("pvfinder_peak: block_dim.x must not exceed " + std::to_string(max_block_dim));
  }

  global_function(pvfinder_peak)(dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
    arguments, m_threshold, m_integral_threshold, m_min_width, m_split_peaks);

  const std::string& dump_dir = m_dump_dir.value();
  if (!dump_dir.empty() && !m_dump_done) {
    const auto zpeaks = make_host_buffer<dev_zpeaks_t>(arguments, context);
    const auto number_of_zpeaks = make_host_buffer<dev_number_of_zpeaks_t>(arguments, context);
    const auto event_list = make_host_buffer<dev_event_list_t>(arguments, context);
    const uint32_t n_events = first<host_number_of_events_t>(arguments);
    std::vector<uint32_t> in_list(n_events, 0);
    for (const auto event : event_list) {
      in_list[event] = 1;
    }
    std::ofstream file {dump_dir + "/allen_zpeaks.bin", std::ios::binary};
    const uint32_t magic = 0xAB1EU;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&n_events), sizeof(n_events));
    for (uint32_t event = 0; event < n_events; ++event) {
      const uint32_t n = in_list[event] ? number_of_zpeaks[event] : 0;
      file.write(reinterpret_cast<const char*>(&n), sizeof(n));
      file.write(
        reinterpret_cast<const char*>(zpeaks.data() + event * PV::max_number_vertices),
        PV::max_number_vertices * sizeof(float));
    }
    if (!file) throw StrException("pvfinder_peak: cannot write " + dump_dir + "/allen_zpeaks.bin");
    info_cout << "[pvfinder_peak] validation dump written to " << dump_dir << " (" << n_events << " events)\n";
    m_dump_done = true;
  }
}

namespace {
  // Scans the peaks of the run of bins at or above threshold that starts at
  // bin `begin`, in increasing z, and calls emit(z) for each accepted one.
  // Bins are processed as in pv-finder's pv_locations_updated (one event's
  // 4000 bins in a row), except that the bin before the first is taken as 0
  // (pv-finder reads the last bin there, a wrap-around of its Python indexing).
  template<typename Emit>
  __device__ void scan_run(
    const float* kde,
    const unsigned begin,
    const float threshold,
    const float integral_threshold,
    const unsigned min_width,
    const bool split_peaks,
    Emit&& emit)
  {
    using namespace PVFinderConstants;
    unsigned width = 0;
    float integral = 0.f;
    float weighted_sum = 0.f;
    bool peak_passed = false;
    for (unsigned i = begin; i < KDE::n_bins; ++i) {
      const float value = kde[i];
      const float previous = i > 0 ? kde[i - 1] : 0.f;
      const bool on = value >= threshold;
      if (on) {
        ++width;
        integral += value;
        weighted_sum += static_cast<float>(i) * value;
        if (previous > value + Peak::split_min_drop && previous > Peak::split_min_ratio * value) {
          peak_passed = true;
        }
      }
      const bool last = i == KDE::n_bins - 1;
      const bool split = split_peaks && peak_passed && previous < value;
      if ((!on || last || split) && width > 0) {
        if (width >= min_width && integral >= integral_threshold) {
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
} // namespace

__global__ void pvfinder_peak::pvfinder_peak(
  pvfinder_peak::Parameters parameters,
  const float threshold,
  const float integral_threshold,
  const unsigned min_width,
  const bool split_peaks)
{
  using namespace PVFinderConstants;
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  const float* event_kde = parameters.dev_pvfinder_kde_output + event_number * KDE::n_bins;
  float* zpeaks = parameters.dev_zpeaks + event_number * PV::max_number_vertices;

  // Each thread owns the runs that start in its contiguous share of the bins.
  // First pass: count their peaks; second pass: write them after the peaks of
  // the threads before it, so the seeds come out in increasing z.
  __shared__ unsigned peak_offset[max_block_dim];

  // The event's KDE in shared memory (16 KB), loaded coalesced: each thread
  // scans its bins, and runs past them, twice.
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
      const bool starts_run = kde[i] >= threshold && (i == 0 || kde[i - 1] < threshold);
      if (starts_run) {
        scan_run(kde, i, threshold, integral_threshold, min_width, split_peaks, emit);
      }
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
  if (threadIdx.x == 0) {
    // At most PV::max_number_vertices seeds, the lowest in z, like pv_beamline_peak.
    parameters.dev_number_of_zpeaks[event_number] = min(peak_offset[blockDim.x - 1], PV::max_number_vertices);
  }

  unsigned index = peak_offset[threadIdx.x] - number_of_peaks;
  for_each_run([&](float z) {
    if (index < PV::max_number_vertices) zpeaks[index] = z;
    ++index;
  });
}
