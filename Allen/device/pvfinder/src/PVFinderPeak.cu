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
  if (m_block_dim.value().x > PVFinderPeakFinding::max_block_dim) {
    throw StrException(
      "pvfinder_peak: block_dim.x must not exceed " + std::to_string(PVFinderPeakFinding::max_block_dim));
  }

  const PVFinderPeakFinding::Cuts cuts {m_threshold, m_integral_threshold, m_min_width, m_split_peaks};
  global_function(pvfinder_peak)(dim3(size<dev_event_list_t>(arguments)), m_block_dim, context)(
    arguments,
    cuts,
    m_seeds.data(context),
    m_truncated.data(context),
    m_histogram_n_seeds.data(context),
    m_histogram_seed_z.data(context));

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

__global__ void pvfinder_peak::pvfinder_peak(
  pvfinder_peak::Parameters parameters,
  const PVFinderPeakFinding::Cuts cuts,
  Allen::Monitoring::AveragingCounter<>::DeviceType dev_n_seeds_counter,
  Allen::Monitoring::Counter<>::DeviceType dev_truncated_counter,
  Allen::Monitoring::Histogram<>::DeviceType dev_n_seeds_histo,
  Allen::Monitoring::Histogram<>::DeviceType dev_seed_z_histo)
{
  using namespace PVFinderConstants;
  const unsigned event_number = parameters.dev_event_list[blockIdx.x];
  float* zpeaks = parameters.dev_zpeaks + event_number * PV::max_number_vertices;
  const unsigned number_of_peaks =
    PVFinderPeakFinding::find_peaks(parameters.dev_pvfinder_kde_output + event_number * KDE::n_bins, zpeaks, cuts);
  const unsigned number_of_seeds = min(number_of_peaks, PV::max_number_vertices);

  for (unsigned i = threadIdx.x; i < number_of_seeds; i += blockDim.x) {
    dev_seed_z_histo.increment(zpeaks[i]);
  }
  if (threadIdx.x == 0) {
    parameters.dev_number_of_zpeaks[event_number] = number_of_seeds;
    dev_n_seeds_counter.add(number_of_seeds);
    dev_n_seeds_histo.increment(number_of_seeds);
    if (number_of_peaks > PV::max_number_vertices) dev_truncated_counter.increment();
  }
}
