/*****************************************************************************\
* (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "DecReporter.cuh"
#include "HltDecReport.cuh"
#include "SelectionsEventModel.cuh"
#include <PrefixSum.cuh>
#include <string>
#include <sstream>

INSTANTIATE_ALGORITHM(dec_reporter::dec_reporter_t)

void dec_reporter::dec_reporter_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  auto const n_lines = first<host_number_of_active_lines_t>(arguments);
  set_size<dev_dec_reports_t>(
    arguments, HltDecReports<false>::size(n_lines) * first<host_number_of_events_t>(arguments));
  set_size<host_dec_reports_t>(
    arguments, HltDecReports<false>::size(n_lines) * first<host_number_of_events_t>(arguments));
  set_size<dev_max_objects_offsets_t>(arguments, n_lines * first<host_number_of_events_t>(arguments) + 1);
  set_size<host_max_objects_t>(arguments, 1);
}

void dec_reporter::dec_reporter_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  Allen::memset_async<host_dec_reports_t>(arguments, 0, context);
  Allen::memset_async<dev_max_objects_offsets_t>(arguments, 0, context);

  global_function(dec_reporter)(dim3(first<host_number_of_events_t>(arguments)), m_block_dim, context)(
    arguments, m_key, m_tck, m_taskID);

  Allen::copy_async<host_dec_reports_t, dev_dec_reports_t>(arguments, context);

  PrefixSum::prefix_sum<dev_max_objects_offsets_t, host_max_objects_t>(*this, arguments, context);

  // Only perform the memcpy if the number of selected objects exceeds the warning limit
  unsigned number_of_events = first<host_number_of_events_t>(arguments);
  unsigned total_warn_limit = m_warn_mean_event_limit * number_of_events;
  if (first<host_max_objects_t>(arguments) > total_warn_limit) {
    warning_cout << "Warning: Maximum number of objects in slice exceeded averaged limit of " << m_warn_mean_event_limit
                 << std::endl;
    unsigned number_of_lines = first<host_number_of_active_lines_t>(arguments);

    // Get the names of the active lines in a std::string vector
    std::vector<std::string> line_names;
    auto char_begin = data<host_names_of_active_lines_t>(arguments);
    auto char_end = char_begin + size<host_names_of_active_lines_t>(arguments);
    std::string strdata(char_begin, char_end);
    std::stringstream data(strdata);
    std::string line_name;
    while (std::getline(data, line_name, ',')) {
      line_names.push_back(line_name);
    }

    auto host_max_objects_offsets = make_host_buffer<dev_max_objects_offsets_t>(arguments, context);

    for (unsigned event = 0; event < number_of_events; ++event) {
      for (unsigned line_index = 0; line_index < number_of_lines; ++line_index) {
        unsigned index = event * number_of_lines + line_index;
        unsigned number_of_candidates = host_max_objects_offsets[index + 1] - host_max_objects_offsets[index];
        // Trigger line limit warning when it exceeds N-candidates in a single event
        if (number_of_candidates > m_warn_line_limit) {
          warning_cout << "  Warning: Event " << event << ", Line " << line_names[line_index] << ": "
                       << number_of_candidates << " objects selected." << std::endl;
        }
      }
    }
  }
}

__global__ void
dec_reporter::dec_reporter(dec_reporter::Parameters parameters, unsigned key, unsigned tck, unsigned task_id)
{
  const auto event_index = blockIdx.x;
  const auto number_of_events = gridDim.x;

  // Selections view
  Selections::ConstSelections selections {
    parameters.dev_selections, parameters.dev_selections_offsets, number_of_events};

  HltDecReports<false> reports(parameters.dev_dec_reports.get(), event_index, parameters.dev_number_of_active_lines[0]);
  unsigned* event_selected_candidates_counts =
    parameters.dev_max_objects_offsets + event_index * parameters.dev_number_of_active_lines[0];

  if (threadIdx.x == 0) {
    // Set TCK and taskID for each event dec report
    reports.set_number_of_lines(parameters.dev_number_of_active_lines[0]);
    reports.set_key(key);
    reports.set_tck(tck);
    reports.set_task_id(task_id);
  }

  __syncthreads();

  for (unsigned line_index = threadIdx.x; line_index < reports.number_of_lines(); line_index += blockDim.x) {
    // Iterate all elements and get a decision for the current {event, line}
    auto span_popcount = selections.count_span_population(line_index, event_index);
    bool final_decision = span_popcount > 0;
    event_selected_candidates_counts[line_index] = span_popcount;

    reports.set_dec_report(
      line_index,
      HltDecReport {
        final_decision,
        std::byte {0},          // error
        static_cast<std::byte>( // number of candidates
          std::min(event_selected_candidates_counts[line_index], 15U)),
        std::byte {1},                                 // execution stage
        static_cast<unsigned short>(line_index + 1)}); // decision ID
  }
}
