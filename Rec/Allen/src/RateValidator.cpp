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
// ----------------------------------------------------------------------------
// Gaudi multi-event replacement for the Allen host_rate_validator algorithm.
// It accumulates the per-line / per-group / inclusive rates of an Allen slice
// from the HLT1 decision reports and prints them in finalize() using the same
// layout as the former RateChecker.
// ----------------------------------------------------------------------------
// Gaudi
#include "Gaudi/Accumulators.h"
#include "GaudiAlg/Consumer.h"
#include "GaudiKernel/StdArrayAsProperty.h"

// Allen
#include "AllenBuffer.cuh"
#include "HltDecReport.cuh"

// Standard
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <format>
#include <iterator>
#include <ranges>
#include <regex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace {
  using RateCounter = Gaudi::Accumulators::Counter<>;
  using RateBuffer = decltype(std::declval<RateCounter&>().buffer());

#ifndef NDEBUG
  std::vector<std::string_view> split_line_names(std::string_view names)
  {
    std::vector<std::string_view> result;
    for (auto subrange : std::views::split(names, ',')) {
      result.emplace_back(std::ranges::data(subrange), std::ranges::size(subrange));
    }
    return result;
  }
#endif

  double rate_binomial_error(size_t n, size_t k) { return 1. / n * std::sqrt(1. * k * (1. - 1. * k / n)); }
} // namespace

class RateValidator final : public Gaudi::Functional::Consumer<void(
                              const Allen::host_buffer<unsigned>&,
                              const Allen::host_buffer<char>&,
                              const Allen::host_buffer<unsigned>&,
                              const Allen::host_buffer<unsigned>&)> {
public:
  RateValidator(const std::string& name, ISvcLocator* pSvcLocator);

  StatusCode initialize() override;

  void operator()(
    const Allen::host_buffer<unsigned>& allen_number_of_events,
    const Allen::host_buffer<char>& allen_names_of_lines,
    const Allen::host_buffer<unsigned>& allen_number_of_active_lines,
    const Allen::host_buffer<unsigned>& allen_dec_reports) const override;

  StatusCode finalize() override;

private:
#ifndef NDEBUG
  bool check_line_names(const Allen::host_buffer<char>& allen_names) const;
#endif

  void print_rate(
    const std::string& line_name,
    size_t longest_string,
    size_t number_of_pass,
    size_t requested_events,
    double in_rate) const;

  Gaudi::Property<std::vector<std::string>> m_line_names {
    this,
    "Hlt1LineNames",
    {},
    "Ordered list of Allen line names"};

  Gaudi::Property<std::string> m_json_string {
    this,
    "json_string",
    "",
    "Names of lines, grouped by type in a JSON compatible string."};

  Gaudi::Property<bool> m_isMultiEvent {this, "IsMultiEvent", true, ""};

  // Plain atomic counters: one increment per fired line (rather than a
  // BinomialCounter's true+false pair), with a single shared event denominator.
  mutable std::deque<RateCounter> m_line_rates {};
  mutable std::deque<RateCounter> m_group_rates {};
  mutable RateCounter m_inclusive_rate {this, "Selected by Hlt1GlobalDecision"};
  mutable RateCounter m_n_events {this, "Number of events"};

  // Read-only after initialize(): for each line, the groups it belongs to.
  std::vector<std::string> m_group_names {};
  std::vector<std::vector<unsigned>> m_line_group_indices {};
};

DECLARE_COMPONENT(RateValidator)

RateValidator::RateValidator(const std::string& name, ISvcLocator* pSvcLocator) :
  Consumer(
    name,
    pSvcLocator,
    // Inputs
    {KeyValue {"allen_number_of_events", ""},
     KeyValue {"allen_names_of_lines", ""},
     KeyValue {"allen_number_of_active_lines", ""},
     KeyValue {"allen_dec_reports", ""}})
{}

StatusCode RateValidator::initialize()
{
  auto status = Consumer::initialize();
  if (!status.isSuccess()) return status;

  // One accumulator per configured line.
  m_line_rates.clear();
  for (const auto& line_name : m_line_names.value()) {
    m_line_rates.emplace_back(this, "Selected by " + line_name);
  }

  // One accumulator per configured group, and for each line the list of groups
  // it contributes to. This avoids recomputing masks for every event.
  const auto& line_names = m_line_names.value();
  m_group_names.clear();
  m_group_rates.clear();
  m_line_group_indices.clear();
  m_line_group_indices.resize(line_names.size());
  if (!m_json_string.value().empty()) {
    std::smatch match;
    std::regex regex_pattern("json:(.*)");
    if (std::regex_search(m_json_string.value(), match, regex_pattern) && match.size() > 1) {
      nlohmann::json json_dict = nlohmann::json::parse(match[1].str());
      for (auto& [key, value] : json_dict.items()) {
        const auto group_index = static_cast<unsigned>(m_group_names.size());
        m_group_names.push_back(key);
        m_group_rates.emplace_back(this, "Selected by " + key + " group");
        for (const auto& line_name : value.get<std::vector<std::string>>()) {
          const auto it = std::find(line_names.begin(), line_names.end(), line_name);
          if (it != line_names.end()) {
            m_line_group_indices[std::distance(line_names.begin(), it)].push_back(group_index);
          }
        }
      }
    }
  }

  return status;
}

#ifndef NDEBUG
bool RateValidator::check_line_names(const Allen::host_buffer<char>& allen_names) const
{
  const auto internal_names = split_line_names(std::string_view {allen_names.data()});
  const auto& property_names = m_line_names.value();

  bool match = true;
  for (std::size_t i = 0; i < property_names.size(); ++i) {
    const std::string_view internal_name = i < internal_names.size() ? internal_names[i] : std::string_view {};
    if (internal_name != property_names[i]) {
      match = false;
      error() << "Mismatch with internal line name: (#, <property>, <internal>): (" << i << ", " << property_names[i]
              << ", " << std::string(internal_name) << ")" << endmsg;
    }
  }
  return match;
}
#endif

void RateValidator::operator()(
  const Allen::host_buffer<unsigned>& allen_number_of_events,
  [[maybe_unused]] const Allen::host_buffer<char>& allen_names_of_lines,
  [[maybe_unused]] const Allen::host_buffer<unsigned>& allen_number_of_active_lines,
  const Allen::host_buffer<unsigned>& allen_dec_reports) const
{
#ifndef NDEBUG
  if (!check_line_names(allen_names_of_lines)) {
    error() << "Mismatch between external (property) and internal (allen) line lists. Misalignment of counters likely."
            << endmsg;
  }
#endif

  const unsigned number_of_events = allen_number_of_events[0];
  assert(allen_number_of_active_lines[0] == m_line_rates.size());

  m_n_events += number_of_events;

  // Thread-local buffers: accumulate without atomics and merge each counter
  // once, when the buffers go out of scope at the end of this call.
  std::vector<RateBuffer> line_buffers;
  line_buffers.reserve(m_line_rates.size());
  for (auto& counter : m_line_rates) {
    line_buffers.emplace_back(counter);
  }

  std::vector<RateBuffer> group_buffers;
  group_buffers.reserve(m_group_rates.size());
  for (auto& counter : m_group_rates) {
    group_buffers.emplace_back(counter);
  }

  auto inclusive_buffer = m_inclusive_rate.buffer();

  // Per-call scratch space, reset for every event, so that concurrent slices do
  // not race and fired groups never leak from one event to the next.
  std::vector<bool> fired_groups(m_group_names.size());

  const std::span<const unsigned> dec_reports_data = allen_dec_reports.get();
  for (auto i_event = 0u; i_event < number_of_events; ++i_event) {
    HltDecReports dec_reports {dec_reports_data, i_event};
    assert(dec_reports.number_of_lines() == m_line_rates.size());

    std::fill(fired_groups.begin(), fired_groups.end(), false);

    bool any_line_fired = false;
    for (HltDecReport dec_report : dec_reports) {
      const auto line_index = dec_report.line_index();
      const auto decision = dec_report.decision();
      if (decision) {
        ++line_buffers[line_index];
        any_line_fired = true;
        for (const auto group_index : m_line_group_indices[line_index]) {
          fired_groups[group_index] = true;
        }
      }
    }

    if (any_line_fired) ++inclusive_buffer;
    for (unsigned i_group = 0; i_group < group_buffers.size(); ++i_group) {
      if (fired_groups[i_group]) ++group_buffers[i_group];
    }
  }
}

StatusCode RateValidator::finalize()
{
  // Assume 30 MHz input rate.
  const double in_rate = 30000.0;
  size_t longest_string = 10;
  for (const auto& line_name : m_line_names.value()) {
    longest_string = std::max(longest_string, line_name.length());
  }
  for (const auto& group_name : m_group_names) {
    longest_string = std::max(longest_string, group_name.length());
  }

  const size_t requested_events = m_n_events.nEntries();

  // Keep the same block delimiters as the former CheckerInvoker report.
  info() << std::format("{} validation:", name()) << endmsg;

  for (unsigned i_line = 0; i_line < m_line_names.value().size(); i_line++) {
    print_rate(
      m_line_names.value()[i_line],
      longest_string,
      m_line_rates[i_line].nEntries(),
      requested_events,
      in_rate); // Print the rate for each line
  }
  for (unsigned i_group = 0; i_group < m_group_names.size(); i_group++) {
    print_rate(
      m_group_names[i_group],
      longest_string,
      m_group_rates[i_group].nEntries(),
      requested_events,
      in_rate); // Print the rate for each group of lines
  }

  print_rate(
    "Inclusive",
    longest_string,
    m_inclusive_rate.nEntries(),
    requested_events,
    in_rate); // Print the inclusive rate

  return Consumer::finalize();
}

void RateValidator::print_rate(
  const std::string& line_name,
  size_t longest_string,
  size_t number_of_pass,
  size_t requested_events,
  double in_rate) const
{
  const std::string padding(longest_string > line_name.size() ? longest_string - line_name.size() : 0, ' ');
  info() << std::format(
              "{}:{} {:>6}/{:>6}, ({:8.2f} +/- {:8.2f}) kHz",
              line_name,
              padding,
              number_of_pass,
              requested_events,
              1. * number_of_pass / requested_events * in_rate,
              rate_binomial_error(requested_events, number_of_pass) * in_rate)
         << endmsg;
}
