/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "RateChecker.h"
#include "ProgramOptions.h"
#include "HltDecReport.cuh"
#include <nlohmann/json.hpp>
#include <regex>

double binomial_error(int n, int k) { return 1. / n * std::sqrt(1. * k * (1. - 1. * k / n)); }

std::vector<char> RateChecker::get_masks_for_lines(
  const std::vector<std::string>& line_names,
  const std::vector<std::string>& lines_to_mask) const
{
  // Set initial masks to 0.
  std::vector<char> masks(line_names.size(), 0);
  for (const auto& name : lines_to_mask) {
    const auto it = std::find(line_names.begin(), line_names.end(), name);
    if (it != line_names.end()) {
      const auto index = std::distance(line_names.begin(), it);
      masks[index] = 1;
    }
  }
  return masks;
}

void RateChecker::accumulate(
  const char* line_names,
  const std::string& json_string,
  std::span<const unsigned> dec_reports_data,
  const unsigned number_of_events)
{
  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_line_counters.size()) {
    // Setup individual lines and their counters
    m_line_names = split_string(line_names, ",");
    m_line_counters = std::vector<unsigned>(m_line_names.size(), 0);
    // Initialize the size for this once
    m_fired_lines.resize(m_line_names.size());
    m_fired_masks.resize(m_line_names.size());
    // Setup the masks for different line groups, if provided.
    if (!json_string.empty()) {
      std::smatch match;
      std::regex regex_pattern("json:(.*)");
      // Check if pattern is valid for grouping lines
      if (std::regex_search(json_string, match, regex_pattern) && match.size() > 1) {
        std::string json_payload = match[1].str();
        nlohmann::json json_dict = nlohmann::json::parse(json_payload);
        for (auto& [key, value] : json_dict.items()) {
          m_group_names.push_back(key);
          m_group_masks.push_back(get_masks_for_lines(m_line_names, value.get<std::vector<std::string>>()));
        }
        m_group_counters = std::vector<unsigned>(m_group_names.size(), 0);
      }
    }
  }

  for (auto i = 0u; i < number_of_events; ++i) {
    HltDecReports dec_reports {dec_reports_data, i};
    assert(dec_reports.number_of_lines() == m_line_names.size());

    // Get the individual lines decision
    for (HltDecReport dec_report : dec_reports) {
      const auto line_index = dec_report.line_index();
      const auto decision = dec_report.decision();
      if (decision) ++m_line_counters[line_index];
      m_fired_lines[line_index] = decision;
    }

    const bool any_line_fired =
      std::any_of(m_fired_lines.begin(), m_fired_lines.end(), [](const char fired) { return fired; });
    if (any_line_fired) {
      ++m_total;
      // Check the counters for each group of lines
      for (unsigned i = 0; i < m_group_masks.size(); ++i) {
        std::transform(
          m_fired_lines.begin(),
          m_fired_lines.end(),
          m_group_masks[i].begin(),
          m_fired_masks.begin(),
          std::logical_and<char>());
        const bool any_in_group_fired =
          std::any_of(m_fired_masks.begin(), m_fired_masks.end(), [](const char fired) { return fired; });
        if (any_in_group_fired) ++m_group_counters[i];
      }
    }
  }
}

void RateChecker::report(const size_t requested_events) const
{
  // Assume 30 MHz input rate.
  const double in_rate = 30000.0;
  size_t longest_string = 10;
  for (const auto& line_name : m_line_names) {
    if (line_name.length() > longest_string) {
      longest_string = line_name.length();
    }
  }
  for (const auto& group_name : m_group_names) {
    if (group_name.length() > longest_string) {
      longest_string = group_name.length();
    }
  }

  for (unsigned i_line = 0; i_line < m_line_names.size(); i_line++) {
    print_rate(
      m_line_names[i_line],
      longest_string,
      m_line_counters[i_line],
      requested_events,
      in_rate); // Print the rate for each line
  }
  for (unsigned i_group = 0; i_group < m_group_names.size(); i_group++) {
    print_rate(
      m_group_names[i_group],
      longest_string,
      m_group_counters[i_group],
      requested_events,
      in_rate); // Print the rate for each group of lines
  }

  print_rate("Inclusive", longest_string, m_total, requested_events,
             in_rate); // Print the inclusive rate
  std::printf("\n");
}

void RateChecker::print_rate(
  const std::string line_name,
  unsigned longest_string,
  const size_t number_of_pass,
  const size_t requested_events,
  const double in_rate) const
{
  std::printf("%s:", line_name.c_str());
  for (unsigned i = 0; i < longest_string - line_name.length(); ++i) {
    std::printf(" ");
  }
  std::printf(
    " %6lu/%6lu, (%8.2f +/- %8.2f) kHz\n",
    number_of_pass,
    requested_events,
    1. * number_of_pass / requested_events * in_rate,
    binomial_error(requested_events, number_of_pass) * in_rate);
}
