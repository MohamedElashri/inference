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
#pragma once

#include <Common.h>
#include <CheckerTypes.h>
#include <CheckerInvoker.h>
#include "BackendCommon.h"
#include <mutex>

double binomial_error(int n, int k);
void print_rate(const size_t number_of_pass, const size_t requested_events, const double in_rate);

class RateChecker : public Checker::BaseChecker {

private:
  // Event counters.
  std::vector<unsigned> m_line_counters;
  std::vector<std::string> m_line_names;
  std::vector<std::string> m_group_names;
  std::vector<unsigned> m_group_counters;
  std::vector<std::vector<char>> m_group_masks;
  std::vector<char> m_fired_lines;
  std::vector<char> m_fired_masks;
  unsigned m_total;

  std::mutex m_mutex;

  std::vector<char> get_masks_for_lines(
    const std::vector<std::string>& line_names,
    const std::vector<std::string>& lines_to_mask) const;

  void print_rate(
    const std::string line_name,
    unsigned longest_string,
    const size_t number_of_pass,
    const size_t requested_events,
    const double in_rate) const;

public:
  RateChecker(CheckerInvoker const*, std::string const&, std::string const&) { m_total = 0; }
  void accumulate(
    const char* names_of_lines,
    const std::string& json_string,
    std::span<const unsigned> dec_reports,
    const unsigned number_of_events);

  void report(const size_t requested_events) const override;
};
