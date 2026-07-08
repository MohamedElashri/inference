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

#if __has_include(<catch2/catch.hpp>)
#include <catch2/catch.hpp>
#else
#include <catch2/catch_test_macros.hpp>
#endif

#include "AllenMonitoring.h"

#include <nlohmann/json.hpp>

#include <vector>

namespace {
  struct TestHistogram {
    std::vector<double> m_bins;
  };
} // namespace

TEST_CASE("unit_tests.monitoring.histogram_bin_counter_handles_missing_histogram", "[monitoring]")
{
  Allen::Monitoring::HistogramBinAsCounter<TestHistogram> counter;

  nlohmann::json j = counter;

  REQUIRE(j.at("type") == "counter:Counter:d");
  REQUIRE(j.at("empty") == true);
  REQUIRE(j.at("nEntries") == 0.0);
}

TEST_CASE("unit_tests.monitoring.histogram_bin_counter_handles_missing_bin", "[monitoring]")
{
  TestHistogram histogram;
  histogram.m_bins = {0.0};

  Allen::Monitoring::HistogramBinAsCounter<TestHistogram> counter;
  counter.m_histo = &histogram;
  counter.m_bin = 0;

  nlohmann::json j = counter;

  REQUIRE(j.at("empty") == true);
  REQUIRE(j.at("nEntries") == 0.0);
}

TEST_CASE("unit_tests.monitoring.histogram_bin_counter_reads_gaudi_bin", "[monitoring]")
{
  TestHistogram histogram;
  histogram.m_bins = {0.0, 42.0};

  Allen::Monitoring::HistogramBinAsCounter<TestHistogram> counter;
  counter.m_histo = &histogram;
  counter.m_bin = 0;

  nlohmann::json j = counter;

  REQUIRE(j.at("empty") == false);
  REQUIRE(j.at("nEntries") == 42.0);
}
