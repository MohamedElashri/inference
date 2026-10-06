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

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
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

TEST_CASE("unit_tests.monitoring.device_axis_flow_bins", "[monitoring]")
{
  const Allen::Monitoring::DeviceAxis<float> axis(4, 0.f, 4.f);

  REQUIRE(axis.index(-1.f) == 0);   // underflow
  REQUIRE(axis.index(0.f) == 1);    // first in-range bin
  REQUIRE(axis.index(0.99f) == 1);  // first in-range bin
  REQUIRE(axis.index(1.f) == 2);    // second in-range bin
  REQUIRE(axis.index(3.999f) == 4); // last in-range bin
  REQUIRE(axis.index(4.f) == 5);    // overflow (maxValue is excluded)
  REQUIRE(axis.index(100.f) == 5);  // overflow
  // NaN fails every comparison and must not produce an out-of-bounds index
  REQUIRE(axis.index(std::numeric_limits<float>::quiet_NaN()) == 5);
}

TEST_CASE("unit_tests.monitoring.device_log_axis_flow_bins", "[monitoring]")
{
  const Allen::Monitoring::DeviceLogAxis axis(2, 1.f, 100.f, 1.f, std::log10(2.f), 0.f);

  REQUIRE(axis.index(0.f) == 0);    // log10(0) = -inf -> underflow
  REQUIRE(axis.index(0.5f) == 0);   // underflow
  REQUIRE(axis.index(1.f) == 1);    // first in-range bin
  REQUIRE(axis.index(30.f) == 2);   // second in-range bin
  REQUIRE(axis.index(100.f) == 3);  // overflow
  REQUIRE(axis.index(1000.f) == 3); // overflow
}

TEST_CASE("unit_tests.monitoring.histogram_flow_bins", "[monitoring]")
{
  // One axis with two in-range bins -> two flow bins + two inner bins
  std::array<unsigned, 4> buffer {};
  Allen::Monitoring::DeviceNDHistogram<unsigned, Allen::Monitoring::Axis<float>> histogram(
    buffer.data(), std::make_tuple(Allen::Monitoring::Axis<float>(2, 0.f, 2.f)));

  histogram.increment(-1.f); // underflow
  histogram.increment(0.5f); // first in-range bin
  histogram.increment(1.5f); // second in-range bin
  histogram.increment(10.f); // overflow

  REQUIRE(buffer[0] == 1);
  REQUIRE(buffer[1] == 1);
  REQUIRE(buffer[2] == 1);
  REQUIRE(buffer[3] == 1);
}

TEST_CASE("unit_tests.monitoring.histogram2d_flow_bins", "[monitoring]")
{
  // Two axes with two in-range bins each -> (2 + 2)^2 bins, x varying fastest
  std::array<unsigned, 16> buffer {};
  const Allen::Monitoring::Axis<float> axis(2, 0.f, 2.f);
  Allen::Monitoring::DeviceNDHistogram<unsigned, Allen::Monitoring::Axis<float>, Allen::Monitoring::Axis<float>>
    histogram(buffer.data(), std::make_tuple(axis, axis));

  histogram.increment(-1.f, 0.5f); // x underflow, y first bin -> 0 + 4 * 1 = 4
  histogram.increment(0.5f, 1.5f); // x first bin, y second bin -> 1 + 4 * 2 = 9
  histogram.increment(10.f, 10.f); // x overflow, y overflow -> 3 + 4 * 3 = 15

  REQUIRE(buffer[4] == 1);
  REQUIRE(buffer[9] == 1);
  REQUIRE(buffer[15] == 1);

  unsigned total = 0;
  for (auto count : buffer) {
    total += count;
  }
  REQUIRE(total == 3);
}
