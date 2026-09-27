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

namespace PVFinderConstants {

  // The z range the networks were trained on: 40 intervals of 100 bins,
  // z in [z_min, z_min + n_intervals * interval_width) mm. Interval i covers
  // [z_min + i * interval_width, z_min + (i + 1) * interval_width); bin b of
  // an event's KDE (b = interval * n_bins_per_interval + bin in interval)
  // covers [z_min + b * bin_width, z_min + (b + 1) * bin_width).
  namespace KDE {
    static constexpr float z_min = -100.f;        // unit: mm
    static constexpr float interval_width = 10.f; // unit: mm
    static constexpr unsigned n_intervals = 40;
    static constexpr unsigned n_bins_per_interval = 100;
    static constexpr unsigned n_bins = n_intervals * n_bins_per_interval;
    static constexpr float bin_width = interval_width / n_bins_per_interval; // unit: mm
    static constexpr float z_max = z_min + n_intervals * interval_width;     // unit: mm
  }                                                                          // namespace KDE

  // Peak finding on the KDE, as in pv-finder's pv_locations_updated with the
  // efficiency_config of its default configuration.
  namespace Peak {
    static constexpr float threshold = 0.07f;         // a bin at or above this value is "on"
    static constexpr float integral_threshold = 0.7f; // minimum sum of a peak's bins
    static constexpr unsigned min_width = 0;          // minimum number of bins of a peak
    // A peak is split where the KDE rises again after falling from bin i - 1
    // to bin i by more than both of these.
    static constexpr float split_min_drop = 0.05f;
    static constexpr float split_min_ratio = 1.1f;
  } // namespace Peak

} // namespace PVFinderConstants
