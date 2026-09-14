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

#include <ostream>
#include <algorithm>

namespace Allen::Store {
  struct AllocationReport {
    size_t total_allocated_bytes = 0;
    size_t total_freed_bytes = 0;
    size_t current_bytes_in_use = 0;
    size_t peak_bytes_in_use = 0;
    size_t n_allocations = 0;
    size_t n_frees = 0;

    void report_allocation(size_t bytes)
    {
      total_allocated_bytes += bytes;
      current_bytes_in_use += bytes;
      n_allocations++;
      peak_bytes_in_use = std::max(peak_bytes_in_use, current_bytes_in_use);
    }

    void report_free(size_t bytes)
    {
      total_freed_bytes += bytes;
      current_bytes_in_use -= bytes;
      n_frees++;
    }

    void max(const AllocationReport& other)
    {
      total_allocated_bytes = std::max(total_allocated_bytes, other.total_allocated_bytes);
      total_freed_bytes = std::max(total_freed_bytes, other.total_freed_bytes);
      current_bytes_in_use = std::max(current_bytes_in_use, other.current_bytes_in_use);
      n_allocations = std::max(n_allocations, other.n_allocations);
      n_frees = std::max(n_frees, other.n_frees);
      peak_bytes_in_use = std::max(peak_bytes_in_use, other.peak_bytes_in_use);
    }

    friend std::ostream& operator<<(std::ostream& os, const AllocationReport& report)
    {
      os << "Allocation Report:\n"
         << "  Total Allocated Bytes (MB): " << static_cast<double>(report.total_allocated_bytes) / (1024 * 1024)
         << "\n"
         << "  Total Freed Bytes (MB): " << static_cast<double>(report.total_freed_bytes) / (1024 * 1024) << "\n"
         << "  Current Bytes in Use (MB): " << static_cast<double>(report.current_bytes_in_use) / (1024 * 1024) << "\n"
         << "  Peak Bytes in Use (MB): " << static_cast<double>(report.peak_bytes_in_use) / (1024 * 1024) << "\n"
         << "  Number of Allocations: " << report.n_allocations << "\n"
         << "  Number of Frees: " << report.n_frees;
      return os;
    }
  };
} // namespace Allen::Store
