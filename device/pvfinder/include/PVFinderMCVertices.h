/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
\*****************************************************************************/
#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace PVFinder {
  struct MCVertex {
    int32_t numberTracks;
    double x, y, z;
  };

  // The serialized MDF MC-PV payload used by Allen's former MCEvent reader:
  // int32 count, then count records of int32 tracks and three float64 coordinates.
  inline std::vector<MCVertex> read_mc_vertices(std::span<const char> payload)
  {
    constexpr size_t record_size = sizeof(int32_t) + 3 * sizeof(double);
    if (payload.size() < sizeof(int32_t)) throw std::runtime_error("truncated MC-PV count");
    int32_t count;
    std::memcpy(&count, payload.data(), sizeof(count));
    if (count < 0 || static_cast<size_t>(count) > (payload.size() - sizeof(count)) / record_size)
      throw std::runtime_error("invalid MC-PV count");
    if (payload.size() != sizeof(count) + static_cast<size_t>(count) * record_size)
      throw std::runtime_error("unexpected MC-PV payload size");
    std::vector<MCVertex> vertices;
    vertices.reserve(count);
    const char* cursor = payload.data() + sizeof(count);
    for (int32_t i = 0; i < count; ++i) {
      MCVertex vertex;
      std::memcpy(&vertex.numberTracks, cursor, sizeof(vertex.numberTracks));
      cursor += sizeof(vertex.numberTracks);
      for (double* value : {&vertex.x, &vertex.y, &vertex.z}) {
        std::memcpy(value, cursor, sizeof(*value));
        cursor += sizeof(*value);
      }
      vertices.push_back(vertex);
    }
    return vertices;
  }
} // namespace PVFinder
