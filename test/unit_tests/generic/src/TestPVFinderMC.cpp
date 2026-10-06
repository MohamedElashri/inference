/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
\*****************************************************************************/
#if __has_include(<catch2/catch.hpp>)
#include <catch2/catch.hpp>
#else
#include <catch2/catch_test_macros.hpp>
#endif
#include "PVFinderMCVertices.h"

namespace {
  template<typename T>
  void append(std::vector<char>& data, T value)
  {
    const auto* bytes = reinterpret_cast<const char*>(&value);
    data.insert(data.end(), bytes, bytes + sizeof(value));
  }
}

TEST_CASE("PVFinder reads serialized MDF MC primary vertices", "[PVFinder]")
{
  std::vector<char> payload;
  append(payload, int32_t {2});
  append(payload, int32_t {7});
  append(payload, double {0.125});
  append(payload, double {-0.25});
  append(payload, double {-83.75});
  append(payload, int32_t {0});
  append(payload, double {1.5});
  append(payload, double {2.75});
  append(payload, double {319.125});
  const auto vertices = PVFinder::read_mc_vertices(payload);
  REQUIRE(vertices.size() == 2);
  REQUIRE(vertices[0].numberTracks == 7);
  REQUIRE(vertices[0].x == 0.125);
  REQUIRE(vertices[0].y == -0.25);
  REQUIRE(vertices[0].z == -83.75);
  REQUIRE(vertices[1].numberTracks == 0);
  REQUIRE(vertices[1].x == 1.5);
  REQUIRE(vertices[1].y == 2.75);
  REQUIRE(vertices[1].z == 319.125);
}

TEST_CASE("PVFinder validates MC primary-vertex payload boundaries", "[PVFinder]")
{
  std::vector<char> payload;
  REQUIRE_THROWS_AS(PVFinder::read_mc_vertices(payload), std::runtime_error);
  append(payload, int32_t {0});
  REQUIRE(PVFinder::read_mc_vertices(payload).empty());
  payload.push_back(0);
  REQUIRE_THROWS_AS(PVFinder::read_mc_vertices(payload), std::runtime_error);
  payload.clear();
  append(payload, int32_t {-1});
  REQUIRE_THROWS_AS(PVFinder::read_mc_vertices(payload), std::runtime_error);
  payload.clear();
  append(payload, int32_t {1});
  REQUIRE_THROWS_AS(PVFinder::read_mc_vertices(payload), std::runtime_error);
  payload.clear();
  append(payload, INT32_MAX);
  REQUIRE_THROWS_AS(PVFinder::read_mc_vertices(payload), std::runtime_error);
}
