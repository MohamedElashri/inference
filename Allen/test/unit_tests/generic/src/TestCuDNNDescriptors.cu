/*****************************************************************************\
* (c) Copyright 2024 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
\*****************************************************************************/
#if __has_include(<catch2/catch.hpp>)
#include <catch2/catch.hpp>
#else
#include <catch2/catch_test_macros.hpp>
#endif

#if __has_include("AllenCuDNN.h")
#include "AllenCuDNN.h"
#endif

#ifdef ALLEN_CUDNN_BACKEND_CUDA

TEST_CASE("cudnn.forward_plan.zero_workspace", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ConvPlanOptions options {};
  options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

  Allen::CuDNN::ForwardConvPlan plan;
  plan.create(handle, {1, 1, 1, 8}, {1, 1, 1, 3}, {0, 1}, {1, 1}, {1, 1}, options);

  REQUIRE(plan.algorithm_policy() == Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace);
  REQUIRE(plan.workspace_policy() == Allen::CuDNN::WorkspacePolicy::ZeroOnly);
  REQUIRE(plan.workspace_bytes() == 0);
}

TEST_CASE("cudnn.backward_data_plan.zero_workspace", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ConvPlanOptions options {};
  options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

  Allen::CuDNN::BackwardDataConvPlan plan;
  plan.create(handle, {1, 1, 1, 2}, {1, 1, 1, 4}, {1, 1, 1, 8}, {0, 0}, {1, 2}, {1, 1}, options);

  REQUIRE(plan.workspace_bytes() == 0);
}

TEST_CASE("cudnn.device_weights.validation", "[AllenCuDNN]") {
  Allen::CuDNN::DeviceWeights weights {"unit"};
  const float host_values[4] = {1.f, 2.f, 3.f, 4.f};

  weights.load_from_buffer("weights", host_values, sizeof(host_values), sizeof(host_values));
  REQUIRE(weights.contains("weights"));
  REQUIRE(weights.size_bytes("weights") == sizeof(host_values));
  REQUIRE(weights.get<float>("weights") != nullptr);

  weights.load_from_buffer(
    "weights",
    host_values,
    sizeof(host_values),
    sizeof(host_values),
    Allen::CuDNN::DuplicateKeyPolicy::ReuseExisting);
  REQUIRE_THROWS(weights.load_from_buffer("bad_size", host_values, sizeof(host_values), sizeof(host_values) + 4));
}
#else
TEST_CASE("cudnn.wrapper_stubs.compile", "[AllenCuDNN]") {
  REQUIRE(true);
}
#endif
