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
  REQUIRE(plan.selection_source() == Allen::CuDNN::AlgorithmSelectionSource::ZeroWorkspace);
  REQUIRE(plan.workspace_bytes() == 0);
  REQUIRE(plan.is_created());

  const auto metadata = plan.metadata();
  REQUIRE(metadata.created);
  REQUIRE(metadata.algorithm_policy == Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace);
  REQUIRE(metadata.workspace_policy == Allen::CuDNN::WorkspacePolicy::ZeroOnly);
  REQUIRE(metadata.workspace_bytes == 0);
}

TEST_CASE("cudnn.backward_data_plan.zero_workspace", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ConvPlanOptions options {};
  options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

  Allen::CuDNN::BackwardDataConvPlan plan;
  plan.create(handle, {1, 1, 1, 2}, {1, 1, 1, 4}, {1, 1, 1, 8}, {0, 0}, {1, 2}, {1, 1}, options);

  REQUIRE(plan.workspace_bytes() == 0);
  REQUIRE(plan.selection_source() == Allen::CuDNN::AlgorithmSelectionSource::ZeroWorkspace);
  REQUIRE(plan.is_created());
}

TEST_CASE("cudnn.forward_plan.bad_shape_failure", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ForwardConvPlan plan;
  REQUIRE_THROWS(plan.create(handle, {1, 2, 1, 8}, {1, 1, 1, 3}, {0, 1}));
}

TEST_CASE("cudnn.workspace_policy.validation", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ConvPlanOptions invalid_options {};
  invalid_options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::Heuristic;
  invalid_options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

  Allen::CuDNN::ForwardConvPlan plan;
  REQUIRE_THROWS(plan.create(handle, {1, 1, 1, 8}, {1, 1, 1, 3}, {0, 1}, {1, 1}, {1, 1}, invalid_options));

  Allen::CuDNN::Workspace external {nullptr, 0};
  REQUIRE_NOTHROW(external.require(0, "unit"));
  REQUIRE_THROWS(external.require(1, "unit"));
}

TEST_CASE("cudnn.device_weights.validation", "[AllenCuDNN]") {
  Allen::CuDNN::DeviceWeights weights {"unit_phase2"};
  const float host_values[4] = {1.f, 2.f, 3.f, 4.f};
  const float replacement_values[2] = {5.f, 6.f};

  weights.load_from_buffer("weights", host_values, sizeof(host_values), sizeof(host_values));
  REQUIRE(weights.contains("weights"));
  REQUIRE(weights.full_key("weights") == "unit_phase2.weights");
  REQUIRE(weights.size_bytes("weights") == sizeof(host_values));
  REQUIRE(weights.get<float>("weights") != nullptr);

  weights.load_from_buffer(
    "weights",
    host_values,
    sizeof(host_values),
    sizeof(host_values),
    Allen::CuDNN::DuplicateKeyPolicy::ReuseExisting);
  REQUIRE(weights.size_bytes("weights") == sizeof(host_values));
  weights.load_from_buffer(
    "weights",
    replacement_values,
    sizeof(replacement_values),
    sizeof(replacement_values),
    Allen::CuDNN::DuplicateKeyPolicy::ReplaceExisting);
  REQUIRE(weights.size_bytes("weights") == sizeof(replacement_values));
  REQUIRE_THROWS(weights.load_from_buffer("bad_size", host_values, sizeof(host_values), sizeof(host_values) + 4));

  weights.register_device_pointer("external", host_values, sizeof(host_values), sizeof(host_values));
  REQUIRE(weights.contains("external"));
  REQUIRE(weights.size_bytes("external") == sizeof(host_values));
}
#else
TEST_CASE("cudnn.wrapper_stubs.compile", "[AllenCuDNN]") {
  REQUIRE(true);
}
#endif
