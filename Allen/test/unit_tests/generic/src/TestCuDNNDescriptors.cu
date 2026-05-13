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

#include <string>

#ifdef ALLEN_CUDNN_BACKEND_CUDA

TEST_CASE("cudnn.forward_plan.zero_workspace", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ConvPlanOptions options {};
  options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

  Allen::CuDNN::ForwardConvPlan plan;
  plan.create(handle, Allen::CuDNN::Conv1DShape::forward(1, 1, 8, 1, 3, 1), options);

  REQUIRE(plan.algorithm_policy() == Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace);
  REQUIRE(plan.workspace_policy() == Allen::CuDNN::WorkspacePolicy::ZeroOnly);
  REQUIRE(plan.selection_source() == Allen::CuDNN::AlgorithmSelectionSource::ZeroWorkspace);
  REQUIRE(std::string(plan.algorithm_name()).find("CUDNN_CONVOLUTION_FWD_ALGO_") == 0);
  REQUIRE(plan.fallback_reason().empty());
  REQUIRE(plan.workspace_bytes() == 0);
  REQUIRE(plan.is_created());

  const auto metadata = plan.metadata();
  REQUIRE(metadata.created);
  REQUIRE(metadata.algorithm_policy == Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace);
  REQUIRE(metadata.workspace_policy == Allen::CuDNN::WorkspacePolicy::ZeroOnly);
  REQUIRE(metadata.workspace_bytes == 0);
  REQUIRE(metadata.algorithm_name == plan.algorithm_name());
  REQUIRE(metadata.fallback_reason.empty());
  REQUIRE(metadata.layout == Allen::CuDNN::TensorLayout::NCHW);
  REQUIRE(metadata.precision.input_output_type == CUDNN_DATA_FLOAT);
  REQUIRE(plan.precision_policy().input_output_type == CUDNN_DATA_FLOAT);
}

TEST_CASE("cudnn.backward_data_plan.zero_workspace", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ConvPlanOptions options {};
  options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

  Allen::CuDNN::BackwardDataConvPlan plan;
  plan.create(handle, Allen::CuDNN::Conv1DShape::backward_data(1, 1, 4, 1, 8, 2, 0, 2), options);

  REQUIRE(plan.workspace_bytes() == 0);
  REQUIRE(plan.selection_source() == Allen::CuDNN::AlgorithmSelectionSource::ZeroWorkspace);
  REQUIRE(std::string(plan.algorithm_name()).find("CUDNN_CONVOLUTION_BWD_DATA_ALGO_") == 0);
  REQUIRE(plan.fallback_reason().empty());
  REQUIRE(plan.is_created());
  REQUIRE(plan.metadata().layout == Allen::CuDNN::TensorLayout::NCHW);
}

TEST_CASE("cudnn.forward_plan.bad_shape_failure", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ForwardConvPlan plan;
  REQUIRE_THROWS(plan.create(handle, {1, 2, 1, 8}, {1, 1, 1, 3}, {0, 1}));

  auto shape = Allen::CuDNN::Conv1DShape::forward(1, 1, 8, 1, 3, 1);
  shape.output = {1, 1, 1, 99};
  shape.has_output = true;
  REQUIRE_THROWS(plan.create(handle, shape));
}

TEST_CASE("cudnn.precision_policy.validation", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ConvPlanOptions fp16_options {};
  fp16_options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  fp16_options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;
  fp16_options.precision = Allen::CuDNN::fp16_precision_policy();

  Allen::CuDNN::ForwardConvPlan fp16_plan;
  fp16_plan.create(handle, Allen::CuDNN::Conv1DShape::forward(1, 1, 8, 1, 3, 1), fp16_options);
  REQUIRE(fp16_plan.data_type() == CUDNN_DATA_HALF);
  REQUIRE(fp16_plan.compute_type() == CUDNN_DATA_FLOAT);
  REQUIRE(fp16_plan.precision_policy().fp16_experimental);

  fp16_options.precision.fp16_experimental = false;
  Allen::CuDNN::ForwardConvPlan invalid_fp16_plan;
  REQUIRE_THROWS(invalid_fp16_plan.create(
    handle, Allen::CuDNN::Conv1DShape::forward(1, 1, 8, 1, 3, 1), fp16_options));

  Allen::CuDNN::ConvPlanOptions no_tf32_options {};
  no_tf32_options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  no_tf32_options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;
  no_tf32_options.precision = Allen::CuDNN::fp32_precision_policy(CUDNN_TENSOR_OP_MATH, false);
  Allen::CuDNN::ForwardConvPlan no_tf32_plan;
  no_tf32_plan.create(handle, Allen::CuDNN::Conv1DShape::forward(1, 1, 8, 1, 3, 1), no_tf32_options);
  REQUIRE(no_tf32_plan.math_type() == CUDNN_DEFAULT_MATH);
}

TEST_CASE("cudnn.workspace_policy.validation", "[AllenCuDNN]") {
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ConvPlanOptions invalid_options {};
  invalid_options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::Heuristic;
  invalid_options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

  Allen::CuDNN::ForwardConvPlan plan;
  REQUIRE_THROWS(
    plan.create(handle, Allen::CuDNN::Conv1DShape::forward(1, 1, 8, 1, 3, 1), invalid_options));

  Allen::CuDNN::Workspace external {nullptr, 0};
  REQUIRE_NOTHROW(external.require(0, "unit"));
  REQUIRE_THROWS(external.require(1, "unit"));
}

TEST_CASE("cudnn.bias_add_plan.metadata", "[AllenCuDNN]") {
  const Allen::CuDNN::TensorShape tensor_shape {2, 3, 4, 5};
  const Allen::CuDNN::TensorShape bias_shape {1, 3, 1, 1};

  Allen::CuDNN::BiasAddPlan plan;
  plan.create(tensor_shape);

  REQUIRE(plan.is_created());
  REQUIRE(plan.tensor_shape() == tensor_shape);
  REQUIRE(plan.bias_shape() == bias_shape);
  REQUIRE(plan.data_type() == CUDNN_DATA_FLOAT);

  const auto metadata = plan.metadata();
  REQUIRE(metadata.created);
  REQUIRE(metadata.layout == Allen::CuDNN::TensorLayout::NCHW);
  REQUIRE(metadata.tensor_shape == tensor_shape);
  REQUIRE(metadata.bias_shape == bias_shape);
  REQUIRE(metadata.precision.input_output_type == CUDNN_DATA_FLOAT);
}

TEST_CASE("cudnn.activation_plan.metadata", "[AllenCuDNN]") {
  const Allen::CuDNN::TensorShape tensor_shape {2, 3, 4, 5};
  Allen::CuDNN::ActivationOptions options {};
  options.mode = Allen::CuDNN::ActivationMode::ClippedRelu;
  options.coefficient = 6.0;

  Allen::CuDNN::ActivationPlan plan;
  plan.create(tensor_shape, options);

  REQUIRE(plan.is_created());
  REQUIRE(plan.mode() == Allen::CuDNN::ActivationMode::ClippedRelu);
  REQUIRE(plan.coefficient() == 6.0);
  REQUIRE(plan.tensor_shape() == tensor_shape);
  REQUIRE(plan.data_type() == CUDNN_DATA_FLOAT);
  REQUIRE(std::string(Allen::CuDNN::to_string(plan.mode())) == "ClippedRelu");

  const auto metadata = plan.metadata();
  REQUIRE(metadata.created);
  REQUIRE(metadata.mode == Allen::CuDNN::ActivationMode::ClippedRelu);
  REQUIRE(metadata.coefficient == 6.0);
  REQUIRE(metadata.layout == Allen::CuDNN::TensorLayout::NCHW);
  REQUIRE(metadata.tensor_shape == tensor_shape);
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
