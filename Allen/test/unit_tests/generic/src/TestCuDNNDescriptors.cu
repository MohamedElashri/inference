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
#define ALLEN_CUDNN_TEST_HAS_HEADER 1
#else
#define ALLEN_CUDNN_TEST_HAS_HEADER 0
#endif

#include <string>

#if ALLEN_CUDNN_TEST_HAS_HEADER

TEST_CASE("cudnn.workspace_planner.non_overlapping_and_overlapping", "[AllenCuDNN]") {
  Allen::CuDNN::WorkspacePlanner planner;
  planner.add("conv_a", 100, 64);
  planner.add("conv_b", 256, 128);
  planner.add("conv_c", 0, 256);

  const auto non_overlapping = planner.non_overlapping_plan();
  REQUIRE(non_overlapping.mode == Allen::CuDNN::WorkspacePlanningMode::NonOverlapping);
  REQUIRE(non_overlapping.total_bytes == 256);
  REQUIRE(non_overlapping.max_required_bytes == 256);
  REQUIRE(non_overlapping.slice("conv_a").offset == 0);
  REQUIRE(non_overlapping.slice("conv_b").offset == 0);

  const auto overlapping = planner.overlapping_plan();
  REQUIRE(overlapping.mode == Allen::CuDNN::WorkspacePlanningMode::Overlapping);
  REQUIRE(overlapping.slice("conv_a").offset == 0);
  REQUIRE(overlapping.slice("conv_b").offset == 128);
  REQUIRE(overlapping.total_bytes == 384);
  REQUIRE(overlapping.max_required_bytes == 256);
  REQUIRE_THROWS(overlapping.slice("missing"));
}

TEST_CASE("cudnn.workspace_arena.validation", "[AllenCuDNN]") {
  Allen::CuDNN::WorkspacePlanner planner;
  planner.add("conv_a", 128, 64);
  planner.add("conv_b", 256, 128);
  const auto plan = planner.overlapping_plan();

  unsigned char backing[512] {};
  Allen::CuDNN::WorkspaceArena arena {{backing, sizeof(backing)}, plan};
  REQUIRE(arena.total_bytes() == plan.total_bytes);
  REQUIRE(arena.slice("conv_a").ptr == static_cast<void*>(backing));
  REQUIRE(arena.slice("conv_a").bytes == 128);
  REQUIRE(arena.slice("conv_b").ptr == static_cast<void*>(backing + 128));
  REQUIRE(arena.slice("conv_b").bytes == 256);

  REQUIRE_THROWS(Allen::CuDNN::WorkspaceArena({backing, 128}, plan));
  REQUIRE_THROWS(Allen::CuDNN::WorkspaceArena({nullptr, 0}, plan));
}

TEST_CASE("cudnn.fused_conv_plan.accepted_post_ops", "[AllenCuDNN]") {
  Allen::CuDNN::FusedConvPlanOptions options {};
  options.conv.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  options.conv.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;
  options.conv.workspace_limit_bytes = 1024;
  options.post_ops = Allen::CuDNN::PostOpSequence::bias_activation(Allen::CuDNN::ActivationMode::Relu);
  options.output_shape = {2, 4, 1, 8};
  options.has_output_shape = true;

  Allen::CuDNN::FusedConvPlan plan;
  plan.create(Allen::CuDNN::Conv1DShape::forward(2, 3, 8, 4, 3, 1), options);

  REQUIRE(plan.is_created());
  REQUIRE(plan.selected_backend() == Allen::CuDNN::FusedConvBackend::MetadataOnly);
  REQUIRE(plan.execution_kind() == Allen::CuDNN::FusedConvExecutionKind::MetadataOnly);
  REQUIRE(plan.workspace_policy() == Allen::CuDNN::WorkspacePolicy::ZeroOnly);
  REQUIRE(plan.workspace_bytes() == 0);
  REQUIRE(plan.workspace_limit_bytes() == 1024);
  REQUIRE(plan.selection_source() == Allen::CuDNN::AlgorithmSelectionSource::Default);
  REQUIRE(plan.output_shape() == options.output_shape);
  REQUIRE(plan.post_ops() == options.post_ops);
  REQUIRE(std::string(Allen::CuDNN::describe_post_ops(plan.post_ops())) == "Conv+ChannelBias+Activation(Relu)");
  REQUIRE_FALSE(plan.fallback_reason().empty());

  const auto metadata = plan.metadata();
  REQUIRE(metadata.created);
  REQUIRE(metadata.post_ops == options.post_ops);
  REQUIRE(metadata.backend_preference == Allen::CuDNN::FusedConvBackendPreference::Auto);
  REQUIRE(metadata.algorithm_policy == Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace);
  REQUIRE(metadata.workspace_policy == Allen::CuDNN::WorkspacePolicy::ZeroOnly);
  REQUIRE(metadata.workspace_limit_bytes == 1024);
  REQUIRE(metadata.selected_backend == Allen::CuDNN::FusedConvBackend::MetadataOnly);
  REQUIRE(metadata.execution_kind == Allen::CuDNN::FusedConvExecutionKind::MetadataOnly);
  REQUIRE(metadata.output_shape == options.output_shape);
  REQUIRE(metadata.layout == Allen::CuDNN::TensorLayout::NCHW);
}

TEST_CASE("cudnn.fused_conv_plan.post_op_validation", "[AllenCuDNN]") {
  Allen::CuDNN::FusedConvPlan plan;
  const auto shape = Allen::CuDNN::Conv1DShape::forward(1, 2, 8, 4, 3, 1);

  Allen::CuDNN::FusedConvPlanOptions conv_only {};
  REQUIRE_NOTHROW(plan.create(shape, conv_only));

  Allen::CuDNN::FusedConvPlanOptions bias_only {};
  bias_only.post_ops = Allen::CuDNN::PostOpSequence::channel_bias();
  REQUIRE_NOTHROW(plan.create(shape, bias_only));

  Allen::CuDNN::FusedConvPlanOptions activation_only {};
  activation_only.post_ops = Allen::CuDNN::PostOpSequence::activation(Allen::CuDNN::ActivationMode::Tanh);
  REQUIRE_NOTHROW(plan.create(shape, activation_only));

  Allen::CuDNN::FusedConvPlanOptions identity_activation {};
  identity_activation.post_ops = Allen::CuDNN::PostOpSequence::activation(Allen::CuDNN::ActivationMode::Identity);
  REQUIRE_NOTHROW(plan.create(shape, identity_activation));
  REQUIRE(std::string(Allen::CuDNN::describe_post_ops(identity_activation.post_ops)) == "Conv+Activation(Identity)");

  Allen::CuDNN::FusedConvPlanOptions activation_then_bias {};
  activation_then_bias.post_ops.ops = {
    Allen::CuDNN::PostOp::activation(),
    Allen::CuDNN::PostOp::channel_bias()};
  REQUIRE_THROWS(plan.create(shape, activation_then_bias));

  Allen::CuDNN::FusedConvPlanOptions duplicate_activation {};
  duplicate_activation.post_ops.ops = {
    Allen::CuDNN::PostOp::activation(),
    Allen::CuDNN::PostOp::activation()};
  REQUIRE_THROWS(plan.create(shape, duplicate_activation));

  Allen::CuDNN::FusedConvPlanOptions residual_add {};
  residual_add.post_ops.ops = {Allen::CuDNN::PostOp::add()};
  REQUIRE_THROWS(plan.create(shape, residual_add));
}

TEST_CASE("cudnn.fused_conv_plan.shape_and_option_validation", "[AllenCuDNN]") {
  Allen::CuDNN::FusedConvPlan plan;

  Allen::CuDNN::FusedConvPlanOptions bad_workspace {};
  bad_workspace.conv.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::Heuristic;
  bad_workspace.conv.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;
  REQUIRE_THROWS(plan.create(Allen::CuDNN::Conv1DShape::forward(1, 2, 8, 4, 3, 1), bad_workspace));

  Allen::CuDNN::FusedConvPlanOptions bad_output {};
  bad_output.output_shape = {1, 4, 1, 99};
  bad_output.has_output_shape = true;
  REQUIRE_THROWS(plan.create(Allen::CuDNN::Conv1DShape::forward(1, 2, 8, 4, 3, 1), bad_output));

  Allen::CuDNN::FusedConvPlanOptions strict_unimplemented {};
  strict_unimplemented.preferred_backend = Allen::CuDNN::FusedConvBackend::LegacyConvPlusCudaPostOp;
  strict_unimplemented.fallback_policy = Allen::CuDNN::FusedConvFallbackPolicy::RequireRequestedBackend;
  REQUIRE_THROWS(plan.create(Allen::CuDNN::Conv1DShape::forward(1, 2, 8, 4, 3, 1), strict_unimplemented));

  Allen::CuDNN::FusedConvPlanOptions forced_frontend {};
  forced_frontend.backend_preference = Allen::CuDNN::FusedConvBackendPreference::ForceCudnnFrontendGraph;
  REQUIRE_THROWS(plan.create(Allen::CuDNN::Conv1DShape::forward(1, 2, 8, 4, 3, 1), forced_frontend));
}

TEST_CASE("cudnn.fused_conv_plan.frontend_graph_capability_metadata", "[AllenCuDNN]") {
  auto shape = Allen::CuDNN::Conv1DShape::forward(1, 2, 8, 4, 3, 1);
  Allen::CuDNN::FusedConvPlanOptions options {};
  options.post_ops = Allen::CuDNN::PostOpSequence::bias_activation(Allen::CuDNN::ActivationMode::Relu);

  const auto capability = Allen::CuDNN::frontend_graph_capability(shape, options);
  REQUIRE_FALSE(capability.available());
  REQUIRE_FALSE(capability.reason.empty());

  options.backend_preference = Allen::CuDNN::FusedConvBackendPreference::PreferCudnnFrontendGraph;
  Allen::CuDNN::FusedConvPlan plan;
  plan.create(shape, options);
  REQUIRE(plan.metadata().backend_preference == Allen::CuDNN::FusedConvBackendPreference::PreferCudnnFrontendGraph);
  REQUIRE(plan.metadata().selected_backend == Allen::CuDNN::FusedConvBackend::MetadataOnly);
  REQUIRE_FALSE(plan.metadata().frontend_graph_capability.reason.empty());
}

#ifdef ALLEN_CUDNN_BACKEND_CUDA

#include <cuda_runtime.h>

#include <cmath>
#include <sstream>
#include <vector>

namespace {
  bool has_cuda_device()
  {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
  }

  void require_cuda(cudaError_t status)
  {
    REQUIRE(status == cudaSuccess);
  }

  std::vector<float> reference_conv_bias_relu_1d(
    const std::vector<float>& input,
    const std::vector<float>& filter,
    const std::vector<float>& bias,
    Allen::CuDNN::TensorShape input_shape,
    Allen::CuDNN::TensorShape filter_shape,
    int pad)
  {
    const int output_width = input_shape.w;
    std::vector<float> output(input_shape.n * filter_shape.n * output_width, 0.f);
    for (int n = 0; n < input_shape.n; ++n) {
      for (int oc = 0; oc < filter_shape.n; ++oc) {
        for (int x = 0; x < output_width; ++x) {
          float value = 0.f;
          for (int ic = 0; ic < input_shape.c; ++ic) {
            for (int k = 0; k < filter_shape.w; ++k) {
              const int in_x = x + k - pad;
              if (in_x < 0 || in_x >= input_shape.w) continue;
              const auto input_index = ((n * input_shape.c + ic) * input_shape.w) + in_x;
              const auto filter_index = ((oc * filter_shape.c + ic) * filter_shape.w) + k;
              value += input[input_index] * filter[filter_index];
            }
          }
          value += bias[oc];
          output[(n * filter_shape.n + oc) * output_width + x] = value > 0.f ? value : 0.f;
        }
      }
    }
    return output;
  }

#define REQUIRE_CUDA_DEVICE()                                                                      \
  do {                                                                                             \
    if (!has_cuda_device()) {                                                                      \
      SUCCEED("Skipping CUDA/cuDNN runtime test: no CUDA-capable device is available");             \
      return;                                                                                      \
    }                                                                                              \
  } while (false)
} // namespace

namespace {
  struct TinyCuDNNClientParameters {
    int events = 1;
    int input_channels = 2;
    int width = 8;
    int output_channels = 4;
    int kernel_width = 3;
    int padding = 1;
  };

  class TinyCuDNNClient {
  public:
    explicit TinyCuDNNClient(TinyCuDNNClientParameters parameters) :
      m_parameters(parameters), m_weights("tiny_cudnn_client")
    {}

    void register_external_weights(const float* dev_filter, size_t filter_bytes)
    {
      m_weights.register_device_pointer(
        "conv.weight",
        dev_filter,
        filter_bytes,
        filter_bytes,
        Allen::CuDNN::DuplicateKeyPolicy::ReplaceExisting);
    }

    void create_metadata_plan()
    {
      Allen::CuDNN::FusedConvPlanOptions options {};
      options.conv.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
      options.conv.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;
      options.backend_preference = Allen::CuDNN::FusedConvBackendPreference::PreferLegacyConvPlusCudaPostOp;
      options.post_ops = Allen::CuDNN::PostOpSequence::bias_activation(Allen::CuDNN::ActivationMode::Relu);

      m_plan.create(
        Allen::CuDNN::Conv1DShape::forward(
          m_parameters.events,
          m_parameters.input_channels,
          m_parameters.width,
          m_parameters.output_channels,
          m_parameters.kernel_width,
          m_parameters.padding),
        options);
    }

    Allen::CuDNN::WorkspacePlan workspace_plan() const
    {
      Allen::CuDNN::WorkspacePlanner planner;
      planner.add("tiny_conv", m_plan.workspace_bytes());
      return planner.non_overlapping_plan();
    }

    std::string metadata_line() const
    {
      const auto metadata = m_plan.metadata();
      std::ostringstream out;
      out << "backend=" << Allen::CuDNN::to_string(metadata.selected_backend)
          << " workspace_policy=" << Allen::CuDNN::to_string(metadata.workspace_policy)
          << " workspace_bytes=" << metadata.workspace_bytes
          << " post_ops=" << Allen::CuDNN::describe_post_ops(metadata.post_ops)
          << " fallback_reason=" << metadata.fallback_reason;
      return out.str();
    }

    const Allen::CuDNN::DeviceWeights& weights() const { return m_weights; }
    const Allen::CuDNN::FusedConvPlan& plan() const { return m_plan; }

  private:
    TinyCuDNNClientParameters m_parameters;
    Allen::CuDNN::DeviceWeights m_weights;
    Allen::CuDNN::FusedConvPlan m_plan;
  };
} // namespace

TEST_CASE("cudnn.generic_client_contract.metadata_only_example", "[AllenCuDNN]") {
  const float external_filter[24] {};

  TinyCuDNNClient client {{1, 2, 8, 4, 3, 1}};
  client.register_external_weights(external_filter, sizeof(external_filter));
  client.create_metadata_plan();

  REQUIRE(client.weights().key_namespace() == "tiny_cudnn_client");
  REQUIRE(client.weights().contains("conv.weight"));
  REQUIRE(client.weights().full_key("conv.weight") == "tiny_cudnn_client.conv.weight");
  REQUIRE(client.weights().size_bytes("conv.weight") == sizeof(external_filter));

  REQUIRE(client.plan().is_created());
  REQUIRE(client.plan().workspace_policy() == Allen::CuDNN::WorkspacePolicy::ZeroOnly);
  REQUIRE(client.plan().workspace_bytes() == 0);
  REQUIRE(client.plan().metadata().layout == Allen::CuDNN::TensorLayout::NCHW);
  REQUIRE(
    client.plan().metadata().post_ops ==
    Allen::CuDNN::PostOpSequence::bias_activation(Allen::CuDNN::ActivationMode::Relu));

  const auto workspace = client.workspace_plan();
  REQUIRE(workspace.mode == Allen::CuDNN::WorkspacePlanningMode::NonOverlapping);
  REQUIRE(workspace.total_bytes == 0);

  const auto log_line = client.metadata_line();
  REQUIRE(log_line.find("workspace_policy=ZeroOnly") != std::string::npos);
  REQUIRE(log_line.find("post_ops=Conv+ChannelBias+Activation(Relu)") != std::string::npos);
}

TEST_CASE("cudnn.fused_conv_plan.legacy_conv_only_matches_forward_plan", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  const auto shape = Allen::CuDNN::Conv1DShape::forward(1, 2, 5, 3, 3, 1);
  Allen::CuDNN::ConvPlanOptions conv_options {};
  conv_options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  conv_options.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;

  Allen::CuDNN::ForwardConvPlan forward_plan;
  forward_plan.create(handle, shape, conv_options);

  Allen::CuDNN::FusedConvPlanOptions fused_options {};
  fused_options.conv = conv_options;
  fused_options.preferred_backend = Allen::CuDNN::FusedConvBackend::LegacyConvPlusCudaPostOp;
  fused_options.fallback_policy = Allen::CuDNN::FusedConvFallbackPolicy::RequireRequestedBackend;

  Allen::CuDNN::FusedConvPlan fused_plan;
  fused_plan.create(handle, shape, fused_options);

  REQUIRE(fused_plan.selected_backend() == Allen::CuDNN::FusedConvBackend::LegacyConvPlusCudaPostOp);
  REQUIRE(fused_plan.execution_kind() == Allen::CuDNN::FusedConvExecutionKind::SingleCall);
  REQUIRE(fused_plan.workspace_bytes() == forward_plan.workspace_bytes());
  REQUIRE(fused_plan.selection_source() == forward_plan.selection_source());
  REQUIRE(fused_plan.metadata().algorithm == forward_plan.algorithm_id());

  const std::vector<float> host_input {
    1.f, -2.f, 3.f, -4.f, 5.f,
    0.5f, 1.f, -1.5f, 2.f, -2.5f};
  const std::vector<float> host_filter {
    1.f, 0.f, -1.f, 0.25f, 0.5f, 0.75f,
    -0.5f, 0.25f, 1.f, 1.5f, -1.f, 0.5f,
    0.75f, -0.25f, 0.5f, -1.f, 1.f, 0.f};
  std::vector<float> forward_output(15, 0.f);
  std::vector<float> fused_output(15, 0.f);

  float *dev_input = nullptr, *dev_filter = nullptr, *dev_forward = nullptr, *dev_fused = nullptr;
  require_cuda(cudaMalloc(&dev_input, host_input.size() * sizeof(float)));
  require_cuda(cudaMalloc(&dev_filter, host_filter.size() * sizeof(float)));
  require_cuda(cudaMalloc(&dev_forward, forward_output.size() * sizeof(float)));
  require_cuda(cudaMalloc(&dev_fused, fused_output.size() * sizeof(float)));
  require_cuda(cudaMemcpy(dev_input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice));
  require_cuda(cudaMemcpy(dev_filter, host_filter.data(), host_filter.size() * sizeof(float), cudaMemcpyHostToDevice));
  require_cuda(cudaMemset(dev_forward, 0, forward_output.size() * sizeof(float)));
  require_cuda(cudaMemset(dev_fused, 0, fused_output.size() * sizeof(float)));

  forward_plan.forward(handle, 1.f, 0.f, dev_input, dev_filter, dev_forward);
  fused_plan.execute(handle, 1.f, 0.f, dev_input, dev_filter, nullptr, dev_fused);
  require_cuda(cudaMemcpy(forward_output.data(), dev_forward, forward_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  require_cuda(cudaMemcpy(fused_output.data(), dev_fused, fused_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < forward_output.size(); ++i) {
    REQUIRE(std::fabs(forward_output[i] - fused_output[i]) < 1e-6f);
  }

  cudaFree(dev_input);
  cudaFree(dev_filter);
  cudaFree(dev_forward);
  cudaFree(dev_fused);
}

TEST_CASE("cudnn.fused_conv_plan.frontend_preference_falls_back_to_legacy", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  const auto shape = Allen::CuDNN::Conv1DShape::forward(1, 2, 5, 3, 3, 1);
  Allen::CuDNN::FusedConvPlanOptions options {};
  options.conv.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  options.conv.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;
  options.backend_preference = Allen::CuDNN::FusedConvBackendPreference::PreferCudnnFrontendGraph;
  options.post_ops = Allen::CuDNN::PostOpSequence::bias_activation(Allen::CuDNN::ActivationMode::Relu);

  Allen::CuDNN::FusedConvPlan plan;
  plan.create(handle, shape, options);

  REQUIRE(plan.selected_backend() == Allen::CuDNN::FusedConvBackend::LegacyConvPlusCudaPostOp);
  REQUIRE(plan.metadata().backend_preference == Allen::CuDNN::FusedConvBackendPreference::PreferCudnnFrontendGraph);
  REQUIRE_FALSE(plan.metadata().frontend_graph_capability.reason.empty());
  REQUIRE(plan.fallback_reason().find("CudnnFrontendGraph unavailable") != std::string::npos);

  options.backend_preference = Allen::CuDNN::FusedConvBackendPreference::ForceCudnnFrontendGraph;
  REQUIRE_THROWS(plan.create(handle, shape, options));
}

TEST_CASE("cudnn.fused_conv_plan.legacy_bias_relu_matches_reference", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  const auto shape = Allen::CuDNN::Conv1DShape::forward(1, 1, 4, 2, 3, 1);
  Allen::CuDNN::FusedConvPlanOptions options {};
  options.conv.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::ZeroWorkspace;
  options.conv.workspace_policy = Allen::CuDNN::WorkspacePolicy::ZeroOnly;
  options.preferred_backend = Allen::CuDNN::FusedConvBackend::LegacyConvPlusCudaPostOp;
  options.fallback_policy = Allen::CuDNN::FusedConvFallbackPolicy::RequireRequestedBackend;
  options.post_ops = Allen::CuDNN::PostOpSequence::bias_activation(Allen::CuDNN::ActivationMode::Relu);

  Allen::CuDNN::FusedConvPlan plan;
  plan.create(handle, shape, options);

  REQUIRE(plan.selected_backend() == Allen::CuDNN::FusedConvBackend::LegacyConvPlusCudaPostOp);
  REQUIRE(plan.execution_kind() == Allen::CuDNN::FusedConvExecutionKind::ConvPlusKernel);
  REQUIRE(plan.metadata().post_ops == options.post_ops);

  const std::vector<float> host_input {1.f, -2.f, 3.f, -4.f};
  const std::vector<float> host_filter {
    1.f, 0.5f, -1.f,
    -0.25f, 1.f, 0.75f};
  const std::vector<float> host_bias {0.5f, -1.f};
  auto expected = reference_conv_bias_relu_1d(host_input, host_filter, host_bias, shape.input, shape.filter, 1);
  std::vector<float> observed(expected.size(), 0.f);

  float *dev_input = nullptr, *dev_filter = nullptr, *dev_bias = nullptr, *dev_output = nullptr;
  require_cuda(cudaMalloc(&dev_input, host_input.size() * sizeof(float)));
  require_cuda(cudaMalloc(&dev_filter, host_filter.size() * sizeof(float)));
  require_cuda(cudaMalloc(&dev_bias, host_bias.size() * sizeof(float)));
  require_cuda(cudaMalloc(&dev_output, observed.size() * sizeof(float)));
  require_cuda(cudaMemcpy(dev_input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice));
  require_cuda(cudaMemcpy(dev_filter, host_filter.data(), host_filter.size() * sizeof(float), cudaMemcpyHostToDevice));
  require_cuda(cudaMemcpy(dev_bias, host_bias.data(), host_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
  require_cuda(cudaMemset(dev_output, 0, observed.size() * sizeof(float)));

  plan.execute(handle, 1.f, 0.f, dev_input, dev_filter, dev_bias, dev_output);
  require_cuda(cudaMemcpy(observed.data(), dev_output, observed.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(std::fabs(expected[i] - observed[i]) < 1e-5f);
  }

  cudaFree(dev_input);
  cudaFree(dev_filter);
  cudaFree(dev_bias);
  cudaFree(dev_output);
}

TEST_CASE("cudnn.forward_plan.zero_workspace", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
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
  REQUIRE_CUDA_DEVICE();
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

TEST_CASE("cudnn.algorithm_cache.forward_plan_hit_miss_and_strict_lookup", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
  auto handle = Allen::CuDNN::HandleProvider::get(0);
  Allen::CuDNN::clear_algorithm_cache();

  Allen::CuDNN::ConvPlanOptions options {};
  options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::Heuristic;
  options.workspace_policy = Allen::CuDNN::WorkspacePolicy::OwnedInitTime;
  options.cache_policy = Allen::CuDNN::AlgorithmCachePolicy::LookupAndPopulate;
  const auto shape = Allen::CuDNN::Conv1DShape::forward(1, 2, 8, 3, 3, 1);

  Allen::CuDNN::ForwardConvPlan first;
  first.create(handle, shape, options);
  REQUIRE(first.metadata().cache.status == Allen::CuDNN::AlgorithmCacheStatus::Miss);
  REQUIRE(first.metadata().cache.created_by == "ForwardConvPlan");
  REQUIRE_FALSE(first.metadata().cache.key.empty());
  REQUIRE(Allen::CuDNN::algorithm_cache_size() == 1);

  Allen::CuDNN::ForwardConvPlan second;
  second.create(handle, shape, options);
  REQUIRE(second.metadata().cache.status == Allen::CuDNN::AlgorithmCacheStatus::Hit);
  REQUIRE(second.algorithm_id() == first.algorithm_id());
  REQUIRE(second.workspace_bytes() == first.workspace_bytes());
  REQUIRE(second.selection_source() == first.selection_source());

  auto different_workspace_limit = options;
  different_workspace_limit.workspace_limit_bytes += 4096;
  Allen::CuDNN::ForwardConvPlan workspace_miss;
  workspace_miss.create(handle, shape, different_workspace_limit);
  REQUIRE(workspace_miss.metadata().cache.status == Allen::CuDNN::AlgorithmCacheStatus::Miss);
  REQUIRE(workspace_miss.metadata().cache.key != first.metadata().cache.key);

  auto different_math = options;
  different_math.precision = Allen::CuDNN::fp32_precision_policy(CUDNN_TENSOR_OP_MATH, false);
  Allen::CuDNN::ForwardConvPlan math_miss;
  math_miss.create(handle, shape, different_math);
  REQUIRE(math_miss.metadata().cache.status == Allen::CuDNN::AlgorithmCacheStatus::Miss);
  REQUIRE(math_miss.metadata().cache.key != first.metadata().cache.key);

  auto strict = options;
  strict.cache_policy = Allen::CuDNN::AlgorithmCachePolicy::StrictLookup;
  Allen::CuDNN::clear_algorithm_cache();
  Allen::CuDNN::ForwardConvPlan strict_plan;
  REQUIRE_THROWS(strict_plan.create(handle, shape, strict));
}

TEST_CASE("cudnn.forward_plan.bad_shape_failure", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::ForwardConvPlan plan;
  REQUIRE_THROWS(plan.create(handle, {1, 2, 1, 8}, {1, 1, 1, 3}, {0, 1}));

  auto shape = Allen::CuDNN::Conv1DShape::forward(1, 1, 8, 1, 3, 1);
  shape.output = {1, 1, 1, 99};
  shape.has_output = true;
  REQUIRE_THROWS(plan.create(handle, shape));
}

TEST_CASE("cudnn.precision_policy.validation", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
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
  REQUIRE(fp16_plan.metadata().precision.input_output_type == CUDNN_DATA_HALF);
  REQUIRE(fp16_plan.metadata().precision.compute_type == CUDNN_DATA_FLOAT);
  REQUIRE(fp16_plan.metadata().precision.math_type == CUDNN_TENSOR_OP_MATH);
  REQUIRE_FALSE(fp16_plan.metadata().precision.allow_tf32);
  REQUIRE(std::string(Allen::CuDNN::describe_precision_policy(fp16_plan.metadata().precision))
            .find("fp16_experimental=true") != std::string::npos);

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
  REQUIRE_FALSE(no_tf32_plan.metadata().precision.allow_tf32);
  REQUIRE(std::string(Allen::CuDNN::describe_precision_policy(no_tf32_plan.metadata().precision))
            .find("tf32=false") != std::string::npos);
}

TEST_CASE("cudnn.workspace_policy.validation", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
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

TEST_CASE("cudnn.pooling_plan.metadata_and_validation", "[AllenCuDNN]") {
  const auto shape = Allen::CuDNN::Pooling1DShape::forward(2, 3, 100, 2, 2);
  const Allen::CuDNN::TensorShape input_shape {2, 3, 1, 100};
  const Allen::CuDNN::TensorShape output_shape {2, 3, 1, 50};
  const std::array<int, 2> window {1, 2};
  const std::array<int, 2> pad {0, 0};
  const std::array<int, 2> stride {1, 2};

  Allen::CuDNN::PoolingPlan plan;
  plan.create(shape);

  REQUIRE(plan.is_created());
  REQUIRE(plan.mode() == Allen::CuDNN::PoolingMode::Max);
  REQUIRE(plan.input_shape() == input_shape);
  REQUIRE(plan.output_shape() == output_shape);
  REQUIRE(plan.window() == window);
  REQUIRE(plan.pad() == pad);
  REQUIRE(plan.stride() == stride);
  REQUIRE(plan.workspace_bytes() == 0);
  REQUIRE(plan.data_type() == CUDNN_DATA_FLOAT);
  REQUIRE(std::string(Allen::CuDNN::to_string(plan.mode())) == "Max");

  const auto metadata = plan.metadata();
  REQUIRE(metadata.created);
  REQUIRE(metadata.mode == Allen::CuDNN::PoolingMode::Max);
  REQUIRE(metadata.layout == Allen::CuDNN::TensorLayout::NCHW);
  REQUIRE(metadata.input_shape == input_shape);
  REQUIRE(metadata.output_shape == output_shape);
  REQUIRE(metadata.workspace_bytes == 0);
  REQUIRE(metadata.precision.input_output_type == CUDNN_DATA_FLOAT);

  auto bad_output = shape;
  bad_output.output = {2, 3, 1, 49};
  bad_output.has_output = true;
  REQUIRE_THROWS(plan.create(bad_output));

  REQUIRE_THROWS(plan.create(Allen::CuDNN::Pooling1DShape::forward(1, 1, 1, 2, 2)));
}

TEST_CASE("cudnn.pooling_plan.max_pool_1d_matches_reference", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
  auto handle = Allen::CuDNN::HandleProvider::get(0);

  Allen::CuDNN::PoolingPlan plan;
  plan.create(Allen::CuDNN::Pooling1DShape::forward(1, 2, 6, 2, 2));

  const std::vector<float> host_input {
    1.f, 3.f, -1.f, 5.f, 2.f, 0.f,
    -2.f, -1.f, 4.f, 3.f, 8.f, 7.f};
  const std::vector<float> expected {3.f, 5.f, 2.f, -1.f, 4.f, 8.f};
  std::vector<float> observed(expected.size(), 0.f);

  float *dev_input = nullptr, *dev_output = nullptr;
  require_cuda(cudaMalloc(&dev_input, host_input.size() * sizeof(float)));
  require_cuda(cudaMalloc(&dev_output, observed.size() * sizeof(float)));
  require_cuda(cudaMemcpy(dev_input, host_input.data(), host_input.size() * sizeof(float), cudaMemcpyHostToDevice));
  require_cuda(cudaMemset(dev_output, 0, observed.size() * sizeof(float)));

  plan.forward(handle, 1.f, dev_input, 0.f, dev_output);
  require_cuda(cudaMemcpy(observed.data(), dev_output, observed.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(std::fabs(expected[i] - observed[i]) < 1e-6f);
  }

  cudaFree(dev_input);
  cudaFree(dev_output);
}

TEST_CASE("cudnn.device_weights.validation", "[AllenCuDNN]") {
  REQUIRE_CUDA_DEVICE();
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

#else
TEST_CASE("cudnn.wrapper_header_unavailable.compile", "[AllenCuDNN]") {
  REQUIRE(true);
}
#endif
