#pragma once

#include "CuDNNDescriptors.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace Allen::CuDNN {

  enum class PostOpKind {
    ChannelBias,
    Activation,
    Add
  };

  enum class FusedConvBackend {
    Auto,
    MetadataOnly,
    LegacyConvPlusCudaPostOp,
    CudnnFrontendGraph,
    PrimitiveSequence,
    ClientFallback
  };

  enum class FusedConvBackendPreference {
    Auto,
    PreferCudnnFrontendGraph,
    PreferLegacyConvPlusCudaPostOp,
    ForceCudnnFrontendGraph,
    ForceLegacyConvPlusCudaPostOp
  };

  enum class FusedConvFallbackPolicy {
    RequireRequestedBackend,
    AllowMetadataOnly,
    AllowClientFallback
  };

  enum class FusedConvExecutionKind {
    MetadataOnly,
    SingleCall,
    ConvPlusKernel,
    PrimitiveSequence,
    ClientFallback
  };

  struct PostOp {
    PostOpKind kind = PostOpKind::Activation;
    ActivationMode activation_mode = ActivationMode::Relu;
    double activation_coefficient = 0.0;

    static PostOp channel_bias() { return {PostOpKind::ChannelBias, ActivationMode::Relu, 0.0}; }
    static PostOp activation(ActivationMode mode = ActivationMode::Relu, double coefficient = 0.0)
    {
      return {PostOpKind::Activation, mode, coefficient};
    }
    static PostOp add() { return {PostOpKind::Add, ActivationMode::Relu, 0.0}; }
  };

  struct PostOpSequence {
    std::vector<PostOp> ops {};

    static PostOpSequence conv_only() { return {}; }
    static PostOpSequence channel_bias() { return {{PostOp::channel_bias()}}; }
    static PostOpSequence activation(ActivationMode mode = ActivationMode::Relu, double coefficient = 0.0)
    {
      return {{PostOp::activation(mode, coefficient)}};
    }
    static PostOpSequence bias_activation(ActivationMode mode = ActivationMode::Relu, double coefficient = 0.0)
    {
      return {{PostOp::channel_bias(), PostOp::activation(mode, coefficient)}};
    }

    bool empty() const { return ops.empty(); }
    size_t size() const { return ops.size(); }
    const PostOp& operator[](size_t index) const { return ops[index]; }
  };

  struct FusedConvPlanOptions {
    ConvPlanOptions conv {};
    PostOpSequence post_ops {};
    TensorShape output_shape {};
    TensorLayout output_layout = TensorLayout::NCHW;
    bool has_output_shape = false;
    FusedConvBackendPreference backend_preference = FusedConvBackendPreference::Auto;
    // Compatibility alias retained for V2.1/V2.2 callers. New callers should use backend_preference.
    FusedConvBackend preferred_backend = FusedConvBackend::Auto;
    FusedConvFallbackPolicy fallback_policy = FusedConvFallbackPolicy::AllowMetadataOnly;
    bool log_plan_creation = false;
  };

  struct FusedConvFrontendGraphCapability {
    bool compile_time_available = false;
    bool runtime_version_supported = false;
    bool device_supported = true;
    bool dtype_supported = false;
    bool layout_supported = false;
    bool post_ops_supported = false;
    size_t runtime_version = 0;
    int device_major = 0;
    int device_minor = 0;
    std::string reason;

    bool available() const
    {
      return compile_time_available && runtime_version_supported && device_supported && dtype_supported &&
             layout_supported && post_ops_supported && reason.empty();
    }
  };

  struct FusedConvMetadata {
    bool created = false;
    FusedConvBackend selected_backend = FusedConvBackend::MetadataOnly;
    FusedConvBackendPreference backend_preference = FusedConvBackendPreference::Auto;
    FusedConvBackend preferred_backend = FusedConvBackend::Auto;
    FusedConvFallbackPolicy fallback_policy = FusedConvFallbackPolicy::AllowMetadataOnly;
    FusedConvExecutionKind execution_kind = FusedConvExecutionKind::MetadataOnly;
    AlgorithmSelectionPolicy algorithm_policy = AlgorithmSelectionPolicy::TimedFind;
    AlgorithmSelectionSource selection_source = AlgorithmSelectionSource::Default;
    WorkspacePolicy workspace_policy = WorkspacePolicy::OwnedInitTime;
    size_t workspace_bytes = 0;
    size_t workspace_limit_bytes = 0;
    std::string algorithm_name;
    std::string fallback_reason;
    PostOpSequence post_ops {};
    TensorShape input_shape {};
    TensorShape filter_shape {};
    TensorShape output_shape {};
    TensorLayout layout = TensorLayout::NCHW;
    PrecisionPolicy precision {};
    AlgorithmCacheMetadata cache {};
    FusedConvFrontendGraphCapability frontend_graph_capability {};
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    int algorithm = 0;
#endif
  };

  inline const char* to_string(PostOpKind kind)
  {
    switch (kind) {
    case PostOpKind::ChannelBias: return "ChannelBias";
    case PostOpKind::Activation: return "Activation";
    case PostOpKind::Add: return "Add";
    }
    return "Unknown";
  }

  inline const char* to_string(FusedConvBackend backend)
  {
    switch (backend) {
    case FusedConvBackend::Auto: return "Auto";
    case FusedConvBackend::MetadataOnly: return "MetadataOnly";
    case FusedConvBackend::LegacyConvPlusCudaPostOp: return "LegacyConvPlusCudaPostOp";
    case FusedConvBackend::CudnnFrontendGraph: return "CudnnFrontendGraph";
    case FusedConvBackend::PrimitiveSequence: return "PrimitiveSequence";
    case FusedConvBackend::ClientFallback: return "ClientFallback";
    }
    return "Unknown";
  }

  inline const char* to_string(FusedConvBackendPreference preference)
  {
    switch (preference) {
    case FusedConvBackendPreference::Auto: return "Auto";
    case FusedConvBackendPreference::PreferCudnnFrontendGraph: return "PreferCudnnFrontendGraph";
    case FusedConvBackendPreference::PreferLegacyConvPlusCudaPostOp: return "PreferLegacyConvPlusCudaPostOp";
    case FusedConvBackendPreference::ForceCudnnFrontendGraph: return "ForceCudnnFrontendGraph";
    case FusedConvBackendPreference::ForceLegacyConvPlusCudaPostOp: return "ForceLegacyConvPlusCudaPostOp";
    }
    return "Unknown";
  }

  inline const char* to_string(FusedConvFallbackPolicy policy)
  {
    switch (policy) {
    case FusedConvFallbackPolicy::RequireRequestedBackend: return "RequireRequestedBackend";
    case FusedConvFallbackPolicy::AllowMetadataOnly: return "AllowMetadataOnly";
    case FusedConvFallbackPolicy::AllowClientFallback: return "AllowClientFallback";
    }
    return "Unknown";
  }

  inline const char* to_string(FusedConvExecutionKind kind)
  {
    switch (kind) {
    case FusedConvExecutionKind::MetadataOnly: return "MetadataOnly";
    case FusedConvExecutionKind::SingleCall: return "SingleCall";
    case FusedConvExecutionKind::ConvPlusKernel: return "ConvPlusKernel";
    case FusedConvExecutionKind::PrimitiveSequence: return "PrimitiveSequence";
    case FusedConvExecutionKind::ClientFallback: return "ClientFallback";
    }
    return "Unknown";
  }

  inline bool operator==(const PostOp& lhs, const PostOp& rhs)
  {
    return lhs.kind == rhs.kind && lhs.activation_mode == rhs.activation_mode &&
           lhs.activation_coefficient == rhs.activation_coefficient;
  }

  inline bool operator==(const PostOpSequence& lhs, const PostOpSequence& rhs)
  {
    return lhs.ops == rhs.ops;
  }

  inline std::string describe_post_ops(const PostOpSequence& sequence)
  {
    if (sequence.ops.empty()) return "Conv";
    std::string description = "Conv";
    for (const auto& op : sequence.ops) {
      description += "+";
      description += to_string(op.kind);
      if (op.kind == PostOpKind::Activation) {
        description += "(";
        description += to_string(op.activation_mode);
        description += ")";
      }
    }
    return description;
  }

  inline FusedConvBackendPreference effective_backend_preference(const FusedConvPlanOptions& options)
  {
    if (options.backend_preference != FusedConvBackendPreference::Auto) {
      return options.backend_preference;
    }
    switch (options.preferred_backend) {
    case FusedConvBackend::Auto:
      return FusedConvBackendPreference::Auto;
    case FusedConvBackend::LegacyConvPlusCudaPostOp:
      return options.fallback_policy == FusedConvFallbackPolicy::RequireRequestedBackend ?
        FusedConvBackendPreference::ForceLegacyConvPlusCudaPostOp :
        FusedConvBackendPreference::PreferLegacyConvPlusCudaPostOp;
    case FusedConvBackend::CudnnFrontendGraph:
      return options.fallback_policy == FusedConvFallbackPolicy::RequireRequestedBackend ?
        FusedConvBackendPreference::ForceCudnnFrontendGraph :
        FusedConvBackendPreference::PreferCudnnFrontendGraph;
    default:
      return FusedConvBackendPreference::Auto;
    }
  }

  inline FusedConvBackend preferred_backend_from_preference(FusedConvBackendPreference preference)
  {
    switch (preference) {
    case FusedConvBackendPreference::PreferCudnnFrontendGraph:
    case FusedConvBackendPreference::ForceCudnnFrontendGraph:
      return FusedConvBackend::CudnnFrontendGraph;
    case FusedConvBackendPreference::PreferLegacyConvPlusCudaPostOp:
    case FusedConvBackendPreference::ForceLegacyConvPlusCudaPostOp:
      return FusedConvBackend::LegacyConvPlusCudaPostOp;
    case FusedConvBackendPreference::Auto:
      return FusedConvBackend::Auto;
    }
    return FusedConvBackend::Auto;
  }

  inline bool forces_backend(FusedConvBackendPreference preference)
  {
    return preference == FusedConvBackendPreference::ForceCudnnFrontendGraph ||
           preference == FusedConvBackendPreference::ForceLegacyConvPlusCudaPostOp;
  }

  inline void validate_post_op_sequence(const PostOpSequence& sequence, const char* owner)
  {
    bool seen_bias = false;
    bool seen_activation = false;
    for (size_t i = 0; i < sequence.ops.size(); ++i) {
      const auto& op = sequence.ops[i];
      switch (op.kind) {
      case PostOpKind::ChannelBias:
        if (seen_bias) {
          throw std::invalid_argument(std::string("AllenCuDNN: ") + owner + " post-op sequence has duplicate bias");
        }
        if (seen_activation) {
          throw std::invalid_argument(
            std::string("AllenCuDNN: ") + owner + " post-op sequence must apply bias before activation");
        }
        seen_bias = true;
        break;
      case PostOpKind::Activation:
        if (seen_activation) {
          throw std::invalid_argument(
            std::string("AllenCuDNN: ") + owner + " post-op sequence has duplicate activation");
        }
        if (i + 1 != sequence.ops.size()) {
          throw std::invalid_argument(
            std::string("AllenCuDNN: ") + owner + " activation post-op must be last");
        }
        seen_activation = true;
        break;
      case PostOpKind::Add:
        throw std::invalid_argument(
          std::string("AllenCuDNN: ") + owner + " residual/add post-op is not enabled without a concrete client");
      }
    }
  }

  inline TensorShape convolution_forward_output_shape(const Conv2DShape& shape, const char* owner)
  {
    const auto output_dim = [owner](int input, int filter, int pad, int stride, int dilation, const char* axis) {
      const int value = (input + 2 * pad - dilation * (filter - 1) - 1) / stride + 1;
      if (value <= 0) {
        throw std::invalid_argument(
          std::string("AllenCuDNN: ") + owner + " computed non-positive output dimension on " + axis);
      }
      return value;
    };

    return {
      shape.input.n,
      shape.filter.n,
      output_dim(shape.input.h, shape.filter.h, shape.pad[0], shape.stride[0], shape.dilation[0], "height"),
      output_dim(shape.input.w, shape.filter.w, shape.pad[1], shape.stride[1], shape.dilation[1], "width")};
  }

  inline FusedConvFrontendGraphCapability frontend_graph_capability(
    Conv2DShape shape,
    const FusedConvPlanOptions& options)
  {
    FusedConvFrontendGraphCapability capability {};
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    capability.compile_time_available = true;
    capability.runtime_version = cudnnGetVersion();
    capability.runtime_version_supported = capability.runtime_version >= 8900;

    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
      cudaDeviceProp properties {};
      if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        capability.device_major = properties.major;
        capability.device_minor = properties.minor;
      }
    }
    capability.device_supported = true;

    capability.dtype_supported =
      options.conv.precision.input_output_type == CUDNN_DATA_FLOAT &&
      options.conv.precision.filter_type == CUDNN_DATA_FLOAT &&
      options.conv.precision.compute_type == CUDNN_DATA_FLOAT;
    capability.layout_supported = shape.layout == TensorLayout::NCHW;
    capability.post_ops_supported = true;
    for (const auto& op : options.post_ops.ops) {
      if (op.kind == PostOpKind::Add) {
        capability.post_ops_supported = false;
      }
      if (op.kind == PostOpKind::Activation &&
          op.activation_mode != ActivationMode::Identity &&
          op.activation_mode != ActivationMode::Relu) {
        capability.post_ops_supported = false;
      }
    }

    if (!capability.compile_time_available) {
      capability.reason = "cuDNN backend graph API headers are not available at compile time";
    }
    else if (!capability.runtime_version_supported) {
      capability.reason = "runtime cuDNN version is too old for the frontend graph backend";
    }
    else if (!capability.dtype_supported) {
      capability.reason = "frontend graph backend accepts only FP32 fused convolution in V2.3";
    }
    else if (!capability.layout_supported) {
      capability.reason = "frontend graph backend accepts only TensorLayout::NCHW in V2.3";
    }
    else if (!capability.post_ops_supported) {
      capability.reason = "frontend graph backend supports only conv, bias, ReLU, and bias+ReLU in V2.3";
    }
    else {
      capability.reason =
        "cuDNN frontend graph backend is capability-detected but disabled for V2.3; earlier PVFinder "
        "NCHW FP32 graph experiments were unsupported or slower than the legacy fused backend";
    }
#else
    (void) shape;
    (void) options;
    capability.reason = "cuDNN frontend graph backend requires ALLEN_CUDNN_BACKEND_CUDA";
#endif
    return capability;
  }

  inline bool fused_plan_creation_logging_enabled(const FusedConvPlanOptions& options)
  {
    return options.log_plan_creation || options.conv.log_plan_creation || std::getenv("ALLEN_CUDNN_VERBOSE") != nullptr;
  }

  inline void log_fused_plan_creation(const char* plan_name, const FusedConvMetadata& info, const FusedConvPlanOptions& options)
  {
    if (!fused_plan_creation_logging_enabled(options)) return;
    std::fprintf(
      stderr,
      "AllenCuDNN: %s created backend=%s backend_preference=%s execution=%s post_ops=%s algorithm=%s"
#ifdef ALLEN_CUDNN_BACKEND_CUDA
      "(%d)"
#endif
      " selection_policy=%s selection_source=%s "
      "workspace_policy=%s workspace_bytes=%zu workspace_limit_bytes=%zu precision={%s}",
      plan_name,
      to_string(info.selected_backend),
      to_string(info.backend_preference),
      to_string(info.execution_kind),
      describe_post_ops(info.post_ops).c_str(),
      info.algorithm_name.c_str(),
#ifdef ALLEN_CUDNN_BACKEND_CUDA
      info.algorithm,
#endif
      to_string(info.algorithm_policy),
      to_string(info.selection_source),
      to_string(info.workspace_policy),
      info.workspace_bytes,
      info.workspace_limit_bytes,
      describe_precision_policy(info.precision).c_str());
    std::fprintf(stderr, " cache=%s", to_string(info.cache.status));
    if (!info.cache.provenance.empty()) {
      std::fprintf(stderr, " cache_provenance=\"%s\"", info.cache.provenance.c_str());
    }
    if (!info.frontend_graph_capability.reason.empty()) {
      std::fprintf(stderr, " frontend_graph=\"%s\"", info.frontend_graph_capability.reason.c_str());
    }
    if (!info.fallback_reason.empty()) {
      std::fprintf(stderr, " fallback_reason=\"%s\"", info.fallback_reason.c_str());
    }
    std::fprintf(stderr, "\n");
  }

#ifdef ALLEN_CUDNN_BACKEND_CUDA
  namespace detail {
    void launch_fused_conv_post_ops_float(
      float* output,
      const float* bias,
      TensorShape shape,
      bool add_bias,
      ActivationMode activation,
      cudaStream_t stream);
  } // namespace detail
#endif

  struct FusedConvPlan {
  private:
    FusedConvPlanOptions m_options {};
    FusedConvMetadata m_metadata {};
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    ForwardConvPlan m_conv_plan {};
#endif

    static bool has_channel_bias(const PostOpSequence& sequence)
    {
      for (const auto& op : sequence.ops) {
        if (op.kind == PostOpKind::ChannelBias) return true;
      }
      return false;
    }

    static ActivationMode activation_mode(const PostOpSequence& sequence)
    {
      for (const auto& op : sequence.ops) {
        if (op.kind == PostOpKind::Activation) return op.activation_mode;
      }
      return ActivationMode::Identity;
    }

    static bool legacy_post_ops_supported(const PostOpSequence& sequence)
    {
      const auto mode = activation_mode(sequence);
      return mode == ActivationMode::Identity || mode == ActivationMode::Relu;
    }

    static bool requires_cuda_backend(FusedConvBackend backend)
    {
      return backend == FusedConvBackend::LegacyConvPlusCudaPostOp ||
             backend == FusedConvBackend::CudnnFrontendGraph ||
             backend == FusedConvBackend::PrimitiveSequence;
    }

    static bool wants_legacy_backend(FusedConvBackend backend)
    {
      return backend == FusedConvBackend::Auto || backend == FusedConvBackend::LegacyConvPlusCudaPostOp;
    }

    static bool wants_legacy_backend(FusedConvBackendPreference preference)
    {
      return preference == FusedConvBackendPreference::Auto ||
             preference == FusedConvBackendPreference::PreferCudnnFrontendGraph ||
             preference == FusedConvBackendPreference::PreferLegacyConvPlusCudaPostOp ||
             preference == FusedConvBackendPreference::ForceLegacyConvPlusCudaPostOp;
    }

    static bool wants_frontend_backend(FusedConvBackendPreference preference)
    {
      return preference == FusedConvBackendPreference::Auto ||
             preference == FusedConvBackendPreference::PreferCudnnFrontendGraph ||
             preference == FusedConvBackendPreference::ForceCudnnFrontendGraph;
    }

    static std::string append_reason(std::string existing, const std::string& addition)
    {
      if (addition.empty()) return existing;
      if (existing.empty()) return addition;
      return existing + "; " + addition;
    }

    void fill_metadata(
      Conv2DShape shape,
      TensorShape output_shape,
      FusedConvPlanOptions options,
      FusedConvBackendPreference backend_preference,
      FusedConvBackend selected_backend,
      FusedConvExecutionKind execution_kind,
      AlgorithmSelectionSource selection_source,
      size_t workspace_bytes,
      const std::string& algorithm_name,
      const std::string& fallback_reason,
      FusedConvFrontendGraphCapability frontend_capability = {})
    {
      m_options = options;
      m_metadata = {};
      m_metadata.created = true;
      m_metadata.selected_backend = selected_backend;
      m_metadata.backend_preference = backend_preference;
      m_metadata.preferred_backend = options.preferred_backend;
      m_metadata.fallback_policy = options.fallback_policy;
      m_metadata.execution_kind = execution_kind;
      m_metadata.algorithm_policy = options.conv.algorithm_policy;
      m_metadata.selection_source = selection_source;
      m_metadata.workspace_policy = options.conv.workspace_policy;
      m_metadata.workspace_bytes = workspace_bytes;
      m_metadata.workspace_limit_bytes = options.conv.workspace_limit_bytes;
      m_metadata.algorithm_name = algorithm_name;
      m_metadata.fallback_reason = fallback_reason;
      m_metadata.post_ops = options.post_ops;
      m_metadata.input_shape = shape.input;
      m_metadata.filter_shape = shape.filter;
      m_metadata.output_shape = output_shape;
      m_metadata.layout = shape.layout;
      m_metadata.precision = options.conv.precision;
      m_metadata.cache = {};
      m_metadata.frontend_graph_capability = frontend_capability;
#ifdef ALLEN_CUDNN_BACKEND_CUDA
      m_metadata.algorithm = 0;
#endif
    }

  public:
    FusedConvPlan() = default;

    void create(Conv2DShape shape, FusedConvPlanOptions options = {})
    {
      validate_forward_shape(shape, "FusedConvPlan");
      validate_post_op_sequence(options.post_ops, "FusedConvPlan");
      validate_layout(options.output_layout, "FusedConvPlan");
      options.conv.precision = normalize_precision_policy(options.conv.precision);
      validate_precision_policy(options.conv.precision, "FusedConvPlan");
      validate_workspace_options(options.conv, "FusedConvPlan");

      const TensorShape computed_output = convolution_forward_output_shape(shape, "FusedConvPlan");
      if (shape.has_output && options.has_output_shape && shape.output != options.output_shape) {
        throw std::invalid_argument(
          "AllenCuDNN: FusedConvPlan output shape is specified both on shape and options with different values");
      }
      const TensorShape requested_output =
        options.has_output_shape ? options.output_shape : (shape.has_output ? shape.output : computed_output);
      validate_tensor_shape(requested_output, "FusedConvPlan output_shape");
      if (requested_output != computed_output) {
        throw std::invalid_argument("AllenCuDNN: FusedConvPlan caller output_shape does not match computed output shape");
      }

      if (options.fallback_policy == FusedConvFallbackPolicy::RequireRequestedBackend &&
          requires_cuda_backend(options.preferred_backend)) {
        throw std::invalid_argument(
          "AllenCuDNN: FusedConvPlan executable backend creation requires the cudnnHandle_t create overload");
      }

      const auto backend_preference = effective_backend_preference(options);
      if (forces_backend(backend_preference)) {
        throw std::invalid_argument(
          "AllenCuDNN: FusedConvPlan forced executable backend creation requires the cudnnHandle_t create overload");
      }
      const auto frontend_capability = Allen::CuDNN::frontend_graph_capability(shape, options);
      fill_metadata(
        shape,
        computed_output,
        options,
        backend_preference,
        FusedConvBackend::MetadataOnly,
        FusedConvExecutionKind::MetadataOnly,
        AlgorithmSelectionSource::Default,
        0,
        "metadata-only",
        "metadata-only fused plan; call the cudnnHandle_t create overload for executable fused backends",
        frontend_capability);
#ifdef ALLEN_CUDNN_BACKEND_CUDA
      m_conv_plan.reset();
#endif
      log_fused_plan_creation("FusedConvPlan", m_metadata, m_options);
    }

#ifdef ALLEN_CUDNN_BACKEND_CUDA
    void create(cudnnHandle_t handle, Conv2DShape shape, FusedConvPlanOptions options = {})
    {
      validate_forward_shape(shape, "FusedConvPlan");
      validate_post_op_sequence(options.post_ops, "FusedConvPlan");
      validate_layout(options.output_layout, "FusedConvPlan");
      options.conv.precision = normalize_precision_policy(options.conv.precision);
      validate_precision_policy(options.conv.precision, "FusedConvPlan");
      validate_workspace_options(options.conv, "FusedConvPlan");

      const TensorShape computed_output = convolution_forward_output_shape(shape, "FusedConvPlan");
      if (shape.has_output && options.has_output_shape && shape.output != options.output_shape) {
        throw std::invalid_argument(
          "AllenCuDNN: FusedConvPlan output shape is specified both on shape and options with different values");
      }
      const TensorShape requested_output =
        options.has_output_shape ? options.output_shape : (shape.has_output ? shape.output : computed_output);
      validate_tensor_shape(requested_output, "FusedConvPlan output_shape");
      if (requested_output != computed_output) {
        throw std::invalid_argument("AllenCuDNN: FusedConvPlan caller output_shape does not match computed output shape");
      }
      shape.output = computed_output;
      shape.has_output = true;
      const auto backend_preference = effective_backend_preference(options);
      const auto frontend_capability = Allen::CuDNN::frontend_graph_capability(shape, options);

      if (options.preferred_backend == FusedConvBackend::MetadataOnly) {
        fill_metadata(
          shape,
          computed_output,
          options,
          backend_preference,
          FusedConvBackend::MetadataOnly,
          FusedConvExecutionKind::MetadataOnly,
          AlgorithmSelectionSource::Default,
          0,
          "metadata-only",
          "metadata-only backend requested",
          frontend_capability);
        m_conv_plan.reset();
        log_fused_plan_creation("FusedConvPlan", m_metadata, m_options);
        return;
      }

      if (backend_preference == FusedConvBackendPreference::ForceCudnnFrontendGraph) {
        const std::string reason =
          frontend_capability.reason.empty() ?
            "CudnnFrontendGraph execution is not enabled for V2.3" :
            std::string("CudnnFrontendGraph unavailable: ") + frontend_capability.reason;
        throw std::invalid_argument(std::string("AllenCuDNN: FusedConvPlan ") + reason);
      }

      const bool fp32 =
        options.conv.precision.input_output_type == CUDNN_DATA_FLOAT &&
        options.conv.precision.filter_type == CUDNN_DATA_FLOAT;
      std::string frontend_fallback_reason;
      if (wants_frontend_backend(backend_preference) && !frontend_capability.available()) {
        frontend_fallback_reason = std::string("CudnnFrontendGraph unavailable: ") + frontend_capability.reason;
      }

      if (!wants_legacy_backend(backend_preference) || !fp32 || !legacy_post_ops_supported(options.post_ops)) {
        std::string reason;
        if (!wants_legacy_backend(backend_preference)) {
          reason = std::string("requested backend preference ") + to_string(backend_preference) +
                   " cannot use LegacyConvPlusCudaPostOp";
        }
        else if (!fp32) {
          reason = "LegacyConvPlusCudaPostOp supports FP32 plans in V2.2/V2.3";
        }
        else {
          reason = "LegacyConvPlusCudaPostOp supports only identity and ReLU activation in V2.2/V2.3";
        }
        if (options.fallback_policy == FusedConvFallbackPolicy::RequireRequestedBackend || forces_backend(backend_preference)) {
          throw std::invalid_argument(std::string("AllenCuDNN: FusedConvPlan ") + reason);
        }
        fill_metadata(
          shape,
          computed_output,
          options,
          backend_preference,
          FusedConvBackend::MetadataOnly,
          FusedConvExecutionKind::MetadataOnly,
          AlgorithmSelectionSource::Default,
          0,
          "metadata-only",
          append_reason(frontend_fallback_reason, reason),
          frontend_capability);
        m_conv_plan.reset();
        log_fused_plan_creation("FusedConvPlan", m_metadata, m_options);
        return;
      }

      auto conv_options = options.conv;
      conv_options.log_plan_creation = conv_options.log_plan_creation || options.log_plan_creation;
      m_conv_plan.create(handle, shape, conv_options);
      const auto conv_metadata = m_conv_plan.metadata();
      fill_metadata(
        shape,
        computed_output,
        options,
        backend_preference,
        FusedConvBackend::LegacyConvPlusCudaPostOp,
        (options.post_ops.empty() || activation_mode(options.post_ops) == ActivationMode::Identity) &&
            !has_channel_bias(options.post_ops) ?
          FusedConvExecutionKind::SingleCall : FusedConvExecutionKind::ConvPlusKernel,
        conv_metadata.selection_source,
        conv_metadata.workspace_bytes,
        conv_metadata.algorithm_name,
        append_reason(frontend_fallback_reason, conv_metadata.fallback_reason),
        frontend_capability);
      m_metadata.algorithm = conv_metadata.algorithm;
      m_metadata.cache = conv_metadata.cache;
      log_fused_plan_creation("FusedConvPlan", m_metadata, m_options);
    }

    void create(
      cudnnHandle_t handle,
      std::array<int, 4> input_shape,
      std::array<int, 4> filter_shape,
      std::array<int, 2> pad = {0, 0},
      std::array<int, 2> stride = {1, 1},
      std::array<int, 2> dilation = {1, 1},
      FusedConvPlanOptions options = {})
    {
      create(
        handle,
        Conv2DShape::forward(make_tensor_shape(input_shape), make_tensor_shape(filter_shape), pad, stride, dilation),
        options);
    }
#endif

    void create(
      std::array<int, 4> input_shape,
      std::array<int, 4> filter_shape,
      std::array<int, 2> pad = {0, 0},
      std::array<int, 2> stride = {1, 1},
      std::array<int, 2> dilation = {1, 1},
      FusedConvPlanOptions options = {})
    {
      create(
        Conv2DShape::forward(make_tensor_shape(input_shape), make_tensor_shape(filter_shape), pad, stride, dilation),
        options);
    }

    bool is_created() const { return m_metadata.created; }
    const FusedConvMetadata& metadata() const { return m_metadata; }
    const FusedConvPlanOptions& options() const { return m_options; }
    const PostOpSequence& post_ops() const { return m_metadata.post_ops; }
    FusedConvBackend selected_backend() const { return m_metadata.selected_backend; }
    FusedConvExecutionKind execution_kind() const { return m_metadata.execution_kind; }
    WorkspacePolicy workspace_policy() const { return m_metadata.workspace_policy; }
    size_t workspace_bytes() const { return m_metadata.workspace_bytes; }
    size_t workspace_limit_bytes() const { return m_metadata.workspace_limit_bytes; }
    AlgorithmSelectionPolicy algorithm_policy() const { return m_metadata.algorithm_policy; }
    AlgorithmSelectionSource selection_source() const { return m_metadata.selection_source; }
    const std::string& fallback_reason() const { return m_metadata.fallback_reason; }
    const AlgorithmCacheMetadata& cache_metadata() const { return m_metadata.cache; }
    const FusedConvFrontendGraphCapability& frontend_graph_capability() const
    {
      return m_metadata.frontend_graph_capability;
    }
    TensorShape output_shape() const { return m_metadata.output_shape; }
    const PrecisionPolicy& precision_policy() const { return m_metadata.precision; }

#ifdef ALLEN_CUDNN_BACKEND_CUDA
    void execute(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const void* dev_input,
      const void* dev_filter,
      const void* dev_bias,
      void* dev_output,
      void* external_workspace = nullptr) const
    {
      if (m_metadata.selected_backend != FusedConvBackend::LegacyConvPlusCudaPostOp) {
        throw std::runtime_error("AllenCuDNN: FusedConvPlan has no executable legacy fused backend");
      }
      m_conv_plan.forward(handle, alpha, beta, dev_input, dev_filter, dev_output, external_workspace);

      const bool add_bias = has_channel_bias(m_metadata.post_ops);
      const auto mode = activation_mode(m_metadata.post_ops);
      if (add_bias && dev_bias == nullptr) {
        throw std::invalid_argument("AllenCuDNN: FusedConvPlan bias post-op requires a non-null bias pointer");
      }
      if (!add_bias && mode == ActivationMode::Identity) return;

      cudaStream_t stream = nullptr;
      ALLEN_CUDNN_CHECK(cudnnGetStream(handle, &stream));
      detail::launch_fused_conv_post_ops_float(
        static_cast<float*>(dev_output),
        static_cast<const float*>(dev_bias),
        m_metadata.output_shape,
        add_bias,
        mode,
        stream);
      detail::cuda_check(cudaGetLastError(), "FusedConvPlan post-op kernel launch failed");
    }

    void execute(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const float* dev_input,
      const float* dev_filter,
      const float* dev_bias,
      float* dev_output,
      void* external_workspace = nullptr) const
    {
      execute(
        handle,
        alpha,
        beta,
        static_cast<const void*>(dev_input),
        static_cast<const void*>(dev_filter),
        static_cast<const void*>(dev_bias),
        static_cast<void*>(dev_output),
        external_workspace);
    }

    void execute(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const void* dev_input,
      const void* dev_filter,
      const void* dev_bias,
      void* dev_output,
      Workspace external_workspace) const
    {
      external_workspace.require(m_metadata.workspace_bytes, "FusedConvPlan");
      execute(handle, alpha, beta, dev_input, dev_filter, dev_bias, dev_output, external_workspace.ptr);
    }

    void execute(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const float* dev_input,
      const float* dev_filter,
      const float* dev_bias,
      float* dev_output,
      Workspace external_workspace) const
    {
      execute(
        handle,
        alpha,
        beta,
        static_cast<const void*>(dev_input),
        static_cast<const void*>(dev_filter),
        static_cast<const void*>(dev_bias),
        static_cast<void*>(dev_output),
        external_workspace);
    }
#endif
  };

} // namespace Allen::CuDNN
