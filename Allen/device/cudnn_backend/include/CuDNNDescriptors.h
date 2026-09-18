#pragma once

#include "CuDNNCheck.h"
#include "CuDNNHandle.h"
#include "CuDNNWorkspace.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>

#ifdef ALLEN_CUDNN_BACKEND_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

namespace Allen::CuDNN {

  enum class AlgorithmSelectionPolicy {
    ZeroWorkspace,
    Heuristic,
    TimedFind
  };

  enum class WorkspacePolicy {
    AllenExternal,
    OwnedInitTime,
    ZeroOnly
  };

  enum class AlgorithmSelectionSource {
    Default,
    ZeroWorkspace,
    Heuristic,
    TimedFind,
    Fallback
  };

  enum class AlgorithmCachePolicy {
    Disabled,
    LookupOnly,
    Populate,
    LookupAndPopulate,
    StrictLookup
  };

  enum class AlgorithmCacheStatus {
    Disabled,
    Miss,
    Hit,
    StrictMiss,
    RejectedIncompatibleEnvironment
  };

  enum class TensorLayout {
    NCHW
  };

  enum class PoolingMode {
    Max,
    AverageCountIncludePadding,
    AverageCountExcludePadding
  };

  enum class ActivationMode {
    Identity,
    Relu,
    Sigmoid,
    Tanh,
    ClippedRelu,
    Elu
  };

  struct TensorShape {
    int n = 0, c = 0, h = 0, w = 0;
    std::array<int, 4> dims() const { return {n, c, h, w}; }
    size_t elements() const { return (size_t) n * c * h * w; }
    bool operator==(const TensorShape& o) const { return n == o.n && c == o.c && h == o.h && w == o.w; }
    bool operator!=(const TensorShape& o) const { return !(*this == o); }
  };

  struct Conv2DShape {
    TensorShape input;
    TensorShape filter;
    TensorShape output;
    std::array<int, 2> pad = {0, 0};
    std::array<int, 2> stride = {1, 1};
    std::array<int, 2> dilation = {1, 1};
    TensorLayout layout = TensorLayout::NCHW;
    bool has_output = false;

    static Conv2DShape forward(
      TensorShape input_shape,
      TensorShape filter_shape,
      std::array<int, 2> pad = {0, 0},
      std::array<int, 2> stride = {1, 1},
      std::array<int, 2> dilation = {1, 1})
    {
      Conv2DShape shape {};
      shape.input = input_shape;
      shape.filter = filter_shape;
      shape.pad = pad;
      shape.stride = stride;
      shape.dilation = dilation;
      return shape;
    }

    static Conv2DShape backward_data(
      TensorShape filter_shape,
      TensorShape input_shape,
      TensorShape output_shape,
      std::array<int, 2> pad = {0, 0},
      std::array<int, 2> stride = {1, 1},
      std::array<int, 2> dilation = {1, 1})
    {
      Conv2DShape shape {};
      shape.input = input_shape;
      shape.filter = filter_shape;
      shape.output = output_shape;
      shape.pad = pad;
      shape.stride = stride;
      shape.dilation = dilation;
      shape.has_output = true;
      return shape;
    }
  };

  struct Conv1DShape {
    static Conv2DShape forward(
      int n,
      int input_channels,
      int width,
      int output_channels,
      int kernel_width,
      int pad = 0,
      int stride = 1,
      int dilation = 1)
    {
      return Conv2DShape::forward(
        {n, input_channels, 1, width},
        {output_channels, input_channels, 1, kernel_width},
        {0, pad},
        {1, stride},
        {1, dilation});
    }

    static Conv2DShape backward_data(
      int n,
      int input_channels,
      int input_width,
      int output_channels,
      int output_width,
      int kernel_width,
      int pad = 0,
      int stride = 1,
      int dilation = 1)
    {
      return Conv2DShape::backward_data(
        {input_channels, output_channels, 1, kernel_width},
        {n, input_channels, 1, input_width},
        {n, output_channels, 1, output_width},
        {0, pad},
        {1, stride},
        {1, dilation});
    }
  };

  struct Pooling2DShape {
    TensorShape input;
    TensorShape output;
    std::array<int, 2> window = {1, 1};
    std::array<int, 2> pad = {0, 0};
    std::array<int, 2> stride = {1, 1};
    TensorLayout layout = TensorLayout::NCHW;
    bool has_output = false;

    static Pooling2DShape forward(
      TensorShape input_shape,
      std::array<int, 2> window_shape,
      std::array<int, 2> pad = {0, 0},
      std::array<int, 2> stride = {1, 1})
    {
      Pooling2DShape shape {};
      shape.input = input_shape;
      shape.window = window_shape;
      shape.pad = pad;
      shape.stride = stride;
      return shape;
    }
  };

  struct Pooling1DShape {
    static Pooling2DShape forward(
      int n,
      int channels,
      int width,
      int window_width,
      int stride = 1,
      int pad = 0)
    {
      return Pooling2DShape::forward(
        {n, channels, 1, width},
        {1, window_width},
        {0, pad},
        {1, stride});
    }
  };

  struct PrecisionPolicy {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    cudnnDataType_t input_output_type = CUDNN_DATA_FLOAT;
    cudnnDataType_t filter_type = CUDNN_DATA_FLOAT;
    cudnnDataType_t compute_type = CUDNN_DATA_FLOAT;
    cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
    bool tensor_ops_enabled = true;
    bool allow_tf32 = true;
    bool fp16_experimental = false;
#else
    int input_output_type = 0;
    int filter_type = 0;
    int compute_type = 0;
    int math_type = 0;
    bool tensor_ops_enabled = false;
    bool allow_tf32 = false;
    bool fp16_experimental = false;
#endif
  };

  struct AlgorithmCacheMetadata {
    AlgorithmCacheStatus status = AlgorithmCacheStatus::Disabled;
    std::string key;
    std::string provenance;
    std::string created_by;
  };

  struct ConvPlanOptions {
    AlgorithmSelectionPolicy algorithm_policy = AlgorithmSelectionPolicy::TimedFind;
    WorkspacePolicy workspace_policy = WorkspacePolicy::OwnedInitTime;
    size_t workspace_limit_bytes = 64ul * 1024 * 1024;
    bool log_plan_creation = false;
    PrecisionPolicy precision {};
    AlgorithmCachePolicy cache_policy = AlgorithmCachePolicy::Disabled;
    bool cache_fallback_results = false;
  };

  struct ConvPlanMetadata {
    AlgorithmSelectionPolicy algorithm_policy = AlgorithmSelectionPolicy::TimedFind;
    AlgorithmSelectionSource selection_source = AlgorithmSelectionSource::Default;
    WorkspacePolicy workspace_policy = WorkspacePolicy::OwnedInitTime;
    size_t workspace_bytes = 0;
    size_t workspace_limit_bytes = 0;
    std::string algorithm_name;
    std::string fallback_reason;
    bool created = false;
    TensorLayout layout = TensorLayout::NCHW;
    PrecisionPolicy precision {};
    AlgorithmCacheMetadata cache {};
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    int algorithm = 0;
#endif
  };

  struct BiasAddOptions {
    TensorLayout layout = TensorLayout::NCHW;
    PrecisionPolicy precision {};
  };

  struct BiasAddMetadata {
    bool created = false;
    TensorLayout layout = TensorLayout::NCHW;
    TensorShape tensor_shape {};
    TensorShape bias_shape {};
    PrecisionPolicy precision {};
  };

  struct ActivationOptions {
    ActivationMode mode = ActivationMode::Relu;
    double coefficient = 0.0;
    TensorLayout layout = TensorLayout::NCHW;
    PrecisionPolicy precision {};
  };

  struct ActivationMetadata {
    bool created = false;
    ActivationMode mode = ActivationMode::Relu;
    double coefficient = 0.0;
    TensorLayout layout = TensorLayout::NCHW;
    TensorShape tensor_shape {};
    PrecisionPolicy precision {};
  };

  struct PoolingOptions {
    PoolingMode mode = PoolingMode::Max;
    TensorLayout layout = TensorLayout::NCHW;
    PrecisionPolicy precision {};
    bool log_plan_creation = false;
  };

  struct PoolingMetadata {
    bool created = false;
    PoolingMode mode = PoolingMode::Max;
    TensorLayout layout = TensorLayout::NCHW;
    TensorShape input_shape {};
    TensorShape output_shape {};
    std::array<int, 2> window = {1, 1};
    std::array<int, 2> pad = {0, 0};
    std::array<int, 2> stride = {1, 1};
    size_t workspace_bytes = 0;
    PrecisionPolicy precision {};
  };

  inline const char* to_string(AlgorithmSelectionPolicy policy) {
    switch (policy) {
    case AlgorithmSelectionPolicy::ZeroWorkspace: return "ZeroWorkspace";
    case AlgorithmSelectionPolicy::Heuristic: return "Heuristic";
    case AlgorithmSelectionPolicy::TimedFind: return "TimedFind";
    }
    return "Unknown";
  }

  inline const char* to_string(AlgorithmSelectionSource source) {
    switch (source) {
    case AlgorithmSelectionSource::Default: return "Default";
    case AlgorithmSelectionSource::ZeroWorkspace: return "ZeroWorkspace";
    case AlgorithmSelectionSource::Heuristic: return "Heuristic";
    case AlgorithmSelectionSource::TimedFind: return "TimedFind";
    case AlgorithmSelectionSource::Fallback: return "Fallback";
    }
    return "Unknown";
  }

  inline const char* to_string(AlgorithmCachePolicy policy) {
    switch (policy) {
    case AlgorithmCachePolicy::Disabled: return "Disabled";
    case AlgorithmCachePolicy::LookupOnly: return "LookupOnly";
    case AlgorithmCachePolicy::Populate: return "Populate";
    case AlgorithmCachePolicy::LookupAndPopulate: return "LookupAndPopulate";
    case AlgorithmCachePolicy::StrictLookup: return "StrictLookup";
    }
    return "Unknown";
  }

  inline const char* to_string(AlgorithmCacheStatus status) {
    switch (status) {
    case AlgorithmCacheStatus::Disabled: return "Disabled";
    case AlgorithmCacheStatus::Miss: return "Miss";
    case AlgorithmCacheStatus::Hit: return "Hit";
    case AlgorithmCacheStatus::StrictMiss: return "StrictMiss";
    case AlgorithmCacheStatus::RejectedIncompatibleEnvironment: return "RejectedIncompatibleEnvironment";
    }
    return "Unknown";
  }

  inline const char* to_string(WorkspacePolicy policy) {
    switch (policy) {
    case WorkspacePolicy::AllenExternal: return "AllenExternal";
    case WorkspacePolicy::OwnedInitTime: return "OwnedInitTime";
    case WorkspacePolicy::ZeroOnly: return "ZeroOnly";
    }
    return "Unknown";
  }

  inline const char* to_string(TensorLayout layout) {
    switch (layout) {
    case TensorLayout::NCHW: return "NCHW";
    }
    return "Unknown";
  }

  inline const char* to_string(PoolingMode mode) {
    switch (mode) {
    case PoolingMode::Max: return "Max";
    case PoolingMode::AverageCountIncludePadding: return "AverageCountIncludePadding";
    case PoolingMode::AverageCountExcludePadding: return "AverageCountExcludePadding";
    }
    return "Unknown";
  }

  inline const char* to_string(ActivationMode mode) {
    switch (mode) {
    case ActivationMode::Identity: return "Identity";
    case ActivationMode::Relu: return "Relu";
    case ActivationMode::Sigmoid: return "Sigmoid";
    case ActivationMode::Tanh: return "Tanh";
    case ActivationMode::ClippedRelu: return "ClippedRelu";
    case ActivationMode::Elu: return "Elu";
    }
    return "Unknown";
  }

  inline std::string describe_precision_policy(const PrecisionPolicy& policy) {
    std::ostringstream out;
    out << "io=" << static_cast<int>(policy.input_output_type)
        << ",filter=" << static_cast<int>(policy.filter_type)
        << ",compute=" << static_cast<int>(policy.compute_type)
        << ",math=" << static_cast<int>(policy.math_type)
        << ",tensor_ops=" << (policy.tensor_ops_enabled ? "true" : "false")
        << ",tf32=" << (policy.allow_tf32 ? "true" : "false")
        << ",fp16_experimental=" << (policy.fp16_experimental ? "true" : "false");
    return out.str();
  }

  inline TensorShape make_tensor_shape(std::array<int, 4> shape) {
    return {shape[0], shape[1], shape[2], shape[3]};
  }

  inline void validate_tensor_shape(TensorShape shape, const char* name) {
    if (shape.n <= 0 || shape.c <= 0 || shape.h <= 0 || shape.w <= 0) {
      throw std::invalid_argument(std::string("AllenCuDNN: invalid non-positive dimension in ") + name);
    }
  }

  inline void validate_shape_4d(std::array<int, 4> shape, const char* name) {
    validate_tensor_shape(make_tensor_shape(shape), name);
  }

  inline void validate_pair_2d(std::array<int, 2> value, const char* name, bool allow_zero) {
    for (int dim : value) {
      if ((allow_zero && dim < 0) || (!allow_zero && dim <= 0)) {
        throw std::invalid_argument(std::string("AllenCuDNN: invalid 2D parameter in ") + name);
      }
    }
  }

  inline void validate_workspace_options(const ConvPlanOptions& options, const char* owner) {
    if (options.workspace_policy == WorkspacePolicy::ZeroOnly &&
        options.algorithm_policy != AlgorithmSelectionPolicy::ZeroWorkspace) {
      throw std::invalid_argument(
        std::string("AllenCuDNN: ") + owner + " ZeroOnly workspace requires ZeroWorkspace algorithm policy");
    }
  }

  inline void validate_layout(TensorLayout layout, const char* owner) {
    if (layout != TensorLayout::NCHW) {
      throw std::invalid_argument(std::string("AllenCuDNN: ") + owner + " only supports NCHW tensor layout");
    }
  }

  inline void validate_forward_shape(const Conv2DShape& shape, const char* owner) {
    validate_layout(shape.layout, owner);
    validate_tensor_shape(shape.input, (std::string(owner) + " input_shape").c_str());
    validate_tensor_shape(shape.filter, (std::string(owner) + " filter_shape").c_str());
    validate_pair_2d(shape.pad, (std::string(owner) + " pad").c_str(), true);
    validate_pair_2d(shape.stride, (std::string(owner) + " stride").c_str(), false);
    validate_pair_2d(shape.dilation, (std::string(owner) + " dilation").c_str(), false);
    if (shape.input.c != shape.filter.c) {
      throw std::invalid_argument("AllenCuDNN: ForwardConvPlan input channels do not match filter channels");
    }
  }

  inline void validate_backward_data_shape(const Conv2DShape& shape, const char* owner) {
    validate_layout(shape.layout, owner);
    validate_tensor_shape(shape.filter, (std::string(owner) + " filter_shape").c_str());
    validate_tensor_shape(shape.input, (std::string(owner) + " input_shape").c_str());
    validate_tensor_shape(shape.output, (std::string(owner) + " output_shape").c_str());
    validate_pair_2d(shape.pad, (std::string(owner) + " pad").c_str(), true);
    validate_pair_2d(shape.stride, (std::string(owner) + " stride").c_str(), false);
    validate_pair_2d(shape.dilation, (std::string(owner) + " dilation").c_str(), false);
    if (shape.filter.n != shape.input.c) {
      throw std::invalid_argument("AllenCuDNN: BackwardDataConvPlan filter output channels do not match input channels");
    }
    if (shape.filter.c != shape.output.c) {
      throw std::invalid_argument("AllenCuDNN: BackwardDataConvPlan filter input channels do not match output channels");
    }
  }

  inline void validate_pooling_shape(const Pooling2DShape& shape, const char* owner) {
    validate_layout(shape.layout, owner);
    validate_tensor_shape(shape.input, (std::string(owner) + " input_shape").c_str());
    validate_pair_2d(shape.window, (std::string(owner) + " window").c_str(), false);
    validate_pair_2d(shape.pad, (std::string(owner) + " pad").c_str(), true);
    validate_pair_2d(shape.stride, (std::string(owner) + " stride").c_str(), false);
    if (shape.has_output) {
      validate_tensor_shape(shape.output, (std::string(owner) + " output_shape").c_str());
    }
  }

  inline TensorShape pooling_forward_output_shape(const Pooling2DShape& shape, const char* owner) {
    validate_pooling_shape(shape, owner);
    const int h_extent = shape.input.h + 2 * shape.pad[0] - shape.window[0];
    const int w_extent = shape.input.w + 2 * shape.pad[1] - shape.window[1];
    if (h_extent < 0 || w_extent < 0) {
      throw std::invalid_argument(std::string("AllenCuDNN: ") + owner + " pooling window exceeds padded input");
    }
    const int output_h = h_extent / shape.stride[0] + 1;
    const int output_w = w_extent / shape.stride[1] + 1;
    if (output_h <= 0 || output_w <= 0) {
      throw std::invalid_argument(std::string("AllenCuDNN: ") + owner + " computed non-positive output dimension");
    }
    return {shape.input.n, shape.input.c, output_h, output_w};
  }

  inline TensorShape channel_bias_shape(TensorShape tensor_shape) {
    return {1, tensor_shape.c, 1, 1};
  }

  inline void validate_bias_shape(TensorShape tensor_shape, TensorShape bias_shape, TensorLayout layout, const char* owner) {
    validate_layout(layout, owner);
    validate_tensor_shape(tensor_shape, (std::string(owner) + " tensor_shape").c_str());
    validate_tensor_shape(bias_shape, (std::string(owner) + " bias_shape").c_str());
    if (bias_shape.n != 1 || bias_shape.c != tensor_shape.c || bias_shape.h != 1 || bias_shape.w != 1) {
      throw std::invalid_argument(std::string("AllenCuDNN: ") + owner + " bias_shape must be {1, C, 1, 1}");
    }
  }

#ifdef ALLEN_CUDNN_BACKEND_CUDA
  inline PrecisionPolicy fp32_precision_policy(
    cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH,
    bool allow_tf32 = true)
  {
    PrecisionPolicy policy {};
    policy.input_output_type = CUDNN_DATA_FLOAT;
    policy.filter_type = CUDNN_DATA_FLOAT;
    policy.compute_type = CUDNN_DATA_FLOAT;
    policy.math_type = math_type;
    policy.tensor_ops_enabled = math_type == CUDNN_TENSOR_OP_MATH;
    policy.allow_tf32 = allow_tf32;
    policy.fp16_experimental = false;
    return policy;
  }

  inline PrecisionPolicy fp16_precision_policy(bool experimental = true) {
    PrecisionPolicy policy {};
    policy.input_output_type = CUDNN_DATA_HALF;
    policy.filter_type = CUDNN_DATA_HALF;
    policy.compute_type = CUDNN_DATA_FLOAT;
    policy.math_type = CUDNN_TENSOR_OP_MATH;
    policy.tensor_ops_enabled = true;
    policy.allow_tf32 = false;
    policy.fp16_experimental = experimental;
    return policy;
  }

  inline void validate_precision_policy(const PrecisionPolicy& policy, const char* owner) {
    if (policy.input_output_type != policy.filter_type) {
      throw std::invalid_argument(std::string("AllenCuDNN: ") + owner + " requires matching input/output and filter data types");
    }
    if (policy.input_output_type == CUDNN_DATA_HALF && !policy.fp16_experimental) {
      throw std::invalid_argument(std::string("AllenCuDNN: ") + owner + " FP16 plans require fp16_experimental=true");
    }
  }

  inline PrecisionPolicy normalize_precision_policy(PrecisionPolicy policy) {
    if (!policy.tensor_ops_enabled ||
        (policy.input_output_type == CUDNN_DATA_FLOAT && !policy.allow_tf32)) {
      policy.math_type = CUDNN_DEFAULT_MATH;
    }
    return policy;
  }

  inline const char* to_string(cudnnConvolutionFwdAlgo_t algo) {
    switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM: return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT: return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
    }
    return "CUDNN_CONVOLUTION_FWD_ALGO_UNKNOWN";
  }

  inline const char* to_string(cudnnConvolutionBwdDataAlgo_t algo) {
    switch (algo) {
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED: return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED";
    }
    return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_UNKNOWN";
  }

  inline cudnnActivationMode_t to_cudnn_activation_mode(ActivationMode mode) {
    switch (mode) {
    case ActivationMode::Identity:
      throw std::invalid_argument("AllenCuDNN: identity activation is not a cuDNN activation descriptor");
    case ActivationMode::Relu: return CUDNN_ACTIVATION_RELU;
    case ActivationMode::Sigmoid: return CUDNN_ACTIVATION_SIGMOID;
    case ActivationMode::Tanh: return CUDNN_ACTIVATION_TANH;
    case ActivationMode::ClippedRelu: return CUDNN_ACTIVATION_CLIPPED_RELU;
    case ActivationMode::Elu: return CUDNN_ACTIVATION_ELU;
    }
    return CUDNN_ACTIVATION_RELU;
  }

  inline cudnnPoolingMode_t to_cudnn_pooling_mode(PoolingMode mode) {
    switch (mode) {
    case PoolingMode::Max: return CUDNN_POOLING_MAX;
    case PoolingMode::AverageCountIncludePadding: return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case PoolingMode::AverageCountExcludePadding: return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    }
    return CUDNN_POOLING_MAX;
  }

  inline bool plan_creation_logging_enabled(const ConvPlanOptions& options) {
    return options.log_plan_creation || std::getenv("ALLEN_CUDNN_VERBOSE") != nullptr;
  }

  inline bool pooling_plan_creation_logging_enabled(const PoolingOptions& options) {
    return options.log_plan_creation || std::getenv("ALLEN_CUDNN_VERBOSE") != nullptr;
  }

  inline void log_plan_creation(const char* plan_name, const ConvPlanMetadata& info, const ConvPlanOptions& options) {
    if (!plan_creation_logging_enabled(options)) return;
    std::fprintf(
      stderr,
      "AllenCuDNN: %s created layout=%s algorithm=%s(%d) selection_policy=%s selection_source=%s "
      "workspace_policy=%s workspace_bytes=%zu workspace_limit_bytes=%zu precision={%s} cache=%s",
      plan_name,
      to_string(info.layout),
      info.algorithm_name.c_str(),
      info.algorithm,
      to_string(info.algorithm_policy),
      to_string(info.selection_source),
      to_string(info.workspace_policy),
      info.workspace_bytes,
      info.workspace_limit_bytes,
      describe_precision_policy(info.precision).c_str(),
      to_string(info.cache.status));
    if (!info.cache.provenance.empty()) {
      std::fprintf(stderr, " cache_provenance=\"%s\"", info.cache.provenance.c_str());
    }
    if (!info.fallback_reason.empty()) {
      std::fprintf(stderr, " fallback_reason=\"%s\"", info.fallback_reason.c_str());
    }
    std::fprintf(stderr, "\n");
  }

  inline void log_pooling_plan_creation(const char* plan_name, const PoolingMetadata& info, const PoolingOptions& options) {
    if (!pooling_plan_creation_logging_enabled(options)) return;
    std::fprintf(
      stderr,
      "AllenCuDNN: %s created layout=%s mode=%s input=%dx%dx%dx%d output=%dx%dx%dx%d "
      "window=%dx%d pad=%dx%d stride=%dx%d workspace_bytes=%zu precision={%s}\n",
      plan_name,
      to_string(info.layout),
      to_string(info.mode),
      info.input_shape.n, info.input_shape.c, info.input_shape.h, info.input_shape.w,
      info.output_shape.n, info.output_shape.c, info.output_shape.h, info.output_shape.w,
      info.window[0], info.window[1],
      info.pad[0], info.pad[1],
      info.stride[0], info.stride[1],
      info.workspace_bytes,
      describe_precision_policy(info.precision).c_str());
  }

  namespace detail {
    inline size_t dtype_size(cudnnDataType_t dtype) {
      return dtype == CUDNN_DATA_HALF ? sizeof(__half) : sizeof(float);
    }

    struct AlgorithmCacheEntry {
      int algorithm = 0;
      size_t workspace_bytes = 0;
      AlgorithmSelectionSource selection_source = AlgorithmSelectionSource::Default;
      std::string algorithm_name;
      std::string fallback_reason;
      std::string provenance;
      std::string created_by;
      std::string device_name;
      size_t cudnn_version = 0;
      int cuda_runtime_version = 0;
    };

    class AlgorithmCacheStore {
    public:
      bool lookup(const std::string& key, AlgorithmCacheEntry& entry) const
      {
        std::lock_guard<std::mutex> lock {m_mutex};
        const auto found = m_entries.find(key);
        if (found == m_entries.end()) return false;
        entry = found->second;
        return true;
      }

      void insert(const std::string& key, AlgorithmCacheEntry entry)
      {
        std::lock_guard<std::mutex> lock {m_mutex};
        m_entries[key] = std::move(entry);
      }

      void clear()
      {
        std::lock_guard<std::mutex> lock {m_mutex};
        m_entries.clear();
      }

      size_t size() const
      {
        std::lock_guard<std::mutex> lock {m_mutex};
        return m_entries.size();
      }

    private:
      mutable std::mutex m_mutex;
      std::unordered_map<std::string, AlgorithmCacheEntry> m_entries;
    };

    inline AlgorithmCacheStore& algorithm_cache_store()
    {
      static AlgorithmCacheStore store;
      return store;
    }

    inline void clear_algorithm_cache() { algorithm_cache_store().clear(); }
    inline size_t algorithm_cache_size() { return algorithm_cache_store().size(); }

    inline std::string current_device_name()
    {
      int device = 0;
      if (cudaGetDevice(&device) != cudaSuccess) return "unknown-device";
      cudaDeviceProp properties {};
      if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return "unknown-device";
      return properties.name;
    }

    inline int cuda_runtime_version()
    {
      int version = 0;
      if (cudaRuntimeGetVersion(&version) != cudaSuccess) return 0;
      return version;
    }

    inline bool cache_entry_environment_matches(const AlgorithmCacheEntry& entry)
    {
      return entry.device_name == current_device_name() &&
             entry.cudnn_version == cudnnGetVersion() &&
             entry.cuda_runtime_version == cuda_runtime_version();
    }

    inline const char* operation_provenance(AlgorithmSelectionSource source)
    {
      switch (source) {
      case AlgorithmSelectionSource::Heuristic: return "heuristic";
      case AlgorithmSelectionSource::TimedFind: return "timed-find";
      case AlgorithmSelectionSource::ZeroWorkspace: return "zero-workspace";
      case AlgorithmSelectionSource::Fallback: return "fallback";
      case AlgorithmSelectionSource::Default: return "default";
      }
      return "unknown";
    }

    inline void append_shape(std::ostringstream& out, const char* name, TensorShape shape)
    {
      out << name << '=' << shape.n << 'x' << shape.c << 'x' << shape.h << 'x' << shape.w << ';';
    }

    inline void append_precision(std::ostringstream& out, const PrecisionPolicy& precision)
    {
      out << "io=" << static_cast<int>(precision.input_output_type)
          << ";filter=" << static_cast<int>(precision.filter_type)
          << ";compute=" << static_cast<int>(precision.compute_type)
          << ";math=" << static_cast<int>(precision.math_type)
          << ";tensor_ops=" << precision.tensor_ops_enabled
          << ";tf32=" << precision.allow_tf32
          << ";fp16_exp=" << precision.fp16_experimental << ';';
    }

    inline std::string algorithm_cache_key(
      const char* operation,
      Conv2DShape shape,
      const ConvPlanOptions& options,
      TensorShape output_shape)
    {
      std::ostringstream out;
      out << "op=" << operation << ';';
      append_shape(out, "input", shape.input);
      append_shape(out, "filter", shape.filter);
      append_shape(out, "output", output_shape);
      out << "pad=" << shape.pad[0] << 'x' << shape.pad[1] << ';'
          << "stride=" << shape.stride[0] << 'x' << shape.stride[1] << ';'
          << "dilation=" << shape.dilation[0] << 'x' << shape.dilation[1] << ';'
          << "layout=" << to_string(shape.layout) << ';';
      append_precision(out, options.precision);
      out << "algo_policy=" << to_string(options.algorithm_policy) << ';'
          << "workspace_policy=" << to_string(options.workspace_policy) << ';'
          << "workspace_limit=" << options.workspace_limit_bytes << ';'
          << "device=" << current_device_name() << ';'
          << "cuda_runtime=" << cuda_runtime_version() << ';'
          << "cudnn=" << cudnnGetVersion() << ';';
      return out.str();
    }

    inline void cuda_check(cudaError_t e, const char* what) {
      if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
      }
    }
  } // namespace detail

  inline void clear_algorithm_cache() { detail::clear_algorithm_cache(); }
  inline size_t algorithm_cache_size() { return detail::algorithm_cache_size(); }
#endif

#ifndef ALLEN_CUDNN_BACKEND_CUDA
  inline PrecisionPolicy fp32_precision_policy(int math_type = 0, bool allow_tf32 = false) {
    PrecisionPolicy policy {};
    policy.math_type = math_type;
    policy.allow_tf32 = allow_tf32;
    return policy;
  }

  inline PrecisionPolicy fp16_precision_policy(bool experimental = true) {
    PrecisionPolicy policy {};
    policy.fp16_experimental = experimental;
    return policy;
  }

  inline void validate_precision_policy(const PrecisionPolicy&, const char*) {}

  inline PrecisionPolicy normalize_precision_policy(PrecisionPolicy policy) { return policy; }

  inline void clear_algorithm_cache() {}
  inline size_t algorithm_cache_size() { return 0; }
#endif

  struct TensorDescriptor {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnTensorDescriptor_t m_desc = nullptr;

  public:
    TensorDescriptor() = default;
    ~TensorDescriptor() { reset(); }
    TensorDescriptor(const TensorDescriptor&) = delete;
    TensorDescriptor& operator=(const TensorDescriptor&) = delete;

    void reset() {
      if (m_desc) {
        cudnnDestroyTensorDescriptor(m_desc);
        m_desc = nullptr;
      }
    }

    void create() {
      reset();
      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_desc));
    }

    void set_4d(std::array<int, 4> shape, cudnnDataType_t dtype) {
      if (!m_desc) create();
      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_desc, CUDNN_TENSOR_NCHW, dtype, shape[0], shape[1], shape[2], shape[3]));
    }

    cudnnTensorDescriptor_t get() const { return m_desc; }
#else
    void reset() {}
    void create() {}
    void set_4d(std::array<int, 4>, int) {}
    void* get() const { return nullptr; }
#endif
  };

  struct FilterDescriptor {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnFilterDescriptor_t m_desc = nullptr;

  public:
    FilterDescriptor() = default;
    ~FilterDescriptor() { reset(); }
    FilterDescriptor(const FilterDescriptor&) = delete;
    FilterDescriptor& operator=(const FilterDescriptor&) = delete;

    void reset() {
      if (m_desc) {
        cudnnDestroyFilterDescriptor(m_desc);
        m_desc = nullptr;
      }
    }

    void create() {
      reset();
      ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_desc));
    }

    void set_4d(std::array<int, 4> shape, cudnnDataType_t dtype) {
      if (!m_desc) create();
      ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_desc, dtype, CUDNN_TENSOR_NCHW, shape[0], shape[1], shape[2], shape[3]));
    }

    cudnnFilterDescriptor_t get() const { return m_desc; }
#else
    void reset() {}
    void create() {}
    void set_4d(std::array<int, 4>, int) {}
    void* get() const { return nullptr; }
#endif
  };

  struct ConvolutionDescriptor {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnConvolutionDescriptor_t m_desc = nullptr;

  public:
    ConvolutionDescriptor() = default;
    ~ConvolutionDescriptor() { reset(); }
    ConvolutionDescriptor(const ConvolutionDescriptor&) = delete;
    ConvolutionDescriptor& operator=(const ConvolutionDescriptor&) = delete;

    void reset() {
      if (m_desc) {
        cudnnDestroyConvolutionDescriptor(m_desc);
        m_desc = nullptr;
      }
    }

    void create() {
      reset();
      ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_desc));
    }

    void set_2d(
      std::array<int, 2> pad,
      std::array<int, 2> stride,
      std::array<int, 2> dilation,
      cudnnDataType_t compute_type,
      cudnnMathType_t math_type)
    {
      if (!m_desc) create();
      ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_desc,
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        CUDNN_CROSS_CORRELATION, compute_type));
      ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(m_desc, math_type));
    }

    cudnnConvolutionDescriptor_t get() const { return m_desc; }
#else
    void reset() {}
    void create() {}
    void set_2d(std::array<int, 2>, std::array<int, 2>, std::array<int, 2>, int, int) {}
    void* get() const { return nullptr; }
#endif
  };

  struct ActivationDescriptor {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnActivationDescriptor_t m_desc = nullptr;

  public:
    ActivationDescriptor() = default;
    ~ActivationDescriptor() { reset(); }
    ActivationDescriptor(const ActivationDescriptor&) = delete;
    ActivationDescriptor& operator=(const ActivationDescriptor&) = delete;

    void reset() {
      if (m_desc) {
        cudnnDestroyActivationDescriptor(m_desc);
        m_desc = nullptr;
      }
    }

    void create() {
      reset();
      ALLEN_CUDNN_CHECK(cudnnCreateActivationDescriptor(&m_desc));
    }

    void set(ActivationMode mode, double coefficient) {
      if (!m_desc) create();
      ALLEN_CUDNN_CHECK(cudnnSetActivationDescriptor(
        m_desc, to_cudnn_activation_mode(mode), CUDNN_NOT_PROPAGATE_NAN, coefficient));
    }

    cudnnActivationDescriptor_t get() const { return m_desc; }
#else
    void reset() {}
    void create() {}
    void set(ActivationMode, double) {}
    void* get() const { return nullptr; }
#endif
  };

  struct PoolingDescriptor {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnPoolingDescriptor_t m_desc = nullptr;

  public:
    PoolingDescriptor() = default;
    ~PoolingDescriptor() { reset(); }
    PoolingDescriptor(const PoolingDescriptor&) = delete;
    PoolingDescriptor& operator=(const PoolingDescriptor&) = delete;

    void reset() {
      if (m_desc) {
        cudnnDestroyPoolingDescriptor(m_desc);
        m_desc = nullptr;
      }
    }

    void create() {
      reset();
      ALLEN_CUDNN_CHECK(cudnnCreatePoolingDescriptor(&m_desc));
    }

    void set_2d(PoolingMode mode, std::array<int, 2> window, std::array<int, 2> pad, std::array<int, 2> stride) {
      if (!m_desc) create();
      ALLEN_CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        m_desc,
        to_cudnn_pooling_mode(mode),
        CUDNN_NOT_PROPAGATE_NAN,
        window[0], window[1],
        pad[0], pad[1],
        stride[0], stride[1]));
    }

    cudnnPoolingDescriptor_t get() const { return m_desc; }
#else
    void reset() {}
    void create() {}
    void set_2d(PoolingMode, std::array<int, 2>, std::array<int, 2>, std::array<int, 2>) {}
    void* get() const { return nullptr; }
#endif
  };

  struct ForwardConvPlan {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    TensorDescriptor m_input_desc;
    FilterDescriptor m_filter_desc;
    ConvolutionDescriptor m_conv_desc;
    TensorDescriptor m_output_desc;
    cudnnConvolutionFwdAlgo_t m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    ConvPlanOptions m_options {};
    AlgorithmSelectionSource m_selection_source = AlgorithmSelectionSource::Default;
    AlgorithmCacheMetadata m_cache_metadata {};
    std::string m_fallback_reason;
    void* m_workspace = nullptr;
    size_t m_ws_bytes = 0;
    bool m_created = false;

    void release_workspace() {
      if (m_workspace) {
        cudaFree(m_workspace);
        m_workspace = nullptr;
      }
      m_ws_bytes = 0;
    }

    void set_workspace(size_t bytes) {
      m_ws_bytes = bytes;
      if (m_options.workspace_policy == WorkspacePolicy::ZeroOnly && bytes != 0) {
        throw std::runtime_error("ForwardConvPlan selected an algorithm that violates ZeroOnly workspace policy");
      }
      if (m_options.workspace_policy == WorkspacePolicy::OwnedInitTime && m_ws_bytes > 0) {
        detail::cuda_check(cudaMalloc(&m_workspace, m_ws_bytes), "ForwardConvPlan workspace allocation failed");
      }
    }

    bool cache_lookup(const std::string& key) {
      m_cache_metadata.key = key;
      if (m_options.cache_policy == AlgorithmCachePolicy::Disabled ||
          m_options.algorithm_policy == AlgorithmSelectionPolicy::ZeroWorkspace) {
        m_cache_metadata.status = AlgorithmCacheStatus::Disabled;
        return false;
      }

      if (m_options.cache_policy == AlgorithmCachePolicy::Populate) {
        m_cache_metadata.status = AlgorithmCacheStatus::Miss;
        return false;
      }

      detail::AlgorithmCacheEntry entry {};
      if (!detail::algorithm_cache_store().lookup(key, entry)) {
        if (m_options.cache_policy == AlgorithmCachePolicy::StrictLookup) {
          m_cache_metadata.status = AlgorithmCacheStatus::StrictMiss;
          throw std::runtime_error("AllenCuDNN: ForwardConvPlan strict algorithm cache lookup missed");
        }
        m_cache_metadata.status = AlgorithmCacheStatus::Miss;
        return false;
      }
      if (!detail::cache_entry_environment_matches(entry)) {
        if (m_options.cache_policy == AlgorithmCachePolicy::StrictLookup) {
          m_cache_metadata.status = AlgorithmCacheStatus::RejectedIncompatibleEnvironment;
          throw std::runtime_error("AllenCuDNN: ForwardConvPlan strict algorithm cache lookup found an incompatible environment");
        }
        m_cache_metadata.status = AlgorithmCacheStatus::RejectedIncompatibleEnvironment;
        return false;
      }

      m_algo = static_cast<cudnnConvolutionFwdAlgo_t>(entry.algorithm);
      m_selection_source = entry.selection_source;
      m_fallback_reason = entry.fallback_reason;
      m_cache_metadata.status = AlgorithmCacheStatus::Hit;
      m_cache_metadata.provenance = entry.provenance;
      m_cache_metadata.created_by = entry.created_by;
      set_workspace(entry.workspace_bytes);
      return true;
    }

    void cache_store(const std::string& key) {
      if (m_options.cache_policy != AlgorithmCachePolicy::Populate &&
          m_options.cache_policy != AlgorithmCachePolicy::LookupAndPopulate) {
        return;
      }
      if (m_selection_source == AlgorithmSelectionSource::Fallback && !m_options.cache_fallback_results) {
        return;
      }
      detail::AlgorithmCacheEntry entry {};
      entry.algorithm = algorithm_id();
      entry.workspace_bytes = m_ws_bytes;
      entry.selection_source = m_selection_source;
      entry.algorithm_name = algorithm_name();
      entry.fallback_reason = m_fallback_reason;
      entry.provenance = detail::operation_provenance(m_selection_source);
      entry.created_by = "ForwardConvPlan";
      entry.device_name = detail::current_device_name();
      entry.cudnn_version = cudnnGetVersion();
      entry.cuda_runtime_version = detail::cuda_runtime_version();
      detail::algorithm_cache_store().insert(key, entry);
      if (m_cache_metadata.status != AlgorithmCacheStatus::Hit &&
          m_cache_metadata.status != AlgorithmCacheStatus::RejectedIncompatibleEnvironment) {
        m_cache_metadata.status = AlgorithmCacheStatus::Miss;
      }
      m_cache_metadata.key = key;
      m_cache_metadata.provenance = entry.provenance;
      m_cache_metadata.created_by = entry.created_by;
    }

    void select_heuristic(cudnnHandle_t handle) {
      static constexpr int kMaxAlgos = 8;
      int returned = 0;
      cudnnConvolutionFwdAlgoPerf_t perf[kMaxAlgos];
      const auto status = cudnnGetConvolutionForwardAlgorithm_v7(
            handle,
            m_input_desc.get(),
            m_filter_desc.get(),
            m_conv_desc.get(),
            m_output_desc.get(),
            kMaxAlgos,
            &returned,
            perf);
      if (status == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < returned; ++i) {
          if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= m_options.workspace_limit_bytes) {
            m_algo = perf[i].algo;
            m_selection_source = AlgorithmSelectionSource::Heuristic;
            set_workspace(perf[i].memory);
            return;
          }
        }
        m_fallback_reason = "heuristic selection returned no successful algorithm within the workspace limit";
      }
      else {
        m_fallback_reason = std::string("cudnnGetConvolutionForwardAlgorithm_v7 failed: ") + cudnnGetErrorString(status);
      }
      m_selection_source = AlgorithmSelectionSource::Fallback;
    }

    void select_timed(
      cudnnHandle_t handle,
      TensorShape input_shape,
      TensorShape filter_shape,
      TensorShape output_shape)
    {
      static constexpr int kMaxAlgos = 8;
      const size_t dtype_bytes = detail::dtype_size(m_options.precision.input_output_type);
      const size_t in_elems = input_shape.elements();
      const size_t filt_elems = filter_shape.elements();
      const size_t out_elems = output_shape.elements();

      void *tmp_in = nullptr, *tmp_filt = nullptr, *tmp_out = nullptr, *search_ws = nullptr;
      detail::cuda_check(cudaMalloc(&tmp_in, in_elems * dtype_bytes), "ForwardConvPlan FindEx input allocation failed");
      detail::cuda_check(cudaMalloc(&tmp_filt, filt_elems * dtype_bytes), "ForwardConvPlan FindEx filter allocation failed");
      detail::cuda_check(cudaMalloc(&tmp_out, out_elems * dtype_bytes), "ForwardConvPlan FindEx output allocation failed");
      detail::cuda_check(cudaMalloc(&search_ws, m_options.workspace_limit_bytes), "ForwardConvPlan FindEx search allocation failed");

      int returned = 0;
      cudnnConvolutionFwdAlgoPerf_t perf[kMaxAlgos];
      const auto status = cudnnFindConvolutionForwardAlgorithmEx(
            handle,
            m_input_desc.get(), tmp_in,
            m_filter_desc.get(), tmp_filt,
            m_conv_desc.get(),
            m_output_desc.get(), tmp_out,
            kMaxAlgos, &returned, perf,
            search_ws, m_options.workspace_limit_bytes);
      if (status == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < returned; ++i) {
          if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= m_options.workspace_limit_bytes) {
            m_algo = perf[i].algo;
            m_selection_source = AlgorithmSelectionSource::TimedFind;
            set_workspace(perf[i].memory);
            break;
          }
        }
        if (m_selection_source != AlgorithmSelectionSource::TimedFind) {
          m_fallback_reason = "timed find returned no successful algorithm within the workspace limit";
          m_selection_source = AlgorithmSelectionSource::Fallback;
        }
      }
      else {
        m_fallback_reason = std::string("cudnnFindConvolutionForwardAlgorithmEx failed: ") + cudnnGetErrorString(status);
        m_selection_source = AlgorithmSelectionSource::Fallback;
      }

      cudaFree(tmp_in);
      cudaFree(tmp_filt);
      cudaFree(tmp_out);
      cudaFree(search_ws);
    }

  public:
    ForwardConvPlan() = default;
    ~ForwardConvPlan() { release_workspace(); }
    ForwardConvPlan(const ForwardConvPlan&) = delete;
    ForwardConvPlan& operator=(const ForwardConvPlan&) = delete;

    void reset() {
      release_workspace();
      m_selection_source = AlgorithmSelectionSource::Default;
      m_cache_metadata = {};
      m_fallback_reason.clear();
      m_created = false;
    }

    void create(
      cudnnHandle_t handle,
      Conv2DShape shape,
      ConvPlanOptions options = {})
    {
      validate_forward_shape(shape, "ForwardConvPlan");
      options.precision = normalize_precision_policy(options.precision);
      validate_precision_policy(options.precision, "ForwardConvPlan");
      validate_workspace_options(options, "ForwardConvPlan");
      release_workspace();
      m_options = options;
      m_selection_source = AlgorithmSelectionSource::Default;
      m_cache_metadata = {};
      m_fallback_reason.clear();

      m_input_desc.set_4d(shape.input.dims(), m_options.precision.input_output_type);
      m_filter_desc.set_4d(shape.filter.dims(), m_options.precision.filter_type);
      m_conv_desc.set_2d(
        shape.pad, shape.stride, shape.dilation, m_options.precision.compute_type, m_options.precision.math_type);

      int on = 0, oc = 0, oh = 0, ow = 0;
      ALLEN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        m_conv_desc.get(), m_input_desc.get(), m_filter_desc.get(), &on, &oc, &oh, &ow));
      const TensorShape output_shape {on, oc, oh, ow};
      if (shape.has_output && shape.output != output_shape) {
        throw std::invalid_argument("AllenCuDNN: ForwardConvPlan caller output_shape does not match cuDNN output shape");
      }
      m_output_desc.set_4d(output_shape.dims(), m_options.precision.input_output_type);

      m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      m_ws_bytes = 0;
      m_selection_source = m_options.algorithm_policy == AlgorithmSelectionPolicy::ZeroWorkspace ?
        AlgorithmSelectionSource::ZeroWorkspace : AlgorithmSelectionSource::Default;
      const std::string cache_key = detail::algorithm_cache_key("forward-convolution", shape, m_options, output_shape);
      const bool cache_hit = cache_lookup(cache_key);
      if (!cache_hit) {
        if (m_options.algorithm_policy == AlgorithmSelectionPolicy::Heuristic) {
          select_heuristic(handle);
        }
        else if (m_options.algorithm_policy == AlgorithmSelectionPolicy::TimedFind) {
          select_timed(handle, shape.input, shape.filter, output_shape);
        }
        cache_store(cache_key);
      }

      m_created = true;
      log_plan_creation("ForwardConvPlan", metadata(), m_options);
    }

    void create(
      cudnnHandle_t handle,
      std::array<int, 4> input_shape,
      std::array<int, 4> filter_shape,
      std::array<int, 2> pad = {0, 0},
      std::array<int, 2> stride = {1, 1},
      std::array<int, 2> dilation = {1, 1},
      ConvPlanOptions options = {})
    {
      create(
        handle,
        Conv2DShape::forward(make_tensor_shape(input_shape), make_tensor_shape(filter_shape), pad, stride, dilation),
        options);
    }

    void create(
      cudnnHandle_t handle,
      std::array<int, 4> input_shape,
      std::array<int, 4> filter_shape,
      std::array<int, 2> pad,
      std::array<int, 2> stride,
      std::array<int, 2> dilation,
      cudnnDataType_t dtype)
    {
      ConvPlanOptions options {};
      options.precision.input_output_type = dtype;
      options.precision.filter_type = dtype;
      options.precision.compute_type = CUDNN_DATA_FLOAT;
      options.precision.math_type = CUDNN_TENSOR_OP_MATH;
      options.precision.tensor_ops_enabled = true;
      options.precision.fp16_experimental = dtype == CUDNN_DATA_HALF;
      options.precision.allow_tf32 = dtype != CUDNN_DATA_HALF;
      create(handle, input_shape, filter_shape, pad, stride, dilation, options);
    }

    size_t workspace_bytes() const { return m_ws_bytes; }
    WorkspacePolicy workspace_policy() const { return m_options.workspace_policy; }
    AlgorithmSelectionPolicy algorithm_policy() const { return m_options.algorithm_policy; }
    AlgorithmSelectionSource selection_source() const { return m_selection_source; }
    const std::string& fallback_reason() const { return m_fallback_reason; }
    bool is_created() const { return m_created; }
    int algorithm_id() const { return static_cast<int>(m_algo); }
    const char* algorithm_name() const { return to_string(m_algo); }
    const PrecisionPolicy& precision_policy() const { return m_options.precision; }
    const AlgorithmCacheMetadata& cache_metadata() const { return m_cache_metadata; }
    cudnnDataType_t data_type() const { return m_options.precision.input_output_type; }
    cudnnDataType_t compute_type() const { return m_options.precision.compute_type; }
    cudnnMathType_t math_type() const { return m_options.precision.math_type; }

    ConvPlanMetadata metadata() const {
      ConvPlanMetadata info {};
      info.algorithm_policy = m_options.algorithm_policy;
      info.selection_source = m_selection_source;
      info.workspace_policy = m_options.workspace_policy;
      info.workspace_bytes = m_ws_bytes;
      info.workspace_limit_bytes = m_options.workspace_limit_bytes;
      info.algorithm_name = algorithm_name();
      info.fallback_reason = m_fallback_reason;
      info.created = m_created;
      info.layout = TensorLayout::NCHW;
      info.precision = m_options.precision;
      info.cache = m_cache_metadata;
      info.algorithm = algorithm_id();
      return info;
    }

    void forward(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const void* dev_input,
      const void* dev_filter,
      void* dev_output,
      void* external_workspace = nullptr) const
    {
      if (m_options.workspace_policy == WorkspacePolicy::AllenExternal) {
        Workspace {external_workspace, external_workspace == nullptr ? 0 : m_ws_bytes}.require(
          m_ws_bytes, "ForwardConvPlan");
      }
      void* workspace = m_options.workspace_policy == WorkspacePolicy::AllenExternal ? external_workspace : m_workspace;
      ALLEN_CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        m_input_desc.get(), dev_input,
        m_filter_desc.get(), dev_filter,
        m_conv_desc.get(),
        m_algo,
        workspace, m_ws_bytes,
        &beta,
        m_output_desc.get(), dev_output));
    }

    void forward(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const void* dev_input,
      const void* dev_filter,
      void* dev_output,
      Workspace external_workspace) const
    {
      external_workspace.require(m_ws_bytes, "ForwardConvPlan");
      forward(handle, alpha, beta, dev_input, dev_filter, dev_output, external_workspace.ptr);
    }

    void forward(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const float* dev_input,
      const float* dev_filter,
      float* dev_output) const
    {
      forward(handle, alpha, beta, (const void*) dev_input, (const void*) dev_filter, (void*) dev_output);
    }

    void forward(const Handle& handle, const float alpha, const float beta, const float* dev_input, const float* dev_filter, float* dev_output) const {
      forward(handle.get(), alpha, beta, dev_input, dev_filter, dev_output);
    }

    void forward_half(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const __half* dev_input,
      const __half* dev_filter,
      __half* dev_output) const
    {
      forward(handle, alpha, beta, (const void*) dev_input, (const void*) dev_filter, (void*) dev_output);
    }

    void forward_half(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const __half* dev_input,
      const __half* dev_filter,
      __half* dev_output,
      Workspace external_workspace) const
    {
      forward(
        handle,
        alpha,
        beta,
        static_cast<const void*>(dev_input),
        static_cast<const void*>(dev_filter),
        static_cast<void*>(dev_output),
        external_workspace);
    }
#else
    void create(void*, Conv2DShape, ConvPlanOptions = {}) {}
    void create(void*, std::array<int, 4>, std::array<int, 4>, std::array<int, 2> = {0, 0}, std::array<int, 2> = {1, 1}, std::array<int, 2> = {1, 1}, ConvPlanOptions = {}) {}
    size_t workspace_bytes() const { return 0; }
    WorkspacePolicy workspace_policy() const { return WorkspacePolicy::ZeroOnly; }
    AlgorithmSelectionPolicy algorithm_policy() const { return AlgorithmSelectionPolicy::ZeroWorkspace; }
    AlgorithmSelectionSource selection_source() const { return AlgorithmSelectionSource::Default; }
    const std::string& fallback_reason() const { static const std::string empty {}; return empty; }
    bool is_created() const { return false; }
    int algorithm_id() const { return 0; }
    const char* algorithm_name() const { return "CUDNN_CONVOLUTION_FWD_ALGO_UNKNOWN"; }
    const PrecisionPolicy& precision_policy() const { static const PrecisionPolicy policy {}; return policy; }
    const AlgorithmCacheMetadata& cache_metadata() const { static const AlgorithmCacheMetadata metadata {}; return metadata; }
    ConvPlanMetadata metadata() const { return {}; }
    void reset() {}
    void forward(void*, float, float, const float*, const float*, float*) const {}
    void forward(const Handle&, float, float, const float*, const float*, float*) const {}
    void forward(void*, float, float, const void*, const void*, void*, Workspace) const {}
    void forward_half(void*, float, float, const void*, const void*, void*) const {}
    void forward_half(void*, float, float, const void*, const void*, void*, Workspace) const {}
#endif
  };

  struct BackwardDataConvPlan {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    FilterDescriptor m_filter_desc;
    ConvolutionDescriptor m_conv_desc;
    TensorDescriptor m_input_desc;
    TensorDescriptor m_output_desc;
    cudnnConvolutionBwdDataAlgo_t m_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    ConvPlanOptions m_options {};
    AlgorithmSelectionSource m_selection_source = AlgorithmSelectionSource::Default;
    AlgorithmCacheMetadata m_cache_metadata {};
    std::string m_fallback_reason;
    void* m_workspace = nullptr;
    size_t m_ws_bytes = 0;
    bool m_created = false;

    void release_workspace() {
      if (m_workspace) {
        cudaFree(m_workspace);
        m_workspace = nullptr;
      }
      m_ws_bytes = 0;
    }

    void set_workspace(size_t bytes) {
      m_ws_bytes = bytes;
      if (m_options.workspace_policy == WorkspacePolicy::ZeroOnly && bytes != 0) {
        throw std::runtime_error("BackwardDataConvPlan selected an algorithm that violates ZeroOnly workspace policy");
      }
      if (m_options.workspace_policy == WorkspacePolicy::OwnedInitTime && m_ws_bytes > 0) {
        detail::cuda_check(cudaMalloc(&m_workspace, m_ws_bytes), "BackwardDataConvPlan workspace allocation failed");
      }
    }

    bool cache_lookup(const std::string& key) {
      m_cache_metadata.key = key;
      if (m_options.cache_policy == AlgorithmCachePolicy::Disabled ||
          m_options.algorithm_policy == AlgorithmSelectionPolicy::ZeroWorkspace) {
        m_cache_metadata.status = AlgorithmCacheStatus::Disabled;
        return false;
      }

      if (m_options.cache_policy == AlgorithmCachePolicy::Populate) {
        m_cache_metadata.status = AlgorithmCacheStatus::Miss;
        return false;
      }

      detail::AlgorithmCacheEntry entry {};
      if (!detail::algorithm_cache_store().lookup(key, entry)) {
        if (m_options.cache_policy == AlgorithmCachePolicy::StrictLookup) {
          m_cache_metadata.status = AlgorithmCacheStatus::StrictMiss;
          throw std::runtime_error("AllenCuDNN: BackwardDataConvPlan strict algorithm cache lookup missed");
        }
        m_cache_metadata.status = AlgorithmCacheStatus::Miss;
        return false;
      }
      if (!detail::cache_entry_environment_matches(entry)) {
        if (m_options.cache_policy == AlgorithmCachePolicy::StrictLookup) {
          m_cache_metadata.status = AlgorithmCacheStatus::RejectedIncompatibleEnvironment;
          throw std::runtime_error(
            "AllenCuDNN: BackwardDataConvPlan strict algorithm cache lookup found an incompatible environment");
        }
        m_cache_metadata.status = AlgorithmCacheStatus::RejectedIncompatibleEnvironment;
        return false;
      }

      m_algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(entry.algorithm);
      m_selection_source = entry.selection_source;
      m_fallback_reason = entry.fallback_reason;
      m_cache_metadata.status = AlgorithmCacheStatus::Hit;
      m_cache_metadata.provenance = entry.provenance;
      m_cache_metadata.created_by = entry.created_by;
      set_workspace(entry.workspace_bytes);
      return true;
    }

    void cache_store(const std::string& key) {
      if (m_options.cache_policy != AlgorithmCachePolicy::Populate &&
          m_options.cache_policy != AlgorithmCachePolicy::LookupAndPopulate) {
        return;
      }
      if (m_selection_source == AlgorithmSelectionSource::Fallback && !m_options.cache_fallback_results) {
        return;
      }
      detail::AlgorithmCacheEntry entry {};
      entry.algorithm = algorithm_id();
      entry.workspace_bytes = m_ws_bytes;
      entry.selection_source = m_selection_source;
      entry.algorithm_name = algorithm_name();
      entry.fallback_reason = m_fallback_reason;
      entry.provenance = detail::operation_provenance(m_selection_source);
      entry.created_by = "BackwardDataConvPlan";
      entry.device_name = detail::current_device_name();
      entry.cudnn_version = cudnnGetVersion();
      entry.cuda_runtime_version = detail::cuda_runtime_version();
      detail::algorithm_cache_store().insert(key, entry);
      if (m_cache_metadata.status != AlgorithmCacheStatus::Hit &&
          m_cache_metadata.status != AlgorithmCacheStatus::RejectedIncompatibleEnvironment) {
        m_cache_metadata.status = AlgorithmCacheStatus::Miss;
      }
      m_cache_metadata.key = key;
      m_cache_metadata.provenance = entry.provenance;
      m_cache_metadata.created_by = entry.created_by;
    }

    void select_heuristic(cudnnHandle_t handle) {
      static constexpr int kMaxAlgos = 8;
      int returned = 0;
      cudnnConvolutionBwdDataAlgoPerf_t perf[kMaxAlgos];
      const auto status = cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle,
            m_filter_desc.get(),
            m_input_desc.get(),
            m_conv_desc.get(),
            m_output_desc.get(),
            kMaxAlgos,
            &returned,
            perf);
      if (status == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < returned; ++i) {
          if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= m_options.workspace_limit_bytes) {
            m_algo = perf[i].algo;
            m_selection_source = AlgorithmSelectionSource::Heuristic;
            set_workspace(perf[i].memory);
            return;
          }
        }
        m_fallback_reason = "heuristic selection returned no successful algorithm within the workspace limit";
      }
      else {
        m_fallback_reason =
          std::string("cudnnGetConvolutionBackwardDataAlgorithm_v7 failed: ") + cudnnGetErrorString(status);
      }
      m_selection_source = AlgorithmSelectionSource::Fallback;
    }

    void select_timed(
      cudnnHandle_t handle,
      TensorShape filter_shape,
      TensorShape input_shape,
      TensorShape output_shape)
    {
      static constexpr int kMaxAlgos = 8;
      const size_t dtype_bytes = detail::dtype_size(m_options.precision.input_output_type);
      const size_t filt_elems = filter_shape.elements();
      const size_t in_elems = input_shape.elements();
      const size_t out_elems = output_shape.elements();

      void *tmp_filt = nullptr, *tmp_in = nullptr, *tmp_out = nullptr, *search_ws = nullptr;
      detail::cuda_check(cudaMalloc(&tmp_filt, filt_elems * dtype_bytes), "BackwardDataConvPlan FindEx filter allocation failed");
      detail::cuda_check(cudaMalloc(&tmp_in, in_elems * dtype_bytes), "BackwardDataConvPlan FindEx input allocation failed");
      detail::cuda_check(cudaMalloc(&tmp_out, out_elems * dtype_bytes), "BackwardDataConvPlan FindEx output allocation failed");
      detail::cuda_check(cudaMalloc(&search_ws, m_options.workspace_limit_bytes), "BackwardDataConvPlan FindEx search allocation failed");

      int returned = 0;
      cudnnConvolutionBwdDataAlgoPerf_t perf[kMaxAlgos];
      const auto status = cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle,
        m_filter_desc.get(), tmp_filt,
        m_input_desc.get(), tmp_in,
        m_conv_desc.get(),
        m_output_desc.get(), tmp_out,
        kMaxAlgos, &returned, perf,
        search_ws, m_options.workspace_limit_bytes);
      if (status == CUDNN_STATUS_SUCCESS) {
        for (int i = 0; i < returned; ++i) {
          if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= m_options.workspace_limit_bytes) {
            m_algo = perf[i].algo;
            m_selection_source = AlgorithmSelectionSource::TimedFind;
            set_workspace(perf[i].memory);
            break;
          }
        }
        if (m_selection_source != AlgorithmSelectionSource::TimedFind) {
          m_fallback_reason = "timed find returned no successful algorithm within the workspace limit";
          m_selection_source = AlgorithmSelectionSource::Fallback;
        }
      }
      else {
        m_fallback_reason =
          std::string("cudnnFindConvolutionBackwardDataAlgorithmEx failed: ") + cudnnGetErrorString(status);
        m_selection_source = AlgorithmSelectionSource::Fallback;
      }

      cudaFree(tmp_filt);
      cudaFree(tmp_in);
      cudaFree(tmp_out);
      cudaFree(search_ws);
    }

  public:
    BackwardDataConvPlan() = default;
    ~BackwardDataConvPlan() { release_workspace(); }
    BackwardDataConvPlan(const BackwardDataConvPlan&) = delete;
    BackwardDataConvPlan& operator=(const BackwardDataConvPlan&) = delete;

    void create(
      cudnnHandle_t handle,
      Conv2DShape shape,
      ConvPlanOptions options = {})
    {
      validate_backward_data_shape(shape, "BackwardDataConvPlan");
      options.precision = normalize_precision_policy(options.precision);
      validate_precision_policy(options.precision, "BackwardDataConvPlan");
      validate_workspace_options(options, "BackwardDataConvPlan");
      release_workspace();
      m_options = options;
      m_selection_source = AlgorithmSelectionSource::Default;
      m_cache_metadata = {};
      m_fallback_reason.clear();
      m_filter_desc.set_4d(shape.filter.dims(), m_options.precision.filter_type);
      m_input_desc.set_4d(shape.input.dims(), m_options.precision.input_output_type);
      m_output_desc.set_4d(shape.output.dims(), m_options.precision.input_output_type);
      m_conv_desc.set_2d(
        shape.pad, shape.stride, shape.dilation, m_options.precision.compute_type, m_options.precision.math_type);

      int on = 0, oc = 0, oh = 0, ow = 0;
      ALLEN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        m_conv_desc.get(), m_output_desc.get(), m_filter_desc.get(), &on, &oc, &oh, &ow));
      const TensorShape expected_input_shape {on, oc, oh, ow};
      if (shape.input != expected_input_shape) {
        throw std::invalid_argument("AllenCuDNN: BackwardDataConvPlan input_shape does not match cuDNN-computed forward output shape");
      }

      m_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      m_ws_bytes = 0;
      m_selection_source = m_options.algorithm_policy == AlgorithmSelectionPolicy::ZeroWorkspace ?
        AlgorithmSelectionSource::ZeroWorkspace : AlgorithmSelectionSource::Default;
      const std::string cache_key = detail::algorithm_cache_key("backward-data-convolution", shape, m_options, shape.output);
      const bool cache_hit = cache_lookup(cache_key);
      if (!cache_hit) {
        if (m_options.algorithm_policy == AlgorithmSelectionPolicy::Heuristic) {
          select_heuristic(handle);
        }
        else if (m_options.algorithm_policy == AlgorithmSelectionPolicy::TimedFind) {
          select_timed(handle, shape.filter, shape.input, shape.output);
        }
        cache_store(cache_key);
      }
      m_created = true;
      log_plan_creation("BackwardDataConvPlan", metadata(), m_options);
    }

    void create(
      cudnnHandle_t handle,
      std::array<int, 4> filter_shape,
      std::array<int, 4> input_shape,
      std::array<int, 4> output_shape,
      std::array<int, 2> pad = {0, 0},
      std::array<int, 2> stride = {1, 1},
      std::array<int, 2> dilation = {1, 1},
      ConvPlanOptions options = {})
    {
      create(
        handle,
        Conv2DShape::backward_data(
          make_tensor_shape(filter_shape), make_tensor_shape(input_shape), make_tensor_shape(output_shape), pad, stride, dilation),
        options);
    }

    size_t workspace_bytes() const { return m_ws_bytes; }
    WorkspacePolicy workspace_policy() const { return m_options.workspace_policy; }
    AlgorithmSelectionPolicy algorithm_policy() const { return m_options.algorithm_policy; }
    AlgorithmSelectionSource selection_source() const { return m_selection_source; }
    const std::string& fallback_reason() const { return m_fallback_reason; }
    bool is_created() const { return m_created; }
    int algorithm_id() const { return static_cast<int>(m_algo); }
    const char* algorithm_name() const { return to_string(m_algo); }
    const PrecisionPolicy& precision_policy() const { return m_options.precision; }
    const AlgorithmCacheMetadata& cache_metadata() const { return m_cache_metadata; }
    cudnnDataType_t data_type() const { return m_options.precision.input_output_type; }
    cudnnDataType_t compute_type() const { return m_options.precision.compute_type; }
    cudnnMathType_t math_type() const { return m_options.precision.math_type; }

    ConvPlanMetadata metadata() const {
      ConvPlanMetadata info {};
      info.algorithm_policy = m_options.algorithm_policy;
      info.selection_source = m_selection_source;
      info.workspace_policy = m_options.workspace_policy;
      info.workspace_bytes = m_ws_bytes;
      info.workspace_limit_bytes = m_options.workspace_limit_bytes;
      info.algorithm_name = algorithm_name();
      info.fallback_reason = m_fallback_reason;
      info.created = m_created;
      info.layout = TensorLayout::NCHW;
      info.precision = m_options.precision;
      info.cache = m_cache_metadata;
      info.algorithm = algorithm_id();
      return info;
    }

    void backward_data(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const void* dev_filter,
      const void* dev_input,
      void* dev_output,
      void* external_workspace = nullptr) const
    {
      if (m_options.workspace_policy == WorkspacePolicy::AllenExternal) {
        Workspace {external_workspace, external_workspace == nullptr ? 0 : m_ws_bytes}.require(
          m_ws_bytes, "BackwardDataConvPlan");
      }
      void* workspace = m_options.workspace_policy == WorkspacePolicy::AllenExternal ? external_workspace : m_workspace;
      ALLEN_CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        m_filter_desc.get(), dev_filter,
        m_input_desc.get(), dev_input,
        m_conv_desc.get(),
        m_algo,
        workspace, m_ws_bytes,
        &beta,
        m_output_desc.get(), dev_output));
    }

    void backward_data(
      cudnnHandle_t handle,
      const float alpha,
      const float beta,
      const void* dev_filter,
      const void* dev_input,
      void* dev_output,
      Workspace external_workspace) const
    {
      external_workspace.require(m_ws_bytes, "BackwardDataConvPlan");
      backward_data(handle, alpha, beta, dev_filter, dev_input, dev_output, external_workspace.ptr);
    }
#else
    void create(void*, Conv2DShape, ConvPlanOptions = {}) {}
    void create(void*, std::array<int, 4>, std::array<int, 4>, std::array<int, 4>, std::array<int, 2> = {0, 0}, std::array<int, 2> = {1, 1}, std::array<int, 2> = {1, 1}, ConvPlanOptions = {}) {}
    size_t workspace_bytes() const { return 0; }
    WorkspacePolicy workspace_policy() const { return WorkspacePolicy::ZeroOnly; }
    AlgorithmSelectionPolicy algorithm_policy() const { return AlgorithmSelectionPolicy::ZeroWorkspace; }
    AlgorithmSelectionSource selection_source() const { return AlgorithmSelectionSource::Default; }
    const std::string& fallback_reason() const { static const std::string empty {}; return empty; }
    bool is_created() const { return false; }
    int algorithm_id() const { return 0; }
    const char* algorithm_name() const { return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_UNKNOWN"; }
    const PrecisionPolicy& precision_policy() const { static const PrecisionPolicy policy {}; return policy; }
    const AlgorithmCacheMetadata& cache_metadata() const { static const AlgorithmCacheMetadata metadata {}; return metadata; }
    ConvPlanMetadata metadata() const { return {}; }
    void backward_data(void*, float, float, const void*, const void*, void*, void* = nullptr) const {}
    void backward_data(void*, float, float, const void*, const void*, void*, Workspace) const {}
#endif
  };

  struct PoolingPlan {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    TensorDescriptor m_input_desc;
    TensorDescriptor m_output_desc;
    PoolingDescriptor m_pooling_desc;
    PoolingOptions m_options {};
    Pooling2DShape m_shape {};
    TensorShape m_output_shape {};
    bool m_created = false;

  public:
    PoolingPlan() = default;
    PoolingPlan(const PoolingPlan&) = delete;
    PoolingPlan& operator=(const PoolingPlan&) = delete;

    void create(Pooling2DShape shape, PoolingOptions options = {}) {
      validate_layout(options.layout, "PoolingPlan");
      validate_pooling_shape(shape, "PoolingPlan");
      options.precision = normalize_precision_policy(options.precision);
      validate_precision_policy(options.precision, "PoolingPlan");

      const TensorShape computed_output = pooling_forward_output_shape(shape, "PoolingPlan");
      if (shape.has_output && shape.output != computed_output) {
        throw std::invalid_argument("AllenCuDNN: PoolingPlan caller output_shape does not match computed output shape");
      }

      m_options = options;
      m_shape = shape;
      m_shape.layout = m_options.layout;
      m_output_shape = computed_output;
      m_input_desc.set_4d(m_shape.input.dims(), m_options.precision.input_output_type);
      m_output_desc.set_4d(m_output_shape.dims(), m_options.precision.input_output_type);
      m_pooling_desc.set_2d(m_options.mode, m_shape.window, m_shape.pad, m_shape.stride);
      m_created = true;
      log_pooling_plan_creation("PoolingPlan", metadata(), m_options);
    }

    void create(std::array<int, 4> input_shape, std::array<int, 2> window, PoolingOptions options = {}) {
      create(Pooling2DShape::forward(make_tensor_shape(input_shape), window), options);
    }

    bool is_created() const { return m_created; }
    PoolingMode mode() const { return m_options.mode; }
    TensorShape input_shape() const { return m_shape.input; }
    TensorShape output_shape() const { return m_output_shape; }
    std::array<int, 2> window() const { return m_shape.window; }
    std::array<int, 2> pad() const { return m_shape.pad; }
    std::array<int, 2> stride() const { return m_shape.stride; }
    const PrecisionPolicy& precision_policy() const { return m_options.precision; }
    cudnnDataType_t data_type() const { return m_options.precision.input_output_type; }
    size_t workspace_bytes() const { return 0; }

    PoolingMetadata metadata() const {
      PoolingMetadata info {};
      info.created = m_created;
      info.mode = m_options.mode;
      info.layout = m_options.layout;
      info.input_shape = m_shape.input;
      info.output_shape = m_output_shape;
      info.window = m_shape.window;
      info.pad = m_shape.pad;
      info.stride = m_shape.stride;
      info.workspace_bytes = 0;
      info.precision = m_options.precision;
      return info;
    }

    void forward(
      cudnnHandle_t handle,
      const float alpha,
      const void* dev_input,
      const float beta,
      void* dev_output) const
    {
      ALLEN_CUDNN_CHECK(cudnnPoolingForward(
        handle,
        m_pooling_desc.get(),
        &alpha,
        m_input_desc.get(), dev_input,
        &beta,
        m_output_desc.get(), dev_output));
    }

    void forward(cudnnHandle_t handle, const float alpha, const float* dev_input, const float beta, float* dev_output) const {
      forward(handle, alpha, (const void*) dev_input, beta, (void*) dev_output);
    }

    void forward_half(
      cudnnHandle_t handle,
      const float alpha,
      const __half* dev_input,
      const float beta,
      __half* dev_output) const
    {
      forward(handle, alpha, (const void*) dev_input, beta, (void*) dev_output);
    }
#else
    void create(Pooling2DShape, PoolingOptions = {}) {}
    void create(std::array<int, 4>, std::array<int, 2>, PoolingOptions = {}) {}
    bool is_created() const { return false; }
    PoolingMode mode() const { return PoolingMode::Max; }
    TensorShape input_shape() const { return {}; }
    TensorShape output_shape() const { return {}; }
    std::array<int, 2> window() const { return {1, 1}; }
    std::array<int, 2> pad() const { return {0, 0}; }
    std::array<int, 2> stride() const { return {1, 1}; }
    const PrecisionPolicy& precision_policy() const { static const PrecisionPolicy policy {}; return policy; }
    int data_type() const { return 0; }
    size_t workspace_bytes() const { return 0; }
    PoolingMetadata metadata() const { return {}; }
    void forward(void*, float, const void*, float, void*) const {}
    void forward(void*, float, const float*, float, float*) const {}
    void forward_half(void*, float, const void*, float, void*) const {}
#endif
  };

  struct BiasAddPlan {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    TensorDescriptor m_bias_desc;
    TensorDescriptor m_tensor_desc;
    BiasAddOptions m_options {};
    TensorShape m_tensor_shape {};
    TensorShape m_bias_shape {};
    bool m_created = false;

  public:
    BiasAddPlan() = default;
    BiasAddPlan(const BiasAddPlan&) = delete;
    BiasAddPlan& operator=(const BiasAddPlan&) = delete;

    void create(TensorShape tensor_shape, BiasAddOptions options = {}) {
      options.precision = normalize_precision_policy(options.precision);
      validate_precision_policy(options.precision, "BiasAddPlan");
      const TensorShape bias_shape = channel_bias_shape(tensor_shape);
      validate_bias_shape(tensor_shape, bias_shape, options.layout, "BiasAddPlan");

      m_options = options;
      m_tensor_shape = tensor_shape;
      m_bias_shape = bias_shape;
      m_bias_desc.set_4d(m_bias_shape.dims(), m_options.precision.input_output_type);
      m_tensor_desc.set_4d(m_tensor_shape.dims(), m_options.precision.input_output_type);
      m_created = true;
    }

    void create(std::array<int, 4> tensor_shape, BiasAddOptions options = {}) {
      create(make_tensor_shape(tensor_shape), options);
    }

    bool is_created() const { return m_created; }
    TensorShape tensor_shape() const { return m_tensor_shape; }
    TensorShape bias_shape() const { return m_bias_shape; }
    const PrecisionPolicy& precision_policy() const { return m_options.precision; }
    cudnnDataType_t data_type() const { return m_options.precision.input_output_type; }

    BiasAddMetadata metadata() const {
      BiasAddMetadata info {};
      info.created = m_created;
      info.layout = m_options.layout;
      info.tensor_shape = m_tensor_shape;
      info.bias_shape = m_bias_shape;
      info.precision = m_options.precision;
      return info;
    }

    void add(
      cudnnHandle_t handle,
      const float alpha,
      const void* dev_bias,
      const float beta,
      void* dev_tensor) const
    {
      ALLEN_CUDNN_CHECK(cudnnAddTensor(
        handle,
        &alpha,
        m_bias_desc.get(), dev_bias,
        &beta,
        m_tensor_desc.get(), dev_tensor));
    }

    void add(cudnnHandle_t handle, const float alpha, const float* dev_bias, const float beta, float* dev_tensor) const {
      add(handle, alpha, (const void*) dev_bias, beta, (void*) dev_tensor);
    }

    void add_half(
      cudnnHandle_t handle,
      const float alpha,
      const __half* dev_bias,
      const float beta,
      __half* dev_tensor) const
    {
      add(handle, alpha, (const void*) dev_bias, beta, (void*) dev_tensor);
    }
#else
    void create(TensorShape, BiasAddOptions = {}) {}
    void create(std::array<int, 4>, BiasAddOptions = {}) {}
    bool is_created() const { return false; }
    TensorShape tensor_shape() const { return {}; }
    TensorShape bias_shape() const { return {}; }
    const PrecisionPolicy& precision_policy() const { static const PrecisionPolicy policy {}; return policy; }
    int data_type() const { return 0; }
    BiasAddMetadata metadata() const { return {}; }
    void add(void*, float, const void*, float, void*) const {}
    void add(void*, float, const float*, float, float*) const {}
    void add_half(void*, float, const void*, float, void*) const {}
#endif
  };

  struct ActivationPlan {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    TensorDescriptor m_input_desc;
    TensorDescriptor m_output_desc;
    ActivationDescriptor m_activation_desc;
    ActivationOptions m_options {};
    TensorShape m_tensor_shape {};
    bool m_created = false;

  public:
    ActivationPlan() = default;
    ActivationPlan(const ActivationPlan&) = delete;
    ActivationPlan& operator=(const ActivationPlan&) = delete;

    void create(TensorShape tensor_shape, ActivationOptions options = {}) {
      validate_layout(options.layout, "ActivationPlan");
      validate_tensor_shape(tensor_shape, "ActivationPlan tensor_shape");
      options.precision = normalize_precision_policy(options.precision);
      validate_precision_policy(options.precision, "ActivationPlan");

      m_options = options;
      m_tensor_shape = tensor_shape;
      m_input_desc.set_4d(m_tensor_shape.dims(), m_options.precision.input_output_type);
      m_output_desc.set_4d(m_tensor_shape.dims(), m_options.precision.input_output_type);
      m_activation_desc.set(m_options.mode, m_options.coefficient);
      m_created = true;
    }

    void create(std::array<int, 4> tensor_shape, ActivationOptions options = {}) {
      create(make_tensor_shape(tensor_shape), options);
    }

    bool is_created() const { return m_created; }
    ActivationMode mode() const { return m_options.mode; }
    double coefficient() const { return m_options.coefficient; }
    TensorShape tensor_shape() const { return m_tensor_shape; }
    const PrecisionPolicy& precision_policy() const { return m_options.precision; }
    cudnnDataType_t data_type() const { return m_options.precision.input_output_type; }

    ActivationMetadata metadata() const {
      ActivationMetadata info {};
      info.created = m_created;
      info.mode = m_options.mode;
      info.coefficient = m_options.coefficient;
      info.layout = m_options.layout;
      info.tensor_shape = m_tensor_shape;
      info.precision = m_options.precision;
      return info;
    }

    void forward(
      cudnnHandle_t handle,
      const float alpha,
      const void* dev_input,
      const float beta,
      void* dev_output) const
    {
      ALLEN_CUDNN_CHECK(cudnnActivationForward(
        handle,
        m_activation_desc.get(),
        &alpha,
        m_input_desc.get(), dev_input,
        &beta,
        m_output_desc.get(), dev_output));
    }

    void forward(cudnnHandle_t handle, const float alpha, const float* dev_input, const float beta, float* dev_output) const {
      forward(handle, alpha, (const void*) dev_input, beta, (void*) dev_output);
    }

    void forward_half(
      cudnnHandle_t handle,
      const float alpha,
      const __half* dev_input,
      const float beta,
      __half* dev_output) const
    {
      forward(handle, alpha, (const void*) dev_input, beta, (void*) dev_output);
    }
#else
    void create(TensorShape, ActivationOptions = {}) {}
    void create(std::array<int, 4>, ActivationOptions = {}) {}
    bool is_created() const { return false; }
    ActivationMode mode() const { return ActivationMode::Relu; }
    double coefficient() const { return 0.0; }
    TensorShape tensor_shape() const { return {}; }
    const PrecisionPolicy& precision_policy() const { static const PrecisionPolicy policy {}; return policy; }
    int data_type() const { return 0; }
    ActivationMetadata metadata() const { return {}; }
    void forward(void*, float, const void*, float, void*) const {}
    void forward(void*, float, const float*, float, float*) const {}
    void forward_half(void*, float, const void*, float, void*) const {}
#endif
  };

  // ---------------------------------------------------------------------------
  // Fixed-shape descriptor wrappers used by PVFinder's production UNet path.
  // ConvDescriptors keeps a pinned IMPLICIT_GEMM default, FP16/BF16 forwards and
  // per-thread workspaces tuned for many concurrent Allen streams;
  // ConvBiasReluGraph provides backend-graph fused Conv+Bias+ReLU for the
  // fixed-shape interface. General callers use the plan API above.
  // ---------------------------------------------------------------------------

  /**
   * @brief RAII wrapper for a cuDNN convolution's four descriptors.
   *
   * Design constraints (Allen compatibility):
   *  - Algorithm selection defaults to IMPLICIT_GEMM. PVFinder's thin,
   *    low-channel shapes favor its small workspace under concurrent load.
   *    Callers can enable bounded, live-verified heuristic selection with
   *    workspace_budget_bytes.
   *  - Workspace is thread_local, not a single shared buffer: cuDNN writes real
   *    per-call intermediate scratch state into it during forward() (im2col buffers,
   *    partial reductions, etc.) -- it is NOT read-only/inert. A single workspace
   *    buffer shared across concurrent threads on the same descriptor would be a data
   *    race whenever the selected algorithm has nonzero workspace_bytes() (IMPLICIT_GEMM
   *    itself is typically zero-workspace for these shapes, but the size is queried
   *    via cudnnGetConvolutionForwardWorkspaceSize() for every shape). Each descriptor
   *    instance selects one algorithm and size at create() time; only the workspace
   *    buffer backing that fixed size is lazily allocated per (thread, instance)
   *    via get_thread_local_workspace(), mirroring the thread_local idiom used elsewhere
   *    in Allen::CuDNN (see CuDNNHandle.h's get_thread_local_handle). Never freed (relies
   *    on process teardown, same as that handle) -- sizes here are a few KB at most.
   *  - Tensor descriptors are set once at create() time with the fixed input shape.
   *    Our UNet shapes are compile-time constants so this is safe.
   *  - forward() accepts a raw cudnnHandle_t so callers can use thread_local handles.
   */
  struct ConvDescriptors {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnTensorDescriptor_t      m_input_desc   = nullptr;
    cudnnFilterDescriptor_t      m_filter_desc  = nullptr;
    cudnnConvolutionDescriptor_t m_conv_desc    = nullptr;
    cudnnTensorDescriptor_t      m_output_desc  = nullptr;
    cudnnConvolutionFwdAlgo_t    m_algo         = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t                       m_ws_bytes     = 0;
    bool m_created = false;

    // Thread-local workspace, keyed by this instance within each thread's own map
    // (the map itself is thread_local, so no cross-thread contention/locking is
    // needed -- each thread only ever touches its own map). Lazily grows to
    // needed_bytes on first use per (thread, instance); never shrinks or frees.
    // cudaMalloc's result MUST be checked here: on failure it leaves the pointer
    // null, and if entry.second were still marked as "sized" despite that, every
    // subsequent calls would hand cudnnConvolutionForward a null workspace
    // with a nonzero requested size -- CUDNN_STATUS_BAD_PARAM, permanently, for
    // the rest of the process's life on that (thread, instance). Only record the
    // size only after allocation succeeds, so a transient failure is
    // retried on the next call instead of being latched in as a fatal state.
    static void* get_thread_local_workspace(const void* instance_key, size_t needed_bytes) {
      if (needed_bytes == 0) return nullptr;
      thread_local std::unordered_map<const void*, std::pair<void*, size_t>> tl_workspaces;
      auto& entry = tl_workspaces[instance_key];
      if (entry.second < needed_bytes) {
        if (entry.first) cudaFree(entry.first);
        entry.first = nullptr;
        detail::cuda_check(cudaMalloc(&entry.first, needed_bytes), "thread-local cuDNN workspace allocation failed");
        entry.second = needed_bytes;
      }
      return entry.first;
    }

  public:
    ConvDescriptors() = default;

    ~ConvDescriptors() {
      if (!m_created) return;
      cudnnDestroyTensorDescriptor(m_input_desc);
      cudnnDestroyFilterDescriptor(m_filter_desc);
      cudnnDestroyConvolutionDescriptor(m_conv_desc);
      cudnnDestroyTensorDescriptor(m_output_desc);
      // Thread-local workspaces (see get_thread_local_workspace) are intentionally
      // not freed here -- they may be owned by threads other than the one running
      // this destructor, and are negligible in size (a few KB per thread).
    }

    // Forces this instance's thread-local workspace to be allocated on the calling
    // thread immediately, rather than lazily on the first forward()/forward_half()
    // call. Needed before capturing a CUDA graph that calls forward() on this
    // descriptor: growing the workspace (cudaMalloc/cudaFree) DURING an active
    // stream capture is unsupported, so callers that capture must pre-warm every
    // descriptor they will use first.
    void ensure_thread_local_workspace() const { get_thread_local_workspace(this, m_ws_bytes); }

    ConvDescriptors(const ConvDescriptors&) = delete;
    ConvDescriptors& operator=(const ConvDescriptors&) = delete;

    // Create descriptors with fixed input shape. Algorithm is pinned to
    // IMPLICIT_GEMM by default (see rationale above) rather than selected via
    // cudnnFindConvolutionForwardAlgorithmEx -- unless workspace_budget_bytes
    // is nonzero, in which case a bounded heuristic search is used instead
    // (see the workspace_budget_bytes parameter doc below).
    // dtype: CUDNN_DATA_FLOAT (default) or CUDNN_DATA_HALF for FP16 Tensor Core path.
    // Compute type is always CUDNN_DATA_FLOAT (FP32 accumulation) for both dtypes.
    void create(
      cudnnHandle_t     handle,
      std::array<int,4> input_shape,            // {N, C_in, H, W} — fixed
      std::array<int,4> filter_shape,           // {K, C_in, R, S}
      std::array<int,2> pad      = {0, 0},
      std::array<int,2> stride   = {1, 1},
      std::array<int,2> dilation = {1, 1},
      cudnnDataType_t   dtype    = CUDNN_DATA_FLOAT,
      // 0 pins IMPLICIT_GEMM. A nonzero value runs
      // cudnnGetConvolutionForwardAlgorithm_v7 (a static heuristic cost
      // model, not a timed benchmark), then executes each
      // within-budget candidate with one real cudnnConvolutionForward() call
      // against scratch buffers before adopting it -- the heuristic can and
      // does report success for algorithms that then fail at real call time
      // for some shape/dtype combinations here (see the loop below). Falls
      // back to pinned IMPLICIT_GEMM if nothing fits the budget and survives
      // verification, or the initial query fails.
      size_t            workspace_budget_bytes = 0)
    {
      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_input_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_filter_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_conv_desc));
      ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_output_desc));

      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_input_desc, CUDNN_TENSOR_NCHW, dtype,
        input_shape[0], input_shape[1], input_shape[2], input_shape[3]));

      ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_filter_desc, dtype, CUDNN_TENSOR_NCHW,
        filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]));

      ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_conv_desc,
        pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
      // CUDNN_TENSOR_OP_MATH: enables TF32 on Ampere+ for FP32, and Tensor Core on all
      // supported GPUs for FP16 (wmma on SM 7.x, HMMA on SM 8.x+).
      ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(m_conv_desc, CUDNN_TENSOR_OP_MATH));

      // Derive and store output descriptor — fixed for this shape.
      int on, oc, oh, ow;
      ALLEN_CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        m_conv_desc, m_input_desc, m_filter_desc, &on, &oc, &oh, &ow));
      ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        m_output_desc, CUDNN_TENSOR_NCHW, dtype, on, oc, oh, ow));

      m_created = true;

      bool picked_by_search = false;
      if (workspace_budget_bytes > 0) {
        static constexpr int kFwdMaxAlgo = 8;
        cudnnConvolutionFwdAlgoPerf_t perf[kFwdMaxAlgo];
        int returned = 0;
        if (cudnnGetConvolutionForwardAlgorithm_v7(
              handle, m_input_desc, m_filter_desc, m_conv_desc, m_output_desc,
              kFwdMaxAlgo, &returned, perf) == CUDNN_STATUS_SUCCESS) {
          // The v7 ranking is a static cost model, so a successful query does
          // not guarantee that the candidate executes for this shape and dtype.
          // Execute each within-budget candidate against correctly sized scratch
          // buffers and adopt the first one that succeeds.
          const size_t elem_size  = (dtype == CUDNN_DATA_HALF) ? sizeof(__half) : sizeof(float);
          const size_t in_elems   = (size_t)input_shape[0]  * input_shape[1]  * input_shape[2]  * input_shape[3];
          const size_t filt_elems = (size_t)filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3];
          const size_t out_elems  = (size_t)on * oc * oh * ow;
          void* dummy_in   = nullptr;
          void* dummy_filt = nullptr;
          void* dummy_out  = nullptr;
          if (cudaMalloc(&dummy_in,   in_elems   * elem_size) == cudaSuccess &&
              cudaMalloc(&dummy_filt, filt_elems * elem_size) == cudaSuccess &&
              cudaMalloc(&dummy_out,  out_elems  * elem_size) == cudaSuccess) {
            cudaMemset(dummy_in,   0, in_elems   * elem_size);
            cudaMemset(dummy_filt, 0, filt_elems * elem_size);
            const float alpha = 1.f, beta = 0.f;
            for (int i = 0; i < returned && !picked_by_search; ++i) {
              if (perf[i].status != CUDNN_STATUS_SUCCESS || perf[i].memory > workspace_budget_bytes) continue;
              void* dummy_ws = nullptr;
              if (perf[i].memory > 0 && cudaMalloc(&dummy_ws, perf[i].memory) != cudaSuccess) continue;
              const cudnnStatus_t trial = cudnnConvolutionForward(
                handle, &alpha, m_input_desc, dummy_in, m_filter_desc, dummy_filt,
                m_conv_desc, perf[i].algo, dummy_ws, perf[i].memory, &beta, m_output_desc, dummy_out);
              if (dummy_ws) cudaFree(dummy_ws);
              if (trial == CUDNN_STATUS_SUCCESS) {
                m_algo        = perf[i].algo;
                m_ws_bytes    = perf[i].memory;
                picked_by_search = true;
              }
            }
          }
          if (dummy_in)   cudaFree(dummy_in);
          if (dummy_filt) cudaFree(dummy_filt);
          if (dummy_out)  cudaFree(dummy_out);
        }
      }
      if (!picked_by_search) {
        m_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        ALLEN_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
          handle, m_input_desc, m_filter_desc, m_conv_desc, m_output_desc, m_algo, &m_ws_bytes));
      }
      // Workspace buffer itself is not allocated here -- see
      // get_thread_local_workspace(): it's lazily allocated per (thread,
      // instance) on first forward()/forward_half() call instead.
    }

    size_t workspace_bytes() const { return m_ws_bytes; }
    int algo_id() const { return static_cast<int>(m_algo); }

    // Forward convolution using a thread_local cudnnHandle_t.
    void forward(
      cudnnHandle_t  handle,
      const float    alpha, const float beta,
      const float*   dev_input,
      const float*   dev_filter,
      float*         dev_output) const
    {
      ALLEN_CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        m_input_desc,  dev_input,
        m_filter_desc, dev_filter,
        m_conv_desc,
        m_algo,
        get_thread_local_workspace(this, m_ws_bytes), m_ws_bytes,
        &beta,
        m_output_desc, dev_output));
    }

    // Compatibility overload for callers that use the Handle wrapper.
    void forward(
      const Handle&  handle,
      const float    alpha, const float beta,
      const float*   dev_input,
      const float*   dev_filter,
      float*         dev_output) const
    {
      forward(handle.get(), alpha, beta, dev_input, dev_filter, dev_output);
    }

    // FP16 forward: for descriptors created with dtype=CUDNN_DATA_HALF.
    // Alpha/beta are float (cuDNN convention for FP16 tensors with FP32 accumulation).
    void forward_half(
      cudnnHandle_t   handle,
      const float     alpha, const float beta,
      const __half*   dev_input,
      const __half*   dev_filter,
      __half*         dev_output) const
    {
      ALLEN_CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        m_input_desc,  dev_input,
        m_filter_desc, dev_filter,
        m_conv_desc,
        m_algo,
        get_thread_local_workspace(this, m_ws_bytes), m_ws_bytes,
        &beta,
        m_output_desc, dev_output));
    }

    // BF16 forward: for descriptors created with dtype=CUDNN_DATA_BFLOAT16.
    // Mirrors forward_half exactly; BF16 shares FP32's exponent range, so it
    // avoids the overflow-to-NaN failure mode FP16 can hit on wide-dynamic-
    // range inputs.
    void forward_bf16(
      cudnnHandle_t         handle,
      const float           alpha, const float beta,
      const __nv_bfloat16*  dev_input,
      const __nv_bfloat16*  dev_filter,
      __nv_bfloat16*        dev_output) const
    {
      ALLEN_CUDNN_CHECK(cudnnConvolutionForward(
        handle,
        &alpha,
        m_input_desc,  dev_input,
        m_filter_desc, dev_filter,
        m_conv_desc,
        m_algo,
        get_thread_local_workspace(this, m_ws_bytes), m_ws_bytes,
        &beta,
        m_output_desc, dev_output));
    }

#else
    void create(cudnnHandle_t, std::array<int,4>, std::array<int,4>,
                std::array<int,2> = {0,0}, std::array<int,2> = {1,1},
                std::array<int,2> = {1,1}, cudnnDataType_t = CUDNN_DATA_FLOAT,
                size_t = 0) {}
    size_t workspace_bytes() const { return 0; }
    int algo_id() const { return 0; }
    void ensure_thread_local_workspace() const {}
    void forward(cudnnHandle_t, float, float,
                 const float*, const float*, float*) const {}
    void forward(const Handle&, float, float,
                 const float*, const float*, float*) const {}
    void forward_half(cudnnHandle_t, float, float,
                      const void*, const void*, void*) const {}
    void forward_bf16(cudnnHandle_t, float, float,
                      const void*, const void*, void*) const {}
#endif
  };

  /**
   * @brief Fused Conv + BiasAdd + ReLU via cuDNN backend graph API.
   *
   * BN parameters are folded into the conv weights/bias by the caller at init
   * time, so no separate batch-norm kernel is needed at runtime. The graph
   * encodes Conv → Pointwise-ADD(bias) → Pointwise-RELU as a single fused op
   * that cuDNN executes without writing the intermediate conv output to global
   * memory. Eliminates the conv-output DRAM round trip that a separate
   * bias+ReLU kernel pass otherwise requires.
   *
   * Tensor strides are set to standard NCHW-contiguous (H=1) layout — the
   * cuDNN backend graph API does not require NHWC; it accepts arbitrary
   * strides via CUDNN_ATTR_TENSOR_STRIDES. What IS GPU/shape-dependent is
   * whether any engine advertises support for this op-graph at all: create()
   * throws std::invalid_argument if the engine-heuristic search returns no
   * usable engine config, so callers on unsupported hardware must catch that
   * and fall back to the separate ConvDescriptors + bias/ReLU kernel path.
   *
   * Workspace is thread_local (see get_thread_local_workspace below), mirroring
   * ConvDescriptors: the execution plan is a read-only compiled artifact safe
   * to share across concurrent threads once created, but the workspace buffer
   * cuDNN scribbles into during execute() is not, so each thread gets its own.
   *
   * UIDs for the variant pack (stable per graph instance):
   *   1 = x (input), 2 = w (fused weights), 3 = b (fused bias), 4 = y (output)
   */
  struct ConvBiasReluGraph {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
  private:
    cudnnBackendDescriptor_t m_exec_plan = nullptr;
    size_t m_ws_bytes  = 0;
    bool   m_created   = false;

    static constexpr int64_t UID_X = 1, UID_W = 2, UID_B = 3, UID_Y = 4;
    static constexpr int64_t UID_ZCONV = 5, UID_ZADD = 6;

    // Thread-local workspace, keyed by this instance within each thread's own
    // map — same idiom as ConvDescriptors::get_thread_local_workspace (see
    // that function's comment for the cudaMalloc-failure handling rationale).
    static void* get_thread_local_workspace(const void* instance_key, size_t needed_bytes) {
      if (needed_bytes == 0) return nullptr;
      thread_local std::unordered_map<const void*, std::pair<void*, size_t>> tl_workspaces;
      auto& entry = tl_workspaces[instance_key];
      if (entry.second < needed_bytes) {
        if (entry.first) cudaFree(entry.first);
        entry.first = nullptr;
        detail::cuda_check(cudaMalloc(&entry.first, needed_bytes), "thread-local cuDNN workspace allocation failed");
        entry.second = needed_bytes;
      }
      return entry.first;
    }

  public:
    ConvBiasReluGraph() = default;

    ~ConvBiasReluGraph() {
      if (!m_created) return;
      cudnnBackendDestroyDescriptor(m_exec_plan);
      // Thread-local workspaces (see get_thread_local_workspace) are intentionally
      // not freed here — same rationale as ConvDescriptors's destructor.
    }

    // Pre-allocate this instance's thread-local workspace on the calling thread.
    void ensure_thread_local_workspace() const { get_thread_local_workspace(this, m_ws_bytes); }

    ConvBiasReluGraph(const ConvBiasReluGraph&) = delete;
    ConvBiasReluGraph& operator=(const ConvBiasReluGraph&) = delete;

    // Build a Conv+BiasAdd+ReLU graph, compile to an execution plan, and
    // allocate its workspace. input_shape/filter_shape are fixed at this call.
    // filter_shape is {K, C_in, 1, R} where R is the 1-D kernel size.
    // Fused weights (BN folded in) and fused bias are passed at execute() time.
    void create(
      cudnnHandle_t     handle,
      std::array<int,4> input_shape,           // {N, C_in, 1, W}
      std::array<int,4> filter_shape,          // {K, C_in, 1, R}
      std::array<int,2> pad      = {0, 0},
      std::array<int,2> stride   = {1, 1},
      std::array<int,2> dilation = {1, 1})
    {
      int N = input_shape[0], C_in = input_shape[1], W_in = input_shape[3];
      int K = filter_shape[0], R = filter_shape[3];
      int W_out = (W_in + 2*pad[1] - dilation[1]*(R-1) - 1) / stride[1] + 1;

      // Helper: create a backend descriptor of the given type.
      auto mk = [](cudnnBackendDescriptorType_t t) {
        cudnnBackendDescriptor_t d = nullptr;
        ALLEN_CUDNN_CHECK(cudnnBackendCreateDescriptor(t, &d));
        return d;
      };

      // Helper: create, configure, and finalize a tensor descriptor.
      auto mk_tensor = [&](int64_t uid, int64_t n, int64_t c, int64_t h, int64_t w, bool is_virtual) {
        auto d = mk(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        int64_t dims[4]    = {n, c, h, w};
        int64_t strides[4] = {c*h*w, h*w, w, 1};
        int64_t align = 4;
        cudnnDataType_t dt = CUDNN_DATA_FLOAT;
        int8_t  virt = is_virtual ? 1 : 0;
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DATA_TYPE,     CUDNN_TYPE_DATA_TYPE, 1, &dt));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_UNIQUE_ID,     CUDNN_TYPE_INT64,     1, &uid));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_DIMENSIONS,    CUDNN_TYPE_INT64,     4, dims));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_STRIDES,       CUDNN_TYPE_INT64,     4, strides));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,     1, &align));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL,    CUDNN_TYPE_BOOLEAN,   1, &virt));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(d));
        return d;
      };

      // Tensor descriptors: real (x, w, b, y) and virtual (zconv, zadd).
      auto x_t     = mk_tensor(UID_X,     N,  C_in, 1, W_in,  false);
      auto w_t     = mk_tensor(UID_W,     K,  C_in, 1, R,     false);
      auto b_t     = mk_tensor(UID_B,     1,  K,    1, 1,     false);
      auto y_t     = mk_tensor(UID_Y,     N,  K,    1, W_out, false);
      auto zconv_t = mk_tensor(UID_ZCONV, N,  K,    1, W_out, true);
      auto zadd_t  = mk_tensor(UID_ZADD,  N,  K,    1, W_out, true);

      // Convolution descriptor (backend variant).
      auto conv_d = mk(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
      {
        cudnnDataType_t       comp  = CUDNN_DATA_FLOAT;
        cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
        int64_t sdims = 2;
        int64_t cpads[2]   = {pad[0],      pad[1]};
        int64_t cstrs[2]   = {stride[0],   stride[1]};
        int64_t cdils[2]   = {dilation[0], dilation[1]};
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_COMP_TYPE,      CUDNN_TYPE_DATA_TYPE,        1, &comp));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_CONV_MODE,      CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,   CUDNN_TYPE_INT64,            1, &sdims));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,   CUDNN_TYPE_INT64,            2, cpads));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,  CUDNN_TYPE_INT64,            2, cpads));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64,            2, cstrs));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_d, CUDNN_ATTR_CONVOLUTION_DILATIONS,      CUDNN_TYPE_INT64,            2, cdils));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(conv_d));
      }

      // Pointwise ADD descriptor (for bias add).
      auto add_pw = mk(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
      {
        cudnnPointwiseMode_t m = CUDNN_POINTWISE_ADD;
        cudnnDataType_t      p = CUDNN_DATA_FLOAT;
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_pw, CUDNN_ATTR_POINTWISE_MODE,      CUDNN_TYPE_POINTWISE_MODE, 1, &m));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE,      1, &p));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(add_pw));
      }

      // Pointwise RELU descriptor.
      auto relu_pw = mk(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
      {
        cudnnPointwiseMode_t m = CUDNN_POINTWISE_RELU_FWD;
        cudnnDataType_t      p = CUDNN_DATA_FLOAT;
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_pw, CUDNN_ATTR_POINTWISE_MODE,      CUDNN_TYPE_POINTWISE_MODE, 1, &m));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_pw, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE,      1, &p));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(relu_pw));
      }

      // Convolution forward operation: x, w → zconv (virtual).
      float falpha = 1.f, fbeta = 0.f;
      auto conv_op = mk(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
      {
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zconv_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_d));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,     CUDNN_TYPE_FLOAT,              1, &falpha));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,      CUDNN_TYPE_FLOAT,              1, &fbeta));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(conv_op));
      }

      // Pointwise ADD operation: zconv + b → zadd (virtual).
      auto add_op = mk(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
      {
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &add_pw));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zconv_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(add_op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zadd_t));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(add_op));
      }

      // Pointwise RELU operation: zadd → y.
      auto relu_op = mk(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
      {
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &relu_pw));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zadd_t));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(relu_op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC,         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y_t));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(relu_op));
      }

      // Operation graph: {conv_op, add_op, relu_op}.
      auto op_graph = mk(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
      {
        cudnnBackendDescriptor_t ops[3] = {conv_op, add_op, relu_op};
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE,             1, &handle));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS,    CUDNN_TYPE_BACKEND_DESCRIPTOR, 3, ops));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(op_graph));
      }

      // Engine heuristics: instant mode (fast lookup, no profiling).
      auto heur = mk(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
      {
        cudnnBackendHeurMode_t hmode = CUDNN_HEUR_MODE_INSTANT;
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_MODE,            CUDNN_TYPE_HEUR_MODE,          1, &hmode));
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(heur));
      }

      // Retrieve engine configs; iterate until one compiles to a valid plan.
      static constexpr int kMaxEngines = 10;
      cudnnBackendDescriptor_t eng_cfgs[kMaxEngines] = {};
      for (int i = 0; i < kMaxEngines; ++i)
        ALLEN_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &eng_cfgs[i]));
      int64_t returned = 0;
      ALLEN_CUDNN_CHECK(cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, kMaxEngines, &returned, eng_cfgs));

      for (int64_t i = 0; i < returned && !m_exec_plan; ++i) {
        cudnnBackendDescriptor_t plan = nullptr;
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan) != CUDNN_STATUS_SUCCESS) continue;
        if (cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE,        CUDNN_TYPE_HANDLE,             1, &handle)        != CUDNN_STATUS_SUCCESS ||
            cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &eng_cfgs[i])   != CUDNN_STATUS_SUCCESS ||
            cudnnBackendFinalize(plan)                                                                                                  != CUDNN_STATUS_SUCCESS) {
          cudnnBackendDestroyDescriptor(plan);
          continue;
        }
        int64_t ws = 0;
        cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, nullptr, &ws);
        m_ws_bytes  = (size_t)ws;
        m_exec_plan = plan;
        // Workspace buffer itself is not allocated here — see
        // get_thread_local_workspace(): it's lazily allocated per (thread,
        // instance) on first execute() call instead, mirroring ConvDescriptors.
      }

      // Free all intermediate descriptors.
      cudnnBackendDestroyDescriptor(x_t);
      cudnnBackendDestroyDescriptor(w_t);
      cudnnBackendDestroyDescriptor(b_t);
      cudnnBackendDestroyDescriptor(y_t);
      cudnnBackendDestroyDescriptor(zconv_t);
      cudnnBackendDestroyDescriptor(zadd_t);
      cudnnBackendDestroyDescriptor(conv_d);
      cudnnBackendDestroyDescriptor(add_pw);
      cudnnBackendDestroyDescriptor(relu_pw);
      cudnnBackendDestroyDescriptor(conv_op);
      cudnnBackendDestroyDescriptor(add_op);
      cudnnBackendDestroyDescriptor(relu_op);
      cudnnBackendDestroyDescriptor(op_graph);
      cudnnBackendDestroyDescriptor(heur);
      for (int i = 0; i < kMaxEngines; ++i)
        if (eng_cfgs[i]) cudnnBackendDestroyDescriptor(eng_cfgs[i]);

      if (!m_exec_plan)
        throw std::invalid_argument("ConvBiasReluGraph::create: no valid engine for Conv+Add+ReLU");
      m_created = true;
    }

    bool is_created() const { return m_created; }
    size_t workspace_bytes() const { return m_ws_bytes; }

    // Execute the fused graph. x/w/b are device pointers; w and b must be the
    // BN-folded fused weights and biases computed at init time.
    void execute(
      cudnnHandle_t handle,
      const float*  x,
      const float*  w,
      const float*  b,
      float*        y) const
    {
      void* workspace = get_thread_local_workspace(this, m_ws_bytes);

      cudnnBackendDescriptor_t vpack = nullptr;
      ALLEN_CUDNN_CHECK(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vpack));

      int64_t uids[4] = {UID_X, UID_W, UID_B, UID_Y};
      void*   ptrs[4] = {const_cast<float*>(x), const_cast<float*>(w),
                         const_cast<float*>(b), y};
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,    CUDNN_TYPE_INT64,     4, uids));
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR,  4, ptrs));
      ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(vpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,     CUDNN_TYPE_VOID_PTR,  1, &workspace));
      ALLEN_CUDNN_CHECK(cudnnBackendFinalize(vpack));
      ALLEN_CUDNN_CHECK(cudnnBackendExecute(handle, m_exec_plan, vpack));
      cudnnBackendDestroyDescriptor(vpack);
    }

#else
    void create(cudnnHandle_t, std::array<int,4>, std::array<int,4>,
                std::array<int,2> = {0,0}, std::array<int,2> = {1,1},
                std::array<int,2> = {1,1}) {}
    bool is_created() const { return false; }
    size_t workspace_bytes() const { return 0; }
    void ensure_thread_local_workspace() const {}
    void execute(cudnnHandle_t, const float*, const float*, const float*, float*) const {}
#endif
  };

} // namespace Allen::CuDNN
