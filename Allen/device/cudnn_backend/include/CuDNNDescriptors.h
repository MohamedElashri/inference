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

#ifdef ALLEN_CUDNN_BACKEND_CUDA
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

  inline bool plan_creation_logging_enabled(const ConvPlanOptions& options) {
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

  using ConvDescriptors = ForwardConvPlan;

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

  // ConvBiasReluGraph: placeholder — implementation removed (cuDNN backend graph
  // API requires NHWC layout and is not supported on all target GPUs).
  struct ConvBiasReluGraph {
    void create(
#ifdef ALLEN_CUDNN_BACKEND_CUDA
      cudnnHandle_t,
#else
      void*,
#endif
      std::array<int, 4>, std::array<int, 4>,
      std::array<int, 2> = {0, 0}, std::array<int, 2> = {1, 1}, std::array<int, 2> = {1, 1})
    {}
    bool is_created() const { return false; }
    void execute(
#ifdef ALLEN_CUDNN_BACKEND_CUDA
      cudnnHandle_t,
#else
      void*,
#endif
      const float*, const float*, const float*, float*) const
    {}
  };

} // namespace Allen::CuDNN
