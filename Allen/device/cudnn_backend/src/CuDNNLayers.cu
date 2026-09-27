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
#include "CuDNNLayers.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace Allen::CuDNN {

  namespace {
    // Spatial parameters as (H, W): 1D values get H = 1 (size, kernel, stride,
    // dilation) or 0 (padding) in front.
    std::vector<int64_t> as_2d(const std::vector<int64_t>& v, int64_t fill, const char* what)
    {
      if (v.size() == 2) return v;
      if (v.size() == 1) return {fill, v[0]};
      throw StrException(std::string("Allen::CuDNN: ") + what + " needs 1 or 2 values");
    }

    __device__ inline float to_float(float v) { return v; }
    __device__ inline float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
    __device__ inline float to_float(__half v) { return __half2float(v); }
    template<typename T>
    __device__ inline T from_float(float v);
    template<>
    __device__ inline float from_float<float>(float v)
    {
      return v;
    }
    template<>
    __device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float v)
    {
      return __float2bfloat16(v);
    }
    template<>
    __device__ inline __half from_float<__half>(float v)
    {
      return __float2half(v);
    }

    // y = activation(y + b[channel]) * scale, in place, arithmetic in float.
    // Channel of element i: (i / spatial) % C (NCHW) or i % C (NHWC).
    template<typename T>
    __global__ void epilogue_kernel(
      T* __restrict__ y,
      const T* __restrict__ bias,
      int64_t total,
      int64_t channels,
      int64_t spatial,
      bool channels_last,
      int activation,
      float parameter,
      float scale)
    {
      const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (i >= total) return;
      float v = to_float(y[i]);
      if (bias != nullptr) {
        const int64_t c = channels_last ? i % channels : (i / spatial) % channels;
        v += to_float(bias[c]);
      }
      switch (static_cast<Activation>(activation)) {
      case Activation::Relu: v = v > 0.f ? v : 0.f; break;
      case Activation::LeakyRelu: v = v > 0.f ? v : v * parameter; break;
      case Activation::Sigmoid: v = 1.f / (1.f + expf(-v)); break;
      case Activation::Tanh: v = tanhf(v); break;
      // Stable softplus: max(v, 0) + log(1 + exp(-|v|)).
      case Activation::Softplus: v = fmaxf(v, 0.f) + logf(1.f + expf(-fabsf(v))); break;
      default: break;
      }
      y[i] = from_float<T>(v * scale);
    }
  } // namespace

  void ConvolutionLayer::create(cudnnHandle_t handle, const ConvolutionSpec& spec, const BuildOptions& options)
  {
    m_spec = spec;
    const auto in = as_2d(spec.input_size, 1, "input_size");
    const auto kernel = as_2d(spec.kernel_size, 1, "kernel_size");
    const ConvolutionParams params {
      as_2d(spec.padding, 0, "padding"), as_2d(spec.stride, 1, "stride"), as_2d(spec.dilation, 1, "dilation")};
    const std::vector<int64_t> x_dims {spec.batch, spec.in_channels, in[0], in[1]};
    const std::vector<int64_t> w_dims =
      spec.transposed ? std::vector<int64_t> {spec.in_channels, spec.out_channels, kernel[0], kernel[1]} :
                        std::vector<int64_t> {spec.out_channels, spec.in_channels, kernel[0], kernel[1]};
    std::vector<int64_t> y_dims {spec.batch, spec.out_channels};
    if (spec.transposed) {
      const auto out = spec.output_size.empty() ? std::vector<int64_t> {} : as_2d(spec.output_size, 1, "output_size");
      for (int i = 0; i < 2; ++i) {
        y_dims.push_back(
          out.empty() ?
            (in[i] - 1) * params.stride[i] - 2 * params.padding[i] + params.dilation[i] * (kernel[i] - 1) + 1 :
            out[i]);
      }
    }
    const bool epilogue = spec.bias || spec.activation != Activation::None || spec.output_scale != 1.f;

    // The whole layer as one graph, or (fused = false) the convolution alone.
    const auto graph = [&](bool fused) {
      Graph g(spec.type, DataType::Float);
      const TensorId x = g.input(x_dims, spec.layout);
      const TensorId w = g.input(w_dims, spec.layout);
      TensorId b {};
      if (fused && spec.bias) b = g.input({1, spec.out_channels, 1, 1}, spec.layout);
      TensorId y = spec.transposed ? g.transposed_convolution(x, w, params, y_dims) : g.convolution(x, w, params);
      m_output_dims = g.dims(y);
      if (fused) {
        if (spec.bias) y = g.add(y, b);
        switch (spec.activation) {
        case Activation::Relu: y = g.relu(y); break;
        case Activation::LeakyRelu: y = g.leaky_relu(y, spec.activation_parameter); break;
        case Activation::Sigmoid: y = g.sigmoid(y); break;
        case Activation::Tanh: y = g.tanh(y); break;
        case Activation::Softplus: y = g.softplus(y); break;
        default: break;
        }
        if (spec.output_scale != 1.f) y = g.scale(y, spec.output_scale);
      }
      g.output(y);
      return g;
    };
    m_fused = epilogue;
    if (epilogue) m_plan = graph(true).try_build(handle, options);
    if (!m_plan.valid()) {
      m_fused = false;
      m_plan = graph(false).build(handle, options);
    }
  }

  void ConvolutionLayer::forward(
    cudnnHandle_t handle,
    const void* x,
    const void* w,
    const void* b,
    void* y,
    void* workspace) const
  {
    const bool epilogue = m_spec.bias || m_spec.activation != Activation::None || m_spec.output_scale != 1.f;
    if (m_fused) {
      if (m_spec.bias)
        m_plan.execute(handle, {x, w, b, y}, workspace);
      else
        m_plan.execute(handle, {x, w, y}, workspace);
      return;
    }
    m_plan.execute(handle, {x, w, y}, workspace);
    if (!epilogue) return;
    cudaStream_t stream = nullptr;
    ALLEN_CUDNN_CHECK(cudnnGetStream(handle, &stream));
    const int64_t spatial = m_output_dims[2] * m_output_dims[3];
    const int64_t total = m_output_dims[0] * m_output_dims[1] * spatial;
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((total + block - 1) / block);
    const bool channels_last = m_spec.layout == Layout::NHWC;
    const int activation = static_cast<int>(m_spec.activation);
    const void* bias = m_spec.bias ? b : nullptr;
    switch (m_spec.type) {
    case DataType::BFloat16:
      epilogue_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(y),
        static_cast<const __nv_bfloat16*>(bias),
        total,
        m_output_dims[1],
        spatial,
        channels_last,
        activation,
        m_spec.activation_parameter,
        m_spec.output_scale);
      break;
    case DataType::Half:
      epilogue_kernel<__half><<<grid, block, 0, stream>>>(
        static_cast<__half*>(y),
        static_cast<const __half*>(bias),
        total,
        m_output_dims[1],
        spatial,
        channels_last,
        activation,
        m_spec.activation_parameter,
        m_spec.output_scale);
      break;
    default:
      epilogue_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<float*>(y),
        static_cast<const float*>(bias),
        total,
        m_output_dims[1],
        spatial,
        channels_last,
        activation,
        m_spec.activation_parameter,
        m_spec.output_scale);
    }
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      throw StrException(std::string("Allen::CuDNN::ConvolutionLayer epilogue: ") + cudaGetErrorString(status));
    }
  }

  std::string ConvolutionLayer::describe() const
  {
    return (m_fused ? "fused graph, " : "convolution graph + epilogue kernel, ") + m_plan.engine();
  }

  void PoolingLayer::create(
    cudnnHandle_t handle,
    std::vector<int64_t> input_dims,
    const PoolingParams& params,
    Layout layout,
    DataType type,
    const BuildOptions& options)
  {
    const PoolingParams p {
      params.mode,
      as_2d(params.window, 1, "window"),
      as_2d(params.stride, 1, "stride"),
      as_2d(params.padding, 0, "padding")};
    Graph g(type, DataType::Float);
    const TensorId x = g.input(std::move(input_dims), layout);
    const TensorId y = g.pooling(x, p);
    m_output_dims = g.dims(y);
    g.output(y);
    m_plan = g.build(handle, options);
  }

  void PoolingLayer::forward(cudnnHandle_t handle, const void* x, void* y, void* workspace) const
  {
    m_plan.execute(handle, {x, y}, workspace);
  }

} // namespace Allen::CuDNN
