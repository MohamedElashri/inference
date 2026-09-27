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
#pragma once

// CNN layers on the cuDNN graph API, for any Allen algorithm with a CNN.
//
// A ConvolutionLayer is a 1D or 2D convolution (or transposed convolution)
// with an optional per-channel bias, activation and output scale, for a fixed
// batch shape. create() (in init()) first asks cuDNN for one fused graph
// doing all of it; when no engine allowed by the options runs that graph (in
// cuDNN 9.6 exact float32 fusions have none, bfloat16 channels-last ones
// have runtime-compiled tensor-core engines), it builds the convolution alone
// and applies bias, activation and scale with one element-wise kernel, in
// place. forward() (in operator()) runs it on the handle's stream.
//
//   ConvolutionSpec spec {.batch = N, .in_channels = C, .out_channels = K,
//                         .input_size = {W}, .kernel_size = {5}, .padding = {2},
//                         .bias = true, .activation = Activation::Relu};
//   layer.create(handle, spec);                                   // init()
//   layer.forward(handle, x, w, b, y, workspace);                 // operator()
//
// Tensors: x [N][C][H][W] (1D: H = 1) in the spec's layout; w [K][C][R][S]
// (PyTorch's Conv layout) for a convolution, [C][K][R][S] (PyTorch's
// ConvTranspose layout) for a transposed one, in the same layout; b [K].

#include "CuDNNGraph.h"

#include <string>
#include <vector>

namespace Allen::CuDNN {

  enum class Activation { None, Relu, LeakyRelu, Sigmoid, Tanh, Softplus };

  struct ConvolutionSpec {
    int64_t batch = 1;
    int64_t in_channels = 1;
    int64_t out_channels = 1;
    // Spatial sizes, 1 (W) or 2 (H, W) values; the same count in the fields below.
    std::vector<int64_t> input_size {1};
    std::vector<int64_t> kernel_size {1};
    std::vector<int64_t> padding {0};
    std::vector<int64_t> stride {1};
    std::vector<int64_t> dilation {1};
    // Transposed convolution; output_size then gives the result's spatial
    // size (default: the usual (in - 1) * stride - 2 * padding + kernel).
    bool transposed = false;
    std::vector<int64_t> output_size {};
    Layout layout = Layout::NCHW;
    DataType type = DataType::Float;
    bool bias = false;
    Activation activation = Activation::None;
    float activation_parameter = 0.f; // LeakyRelu: the slope for negative inputs
    float output_scale = 1.f;         // applied last
  };

  class ConvolutionLayer {
  public:
    void create(cudnnHandle_t handle, const ConvolutionSpec& spec, const BuildOptions& options = {});
    // b is ignored without bias; workspace: workspace_size() bytes.
    void forward(cudnnHandle_t handle, const void* x, const void* w, const void* b, void* y, void* workspace) const;
    size_t workspace_size() const { return m_plan.workspace_size(); }
    // [N][K][H][W] of the result.
    const std::vector<int64_t>& output_dims() const { return m_output_dims; }
    bool fused() const { return m_fused; }
    // For logs: fused or not, and the engine.
    std::string describe() const;

  private:
    ConvolutionSpec m_spec;
    Plan m_plan;
    bool m_fused = false;
    std::vector<int64_t> m_output_dims;
  };

  class PoolingLayer {
  public:
    // x [N][C][H][W] (1D: H = 1); window, stride, padding: 1 or 2 values.
    void create(
      cudnnHandle_t handle,
      std::vector<int64_t> input_dims,
      const PoolingParams& params,
      Layout layout = Layout::NCHW,
      DataType type = DataType::Float,
      const BuildOptions& options = {});
    void forward(cudnnHandle_t handle, const void* x, void* y, void* workspace) const;
    size_t workspace_size() const { return m_plan.workspace_size(); }
    const std::vector<int64_t>& output_dims() const { return m_output_dims; }
    std::string describe() const { return m_plan.engine(); }

  private:
    Plan m_plan;
    std::vector<int64_t> m_output_dims;
  };

} // namespace Allen::CuDNN
