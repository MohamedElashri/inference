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

// cuDNN graph API for Allen algorithms.
//
// A Graph describes a computation as operations on tensors: convolutions
// (forward and transposed), pointwise operations (bias, activations,
// scaling), pooling and matrix products. Tensors are inputs and outputs
// (bound to device memory when the plan runs) or virtual (intermediates that
// cuDNN may keep on chip). build() asks cuDNN's heuristics for engines that
// can run the whole graph as one fused operation, keeps the ones the options
// allow (numerics, determinism, workspace), and returns a Plan. Plans are
// cached process-wide by the graph's signature: building one can compile a
// kernel at run time (about a second), so build in init().
//
//   Allen::CuDNN::Graph g;
//   auto x = g.input({N, C, 1, W}, Layout::NCHW);
//   auto w = g.input({K, C, 1, R}, Layout::NCHW);
//   auto b = g.input({1, K, 1, 1}, Layout::NCHW);
//   auto y = g.relu(g.add(g.convolution(x, w, {.padding = {0, R / 2}}), b));
//   g.output(y);
//   Plan plan = g.build(handle);                         // in init()
//   plan.execute(handle, {x_ptr, w_ptr, b_ptr, y_ptr}, workspace);   // in operator()
//
// execute() takes the device pointers of the graph's inputs and outputs in
// the order they were declared (Graph::bound_tensors()). The workspace must
// hold plan.workspace_size() bytes; request it as an algorithm argument.

#include "CuDNNCheck.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace Allen::CuDNN {

  enum class DataType { Float, Half, BFloat16 };
  // Memory order of a 4D tensor [N][C][H][W]: channels first or channels last.
  enum class Layout { NCHW, NHWC };

  // One tensor of a graph.
  struct TensorId {
    int64_t uid = -1;
  };

  struct ConvolutionParams {
    std::vector<int64_t> padding {0, 0}; // per spatial dimension (H, W), both sides
    std::vector<int64_t> stride {1, 1};
    std::vector<int64_t> dilation {1, 1};
  };

  enum class PoolingMode { Max, Average };
  struct PoolingParams {
    PoolingMode mode = PoolingMode::Max;
    std::vector<int64_t> window {1, 2};
    std::vector<int64_t> stride {1, 2};
    std::vector<int64_t> padding {0, 0};
  };

  struct BuildOptions {
    // Heuristic's choices tried in their order; with more than one that
    // builds, they are timed (on scratch memory) and the fastest is kept.
    int max_candidates = 1;
    // Allow engines that lower the precision of the inputs (TF32 tensor cores
    // for float32) or of reductions. Off: float32 graphs are exact float32.
    bool allow_reduced_precision = false;
    // Allow non-deterministic engines (atomics). Off: identical results run to run.
    bool allow_nondeterministic = false;
    // Allow engines compiled at run time when the plan is built.
    bool allow_runtime_compilation = true;
    // Largest workspace an engine may need, in bytes.
    size_t max_workspace = size_t(256) << 20;
  };

  class Plan {
  public:
    Plan() = default;
    // Runs the plan on the handle's stream. pointers: device memory of the
    // graph's bound tensors, in declaration order. Any handle of the device
    // works: each gets its own execution plan of the same engine on first use
    // (build with the init() handle, execute with the stream's). A handle is
    // used by one thread at a time, as cuDNN requires anyway.
    void execute(cudnnHandle_t handle, std::initializer_list<const void*> pointers, void* workspace) const;
    void execute(cudnnHandle_t handle, const std::vector<const void*>& pointers, void* workspace) const;
    size_t workspace_size() const;
    // The engine chosen, for logs: cuDNN's name and knobs.
    const std::string& engine() const;
    bool valid() const { return static_cast<bool>(m_impl); }

    struct Impl;

  private:
    friend class Graph;
    std::shared_ptr<const Impl> m_impl;
  };

  class Graph {
  public:
    // io: storage type of the tensors; compute: type of the arithmetic.
    explicit Graph(DataType io = DataType::Float, DataType compute = DataType::Float);
    ~Graph();
    Graph(Graph&&) noexcept;
    Graph& operator=(Graph&&) noexcept;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    // A tensor bound at execution. dims: [N][C][H][W] (4D), or any rank with
    // explicit strides (row major when empty).
    TensorId input(std::vector<int64_t> dims, Layout layout);
    TensorId input(std::vector<int64_t> dims, std::vector<int64_t> strides = {});
    // Makes t a bound tensor (the graph's result), stored with the io type.
    void output(TensorId t);

    // Operations; each returns its (virtual) result.
    TensorId convolution(TensorId x, TensorId w, const ConvolutionParams& params);
    // Transposed convolution (the gradient of a convolution with respect to its
    // input); y_dims: the result's dimensions, which the parameters alone
    // do not determine.
    TensorId
    transposed_convolution(TensorId x, TensorId w, const ConvolutionParams& params, std::vector<int64_t> y_dims);
    TensorId add(TensorId a, TensorId b); // b broadcasts over dimensions of size 1
    TensorId mul(TensorId a, TensorId b);
    TensorId scale(TensorId a, float factor);
    TensorId relu(TensorId a);
    TensorId leaky_relu(TensorId a, float slope);
    TensorId sigmoid(TensorId a);
    TensorId tanh(TensorId a);
    TensorId softplus(TensorId a);
    TensorId pooling(TensorId x, const PoolingParams& params);
    TensorId matmul(TensorId a, TensorId b); // [B][M][K] x [B][K][N]

    // The bound tensors (inputs and outputs), in declaration order.
    std::vector<TensorId> bound_tensors() const;
    const std::vector<int64_t>& dims(TensorId t) const;
    // A canonical description: the cache key of the plans.
    std::string signature() const;

    // Builds (or takes from the cache) a plan for the whole graph. Throws
    // StrException naming the graph when no engine allowed by the options can
    // run it; try_build returns an invalid Plan instead.
    Plan build(cudnnHandle_t handle, const BuildOptions& options = {}) const;
    Plan try_build(cudnnHandle_t handle, const BuildOptions& options = {}) const;

    struct Impl;

  private:
    std::unique_ptr<Impl> m_impl;
  };

  // Plans built so far by this process (all graphs), for logs.
  size_t plan_cache_size();

} // namespace Allen::CuDNN
