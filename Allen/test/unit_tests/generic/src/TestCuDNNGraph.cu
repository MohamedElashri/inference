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
// Allen's cuDNN library (device/cudnn_backend) against double-precision
// references on the host. Skipped without a CUDA device.
#if __has_include(<catch2/catch.hpp>)
#include <catch2/catch.hpp>
#else
#include <catch2/catch_test_macros.hpp>
#endif

#ifdef ALLEN_WITH_CUDNN
#include "AllenCuDNN.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace Allen::CuDNN;

namespace {
  bool have_device()
  {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }

  std::vector<float> random_values(size_t n, unsigned seed)
  {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v)
      x = d(g);
    return v;
  }

  struct DeviceBuffer {
    void* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { REQUIRE(cudaMalloc(&p, std::max<size_t>(bytes, 1)) == cudaSuccess); }
    template<typename T>
    explicit DeviceBuffer(const std::vector<T>& h) : DeviceBuffer(h.size() * sizeof(T))
    {
      REQUIRE(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    ~DeviceBuffer() { cudaFree(p); }
    template<typename T>
    std::vector<T> get(size_t n) const
    {
      std::vector<T> h(n);
      REQUIRE(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess);
      return h;
    }
  };

  double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
  {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i)
      m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
  }

  // [N][C][H][W] <-> [N][H][W][C]
  std::vector<float> to_nhwc(const std::vector<float>& x, int N, int C, int H, int W)
  {
    std::vector<float> y(x.size());
    for (int n = 0; n < N; ++n)
      for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
            y[((n * H + h) * W + w) * C + c] = x[((n * C + c) * H + h) * W + w];
    return y;
  }

  // Cross-correlation, NCHW, w [K][C][R][S], symmetric padding, stride 1.
  std::vector<float> reference_convolution(
    const std::vector<float>& x,
    const std::vector<float>& w,
    int N,
    int C,
    int H,
    int W,
    int K,
    int R,
    int S,
    int ph,
    int pw)
  {
    const int Ho = H + 2 * ph - R + 1, Wo = W + 2 * pw - S + 1;
    std::vector<float> y((size_t) N * K * Ho * Wo);
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k)
        for (int oh = 0; oh < Ho; ++oh)
          for (int ow = 0; ow < Wo; ++ow) {
            double s = 0;
            for (int c = 0; c < C; ++c)
              for (int r = 0; r < R; ++r)
                for (int q = 0; q < S; ++q) {
                  const int ih = oh + r - ph, iw = ow + q - pw;
                  if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    s += (double) x[((n * C + c) * H + ih) * W + iw] * w[((k * C + c) * R + r) * S + q];
                }
            y[((n * K + k) * Ho + oh) * Wo + ow] = (float) s;
          }
    return y;
  }

  const ConvolutionParams pad_1_1 {{1, 1}, {1, 1}, {1, 1}};
} // namespace

TEST_CASE("cudnn.graph.convolution_matches_reference", "[AllenCuDNN]")
{
  if (!have_device()) return;
  const int N = 2, C = 3, H = 5, W = 7, K = 4, R = 3, S = 3;
  const auto x = random_values(N * C * H * W, 1), w = random_values(K * C * R * S, 2);
  const auto ref = reference_convolution(x, w, N, C, H, W, K, R, S, 1, 1);
  cudnnHandle_t h = handle(nullptr);
  for (const Layout layout : {Layout::NCHW, Layout::NHWC}) {
    const bool nhwc = layout == Layout::NHWC;
    Graph g;
    const TensorId tx = g.input({N, C, H, W}, layout), tw = g.input({K, C, R, S}, layout);
    g.output(g.convolution(tx, tw, pad_1_1));
    const Plan plan = g.build(h);
    DeviceBuffer dx(nhwc ? to_nhwc(x, N, C, H, W) : x), dw(nhwc ? to_nhwc(w, K, C, R, S) : w);
    DeviceBuffer dy(ref.size() * 4), ws(plan.workspace_size());
    plan.execute(h, {dx.p, dw.p, dy.p}, ws.p);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto want = nhwc ? to_nhwc(ref, N, K, H, W) : ref;
    CHECK(max_abs_diff(dy.get<float>(ref.size()), want) < 1e-5);
  }
}

TEST_CASE("cudnn.graph.transposed_convolution_matches_reference", "[AllenCuDNN]")
{
  if (!have_device()) return;
  // ConvTranspose1d(k 2, stride 2): y[n][o][2 i + t] = sum_c x[n][c][i] w[c][o][t]
  const int N = 3, C = 4, O = 5, W = 6;
  const auto x = random_values(N * C * W, 3), w = random_values(C * O * 2, 4);
  std::vector<float> ref((size_t) N * O * 2 * W, 0.f);
  for (int n = 0; n < N; ++n)
    for (int o = 0; o < O; ++o)
      for (int i = 0; i < W; ++i)
        for (int t = 0; t < 2; ++t) {
          double s = 0;
          for (int c = 0; c < C; ++c)
            s += (double) x[(n * C + c) * W + i] * w[(c * O + o) * 2 + t];
          ref[(n * O + o) * 2 * W + 2 * i + t] = (float) s;
        }
  cudnnHandle_t h = handle(nullptr);
  Graph g;
  const TensorId tx = g.input({N, C, 1, W}, Layout::NCHW), tw = g.input({C, O, 1, 2}, Layout::NCHW);
  g.output(g.transposed_convolution(tx, tw, {{0, 0}, {1, 2}, {1, 1}}, {N, O, 1, 2 * W}));
  const Plan plan = g.build(h);
  DeviceBuffer dx(x), dw(w), dy(ref.size() * 4), ws(plan.workspace_size());
  plan.execute(h, {dx.p, dw.p, dy.p}, ws.p);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(max_abs_diff(dy.get<float>(ref.size()), ref) < 1e-5);
}

TEST_CASE("cudnn.graph.max_pooling_matches_reference", "[AllenCuDNN]")
{
  if (!have_device()) return;
  const int N = 2, C = 3, W = 10;
  const auto x = random_values(N * C * W, 5);
  std::vector<float> ref((size_t) N * C * W / 2);
  for (size_t i = 0; i < ref.size(); ++i)
    ref[i] = std::max(x[2 * i], x[2 * i + 1]);
  cudnnHandle_t h = handle(nullptr);
  PoolingLayer pool;
  pool.create(h, {N, C, 1, W}, {PoolingMode::Max, {2}, {2}, {0}});
  REQUIRE(pool.output_dims() == std::vector<int64_t> {N, C, 1, W / 2});
  DeviceBuffer dx(x), dy(ref.size() * 4), ws(pool.workspace_size());
  pool.forward(h, dx.p, dy.p, ws.p);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(max_abs_diff(dy.get<float>(ref.size()), ref) == 0.0);
}

TEST_CASE("cudnn.graph.matmul_matches_reference", "[AllenCuDNN]")
{
  if (!have_device()) return;
  const int M = 8, K = 16, N = 4;
  const auto a = random_values(M * K, 6), b = random_values(K * N, 7);
  std::vector<float> ref((size_t) M * N);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0;
      for (int k = 0; k < K; ++k)
        s += (double) a[i * K + k] * b[k * N + j];
      ref[i * N + j] = (float) s;
    }
  cudnnHandle_t h = handle(nullptr);
  Graph g;
  const TensorId ta = g.input({1, M, K}), tb = g.input({1, K, N});
  g.output(g.matmul(ta, tb));
  const Plan plan = g.try_build(h);
  if (!plan.valid()) return; // no exact float32 matmul engine on this device
  DeviceBuffer da(a), db(b), dc(ref.size() * 4), ws(plan.workspace_size());
  plan.execute(h, {da.p, db.p, dc.p}, ws.p);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(max_abs_diff(dc.get<float>(ref.size()), ref) < 1e-5);
}

TEST_CASE("cudnn.layer.convolution_bias_activation_scale", "[AllenCuDNN]")
{
  if (!have_device()) return;
  const int N = 2, C = 3, W = 20, K = 4, R = 5;
  const auto x = random_values(N * C * W, 8), w = random_values(K * C * R, 9), b = random_values(K, 10);
  const auto conv = reference_convolution(x, w, N, C, 1, W, K, 1, R, 0, 2);
  cudnnHandle_t h = handle(nullptr);
  for (const Activation act : {Activation::Relu, Activation::Softplus, Activation::LeakyRelu}) {
    const float scale = act == Activation::Softplus ? 0.001f : 1.f;
    std::vector<float> ref(conv.size());
    for (size_t i = 0; i < ref.size(); ++i) {
      float v = conv[i] + b[(i / W) % K];
      if (act == Activation::Relu) v = std::max(v, 0.f);
      if (act == Activation::LeakyRelu) v = v > 0.f ? v : 0.01f * v;
      if (act == Activation::Softplus) v = std::max(v, 0.f) + std::log1p(std::exp(-std::fabs(v)));
      ref[i] = v * scale;
    }
    ConvolutionLayer layer;
    layer.create(
      h,
      {.batch = N,
       .in_channels = C,
       .out_channels = K,
       .input_size = {W},
       .kernel_size = {R},
       .padding = {2},
       .bias = true,
       .activation = act,
       .activation_parameter = 0.01f,
       .output_scale = scale});
    REQUIRE(layer.output_dims() == std::vector<int64_t> {N, K, 1, W});
    DeviceBuffer dx(x), dw(w), db(b), dy(ref.size() * 4), ws(layer.workspace_size());
    layer.forward(h, dx.p, dw.p, db.p, dy.p, ws.p);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    CHECK(max_abs_diff(dy.get<float>(ref.size()), ref) < 1e-5);
  }
}

TEST_CASE("cudnn.layer.bfloat16_channels_last_is_fused", "[AllenCuDNN]")
{
  if (!have_device()) return;
  int device = 0, major = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (major < 8) return; // bfloat16 tensor cores
  const int N = 64, C = 16, W = 100, K = 16, R = 5;
  auto x = random_values(N * C * W, 11), w = random_values(K * C * R, 12), b = random_values(K, 13);
  const auto round = [](std::vector<float>& v) {
    for (auto& e : v)
      e = __bfloat162float(__float2bfloat16(e));
  };
  round(x);
  round(w);
  round(b);
  const auto conv = reference_convolution(x, w, N, C, 1, W, K, 1, R, 0, 2);
  std::vector<float> ref(conv.size());
  for (size_t i = 0; i < ref.size(); ++i)
    ref[i] = std::max(conv[i] + b[(i / W) % K], 0.f);
  const auto to_bf16 = [](const std::vector<float>& v) {
    std::vector<__nv_bfloat16> r(v.size());
    for (size_t i = 0; i < v.size(); ++i)
      r[i] = __float2bfloat16(v[i]);
    return r;
  };
  cudnnHandle_t h = handle(nullptr);
  ConvolutionLayer layer;
  layer.create(
    h,
    {.batch = N,
     .in_channels = C,
     .out_channels = K,
     .input_size = {W},
     .kernel_size = {R},
     .padding = {2},
     .layout = Layout::NHWC,
     .type = DataType::BFloat16,
     .bias = true,
     .activation = Activation::Relu});
  CHECK(layer.fused());
  DeviceBuffer dx(to_bf16(to_nhwc(x, N, C, 1, W))), dw(to_bf16(to_nhwc(w, K, C, 1, R))), db(to_bf16(b));
  DeviceBuffer dy(ref.size() * 2), ws(layer.workspace_size());
  layer.forward(h, dx.p, dw.p, db.p, dy.p, ws.p);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  const auto got = dy.get<__nv_bfloat16>(ref.size());
  const auto want = to_nhwc(ref, N, K, 1, W);
  double worst = 0; // relative to bfloat16's resolution
  for (size_t i = 0; i < got.size(); ++i)
    worst = std::max(worst, std::fabs(__bfloat162float(got[i]) - want[i]) / (std::fabs(want[i]) + 1.0));
  CHECK(worst < 1e-2);
}

TEST_CASE("cudnn.graph.plan_cache_and_errors", "[AllenCuDNN]")
{
  if (!have_device()) return;
  cudnnHandle_t h = handle(nullptr);
  const auto make = [](int64_t pad) {
    Graph g;
    const TensorId x = g.input({1, 2, 1, 9}, Layout::NCHW), w = g.input({3, 2, 1, 3}, Layout::NCHW);
    g.output(g.convolution(x, w, {{0, pad}, {1, 1}, {1, 1}}));
    return g;
  };
  CHECK(make(1).signature() != make(0).signature());
  const Plan first = make(1).build(h);
  const size_t cached = plan_cache_size();
  const Plan again = make(1).build(h); // same graph: from the cache
  CHECK(plan_cache_size() == cached);
  CHECK(again.engine() == first.engine());
  DeviceBuffer x(64), w(64);
  CHECK_THROWS_AS(first.execute(h, {x.p, w.p}, nullptr), StrException); // three bound tensors, two pointers
}

#endif // ALLEN_WITH_CUDNN
