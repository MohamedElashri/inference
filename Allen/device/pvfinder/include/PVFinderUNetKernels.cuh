#pragma once
#include "AlgorithmTypes.cuh"
#include <cmath>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Lightweight CUDA kernels for UNet layer primitives.
// All tensors use NCW layout (cuDNN NCHW with H=1).
// ---------------------------------------------------------------------------

namespace pvfinder_unet {

// ---------------------------------------------------------------------------
// Bias add: output[n,c,w] += bias[c]
// Operates flat: elem = n*C*W + c*W + w
// ---------------------------------------------------------------------------
__global__ void bias_add_kernel(
    float* __restrict__ tensor,
    const float* __restrict__ bias,
    int C, int W, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (i / W) % C;
    tensor[i] += bias[c];
}

// ---------------------------------------------------------------------------
// MaxPool1d(kernel=2, stride=2): [N, C, W] -> [N, C, W/2]
// ---------------------------------------------------------------------------
__global__ void maxpool1d_2_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int N, int C, int W_in)
{
    // dst has W_out = W_in/2
    int W_out = W_in / 2;
    int total = N * C * W_out;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int w_out = i % W_out;
    int c     = (i / W_out) % C;
    int n     = i / (C * W_out);
    int base  = (n * C + c) * W_in + w_out * 2;
    dst[i] = fmaxf(src[base], src[base + 1]);
}

// ---------------------------------------------------------------------------
// Concat along channel dim: [N, C1, W] cat [N, C2, W] -> [N, C1+C2, W]
// Fills the dst buffer: first C1 channels from a, then C2 from b.
// ---------------------------------------------------------------------------
__global__ void concat_channels_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ dst,
    int N, int C1, int C2, int W)
{
    int C_out = C1 + C2;
    int total = N * C_out * W;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int w = i % W;
    int c = (i / W) % C_out;
    int n = i / (C_out * W);
    if (c < C1) {
        dst[i] = a[(n * C1 + c) * W + w];
    } else {
        dst[i] = b[(n * C2 + (c - C1)) * W + w];
    }
}

// ---------------------------------------------------------------------------
// Softplus * scale: y = log(1 + exp(x)) * scale, in-place
// Uses numerically stable form: log(1+exp(x)) = x + log(1+exp(-x)) for x>0
// ---------------------------------------------------------------------------
__global__ void softplus_scale_kernel(float* __restrict__ x, float scale, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = x[i];
    x[i] = (v > 0.f ? v + logf(1.f + expf(-v)) : logf(1.f + expf(v))) * scale;
}

// ---------------------------------------------------------------------------
// Squeeze channel dim: copy [N, 1, W] -> [N, W] (flat, no-op on data)
// Used to write final output KDE tensor.
// ---------------------------------------------------------------------------
__global__ void squeeze_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Element-wise in-place accumulate: a[i] += b[i]
// Used to sum the two out_intermediate half-conv outputs.
// ---------------------------------------------------------------------------
__global__ void accumulate_add_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    a[i] += b[i];
}

// ---------------------------------------------------------------------------
// Bias add + ReLU: y = relu(tensor + bias[c])
// Used after BN-folded convolutions — BN absorbed into weights at init,
// so only bias + ReLU remain at runtime.
// ---------------------------------------------------------------------------
__global__ void bias_relu_kernel(
    float* __restrict__ tensor,
    const float* __restrict__ bias,
    int C, int W, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (i / W) % C;
    float v = tensor[i] + bias[c];
    tensor[i] = v > 0.f ? v : 0.f;
}

// ---------------------------------------------------------------------------
// BN weight folding: fuse BN into conv weights + bias at init time.
// After folding, inference is y = relu(conv(x, w_fused) + b_fused) — no
// separate BN kernel needed at runtime.
//
// scale[k] = gamma[k] / sqrt(var[k] + eps)
// w_fused[k,...] = scale[k] * w[k,...]
// b_fused[k]     = scale[k] * (b[k] - mean[k]) + beta[k]
//
// Launch: <<<K, 256>>> where K = number of output channels.
// ---------------------------------------------------------------------------
__global__ void fold_bn_into_conv_kernel(
    float* __restrict__ w_fused,
    float* __restrict__ b_fused,
    const float* __restrict__ w,
    const float* __restrict__ b,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps, int K, int CxHxW)
{
    int k = blockIdx.x;
    if (k >= K) return;
    float scale = gamma[k] * rsqrtf(var[k] + eps);
    if (threadIdx.x == 0) b_fused[k] = scale * (b[k] - mean[k]) + beta[k];
    for (int i = threadIdx.x; i < CxHxW; i += blockDim.x)
        w_fused[k * CxHxW + i] = scale * w[k * CxHxW + i];
}

// ---------------------------------------------------------------------------
// Convenience: launch helpers called from host code
// ---------------------------------------------------------------------------
inline void launch_bias_relu(
    float* tensor, const float* bias,
    int C, int W, int N,
    const dim3& block, const Allen::Context& ctx)
{
    int total = N * C * W;
    dim3 grid((total + block.x - 1) / block.x);
    bias_relu_kernel<<<grid, block, 0, ctx.stream()>>>(
        tensor, bias, C, W, total);
}

inline void launch_bias_add(
    float* tensor, const float* bias,
    int C, int W, int N,
    const dim3& block, const Allen::Context& ctx)
{
    int total = N * C * W;
    dim3 grid((total + block.x - 1) / block.x);
    bias_add_kernel<<<grid, block, 0, ctx.stream()>>>(
        tensor, bias, C, W, total);
}

inline void launch_maxpool(
    const float* src, float* dst,
    int N, int C, int W_in,
    const dim3& block, const Allen::Context& ctx)
{
    int W_out = W_in / 2;
    int total = N * C * W_out;
    dim3 grid((total + block.x - 1) / block.x);
    maxpool1d_2_kernel<<<grid, block, 0, ctx.stream()>>>(
        src, dst, N, C, W_in);
}

inline void launch_concat(
    const float* a, const float* b, float* dst,
    int N, int C1, int C2, int W,
    const dim3& block, const Allen::Context& ctx)
{
    int total = N * (C1 + C2) * W;
    dim3 grid((total + block.x - 1) / block.x);
    concat_channels_kernel<<<grid, block, 0, ctx.stream()>>>(
        a, b, dst, N, C1, C2, W);
}

// Skip-connection ablation: a[i] += b[i] in place (same shape tensors).
// Used to replace a channel-concat skip with an element-wise add skip.
inline void launch_accumulate(
    float* a, const float* b, int total,
    const dim3& block, const Allen::Context& ctx)
{
    dim3 grid((total + block.x - 1) / block.x);
    accumulate_add_kernel<<<grid, block, 0, ctx.stream()>>>(a, b, total);
}

inline void launch_softplus_scale(
    float* x, float scale, int total,
    const dim3& block, const Allen::Context& ctx)
{
    dim3 grid((total + block.x - 1) / block.x);
    softplus_scale_kernel<<<grid, block, 0, ctx.stream()>>>(x, scale, total);
}

// ---------------------------------------------------------------------------
// Phase 3 (optimization_plan.md): fused Conv1d(k=5,pad=2,same-width) + bias
// + ReLU for one CBR layer, replacing cuDNN + a separate bias_relu_kernel
// pass for that layer. Keeps the [C,W] activation slice resident in shared
// memory across the conv and its epilogue -- the conv's raw output never
// makes a DRAM round trip. Weights/bias are read from global memory via the
// read-only cache (__ldg): the byte/FLOP census found weight traffic
// negligible already (tiny, reused, effectively cached across the whole
// grid), so only the activation tile needs to be shared-memory-resident to
// capture the targeted traffic reduction -- this also sidesteps any
// static-shared-memory-size concern for larger N_FEAT builds (e.g. 64ch),
// since only the input tile (a few KB) lives in __shared__.
//
// One block per (chunk-relative) slice n. C/W/PAD/K are compile-time
// constants (this UNet's shapes are all fixed at compile time already).
// Cross-correlation semantics (no kernel flip), matching cuDNN's
// CUDNN_CROSS_CORRELATION mode used elsewhere in this UNet.
// ---------------------------------------------------------------------------
template <int C, int W, int PAD, int K>
__global__ void fused_conv_bias_relu_same_width_kernel(
    const float* __restrict__ src,     // [N, C, W]
    float* __restrict__ dst,           // [N, C, W]
    const float* __restrict__ weight,  // [C_out=C, C_in=C, K]
    const float* __restrict__ bias)    // [C]
{
    constexpr int W_PAD = W + 2 * PAD;
    __shared__ float s_in[C * W_PAD];

    const int n = blockIdx.x;
    const float* src_n = src + (size_t) n * C * W;
    float* dst_n = dst + (size_t) n * C * W;

    // Cooperative load of this slice's input into shared memory, zero-padded
    // at the halo (out-of-range) positions.
    for (int i = threadIdx.x; i < C * W_PAD; i += blockDim.x) {
        const int c = i / W_PAD;
        const int w = i % W_PAD - PAD;
        s_in[i] = (w >= 0 && w < W) ? src_n[c * W + w] : 0.f;
    }
    __syncthreads();

    // Each thread computes one or more (k_out, w_out) output elements.
    for (int idx = threadIdx.x; idx < C * W; idx += blockDim.x) {
        const int k_out = idx / W;
        const int w_out = idx % W;
        float acc = 0.f;
        for (int c = 0; c < C; ++c) {
            const float* s_row = s_in + c * W_PAD + w_out;   // padded-index base for this (c, w_out)
            const float* w_row = weight + (size_t) (k_out * C + c) * K;
            #pragma unroll
            for (int j = 0; j < K; ++j) acc += s_row[j] * __ldg(&w_row[j]);
        }
        acc += __ldg(&bias[k_out]);
        dst_n[k_out * W + w_out] = acc > 0.f ? acc : 0.f;
    }
}

// rcbn3-shaped instantiation: Conv(N_FEAT->N_FEAT, k=5, pad=2) at W_HALF,
// same width in and out. N is the number of slices in this chunk
// (N_CHUNK_INTERVALS); one block per slice.
inline void launch_fused_rcbn3(
    const float* src, float* dst,
    const float* weight, const float* bias,
    int N, const dim3& block, const Allen::Context& ctx)
{
    fused_conv_bias_relu_same_width_kernel<N_FEAT, W_HALF, 2, 5>
        <<<N, block, 0, ctx.stream()>>>(src, dst, weight, bias);
}

// ---------------------------------------------------------------------------
// Phase M FP16 kernels — used when m_use_fp16=true for Tensor Core benchmarking.
// All weights/biases must be __half (converted from FP32 BN-folded weights at init).
// ---------------------------------------------------------------------------

__global__ void f32_to_f16_kernel(__half* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}

__global__ void f16_to_f32_kernel(float* __restrict__ dst, const __half* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}

__global__ void bias_relu_half_kernel(
    __half* __restrict__ tensor,
    const __half* __restrict__ bias,
    int C, int W, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (i / W) % C;
    float v = __half2float(tensor[i]) + __half2float(bias[c]);
    tensor[i] = __float2half(v > 0.f ? v : 0.f);
}

__global__ void maxpool1d_2_half_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int N, int C, int W_in)
{
    int W_out = W_in / 2;
    int total = N * C * W_out;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int w_out = i % W_out;
    int c     = (i / W_out) % C;
    int n     = i / (C * W_out);
    int base  = (n * C + c) * W_in + w_out * 2;
    dst[i] = __hgt(src[base], src[base + 1]) ? src[base] : src[base + 1];
}

__global__ void concat_channels_half_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ dst,
    int N, int C1, int C2, int W)
{
    int C_out = C1 + C2;
    int total = N * C_out * W;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int w = i % W;
    int c = (i / W) % C_out;
    int n = i / (C_out * W);
    if (c < C1) dst[i] = a[(n * C1 + c) * W + w];
    else        dst[i] = b[(n * C2 + (c - C1)) * W + w];
}

inline void launch_f32_to_f16(
    __half* dst, const float* src, int n,
    const dim3& block, const Allen::Context& ctx)
{
    dim3 grid((n + block.x - 1) / block.x);
    f32_to_f16_kernel<<<grid, block, 0, ctx.stream()>>>(dst, src, n);
}

inline void launch_f16_to_f32(
    float* dst, const __half* src, int n,
    const dim3& block, const Allen::Context& ctx)
{
    dim3 grid((n + block.x - 1) / block.x);
    f16_to_f32_kernel<<<grid, block, 0, ctx.stream()>>>(dst, src, n);
}

inline void launch_bias_relu_half(
    __half* tensor, const __half* bias,
    int C, int W, int N,
    const dim3& block, const Allen::Context& ctx)
{
    int total = N * C * W;
    dim3 grid((total + block.x - 1) / block.x);
    bias_relu_half_kernel<<<grid, block, 0, ctx.stream()>>>(tensor, bias, C, W, total);
}

inline void launch_maxpool_half(
    const __half* src, __half* dst,
    int N, int C, int W_in,
    const dim3& block, const Allen::Context& ctx)
{
    int W_out = W_in / 2;
    int total = N * C * W_out;
    dim3 grid((total + block.x - 1) / block.x);
    maxpool1d_2_half_kernel<<<grid, block, 0, ctx.stream()>>>(src, dst, N, C, W_in);
}

inline void launch_concat_half(
    const __half* a, const __half* b, __half* dst,
    int N, int C1, int C2, int W,
    const dim3& block, const Allen::Context& ctx)
{
    int total = N * (C1 + C2) * W;
    dim3 grid((total + block.x - 1) / block.x);
    concat_channels_half_kernel<<<grid, block, 0, ctx.stream()>>>(a, b, dst, N, C1, C2, W);
}

} // namespace pvfinder_unet
