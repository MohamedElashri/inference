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
#include "AlgorithmTypes.cuh"
#include <cmath>

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
// Softplus * scale: y = log(1 + exp(x)) * scale, in-place
// Uses numerically stable form: log(1+exp(x)) = x + log(1+exp(-x)) for x>0
// ---------------------------------------------------------------------------
__global__ void softplus_scale_kernel(float* __restrict__ x, float scale, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = x[i];
    // Branchless stable softplus. Same floating-point operations as
    // v > 0 ? v + log(1 + exp(-v)) : log(1 + exp(v)) for every input
    // (for v <= 0 it only adds +0), so the output is bit-identical.
    x[i] = (fmaxf(v, 0.f) + logf(1.f + expf(-fabsf(v)))) * scale;
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
// Compact KDE rows back to [slot = event * 40 + interval][100]: a slot the
// UNet ran on copies its row, a skipped slot (slot_row < 0) gets the UNet's
// zero-input response `empty`. One thread per float4 (100 bins = 25 float4).
// ---------------------------------------------------------------------------
__global__ void expand_kde_rows_kernel(
    const float* __restrict__ rows,
    const int* __restrict__ slot_row,
    const float* __restrict__ empty,
    float* __restrict__ kde,
    int n_slots)
{
    constexpr int Q = 100 / 4;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_slots * Q) return;
    const int slot = i / Q;
    const int row = slot_row[slot];
    const float* src = row >= 0 ? rows + (size_t)row * 100 : empty;
    reinterpret_cast<float4*>(kde + (size_t)slot * 100)[i % Q] = reinterpret_cast<const float4*>(src)[i % Q];
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

inline void launch_softplus_scale(
    float* x, float scale, int total,
    const dim3& block, const Allen::Context& ctx)
{
    dim3 grid((total + block.x - 1) / block.x);
    softplus_scale_kernel<<<grid, block, 0, ctx.stream()>>>(x, scale, total);
}

} // namespace pvfinder_unet
