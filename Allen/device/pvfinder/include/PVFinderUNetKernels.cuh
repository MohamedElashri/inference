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
// BatchNorm1d inference: y = gamma*(x - mean)/sqrt(var+eps) + beta
// Per-channel parameters; operates over flat NCW tensor.
// ---------------------------------------------------------------------------
__global__ void batchnorm_inference_kernel(
    float* __restrict__ tensor,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    int C, int W, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (i / W) % C;
    float inv_std = rsqrtf(var[c] + eps);
    tensor[i] = gamma[c] * (tensor[i] - mean[c]) * inv_std + beta[c];
}

// ---------------------------------------------------------------------------
// ReLU in-place
// ---------------------------------------------------------------------------
__global__ void relu_inplace_kernel(float* __restrict__ x, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    x[i] = fmaxf(x[i], 0.f);
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
__global__ void concat_channels_a_kernel(
    const float* __restrict__ a,
    float* __restrict__ dst,
    int N, int C1, int C_out, int W)
{
    int total = N * C1 * W;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int w = i % W;
    int c = (i / W) % C1;
    int n = i / (C1 * W);
    dst[(n * C_out + c) * W + w] = a[i];
}

__global__ void concat_channels_b_kernel(
    const float* __restrict__ b,
    float* __restrict__ dst,
    int N, int C1, int C2, int C_out, int W)
{
    int total = N * C2 * W;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int w = i % W;
    int c = (i / W) % C2;
    int n = i / (C2 * W);
    dst[(n * C_out + C1 + c) * W + w] = b[i];
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
    float vpos = fmaxf(v, 0.f);
    float vneg = fminf(v, 0.f);
    x[i] = (vpos + log1pf(expf(-fabsf(v)))) * scale;
    (void)vneg;
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
// Convenience: launch helpers called from host code
// ---------------------------------------------------------------------------
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

inline void launch_batchnorm(
    float* tensor,
    const float* gamma, const float* beta,
    const float* mean,  const float* var, float eps,
    int C, int W, int N,
    const dim3& block, const Allen::Context& ctx)
{
    int total = N * C * W;
    dim3 grid((total + block.x - 1) / block.x);
    batchnorm_inference_kernel<<<grid, block, 0, ctx.stream()>>>(
        tensor, gamma, beta, mean, var, eps, C, W, total);
}

inline void launch_relu(float* x, int total,
    const dim3& block, const Allen::Context& ctx)
{
    dim3 grid((total + block.x - 1) / block.x);
    relu_inplace_kernel<<<grid, block, 0, ctx.stream()>>>(x, total);
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
    const int C_out = C1 + C2;
    {
        int total = N * C1 * W;
        dim3 grid((total + block.x - 1) / block.x);
        concat_channels_a_kernel<<<grid, block, 0, ctx.stream()>>>(
            a, dst, N, C1, C_out, W);
    }
    {
        int total = N * C2 * W;
        dim3 grid((total + block.x - 1) / block.x);
        concat_channels_b_kernel<<<grid, block, 0, ctx.stream()>>>(
            b, dst, N, C1, C2, C_out, W);
    }
}

inline void launch_softplus_scale(
    float* x, float scale, int total,
    const dim3& block, const Allen::Context& ctx)
{
    dim3 grid((total + block.x - 1) / block.x);
    softplus_scale_kernel<<<grid, block, 0, ctx.stream()>>>(x, scale, total);
}

} // namespace pvfinder_unet
