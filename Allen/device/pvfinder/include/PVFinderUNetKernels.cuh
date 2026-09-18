#pragma once
#include "AlgorithmTypes.cuh"
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
// Fused bias + ReLU + max-pool.
//
// The unfused sequence is: conv writes its raw output, bias_relu reads and
// rewrites it in place, then max-pool reads it again and writes the halved
// tensor. The full-resolution activation therefore crosses DRAM three times
// after the convolution. Here it crosses once: each thread owns one output
// element, reads the two inputs that feed it, applies bias and ReLU to both,
// and writes only the pooled result. The raw conv output is never rewritten.
//
// Arithmetically identical to bias_relu followed by maxpool1d_2, because
// max(relu(a+b), relu(c+b)) == relu(max(a,c)+b) is not assumed -- both inputs
// are biased and rectified independently before the max, exactly as before.
// ---------------------------------------------------------------------------
__global__ void bias_relu_maxpool_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ bias,
    int N, int C, int W_in)
{
    const int W_out = W_in / 2;
    const int total = N * C * W_out;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int w_out = i % W_out;
    const int c     = (i / W_out) % C;
    const int n     = i / (C * W_out);
    const int base  = (n * C + c) * W_in + w_out * 2;
    const float b   = bias[c];
    float a0 = src[base]     + b; a0 = a0 > 0.f ? a0 : 0.f;
    float a1 = src[base + 1] + b; a1 = a1 > 0.f ? a1 : 0.f;
    dst[i] = fmaxf(a0, a1);
}

inline void launch_bias_relu_maxpool(
    const float* src, float* dst, const float* bias,
    int N, int C, int W_in, dim3 block, const Allen::Context& ctx)
{
    const int total = N * C * (W_in / 2);
    const dim3 grid((unsigned(total) + block.x - 1) / block.x);
    bias_relu_maxpool_kernel<<<grid, block, 0, ctx.stream()>>>(src, dst, bias, N, C, W_in);
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
// up1's ConvTranspose1d(k=2,s=2) + Conv1d(k=5,pad=2) merge.
//
// ConvTranspose1d (PyTorch weight layout [C_in,C_out,K1=2], stride 2, no
// padding) maps each input position i to exactly two output positions,
// 2i and 2i+1 -- so composing it with the following Conv1d(k=5,pad=2)
// gives a merged operator whose kernel is genuinely different for even
// vs. odd output positions (a "polyphase" structure, not a single fixed
// kernel): each output position depends on only 3 input positions
// (t-1,t,t+1 where t = m/2 or (m-1)/2), with different tap weights per
// parity. Derived by direct forward composition (no impulse-response
// inversion, so no flip subtlety), verified against real trained weights:
// interior exact to float noise, edge (positions 0,1,W_out-2,W_out-1) needs
// the literal two-stage formula, because composing two independently
// zero-padded stages isn't reducible to one fixed kernel at the boundary.
// Measured as a net throughput regression versus the unmerged path --
// kept for reference, not the default.
// ---------------------------------------------------------------------------

// Fold kernel: derives K_even[C,C,3], K_odd[C,C,3], merged_bias[C] from the
// trained (unmerged) ConvTranspose weight/bias (w_ct/b_ct) and the
// BN-folded Conv1d weight/bias (w_conv/b_conv, i.e. up1c_w_f/up1c_b_f --
// the same fused weights run_convbnrelu already uses).
//
// Weight layouts (match PyTorch/Allen exactly):
//   w_ct   : [C_in][C_out][K1=2]   flat = (c_in*C + c_out)*2 + k1   (ConvTranspose1d convention)
//   w_conv : [C_out2][C_mid][K2=5] flat = (c_out2*C + c_mid)*5 + j  (Conv1d convention)
//   K_even, K_odd : [C_out2][C_in][3]  flat = (c_out2*C + c_in)*3 + offset_idx (offset -1,0,+1 -> idx 0,1,2)
__global__ void fold_up1_merge_kernel(
    float* __restrict__ K_even,
    float* __restrict__ K_odd,
    float* __restrict__ merged_bias,
    const float* __restrict__ w_ct,
    const float* __restrict__ b_ct,
    const float* __restrict__ w_conv,
    const float* __restrict__ b_conv,
    int C)
{
    int c_out2 = blockIdx.x;
    if (c_out2 >= C) return;

    for (int c_in = threadIdx.x; c_in < C; c_in += blockDim.x) {
        float e_m1 = 0.f, e_0 = 0.f, e_p1 = 0.f;   // even-phase taps, offsets -1,0,+1
        float o_m1 = 0.f, o_0 = 0.f, o_p1 = 0.f;   // odd-phase taps
        for (int c_mid = 0; c_mid < C; ++c_mid) {
            // w_conv[c_out2, c_mid, j], j=0..4
            const float w2_0 = w_conv[(c_out2 * C + c_mid) * 5 + 0];
            const float w2_1 = w_conv[(c_out2 * C + c_mid) * 5 + 1];
            const float w2_2 = w_conv[(c_out2 * C + c_mid) * 5 + 2];
            const float w2_3 = w_conv[(c_out2 * C + c_mid) * 5 + 3];
            const float w2_4 = w_conv[(c_out2 * C + c_mid) * 5 + 4];
            // w_ct[c_in, c_mid, phase], phase=0,1
            const float w1_0 = w_ct[(c_in * C + c_mid) * 2 + 0];
            const float w1_1 = w_ct[(c_in * C + c_mid) * 2 + 1];

            e_m1 += w2_0 * w1_0 + w2_1 * w1_1;
            e_0  += w2_2 * w1_0 + w2_3 * w1_1;
            e_p1 += w2_4 * w1_0;

            o_m1 += w2_0 * w1_1;
            o_0  += w2_1 * w1_0 + w2_2 * w1_1;
            o_p1 += w2_3 * w1_0 + w2_4 * w1_1;
        }
        int base = (c_out2 * C + c_in) * 3;
        K_even[base + 0] = e_m1; K_even[base + 1] = e_0; K_even[base + 2] = e_p1;
        K_odd [base + 0] = o_m1; K_odd [base + 1] = o_0;  K_odd [base + 2] = o_p1;
    }

    if (threadIdx.x == 0) {
        float acc = 0.f;
        for (int c_mid = 0; c_mid < C; ++c_mid) {
            float wsum = 0.f;
            for (int j = 0; j < 5; ++j) wsum += w_conv[(c_out2 * C + c_mid) * 5 + j];
            acc += wsum * b_ct[c_mid];
        }
        merged_bias[c_out2] = acc + b_conv[c_out2];
    }
}

// Forward kernel: computes z[n, c_out2, m] directly from x (ConvTranspose's
// input, [N,C,L_in]) -- fast phase-dependent 3-tap path for interior m,
// exact literal two-stage formula (using the original unmerged weights)
// for the boundary. One thread per (event, output position m in
// [0,2*L_in)); each thread loops over all C output channels.
__global__ void up1_merge_kernel(
    const float* __restrict__ x,          // [N, C, L_in]
    float* __restrict__ z,                // [N, C, 2*L_in]
    const float* __restrict__ K_even,     // [C, C, 3]
    const float* __restrict__ K_odd,      // [C, C, 3]
    const float* __restrict__ merged_bias,// [C]
    const float* __restrict__ w_ct,       // [C, C, 2] -- boundary only
    const float* __restrict__ b_ct,       // [C]
    const float* __restrict__ w_conv,     // [C, C, 5]
    const float* __restrict__ b_conv,     // [C]
    int C, int L_in)
{
    int L_out = 2 * L_in;
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (m >= L_out) return;

    const float* x_n = x + (size_t)n * C * L_in;
    float* z_n = z + (size_t)n * C * L_out;

    if (m >= 2 && m < L_out - 2) {
        // Fast interior path: 3-tap phase-dependent kernel.
        int t = m / 2;
        const float* K = (m & 1) ? K_odd : K_even;
        for (int c_out2 = 0; c_out2 < C; ++c_out2) {
            float acc = merged_bias[c_out2];
            const float* Kc = K + (size_t)c_out2 * C * 3;
            for (int c_in = 0; c_in < C; ++c_in) {
                acc += Kc[c_in * 3 + 0] * x_n[c_in * L_in + (t - 1)]
                     + Kc[c_in * 3 + 1] * x_n[c_in * L_in + t]
                     + Kc[c_in * 3 + 2] * x_n[c_in * L_in + (t + 1)];
            }
            z_n[c_out2 * L_out + m] = fmaxf(acc, 0.f);  // ReLU folded in (run_convbnrelu applies it too)
        }
    } else {
        // Exact boundary: literal nested zero-clipped two-stage formula --
        // NOT an approximation, the same computation the unmerged path
        // does. Only one level of clipping needed (on p, the ConvTranspose
        // output index) since ConvTranspose1d itself has no boundary loss
        // -- every input position maps to exactly two valid output
        // positions, unlike Conv1d's own zero-padding.
        for (int c_out2 = 0; c_out2 < C; ++c_out2) {
            float acc = b_conv[c_out2];
            for (int j = 0; j < 5; ++j) {
                int p = m + j - 2;
                if (p < 0 || p >= L_out) continue;
                int i = p / 2;
                int phase = p & 1;
                // y1[c_mid, p] = b_ct[c_mid] + sum_cin w_ct[cin,c_mid,phase] * x[cin,i]
                // acc += sum_cmid w_conv[c_out2,c_mid,j] * y1[c_mid,p]
                for (int c_mid = 0; c_mid < C; ++c_mid) {
                    float y1_val = b_ct[c_mid];
                    for (int c_in = 0; c_in < C; ++c_in)
                        y1_val += w_ct[(c_in * C + c_mid) * 2 + phase] * x_n[c_in * L_in + i];
                    acc += w_conv[(c_out2 * C + c_mid) * 5 + j] * y1_val;
                }
            }
            z_n[c_out2 * L_out + m] = fmaxf(acc, 0.f);  // ReLU folded in
        }
    }
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

inline void launch_fold_up1_merge(
    float* K_even, float* K_odd, float* merged_bias,
    const float* w_ct, const float* b_ct, const float* w_conv, const float* b_conv,
    int C, cudaStream_t stream)
{
    fold_up1_merge_kernel<<<C, 32, 0, stream>>>(
        K_even, K_odd, merged_bias, w_ct, b_ct, w_conv, b_conv, C);
}

inline void launch_up1_merge(
    const float* x, float* z,
    const float* K_even, const float* K_odd, const float* merged_bias,
    const float* w_ct, const float* b_ct, const float* w_conv, const float* b_conv,
    int C, int L_in, int N,
    const dim3& block, const Allen::Context& ctx)
{
    int L_out = 2 * L_in;
    dim3 grid((L_out + block.x - 1) / block.x, N);
    up1_merge_kernel<<<grid, block, 0, ctx.stream()>>>(
        x, z, K_even, K_odd, merged_bias, w_ct, b_ct, w_conv, b_conv, C, L_in);
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

// ---------------------------------------------------------------------------
// Fused Conv1d(k=5,pad=2,same-width) + bias + ReLU for one CBR layer,
// replacing cuDNN + a separate bias_relu_kernel
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
// (unet_batch_events * N_INTERVALS); one block per slice.
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

// ---------------------------------------------------------------------------
// BF16 kernels — used when m_use_bf16=true. Mirrors the FP16 kernels above
// exactly (same math, same launch shapes); the only difference is the
// storage type and its conversion intrinsics (__bfloat162float/
// __float2bfloat16 vs. __half2float/__float2half). Motivation: FP16
// produces real NaN on real data (input values up to ~109,000 exceed
// FP16's ~65504 max representable magnitude at the very first f32->half
// cast); BF16 shares FP32's exponent range, so that specific overflow
// cannot recur here. Written as separate kernels rather than templating
// the FP16 ones above, to keep zero risk to the existing, already-validated
// FP16 path.
// ---------------------------------------------------------------------------

__global__ void f32_to_bf16_kernel(__nv_bfloat16* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(src[i]);
}

__global__ void bf16_to_f32_kernel(float* __restrict__ dst, const __nv_bfloat16* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

__global__ void bias_relu_bf16_kernel(
    __nv_bfloat16* __restrict__ tensor,
    const __nv_bfloat16* __restrict__ bias,
    int C, int W, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (i / W) % C;
    float v = __bfloat162float(tensor[i]) + __bfloat162float(bias[c]);
    tensor[i] = __float2bfloat16(v > 0.f ? v : 0.f);
}

__global__ void maxpool1d_2_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
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
    // Compare in float space rather than via a bf16 comparison intrinsic --
    // avoids depending on __hgt's bf16 overload being available in every
    // CUDA version this builds against; bf16's reduced mantissa makes this
    // comparison exact either way (no rounding happens by going through
    // float here, since both operands are already bf16-precision values).
    float a = __bfloat162float(src[base]), b = __bfloat162float(src[base + 1]);
    dst[i] = a > b ? src[base] : src[base + 1];
}

inline void launch_f32_to_bf16(
    __nv_bfloat16* dst, const float* src, int n,
    const dim3& block, const Allen::Context& ctx)
{
    dim3 grid((n + block.x - 1) / block.x);
    f32_to_bf16_kernel<<<grid, block, 0, ctx.stream()>>>(dst, src, n);
}

inline void launch_bf16_to_f32(
    float* dst, const __nv_bfloat16* src, int n,
    const dim3& block, const Allen::Context& ctx)
{
    dim3 grid((n + block.x - 1) / block.x);
    bf16_to_f32_kernel<<<grid, block, 0, ctx.stream()>>>(dst, src, n);
}

inline void launch_bias_relu_bf16(
    __nv_bfloat16* tensor, const __nv_bfloat16* bias,
    int C, int W, int N,
    const dim3& block, const Allen::Context& ctx)
{
    int total = N * C * W;
    dim3 grid((total + block.x - 1) / block.x);
    bias_relu_bf16_kernel<<<grid, block, 0, ctx.stream()>>>(tensor, bias, C, W, total);
}

inline void launch_maxpool_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int N, int C, int W_in,
    const dim3& block, const Allen::Context& ctx)
{
    int W_out = W_in / 2;
    int total = N * C * W_out;
    dim3 grid((total + block.x - 1) / block.x);
    maxpool1d_2_bf16_kernel<<<grid, block, 0, ctx.stream()>>>(src, dst, N, C, W_in);
}

} // namespace pvfinder_unet
