#pragma once

// ---------------------------------------------------------------------------
// Storage-type-generic kernels for the UNet's reduced-precision paths.
//
// A reduced-precision path keeps every activation in its storage type T
// (only BF16 uses these today) and does all arithmetic in FP32, so the
// pipeline needs no separate F32 <-> T conversion passes:
//   - conv_transpose_k2s2_kernel<T> replaces cuDNN's FP32 ConvTranspose and
//     the conversions around it;
//   - output_stage_kernel<T> replaces the FP32 output stage (conversion,
//     out_intermediate, its bias, outc, its bias, softplus, copy) with one
//     kernel that reads T and writes the FP32 KDE, applying the two
//     convolutions as their exact composition.
// The FP32 path does not use any of this. A new storage type needs only a
// StorageType specialisation.
// ---------------------------------------------------------------------------

#include "AlgorithmTypes.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>

namespace pvfinder_unet {

template <typename T>
struct StorageType;

template <>
struct StorageType<float> {
    __device__ __forceinline__ static float load(float v) { return v; }
    __device__ __forceinline__ static float store(float v) { return v; }
};

template <>
struct StorageType<__nv_bfloat16> {
    __device__ __forceinline__ static float load(__nv_bfloat16 v) { return __bfloat162float(v); }
    __device__ __forceinline__ static __nv_bfloat16 store(float v) { return __float2bfloat16(v); }
};

template <>
struct StorageType<__half> {
    __device__ __forceinline__ static float load(__half v) { return __half2float(v); }
    __device__ __forceinline__ static __half store(float v) { return __float2half(v); }
};

// ---------------------------------------------------------------------------
// ConvTranspose1d(C -> C, kernel 2, stride 2, no padding), T in and out,
// FP32 accumulation. With kernel == stride every output bin depends on one
// input bin: out[n, o, 2i + p] = b[o] + sum_c w[c, o, p] * in[n, c, i].
// w is PyTorch's ConvTranspose1d layout [C][C][2], b is [C], both FP32 (the
// same device weights the FP32 path gives cuDNN), staged in shared memory.
// One thread per input position (n, i): it loads the C inputs once and
// writes all 2 * C outputs they feed.
// in_bias, when given, is the producing layer's bias: the input is then that
// layer's raw convolution output and relu(in + in_bias[c]) is applied once,
// on load, which folds the producer's bias + ReLU pass into this kernel.
// ---------------------------------------------------------------------------
template <typename T, int C>
__global__ void conv_transpose_k2s2_kernel(
    const T* __restrict__ in, T* __restrict__ out,
    const float* __restrict__ w, const float* __restrict__ b,
    const T* __restrict__ in_bias, int N, int W_in)
{
    __shared__ float s_w[C * C * 2];
    __shared__ float s_b[C];
    for (int i = threadIdx.x; i < C * C * 2; i += blockDim.x) s_w[i] = w[i];
    for (int i = threadIdx.x; i < C; i += blockDim.x) s_b[i] = b[i];
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * W_in) return;
    const int i = idx % W_in;
    const int n = idx / W_in;
    const int W_out = 2 * W_in;

    float x[C];
    const T* src = in + (size_t)n * C * W_in + i;
#pragma unroll
    for (int c = 0; c < C; ++c) {
        float v = StorageType<T>::load(src[(size_t)c * W_in]);
        if (in_bias != nullptr) v = fmaxf(v + StorageType<T>::load(in_bias[c]), 0.0f);
        x[c] = v;
    }
    T* dst = out + (size_t)n * C * W_out + 2 * i;
    for (int o = 0; o < C; ++o) {
        float a0 = s_b[o], a1 = s_b[o];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            a0 += s_w[(c * C + o) * 2] * x[c];
            a1 += s_w[(c * C + o) * 2 + 1] * x[c];
        }
        dst[(size_t)o * W_out] = StorageType<T>::store(a0);
        dst[(size_t)o * W_out + 1] = StorageType<T>::store(a1);
    }
}

template <typename T, int C>
inline void launch_conv_transpose_k2s2(
    const T* in, T* out, const float* w, const float* b, const T* in_bias,
    int N, int W_in, const Allen::Context& ctx)
{
    const int total = N * W_in;
    constexpr int threads = 128;
    conv_transpose_k2s2_kernel<T, C><<<(total + threads - 1) / threads, threads, 0, ctx.stream()>>>(
        in, out, w, b, in_bias, N, W_in);
}

// ---------------------------------------------------------------------------
// Output stage from a T activation to the FP32 KDE:
//   h      = out_intermediate(x)      Conv1d(C -> C, k = 5, pad 2) + bias
//   logit  = outc(h)                  Conv1d(C -> 1, k = 5, pad 2) + bias
//   kde    = softplus(logit) * scale
// There is no activation between the two convolutions, so they compose into
// one Conv1d(C -> 1, k = 9) acting on x: 9 * C multiply-adds per bin instead
// of 5 * C * C + 5 * C. At the two bins at each edge outc's zero padding
// applies to h, not x, so only the outc taps landing inside [0, W) count;
// each of those four bins gets its own 9-tap filter and bias with just those
// taps, which keeps the composition exact and every thread on the same code.
// make_output_stage_params() builds the five filter sets once, in double
// precision, from the two layers' FP32 weights.
//
// Parameter block (OutputStage<C>::n_params floats): for set s = 0 (interior),
// 1, 2 (bins 0, 1), 3, 4 (bins W - 2, W - 1): F[s][c][9] then bias[s].
//   F[s][c][k1 + k2] = sum over the set's valid k2 of sum_o w_outc[o][k2] * w_oint[o][c][k1]
//   bias[s]          = b_outc + sum over valid k2 of sum_o w_outc[o][k2] * b_oint[o]
// One block per sample: x is staged in shared memory once (activated once
// when x_bias is given, which folds the producing layer's bias + ReLU as in
// conv_transpose_k2s2_kernel; inside the interval only, the zero padding
// stays zero, as after a separate activation pass), then one thread per bin.
// ---------------------------------------------------------------------------
template <int C>
struct OutputStage {
    static constexpr int K = 5, TAPS = 2 * K - 1, PAD = 2, SETS = 5;
    static constexpr int filter_floats = C * TAPS;
    static constexpr int bias_offset = SETS * filter_floats;
    static constexpr int n_params = bias_offset + SETS;
};

// Host side: composes the parameter block from the layers' weights, in
// PyTorch's Conv1d layout ([out][in][5]).
template <int C>
inline std::vector<float> make_output_stage_params(
    const std::vector<float>& w_oint, const std::vector<float>& b_oint,
    const std::vector<float>& w_outc, float b_outc)
{
    using S = OutputStage<C>;
    // V[k2][c][k1] = sum_o w_outc[o][k2] * w_oint[o][c][k1], U[k2] = sum_o w_outc[o][k2] * b_oint[o]
    std::vector<double> V(S::K * C * S::K, 0.0), U(S::K, 0.0);
    for (int k2 = 0; k2 < S::K; ++k2) {
        for (int o = 0; o < C; ++o) {
            const double wo = w_outc[o * S::K + k2];
            U[k2] += wo * b_oint[o];
            for (int c = 0; c < C; ++c)
                for (int k1 = 0; k1 < S::K; ++k1)
                    V[(k2 * C + c) * S::K + k1] += wo * w_oint[(o * C + c) * S::K + k1];
        }
    }
    // Valid outc taps per set: k2 with 0 <= j + k2 - PAD < W for the set's bin j.
    const int first_k2[S::SETS] = {0, S::PAD, S::PAD - 1, 0, 0};           // interior, j = 0, 1, W-2, W-1
    const int last_k2[S::SETS] = {S::K - 1, S::K - 1, S::K - 1, S::K - 2, S::K - 3};
    std::vector<float> p(S::n_params, 0.0f);
    for (int set = 0; set < S::SETS; ++set) {
        std::vector<double> F(S::filter_floats, 0.0);
        double bias = b_outc;
        for (int k2 = first_k2[set]; k2 <= last_k2[set]; ++k2) {
            bias += U[k2];
            for (int c = 0; c < C; ++c)
                for (int k1 = 0; k1 < S::K; ++k1) F[c * S::TAPS + k1 + k2] += V[(k2 * C + c) * S::K + k1];
        }
        for (int i = 0; i < S::filter_floats; ++i) p[set * S::filter_floats + i] = (float) F[i];
        p[S::bias_offset + set] = (float) bias;
    }
    return p;
}

template <typename T, int C, int W>
__global__ void output_stage_kernel(
    const T* __restrict__ x, const T* __restrict__ x_bias, const float* __restrict__ params,
    float* __restrict__ kde, float scale)
{
    using S = OutputStage<C>;
    constexpr int HALO = 2 * S::PAD, WP = W + 2 * HALO;
    __shared__ float s_p[S::n_params];
    __shared__ float s_x[C][WP];   // x, activated once if x_bias, zero-padded by 4 on each side

    const int n = blockIdx.x;
    const T* xn = x + (size_t)n * C * W;
    for (int i = threadIdx.x; i < S::n_params; i += blockDim.x) s_p[i] = params[i];
    for (int c = 0; c < C; ++c) {
        for (int q = threadIdx.x; q < WP; q += blockDim.x) {
            const int p = q - HALO;
            float v = 0.0f;
            if (p >= 0 && p < W) {
                v = StorageType<T>::load(xn[c * W + p]);
                if (x_bias != nullptr) v = fmaxf(v + StorageType<T>::load(x_bias[c]), 0.0f);
            }
            s_x[c][q] = v;
        }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < W; j += blockDim.x) {
        const int set = j == 0 ? 1 : j == 1 ? 2 : j == W - 2 ? 3 : j == W - 1 ? 4 : 0;
        const float* f = s_p + set * S::filter_floats;
        // Four independent partial sums keep the multiply-add chains short.
        float acc[4] = {s_p[S::bias_offset + set], 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int t = 0; t < S::TAPS; ++t) acc[c & 3] += f[c * S::TAPS + t] * s_x[c][j + t];
        }
        const float v = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        // Same softplus as softplus_scale_kernel.
        kde[(size_t)n * W + j] = (fmaxf(v, 0.f) + logf(1.f + expf(-fabsf(v)))) * scale;
    }
}

template <typename T, int C, int W>
inline void launch_output_stage(const T* x, const T* x_bias, const float* params, float* kde, float scale,
                                int N, const Allen::Context& ctx)
{
    // One block per sample; W = 100 bins, so 128 threads cover it in one pass.
    output_stage_kernel<T, C, W><<<N, 128, 0, ctx.stream()>>>(x, x_bias, params, kde, scale);
}

// ---------------------------------------------------------------------------
// bias + ReLU + MaxPool1d(2), T in and out: out[n, c, i] =
// max(relu(in[n, c, 2i] + b[c]), relu(in[n, c, 2i + 1] + b[c])), in being a
// convolution's raw output. Same values as a separate bias + ReLU pass
// followed by a max-pool (ReLU and rounding to T are both monotonic, so the
// max commutes with them), in one read of the full-resolution activation.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void bias_relu_maxpool2_kernel(
    const T* __restrict__ in, T* __restrict__ out, const T* __restrict__ bias,
    int N, int C, int W_in)
{
    const int W_out = W_in / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * W_out) return;
    const int i = idx % W_out;
    const int c = (idx / W_out) % C;
    const T* src = in + (size_t)(idx / W_out) * W_in + 2 * i;
    const float b = StorageType<T>::load(bias[c]);
    const float v = fmaxf(fmaxf(StorageType<T>::load(src[0]), StorageType<T>::load(src[1])) + b, 0.0f);
    out[idx] = StorageType<T>::store(v);
}

template <typename T>
inline void launch_bias_relu_maxpool2(
    const T* in, T* out, const T* bias, int N, int C, int W_in, const Allen::Context& ctx)
{
    const int total = N * C * (W_in / 2);
    constexpr int threads = 256;
    bias_relu_maxpool2_kernel<T><<<(total + threads - 1) / threads, threads, 0, ctx.stream()>>>(
        in, out, bias, N, C, W_in);
}

// ===========================================================================
// Channels-last (NWC: [N][W][C]) variants for the BF16 path when its
// convolutions run as cuDNN graph-API fused Conv+Bias+ReLU
// (PVFinderConvGraph.cuh), which needs that layout. Same arithmetic as the
// NCW kernels above; only the indexing differs.
// ===========================================================================

// [N][C][W] (Tin) -> [N][W][C] (T), one thread per output element.
template <typename Tin, typename T>
__global__ void ncw_to_nwc_kernel(const Tin* __restrict__ in, T* __restrict__ out, int N, int C, int W)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * W) return;
    const int c = idx % C, w = (idx / C) % W, n = idx / (C * W);
    out[idx] = StorageType<T>::store(StorageType<Tin>::load(in[((size_t)n * C + c) * W + w]));
}

template <typename Tin, typename T>
inline void launch_ncw_to_nwc(const Tin* in, T* out, int N, int C, int W, const Allen::Context& ctx)
{
    const int total = N * C * W;
    ncw_to_nwc_kernel<Tin, T><<<(total + 255) / 256, 256, 0, ctx.stream()>>>(in, out, N, C, W);
}

// MaxPool1d(2) in NWC.
template <typename T>
__global__ void maxpool2_nwc_kernel(const T* __restrict__ in, T* __restrict__ out, int N, int C, int W_in)
{
    const int W_out = W_in / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * W_out * C) return;
    const int c = idx % C, i = (idx / C) % W_out, n = idx / (C * W_out);
    const T* src = in + ((size_t)n * W_in + 2 * i) * C + c;
    const float a = StorageType<T>::load(src[0]), b = StorageType<T>::load(src[C]);
    out[idx] = a > b ? src[0] : src[C];
}

template <typename T>
inline void launch_maxpool2_nwc(const T* in, T* out, int N, int C, int W_in, const Allen::Context& ctx)
{
    const int total = N * C * (W_in / 2);
    maxpool2_nwc_kernel<T><<<(total + 255) / 256, 256, 0, ctx.stream()>>>(in, out, N, C, W_in);
}

// ConvTranspose1d(C -> C, kernel 2, stride 2) in NWC; same maths as
// conv_transpose_k2s2_kernel: one thread per input position (n, i), which
// reads its C contiguous inputs and writes the 2 x C contiguous outputs,
// both as 16-byte vector accesses (a row of C values is a whole number of
// 16-byte chunks: 32 bytes for 16 BF16 channels).
template <typename T, int C>
__global__ void conv_transpose_k2s2_nwc_kernel(
    const T* __restrict__ in, T* __restrict__ out,
    const float* __restrict__ w, const float* __restrict__ b, int N, int W_in)
{
    constexpr int CHUNKS = C * (int) sizeof(T) / 16;
    static_assert(C * sizeof(T) % 16 == 0, "NWC rows must be whole 16-byte chunks");
    union Row { uint4 v[CHUNKS]; T e[C]; };

    __shared__ float s_w[C * C * 2];
    __shared__ float s_b[C];
    for (int i = threadIdx.x; i < C * C * 2; i += blockDim.x) s_w[i] = w[i];
    for (int i = threadIdx.x; i < C; i += blockDim.x) s_b[i] = b[i];
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * W_in) return;
    Row r;
    const uint4* src = reinterpret_cast<const uint4*>(in + (size_t)idx * C);
#pragma unroll
    for (int k = 0; k < CHUNKS; ++k) r.v[k] = src[k];
    float x[C];
#pragma unroll
    for (int c = 0; c < C; ++c) x[c] = StorageType<T>::load(r.e[c]);
    // (n, i) -> output rows 2i and 2i + 1 of sample n: (n * 2 W_in + 2i) * C = 2 * idx * C
    uint4* dst = reinterpret_cast<uint4*>(out + (size_t)idx * 2 * C);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
#pragma unroll
        for (int o = 0; o < C; ++o) {
            float a = s_b[o];
#pragma unroll
            for (int c = 0; c < C; ++c) a += s_w[(c * C + o) * 2 + p] * x[c];
            r.e[o] = StorageType<T>::store(a);
        }
#pragma unroll
        for (int k = 0; k < CHUNKS; ++k) dst[p * CHUNKS + k] = r.v[k];
    }
}

template <typename T, int C>
inline void launch_conv_transpose_k2s2_nwc(
    const T* in, T* out, const float* w, const float* b, int N, int W_in, const Allen::Context& ctx)
{
    const int total = N * W_in;
    conv_transpose_k2s2_nwc_kernel<T, C><<<(total + 127) / 128, 128, 0, ctx.stream()>>>(in, out, w, b, N, W_in);
}

// Output stage reading an NWC activation (already biased and activated by
// the fused convolution); otherwise identical to output_stage_kernel.
template <typename T, int C, int W>
__global__ void output_stage_nwc_kernel(
    const T* __restrict__ x, const float* __restrict__ params, float* __restrict__ kde, float scale)
{
    using S = OutputStage<C>;
    constexpr int HALO = 2 * S::PAD, WP = W + 2 * HALO;
    __shared__ float s_p[S::n_params];
    __shared__ float s_x[C][WP];

    const int n = blockIdx.x;
    const T* xn = x + (size_t)n * W * C;
    for (int i = threadIdx.x; i < S::n_params; i += blockDim.x) s_p[i] = params[i];
    for (int i = threadIdx.x; i < C * WP; i += blockDim.x) {
        const int q = i / C, c = i % C, p = q - HALO;   // consecutive threads read consecutive channels
        s_x[c][q] = (p >= 0 && p < W) ? StorageType<T>::load(xn[p * C + c]) : 0.0f;
    }
    __syncthreads();

    for (int j = threadIdx.x; j < W; j += blockDim.x) {
        const int set = j == 0 ? 1 : j == 1 ? 2 : j == W - 2 ? 3 : j == W - 1 ? 4 : 0;
        const float* f = s_p + set * S::filter_floats;
        float acc[4] = {s_p[S::bias_offset + set], 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int t = 0; t < S::TAPS; ++t) acc[c & 3] += f[c * S::TAPS + t] * s_x[c][j + t];
        }
        const float v = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        kde[(size_t)n * W + j] = (fmaxf(v, 0.f) + logf(1.f + expf(-fabsf(v)))) * scale;
    }
}

template <typename T, int C, int W>
inline void launch_output_stage_nwc(const T* x, const float* params, float* kde, float scale, int N,
                                    const Allen::Context& ctx)
{
    output_stage_nwc_kernel<T, C, W><<<N, 128, 0, ctx.stream()>>>(x, params, kde, scale);
}

} // namespace pvfinder_unet
