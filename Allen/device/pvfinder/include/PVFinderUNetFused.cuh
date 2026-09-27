/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2.0, copied verbatim in the file "COPYING".                         *
*                                                                             *
* In applying this licence, CERN does not waive any privileges or immunities  *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#pragma once

// The whole UNet in one kernel, for pvfinder_unet with precision = bfloat16.
// It uses no cuDNN: the convolutions are written directly with the BF16
// tensor-core instructions (mma.sync, ldmatrix; compute capability 8.0 or
// newer), for exactly these shapes (16 feature maps, 4 input channels).
//
// One warp takes one interval (a row of the channels-last BF16 input, W_IN
// bins x N_BATCH_CHANNELS) through every layer with its activations in shared
// memory, so nothing but the input row and the 100 KDE values touches global
// memory. Each convolution is an implicit GEMM on BF16 tensor cores
// (mma.sync m16n8k16, FP32 accumulation): out[w][k] = sum over taps r and
// channels c of act[w + r - pad][c] * W[k][r][c], i.e. for 16 channels one
// 16-deep k-step per tap over the activation shifted by r - pad.
//
// Numerics, against the float32 model: the BatchNorm-folded convolution
// weights and every activation are rounded to BF16 (8 significant bits);
// bias and ReLU are applied in FP32 before an activation is rounded; sums
// accumulate in FP32. Max-pooling works on the rounded values (the same
// result, rounding being monotonic). The ConvTransposes multiply BF16
// activations by BF16 weights (exact for a checkpoint whose weights are BF16).
// The output stage (out_intermediate and outc composed into one 9-tap filter,
// PVFinderUNetOutputStage.cuh) runs the interior bins on tensor cores with
// the filter split into BF16 hi + lo parts (about 16 significant bits) and the
// four edge bins in FP32.
//
// Shared memory: the block's weights (staged once, as one image prepared on
// the host, see make_fused_unet_blob) and, per warp, two activation buffers
// of ACT_ROWS rows x 16 channels plus the input row. Activation rows are 8
// 32-bit words (pairs of channels); word bit 2 is flipped on rows whose bit 2
// is set, so a fragment's 8 rows x 4 words hit 32 different banks.

#include <cuda_bf16.h>
#include <cstring>
#include <vector>
#include "PVFinderUNetOutputStage.cuh"

namespace pvfinder_unet {
  namespace fused {

    constexpr int C = 16;                     // feature maps (N_FEAT)
    constexpr int CIN = 4;                    // input latent channels (N_BATCH_CHANNELS)
    constexpr int W1 = 100, W2 = 50, W3 = 25; // widths: input, after one and two pools
    // 4 warps per block (about 59 KB of shared memory): with the grid capped
    // (pvfinder_unet.fused_grid_fraction) the other streams' kernels share SMs
    // with it better than with 8-warp blocks of 95 KB (3.7% vs 4.0% loss at 16
    // streams for the same number of warps).
    constexpr int WARPS = 4;
    constexpr int THREADS = WARPS * 32;

    // Activation buffers: rows [-ACT_LO, ACT_ROWS - ACT_LO). The low margin is
    // never written (zero padding for pads up to 3 and the output stage's halo
    // of 4); after each layer the rows from the layer's width up to 20 past it
    // are zeroed, which covers every read past the end (m-tiles round the width
    // up to a multiple of 16, plus the pad).
    constexpr int ACT_LO = 4;
    constexpr int ACT_ROWS = ACT_LO + 112 + 8; // 124
    constexpr int ACT_WORDS = ACT_ROWS * 8;    // 32-bit words per buffer
    // Input row, dense [bin][CIN]: elements from (-12 bins) up to past 112 + 16.
    constexpr int IN_LO = 12 * CIN; // elements before bin 0
    constexpr int IN_ELEMS = 576;   // covers every A-fragment read of rcbn1

    // Convolution weights, [layer][out channel][k] with k = tap * C_in + channel,
    // zero-padded to a whole number of k-steps, rows padded by 8 (bank spread).
    struct ConvShape {
      int c_in, r, pad;
    };
    constexpr ConvShape CONV[5] = {{CIN, 25, 12}, {C, 7, 3}, {C, 5, 2}, {C, 5, 2}, {C, 5, 2}};
    constexpr int ksteps(int l) { return (CONV[l].c_in * CONV[l].r + 15) / 16; }
    constexpr int kstride(int l) { return ksteps(l) * 16 + 8; }
    constexpr int conv_offset(int l) { return l == 0 ? 0 : conv_offset(l - 1) + C * kstride(l - 1); }
    constexpr int CONV_ELEMS = conv_offset(5);
    // ConvTranspose weights, [layer][phase][out channel][in channel], rows padded to 24.
    constexpr int CT_STRIDE = 24;
    constexpr int CT_OFFSET = CONV_ELEMS;
    constexpr int CT_ELEMS = 2 * 2 * C * CT_STRIDE;
    // Output stage, interior filter as per-lane mma B fragments: [8 words][32 lanes]
    // = {hi, lo} x {taps 0-7, taps 8-15} x {channels 2q.., 2q+8..} (see output_stage).
    constexpr int OB_OFFSET = CONV_ELEMS + CT_ELEMS;
    constexpr int OB_ELEMS = 8 * 32 * 2;
    constexpr int BF16_ELEMS = CONV_ELEMS + CT_ELEMS + OB_ELEMS;
    static_assert(BF16_ELEMS % 8 == 0, "float block must stay 16-byte aligned");
    // Floats: conv biases [5][C], ConvTranspose biases [2][C], output-stage parameters.
    constexpr int F_CONV_BIAS = 0;
    constexpr int F_CT_BIAS = 5 * C;
    constexpr int F_OUT = F_CT_BIAS + 2 * C;
    constexpr int F_FLOATS = F_OUT + OutputStage<C>::n_params;
    constexpr int BLOB_BYTES = ((BF16_ELEMS * 2 + F_FLOATS * 4) + 15) / 16 * 16;
    constexpr int WARP_BYTES = 2 * ACT_WORDS * 4 + IN_ELEMS * 2;
    constexpr int SMEM_BYTES = BLOB_BYTES + WARPS * WARP_BYTES;

    __device__ __forceinline__ unsigned pack(float lo, float hi)
    {
      const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
      return *reinterpret_cast<const unsigned*>(&v);
    }

    __device__ __forceinline__ float2 unpack(unsigned u)
    {
      return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
    }

    __device__ __forceinline__ void mma(float d[4], const unsigned a[4], unsigned b0, unsigned b1)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
      __trap(); // the host refuses the fused kernel below compute capability 8.0
#endif
    }

    __device__ __forceinline__ void mma_k8(float d[4], const unsigned a[2], unsigned b)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                   "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a[0]), "r"(a[1]), "r"(b));
#else
      __trap();
#endif
    }

    __device__ __forceinline__ void ldsm_x2(unsigned r[2], const void* p)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
#else
      __trap();
#endif
    }

    // Four 8x8 BF16 matrices from shared memory in one instruction: lane l gives
    // the address of row l % 8 of matrix l / 8; register i gets matrix i in the
    // mma fragment layout (row l / 4, columns 2 (l % 4) and 2 (l % 4) + 1).
    __device__ __forceinline__ void ldsm_x4(unsigned r[4], const void* p)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                   : "r"(a));
#else
      __trap();
#endif
    }

    // Word index of (row, channel pair p) in an activation buffer.
    __device__ __forceinline__ int act_word(int row, int p)
    {
      const int r = row + ACT_LO;
      return r * 8 + (p ^ (((r >> 2) & 1) << 2));
    }

    // Zero rows [from, ACT_ROWS - ACT_LO) of an activation buffer.
    __device__ __forceinline__ void zero_tail(unsigned* act, int from, int lane)
    {
      for (int i = (from + ACT_LO) * 8 + lane; i < ACT_WORDS; i += 32)
        act[i] = 0u;
    }

    // One convolution layer of width W (input and output), bias + ReLU, BF16
    // output, optionally max-pooled by 2. L = layer index into CONV.
    template<int L, int W, bool POOL>
    __device__ __forceinline__ void conv_layer(
      const unsigned* __restrict__ in_act,
      const unsigned* __restrict__ in_row,
      unsigned* __restrict__ out,
      const __nv_bfloat16* s_w,
      const float* s_bias,
      int lane)
    {
      constexpr int CI = CONV[L].c_in, PAD = CONV[L].pad, KS = ksteps(L);
      // A last k-step with at most 8 real k (rcbn1: K = 100) is an 8-deep mma.
      constexpr int K_REAL = CI * CONV[L].r;
      constexpr bool TAIL8 = K_REAL % 16 != 0 && K_REAL % 16 <= 8;
      constexpr int KS16 = TAIL8 ? KS - 1 : KS;
      constexpr int MT = (W + 15) / 16;
      const int g = lane >> 2, q = lane & 3;

      float acc[MT][2][4];
#pragma unroll
      for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int i = 0; i < 4; ++i)
            acc[m][n][i] = 0.0f;

      // ldmatrix row of this lane: B (weights [n][k]): n = l % 8 + 8 (l / 16),
      // k half (l / 8) % 2; A (activations [w][c]): w = l % 8 + 8 ((l / 8) % 2),
      // channel half l / 16.
      const int b_row = (lane & 7) + ((lane >> 4) & 1) * 8, b_half = (lane >> 3) & 1;
      const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_half = lane >> 4;
      if constexpr (TAIL8) {
        // B[k][n] for k = KS16 * 16 + 2q (+1): n-tile 0 from lanes 0-7, n-tile 1 from lanes 8-15.
        unsigned bt[2];
        ldsm_x2(bt, s_w + conv_offset(L) + ((lane & 7) + ((lane >> 3) & 1) * 8) * kstride(L) + KS16 * 16);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          const int e0 = IN_LO + (m * 16 + g - PAD) * CI + KS16 * 16 + 2 * q; // dense input (CI < C)
          const unsigned a[2] = {in_row[e0 / 2], in_row[(e0 + 8 * CI) / 2]};
          mma_k8(acc[m][0], a, bt[0]);
          mma_k8(acc[m][1], a, bt[1]);
        }
      }
#pragma unroll
      for (int s = 0; s < KS16; ++s) {
        // B fragments: B[k][n] = W[n][k], k = s * 16 + 2q (+1) (+8), n = tile * 8 + g
        unsigned bq[4];
        ldsm_x4(bq, s_w + conv_offset(L) + b_row * kstride(L) + s * 16 + b_half * 8);
        const unsigned b00 = bq[0], b01 = bq[1], b10 = bq[2], b11 = bq[3];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          unsigned a[4];
          const int r0 = m * 16 + g; // output rows r0 and r0 + 8
          if constexpr (CI == C) {
            // k-step s is tap s: A[w][c] = act[w + s - PAD][c]
            ldsm_x4(a, in_act + act_word(m * 16 + a_row + s - PAD, 4 * a_half));
          }
          else {
            // dense [bin][CI] input: A[w][k] = in[(w - PAD) * CI + k]
            const int e0 = IN_LO + (r0 - PAD) * CI + s * 16 + 2 * q;
            const int e1 = e0 + 8 * CI;
            a[0] = in_row[e0 / 2];
            a[1] = in_row[e1 / 2];
            a[2] = in_row[e0 / 2 + 4];
            a[3] = in_row[e1 / 2 + 4];
          }
          mma(acc[m][0], a, b00, b01);
          mma(acc[m][1], a, b10, b11);
        }
      }

      // Epilogue: bias + ReLU, round to BF16 (max-pool on the rounded values).
#pragma unroll
      for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const int ch = n * 8 + 2 * q;
          const float b0 = s_bias[L * C + ch], b1 = s_bias[L * C + ch + 1];
          const float* d = acc[m][n];
          float v00 = fmaxf(d[0] + b0, 0.0f), v01 = fmaxf(d[1] + b1, 0.0f);
          float v10 = fmaxf(d[2] + b0, 0.0f), v11 = fmaxf(d[3] + b1, 0.0f);
          const int r0 = m * 16 + g, r1 = r0 + 8;
          if constexpr (POOL) {
            // rows r and r ^ 1 sit in lanes l and l ^ 4
            v00 = fmaxf(v00, __shfl_xor_sync(0xffffffffu, v00, 4));
            v01 = fmaxf(v01, __shfl_xor_sync(0xffffffffu, v01, 4));
            v10 = fmaxf(v10, __shfl_xor_sync(0xffffffffu, v10, 4));
            v11 = fmaxf(v11, __shfl_xor_sync(0xffffffffu, v11, 4));
            if ((g & 1) == 0) {
              if (r0 < W) out[act_word(r0 / 2, n * 4 + q)] = pack(v00, v01);
              if (r1 < W) out[act_word(r1 / 2, n * 4 + q)] = pack(v10, v11);
            }
          }
          else {
            if (r0 < W) out[act_word(r0, n * 4 + q)] = pack(v00, v01);
            if (r1 < W) out[act_word(r1, n * 4 + q)] = pack(v10, v11);
          }
        }
      }
      zero_tail(out, POOL ? W / 2 : W, lane);
      __syncwarp();
    }

    // ConvTranspose, kernel 2, stride 2: out[2i + p][o] = b[o] + sum_c in[i][c] Wt[p][o][c].
    // T = ConvTranspose index (0: up1, 1: up2), WI = input width.
    template<int T, int WI>
    __device__ __forceinline__ void conv_transpose_layer(
      const unsigned* __restrict__ in_act,
      unsigned* __restrict__ out,
      const __nv_bfloat16* s_bf,
      const float* s_f,
      int lane)
    {
      constexpr int MT = (WI + 15) / 16;
      const int g = lane >> 2, q = lane & 3;
      const unsigned* wt = reinterpret_cast<const unsigned*>(s_bf + CT_OFFSET + T * 2 * C * CT_STRIDE);
      const float* bias = s_f + F_CT_BIAS + T * C;
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        const int r0 = m * 16 + g, r1 = r0 + 8;
        unsigned a[4];
        a[0] = in_act[act_word(r0, q)];
        a[1] = in_act[act_word(r1, q)];
        a[2] = in_act[act_word(r0, q + 4)];
        a[3] = in_act[act_word(r1, q + 4)];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            const unsigned* wr = wt + (p * C + n * 8 + g) * (CT_STRIDE / 2);
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma(d, a, wr[q], wr[q + 4]);
            const int ch = n * 8 + 2 * q;
            const float b0 = bias[ch], b1 = bias[ch + 1];
            if (r0 < WI) out[act_word(2 * r0 + p, n * 4 + q)] = pack(d[0] + b0, d[1] + b1);
            if (r1 < WI) out[act_word(2 * r1 + p, n * 4 + q)] = pack(d[2] + b0, d[3] + b1);
          }
        }
      }
      zero_tail(out, 2 * WI, lane);
      __syncwarp();
    }

    // The output stage (PVFinderUNetOutputStage.cuh): one 9-tap filter over
    // the 16 channels, bias, softplus. For the interior bins (the 5 filter sets differ
    // only at the 4 edge bins), Y[w][t] = sum_c act[w][c] F[c][t] on tensor cores,
    // with F split into BF16 hi + lo parts (about 16 significant bits, as good as
    // FP32 here), staged in FP32 in the free buffer, then out[j] = bias +
    // sum_t Y[j + t - 4][t]. The edge bins keep the scalar FP32 loop.
    __device__ __forceinline__ void output_stage(
      const unsigned* __restrict__ act,
      float* __restrict__ ybuf,
      const __nv_bfloat16* s_bf,
      const float* s_f,
      float* __restrict__ kde,
      float scale,
      int lane)
    {
      using S = OutputStage<C>;
      constexpr int YS = S::TAPS;     // Y row stride (odd: a lane per row reads conflict-free)
      constexpr int YLO = 2 * S::PAD; // rows -4..-1 are the zero padding
      const float* params = s_f + F_OUT;
      const int g = lane >> 2, q = lane & 3;
      float* Y = ybuf + YLO * YS;
      for (int i = lane; i < YLO * YS; i += 32)
        ybuf[i] = 0.0f;
      const unsigned* ob = reinterpret_cast<const unsigned*>(s_bf + OB_OFFSET) + lane;
      const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_half = lane >> 4;
#pragma unroll
      for (int m = 0; m < (W1 + 15) / 16; ++m) {
        unsigned a[4];
        ldsm_x4(a, act + act_word(m * 16 + a_row, 4 * a_half));
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          mma(d, a, ob[(0 * 4 + n * 2 + 0) * 32], ob[(0 * 4 + n * 2 + 1) * 32]); // hi
          mma(d, a, ob[(1 * 4 + n * 2 + 0) * 32], ob[(1 * 4 + n * 2 + 1) * 32]); // lo
          const int t0 = n * 8 + 2 * q, r0 = m * 16 + g, r1 = r0 + 8;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int r = i < 2 ? r0 : r1, t = t0 + (i & 1);
            if (t < S::TAPS && r < W1 + YLO) Y[r * YS + t] = d[i];
          }
        }
      }
      __syncwarp();
      // Interior bins from Y.
      for (int j = 2 + lane; j < W1 - 2; j += 32) {
        float acc[3] = {params[S::bias_offset], 0.0f, 0.0f};
#pragma unroll
        for (int t = 0; t < S::TAPS; ++t)
          acc[t % 3] += Y[(j + t - YLO) * YS + t];
        const float v = acc[0] + acc[1] + acc[2];
        kde[j] = (fmaxf(v, 0.f) + logf(1.f + expf(-fabsf(v)))) * scale;
      }
      // Edge bins 0, 1, W-2, W-1 (sets 1-4), exact FP32: lane = 8 e + channel pair,
      // then a sum over the 8 lanes of each bin.
      {
        const int e = lane >> 3, p = lane & 7;
        const int j = e < 2 ? e : W1 - 4 + e, set = e + 1;
        const float* f = params + set * S::filter_floats;
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < S::TAPS; ++t) {
          const float2 x = unpack(act[act_word(j + t - YLO, p)]);
          acc += f[(2 * p) * S::TAPS + t] * x.x + f[(2 * p + 1) * S::TAPS + t] * x.y;
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        acc += __shfl_xor_sync(0xffffffffu, acc, 2);
        acc += __shfl_xor_sync(0xffffffffu, acc, 4);
        if (p == 0) {
          const float v = params[S::bias_offset + set] + acc;
          kde[j] = (fmaxf(v, 0.f) + logf(1.f + expf(-fabsf(v)))) * scale;
        }
      }
    }

    // rows: number of input rows; in: [rows][W1][CIN] BF16; kde: [rows][W1], or
    // with row_slot: [slots][W1], row r going to slot row_slot[r], and every slot
    // with slot_row[s] < 0 (n_slots of them in all) getting empty[W1] (the
    // response to an empty interval): the whole KDE in one launch.
    __global__ void __launch_bounds__(THREADS) fused_unet_bf16_kernel(
      const __nv_bfloat16* __restrict__ in,
      const unsigned char* __restrict__ blob,
      float* __restrict__ kde,
      float scale,
      int rows,
      const int* __restrict__ row_slot,
      const int* __restrict__ slot_row,
      const float* __restrict__ empty,
      int n_slots)
    {
      extern __shared__ __align__(16) unsigned char smem[];
      {
        const uint4* src = reinterpret_cast<const uint4*>(blob);
        uint4* dst = reinterpret_cast<uint4*>(smem);
        for (int i = threadIdx.x; i < BLOB_BYTES / 16; i += THREADS)
          dst[i] = src[i];
      }
      const __nv_bfloat16* s_bf = reinterpret_cast<const __nv_bfloat16*>(smem);
      const float* s_f = reinterpret_cast<const float*>(smem + BF16_ELEMS * 2);
      const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
      unsigned* buf0 = reinterpret_cast<unsigned*>(smem + BLOB_BYTES + warp * WARP_BYTES);
      unsigned* buf1 = buf0 + ACT_WORDS;
      unsigned* in_row = buf1 + ACT_WORDS;
      // Margins start zeroed; later layers only rewrite them with zeros.
      for (int i = lane; i < 2 * ACT_WORDS + IN_ELEMS / 2; i += 32)
        buf0[i] = 0u;
      __syncthreads();

      for (int row = blockIdx.x * WARPS + warp; row < rows; row += gridDim.x * WARPS) {
        // Input row: W1 * CIN BF16 = 50 x 16 bytes, to elements [IN_LO, IN_LO + 400).
        {
          const uint4* src = reinterpret_cast<const uint4*>(in + (size_t) row * W1 * CIN);
          uint4* dst = reinterpret_cast<uint4*>(in_row + IN_LO / 2);
          for (int i = lane; i < W1 * CIN / 8; i += 32)
            dst[i] = src[i];
        }
        __syncwarp();
        conv_layer<0, W1, false>(nullptr, in_row, buf0, s_bf, s_f + F_CONV_BIAS, lane); // rcbn1
        conv_layer<1, W1, true>(buf0, nullptr, buf1, s_bf, s_f + F_CONV_BIAS, lane);    // rcbn2 + pool
        conv_layer<2, W2, true>(buf1, nullptr, buf0, s_bf, s_f + F_CONV_BIAS, lane);    // rcbn3 + pool
        conv_transpose_layer<0, W3>(buf0, buf1, s_bf, s_f, lane);                       // up1 (25 -> 50)
        conv_layer<3, W2, false>(buf1, nullptr, buf0, s_bf, s_f + F_CONV_BIAS, lane);   // up1c
        conv_transpose_layer<1, W2>(buf0, buf1, s_bf, s_f, lane);                       // up2 (50 -> 100)
        conv_layer<4, W1, false>(buf1, nullptr, buf0, s_bf, s_f + F_CONV_BIAS, lane);   // up2c
        const size_t out = row_slot != nullptr ? (size_t) row_slot[row] : (size_t) row;
        output_stage(buf0, reinterpret_cast<float*>(buf1), s_bf, s_f, kde + out * W1, scale, lane);
        __syncwarp(); // buf0 / in_row are rewritten by the next row
      }
      if (row_slot != nullptr) {
        // Slots without a row: the empty-interval response, 25 float4 per slot.
        const float4* e4 = reinterpret_cast<const float4*>(empty);
        for (int i = blockIdx.x * THREADS + threadIdx.x; i < n_slots * (W1 / 4); i += gridDim.x * THREADS) {
          const int slot = i / (W1 / 4), k = i % (W1 / 4);
          if (slot_row[slot] < 0) reinterpret_cast<float4*>(kde + (size_t) slot * W1)[k] = e4[k];
        }
      }
    }

    // Host side: the kernel's shared-memory image from the layers' weights.
    //   conv_w[l]: BN-folded, [C][R][C_in] (taps outer, channels inner), FP32 values
    //   conv_b[l]: [C]; ct_w[t]: PyTorch ConvTranspose1d layout [C_in][C_out][2]; ct_b[t]: [C]
    //   out_params: OutputStage<C> parameter block
    inline std::vector<unsigned char> make_fused_unet_blob(
      const std::vector<float> conv_w[5],
      const std::vector<float> conv_b[5],
      const std::vector<float> ct_w[2],
      const std::vector<float> ct_b[2],
      const std::vector<float>& out_params)
    {
      std::vector<unsigned char> blob(BLOB_BYTES, 0);
      __nv_bfloat16* bf = reinterpret_cast<__nv_bfloat16*>(blob.data());
      float* f = reinterpret_cast<float*>(blob.data() + BF16_ELEMS * 2);
      for (int i = 0; i < BF16_ELEMS; ++i)
        bf[i] = __float2bfloat16(0.0f);
      for (int l = 0; l < 5; ++l) {
        const int k_real = CONV[l].c_in * CONV[l].r;
        for (int n = 0; n < C; ++n)
          for (int k = 0; k < k_real; ++k)
            bf[conv_offset(l) + n * kstride(l) + k] = __float2bfloat16(conv_w[l][(size_t) n * k_real + k]);
        for (int n = 0; n < C; ++n)
          f[F_CONV_BIAS + l * C + n] = conv_b[l][n];
      }
      for (int t = 0; t < 2; ++t) {
        for (int p = 0; p < 2; ++p)
          for (int o = 0; o < C; ++o)
            for (int c = 0; c < C; ++c)
              bf[CT_OFFSET + ((t * 2 + p) * C + o) * CT_STRIDE + c] = __float2bfloat16(ct_w[t][(c * C + o) * 2 + p]);
        for (int o = 0; o < C; ++o)
          f[F_CT_BIAS + t * C + o] = ct_b[t][o];
      }
      for (int i = 0; i < OutputStage<C>::n_params; ++i)
        f[F_OUT + i] = out_params[i];
      // Interior output filter F[c][t] (set 0) as B fragments, B[k = c][n = t], split
      // into hi = bf16(F) and lo = bf16(F - hi): word (part * 4 + n * 2 + h) * 32 + lane,
      // lane = 4 g + q holds channels {2q, 2q+1} + 8 h of tap n * 8 + g.
      using S = OutputStage<C>;
      unsigned* ob = reinterpret_cast<unsigned*>(bf + OB_OFFSET);
      for (int part = 0; part < 2; ++part)
        for (int n = 0; n < 2; ++n)
          for (int h = 0; h < 2; ++h)
            for (int lane = 0; lane < 32; ++lane) {
              const int g = lane >> 2, q = lane & 3, t = n * 8 + g;
              __nv_bfloat16 v[2];
              for (int e = 0; e < 2; ++e) {
                const int c = 2 * q + 8 * h + e;
                const float x = t < S::TAPS ? out_params[c * S::TAPS + t] : 0.0f;
                const __nv_bfloat16 hi = __float2bfloat16(x);
                v[e] = part == 0 ? hi : __float2bfloat16(x - __bfloat162float(hi));
              }
              unsigned w;
              std::memcpy(&w, v, sizeof(w));
              ob[(part * 4 + n * 2 + h) * 32 + lane] = w;
            }
      return blob;
    }

  } // namespace fused
} // namespace pvfinder_unet
