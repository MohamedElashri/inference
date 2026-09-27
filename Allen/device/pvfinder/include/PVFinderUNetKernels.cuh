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
// Small kernels of the UNet outside cuDNN.
// All tensors use NCW layout (cuDNN NCHW with H=1).
// ---------------------------------------------------------------------------

namespace pvfinder_unet {

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
    const float* src = row >= 0 ? rows + (size_t) row * 100 : empty;
    reinterpret_cast<float4*>(kde + (size_t) slot * 100)[i % Q] = reinterpret_cast<const float4*>(src)[i % Q];
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
  // Launch: one block per output channel (K blocks).
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
    float eps,
    int K,
    int CxHxW)
  {
    int k = blockIdx.x;
    if (k >= K) return;
    float scale = gamma[k] * rsqrtf(var[k] + eps);
    if (threadIdx.x == 0) b_fused[k] = scale * (b[k] - mean[k]) + beta[k];
    for (int i = threadIdx.x; i < CxHxW; i += blockDim.x)
      w_fused[k * CxHxW + i] = scale * w[k * CxHxW + i];
  }

} // namespace pvfinder_unet
