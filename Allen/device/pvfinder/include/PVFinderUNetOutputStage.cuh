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

#include <vector>

namespace pvfinder_unet {

// The UNet's output stage as one convolution:
//   h      = out_intermediate(x)      Conv1d(C -> C, k = 5, pad 2) + bias
//   logit  = outc(h)                  Conv1d(C -> 1, k = 5, pad 2) + bias
//   kde    = softplus(logit) * scale
// There is no activation between the two convolutions, so they compose into
// one Conv1d(C -> 1, k = 9) acting on x: 9 * C multiply-adds per bin instead
// of 5 * C * C + 5 * C. At the two bins at each edge outc's zero padding
// applies to h, not x, so only the outc taps landing inside [0, W) count;
// each of those four bins gets its own 9-tap filter and bias with just those
// taps, which keeps the composition exact. make_output_stage_params() builds
// the five filter sets once, in double precision, from the two layers' FP32
// weights; the fused BF16 kernel (PVFinderUNetFused.cuh) applies them.
//
// Parameter block (OutputStage<C>::n_params floats): for set s = 0 (interior),
// 1, 2 (bins 0, 1), 3, 4 (bins W - 2, W - 1): F[s][c][9] then bias[s].
//   F[s][c][k1 + k2] = sum over the set's valid k2 of sum_o w_outc[o][k2] * w_oint[o][c][k1]
//   bias[s]          = b_outc + sum over valid k2 of sum_o w_outc[o][k2] * b_oint[o]
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

} // namespace pvfinder_unet
