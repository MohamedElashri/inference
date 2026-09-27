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

// ---------------------------------------------------------------------------
// Conv1d + bias + ReLU as one cuDNN graph-API operation graph, channels-last
// (NWC: [N][W][C], stored as NHWC with H = 1), reduced-precision storage
// (BF16 today) and FP32 compute. For the UNet's 16-channel convolutions the
// graph API offers runtime-compiled, tensor-core engines for this fused graph
// that are about four times faster than the legacy implicit-GEMM
// convolution in NCHW (benchmarks/campaigns/probe_cudnn_graph_bf16.cu); the
// fused graph is not supported at all in NCHW.
//
// create() asks the heuristics (mode A) for the engine configurations in
// their ranked order and builds plans from the first max_candidates that
// finalize; with more than one it times them on the calling stream with
// scratch buffers and keeps the fastest. Runtime-compiled engines are
// compiled when their plan is built, about a second each, so this belongs
// at initialisation, and the default is the heuristic's first choice (the
// fastest engine for every UNet shape in the probe). execute() binds the
// real buffers.
// Weights are [K][R][C] (KRSC with S = 1), bias [K], x [N][W][C_in], y [N][W_out][K].
// ---------------------------------------------------------------------------

#include "AllenCuDNN.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pvfinder_unet {

#ifdef ALLEN_CUDNN_BACKEND_CUDA
class ConvBiasReluGraphNWC {
public:
    ConvBiasReluGraphNWC() = default;
    ConvBiasReluGraphNWC(const ConvBiasReluGraphNWC&) = delete;
    ConvBiasReluGraphNWC& operator=(const ConvBiasReluGraphNWC&) = delete;
    ~ConvBiasReluGraphNWC() { if (m_plan) cudnnBackendDestroyDescriptor(m_plan); }

    // Returns a one-line description of the chosen engine (for the log).
    std::string create(cudnnHandle_t handle, cudaStream_t stream, cudnnDataType_t dtype,
                       int N, int C_in, int K, int W, int R, int pad, int max_candidates = 1)
    {
        const int W_out = W + 2 * pad - R + 1;
        const int64_t xd[4] = {N, C_in, 1, W}, wd[4] = {K, C_in, 1, R}, bd[4] = {1, K, 1, 1}, yd[4] = {N, K, 1, W_out};
        Desc x = tensor(UID_X, xd, dtype, false), w = tensor(UID_W, wd, dtype, false);
        Desc b = tensor(UID_B, bd, dtype, false), y = tensor(UID_Y, yd, dtype, false);
        Desc zc = tensor(UID_ZC, yd, CUDNN_DATA_FLOAT, true), za = tensor(UID_ZA, yd, CUDNN_DATA_FLOAT, true);

        Desc conv = mk(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        {
            cudnnDataType_t comp = CUDNN_DATA_FLOAT;
            cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
            int64_t sd = 2, pads[2] = {0, pad}, one[2] = {1, 1};
            set(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp);
            set(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
            set(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &sd);
            set(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
            set(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, pads);
            set(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, one);
            set(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, one);
            ALLEN_CUDNN_CHECK(cudnnBackendFinalize(conv));
        }
        float alpha = 1.f, beta = 0.f;
        Desc cop = mk(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
        set(cop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
        set(cop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w);
        set(cop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &zc);
        set(cop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv);
        set(cop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
        set(cop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(cop));
        Desc add = pointwise(CUDNN_POINTWISE_ADD), relu = pointwise(CUDNN_POINTWISE_RELU_FWD);
        Desc aop = pw_op(add, zc, b, za), rop = pw_op(relu, za, nullptr, y);

        Desc graph = mk(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        Desc ops[3] = {cop, aop, rop};
        set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
        set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 3, ops);
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(graph));

        Desc heur = mk(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        cudnnBackendHeurMode_t hm = CUDNN_HEUR_MODE_A;
        set(heur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
        set(heur, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &hm);
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(heur));
        int64_t count = 0;
        cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr);
        std::vector<Desc> cfgs(count);
        for (auto& c : cfgs) c = mk(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        int64_t got = 0;
        cudnnBackendGetAttribute(heur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &got, cfgs.data());

        // Scratch buffers to time the candidates on (contents do not matter).
        const size_t esz = dtype == CUDNN_DATA_FLOAT ? 4 : 2;
        void *sx, *sw, *sb, *sy, *sws = nullptr;
        cudaMalloc(&sx, (size_t)N * C_in * W * esz); cudaMalloc(&sw, (size_t)K * C_in * R * esz);
        cudaMalloc(&sb, (size_t)K * esz); cudaMalloc(&sy, (size_t)N * K * W_out * esz);
        cudaMemsetAsync(sx, 0, (size_t)N * C_in * W * esz, stream); cudaMemsetAsync(sw, 0, (size_t)K * C_in * R * esz, stream);
        cudaMemsetAsync(sb, 0, (size_t)K * esz, stream);
        size_t sws_bytes = 0;
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best_us = 1e30f; std::string best_note;
        int built = 0;
        for (int64_t i = 0; i < got && built < max_candidates; ++i) {
            Desc plan = nullptr;
            if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan) != CUDNN_STATUS_SUCCESS) continue;
            if (cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle) != CUDNN_STATUS_SUCCESS ||
                cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cfgs[i]) != CUDNN_STATUS_SUCCESS ||
                cudnnBackendFinalize(plan) != CUDNN_STATUS_SUCCESS) {
                cudnnBackendDestroyDescriptor(plan);
                continue;
            }
            ++built;
            int64_t ws = 0, n = 0;
            cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &n, &ws);
            if ((size_t)ws > sws_bytes) {
                if (sws) cudaFree(sws);
                cudaMalloc(&sws, (size_t)ws); sws_bytes = (size_t)ws;
            }
            float us = 1e30f;
            if (max_candidates == 1) {
                us = 0.f;   // nothing to compare against
            }
            else if (run(handle, plan, sx, sw, sb, sy, sws) && run(handle, plan, sx, sw, sb, sy, sws)) {
                std::vector<float> t;
                for (int r = 0; r < 20; ++r) {
                    cudaEventRecord(e0, stream);
                    run(handle, plan, sx, sw, sb, sy, sws);
                    cudaEventRecord(e1, stream);
                    cudaEventSynchronize(e1);
                    float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); t.push_back(ms * 1000.f);
                }
                std::sort(t.begin(), t.end());
                us = t[t.size() / 2];
            }
            if (us < best_us) {
                if (m_plan) cudnnBackendDestroyDescriptor(m_plan);
                m_plan = plan; m_ws_bytes = (size_t)ws; best_us = us;
                best_note = "candidate " + std::to_string(i) + " of " + std::to_string(got);
            }
            else {
                cudnnBackendDestroyDescriptor(plan);
            }
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(sx); cudaFree(sw); cudaFree(sb); cudaFree(sy); if (sws) cudaFree(sws);
        for (auto& c : cfgs) cudnnBackendDestroyDescriptor(c);
        for (Desc d : {heur, graph, cop, aop, rop, add, relu, conv, x, w, b, y, zc, za}) cudnnBackendDestroyDescriptor(d);
        if (!m_plan) throw StrException("ConvBiasReluGraphNWC: no engine for the fused Conv+Bias+ReLU graph");
        return best_note + (max_candidates > 1 ? ", " + std::to_string(best_us) + " us" : std::string()) +
               ", workspace " + std::to_string(m_ws_bytes) + " B";
    }

    void execute(cudnnHandle_t handle, const void* x, const void* w, const void* b, void* y) const
    {
        void* ws = thread_local_workspace(this, m_ws_bytes);
        if (!run(handle, m_plan, x, w, b, y, ws)) {
            throw StrException("ConvBiasReluGraphNWC: cudnnBackendExecute failed");
        }
    }

    void ensure_thread_local_workspace() const { thread_local_workspace(this, m_ws_bytes); }

private:
    using Desc = cudnnBackendDescriptor_t;
    enum : int64_t { UID_X = 1, UID_W, UID_B, UID_Y, UID_ZC, UID_ZA };
    Desc m_plan = nullptr;
    size_t m_ws_bytes = 0;

    static Desc mk(cudnnBackendDescriptorType_t t) { Desc d; ALLEN_CUDNN_CHECK(cudnnBackendCreateDescriptor(t, &d)); return d; }
    static void set(Desc d, cudnnBackendAttributeName_t a, cudnnBackendAttributeType_t t, int64_t n, const void* v) {
        ALLEN_CUDNN_CHECK(cudnnBackendSetAttribute(d, a, t, n, v));
    }
    // NHWC strides with H = 1: element (n, c, 0, w) at n * W * C + w * C + c.
    static Desc tensor(int64_t uid, const int64_t dims[4], cudnnDataType_t dt, bool virt) {
        Desc d = mk(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        const int64_t c = dims[1], h = dims[2], w = dims[3];
        const int64_t strides[4] = {h * w * c, 1, w * c, c};
        int64_t align = 16; int8_t v = virt;
        set(d, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
        set(d, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
        set(d, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims);
        set(d, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides);
        set(d, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
        set(d, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &v);
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(d));
        return d;
    }
    static Desc pointwise(cudnnPointwiseMode_t m) {
        Desc d = mk(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        cudnnDataType_t p = CUDNN_DATA_FLOAT;
        set(d, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &m);
        set(d, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &p);
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(d));
        return d;
    }
    static Desc pw_op(Desc pw, Desc x, Desc b, Desc y) {
        Desc d = mk(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
        set(d, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pw);
        set(d, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
        if (b) set(d, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b);
        set(d, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);
        ALLEN_CUDNN_CHECK(cudnnBackendFinalize(d));
        return d;
    }
    static bool run(cudnnHandle_t handle, Desc plan, const void* x, const void* w, const void* b, void* y, void* ws) {
        Desc vp = nullptr;
        if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &vp) != CUDNN_STATUS_SUCCESS) return false;
        int64_t uids[4] = {UID_X, UID_W, UID_B, UID_Y};
        void* ptrs[4] = {const_cast<void*>(x), const_cast<void*>(w), const_cast<void*>(b), y};
        bool ok = cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 4, uids) == CUDNN_STATUS_SUCCESS &&
                  cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 4, ptrs) == CUDNN_STATUS_SUCCESS &&
                  cudnnBackendSetAttribute(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws) == CUDNN_STATUS_SUCCESS &&
                  cudnnBackendFinalize(vp) == CUDNN_STATUS_SUCCESS &&
                  cudnnBackendExecute(handle, plan, vp) == CUDNN_STATUS_SUCCESS;
        cudnnBackendDestroyDescriptor(vp);
        return ok;
    }
    // Per (thread, instance) workspace, like ConvDescriptors' (never freed).
    static void* thread_local_workspace(const void* key, size_t bytes) {
        if (bytes == 0) return nullptr;
        thread_local std::unordered_map<const void*, std::pair<void*, size_t>> tl;
        auto& e = tl[key];
        if (e.second < bytes) {
            if (e.first) cudaFree(e.first);
            if (cudaMalloc(&e.first, bytes) != cudaSuccess) throw StrException("ConvBiasReluGraphNWC: workspace allocation failed");
            e.second = bytes;
        }
        return e.first;
    }
};
#endif

} // namespace pvfinder_unet
