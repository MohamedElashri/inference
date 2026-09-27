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
#include "PVFinderFCAggregation.cuh"

INSTANTIATE_ALGORITHM(pvfinder_fc_aggregation::pvfinder_fc_aggregation_t)

// The FC kernels use CUDA-only features (bfloat16 tensor cores, warp
// intrinsics); other targets build a stub that refuses to run.
#if defined(TARGET_DEVICE_CUDA)
#include "PVFinderWeightRegistry.h"
#include "PVFinderTrackFeatures.cuh"

#include <cuda_bf16.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace pvfinder_fc_aggregation {
// Track-to-interval assignment and selection, as in the training data
// (pv-finder_v2 tools/split_data_intervals.py, checked against the t2hists
// arrays): interval i covers z in [-100 + 10 i, -90 + 10 i), extended by
// 2.5 mm on both sides, so a track within 2.5 mm of an edge also feeds the
// neighbouring interval (at most two intervals per track). Only tracks with
// sigma_z < 2 mm and |x| / sigma_x, |y| / sigma_y < 4 are used, with
// sigma = 1 / sqrt(|A|), 1 / sqrt(|B|), 1 / sqrt(|C|).
constexpr float PVF_Z_MIN = PVFinderConstants::KDE::z_min;
constexpr float PVF_INTERVAL_WIDTH = PVFinderConstants::KDE::interval_width;
constexpr float PVF_INTERVAL_EXTENSION = 2.5f;

__device__ bool pvfinder_track_selected(const float* feat) {
    const float x = feat[0], y = feat[1], z = feat[2];
    const float A = feat[3], B = feat[4], C = feat[5];
    if (!(z > PVF_Z_MIN - PVF_INTERVAL_EXTENSION && z < PVFinderConstants::KDE::z_max + PVF_INTERVAL_EXTENSION))
        return false;
    const float sigma_x = sqrtf(fabsf(1.0f / A));
    const float sigma_y = sqrtf(fabsf(1.0f / B));
    const float sigma_z = sqrtf(fabsf(1.0f / C));
    return sigma_z < 2.0f && fabsf(x / sigma_x) < 4.0f && fabsf(y / sigma_y) < 4.0f;
}

__device__ void assign_intervals(float z_poca, int* intervals, int* num_intervals) {
    const int base = (int)floorf((z_poca - PVF_Z_MIN) / PVF_INTERVAL_WIDTH);
    int n = 0;
    for (int i = base - 1; i <= base + 1; ++i) {
        if (i < 0 || i >= (int) N_INTERVALS) continue;
        const float lo = PVF_Z_MIN + PVF_INTERVAL_WIDTH * i;
        if (z_poca > lo - PVF_INTERVAL_EXTENSION && z_poca < lo + PVF_INTERVAL_WIDTH + PVF_INTERVAL_EXTENSION) {
            intervals[n++] = i;
        }
    }
    *num_intervals = n;
}

// Network input for one (track, interval) entry, in the training order:
// (z - interval lower edge, x, y, A, B, C, D, E, F).
__device__ __forceinline__ void pvfinder_interval_input(const float* feat, int interval, float* in) {
    in[0] = feat[2] - (PVF_Z_MIN + PVF_INTERVAL_WIDTH * interval);
    in[1] = feat[0];
    in[2] = feat[1];
    for (int k = 3; k < 9; ++k) in[k] = feat[k];
}

// Exact, branchless and overflow-safe softplus: log(1 + exp(x)).
__device__ float pvfinder_softplus(float x) {
    return fmaxf(x, 0.0f) + logf(1.0f + expf(-fabsf(x)));
}

// Inline LeakyReLU and linear layer — runs in registers, no global writes.
__device__ __forceinline__ float pvfinder_leaky_relu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}

// A zero w_stride uses in_f as the row stride. A nonzero stride selects a
// leading submatrix from wider stored weights for the hidden-width throughput
// probe.

// ---------------------------------------------------------------------------
// CSR index builder kernel.
//
// Grid: (n_events)  blockDim: 256
//
// For each event, builds a CSR (compressed sparse row) representation that
// maps each interval to a contiguous range of track indices:
//
//   interval_start[ev * 42 + i]          = start offset in track_idx[]
//   interval_start[ev * CSR_STRIDE + CSR_TOTAL]         = total entries (sentinel)
//   track_idx[track_idx_base + start..end] = local track indices for interval i
//
// Boundary tracks (assigned to 2 intervals) appear twice.
// Tracks failing pvfinder_track_selected are omitted entirely.
//
// Three shared-memory passes:
//   1. Histogram: count tracks per interval (pass1 atomic into s_counts[40])
//   2. Exclusive prefix sum: compute s_start[41] from s_counts
//   3. Scatter:   fill track_idx[] advancing s_cursor[40] atomically
// ---------------------------------------------------------------------------
__global__ void pvfinder_build_csr_kernel(pvfinder_fc_aggregation_t::Parameters parameters)
{
    const unsigned event_number      = parameters.dev_event_list[blockIdx.x];
    const unsigned thread_id         = threadIdx.x;
    const auto     velo_tracks_view  = parameters.dev_velo_tracks_view[event_number];
    const unsigned num_tracks        = velo_tracks_view.size();
    const unsigned event_track_offset = velo_tracks_view.offset();

    __shared__ int s_counts[N_INTERVALS];   // histogram
    __shared__ int s_start[N_INTERVALS + 1];    // exclusive prefix sum → CSR start offsets
    __shared__ int s_cursor[N_INTERVALS];   // per-interval fill cursors (advanced atomically)

    // Up to CACHE tracks and entries, pass 1 keeps each track's interval
    // assignment and z in shared memory, so the features are read once; the
    // scatter then goes to shared memory and each
    // interval's tracks are ranked there and written to global memory once.
    // Larger events (none in the 2024 minimum-bias sample: at most 661 tracks,
    // 756 entries) take the uncached path, with the same result.
    constexpr int CACHE = 1024;
    __shared__ int s_trk[CACHE];     // -1 not selected, else n | iv0 << 8 | iv1 << 16
    __shared__ float s_trk_z[CACHE];
    __shared__ int s_ent[CACHE];     // scatter order: local track index per entry
    __shared__ float s_ent_z[CACHE]; // and its z
    const bool cached = num_tracks <= (unsigned) CACHE;

    for (int i = thread_id; i < (int) N_INTERVALS; i += blockDim.x) s_counts[i] = 0;
    __syncthreads();

    // Pass 1 — the track's features (written for the FC kernels), and how
    // many (track, interval) entries per interval
    const auto velo_states_view = parameters.dev_velo_states_view[event_number];
    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        const unsigned gtidx  = event_track_offset + i;
        float feat[9];
        pvfinder_track_features::compute(velo_states_view.state(velo_tracks_view.track(i).track_index()), feat);
        float* g_feat = parameters.dev_pvfinder_track_features + gtidx * 9;
#pragma unroll
        for (int k = 0; k < 9; ++k) g_feat[k] = feat[k];
        if (!pvfinder_track_selected(feat)) {
            if (cached) s_trk[i] = -1;
            continue;
        }
        const float    z_poca = feat[2];
        int ivals[2]; int n = 0;
        assign_intervals(z_poca, ivals, &n);
        for (int j = 0; j < n; ++j)
            atomicAdd(&s_counts[ivals[j]], 1);
        if (cached) {
            s_trk[i] = n | (n > 0 ? ivals[0] << 8 : 0) | (n > 1 ? ivals[1] << 16 : 0);
            s_trk_z[i] = z_poca;
        }
    }
    __syncthreads();

    // Pass 2 — exclusive prefix sum (single-threaded; only 40 elements)
    if (thread_id == 0) {
        int acc = 0;
        for (int i = 0; i < (int) N_INTERVALS; ++i) {
            s_start[i]  = acc;
            s_cursor[i] = acc;
            acc += s_counts[i];
        }
        s_start[N_INTERVALS] = acc;  // sentinel
    }
    __syncthreads();

    // Write interval_start[] to global memory
    int* g_start = parameters.dev_pvfinder_interval_start + event_number * CSR_STRIDE;
    for (int i = thread_id; i <= (int) N_INTERVALS; i += blockDim.x)
        g_start[i] = s_start[i];
    // index 41 = total track_idx entries for this event (= s_start[40])
    if (thread_id == 0) g_start[CSR_TOTAL] = s_start[N_INTERVALS];

    // Pass 3 — scatter: local track indices per interval, in shared memory
    // when cached, else to global memory (the order is fixed below)
    int* g_idx = parameters.dev_pvfinder_track_idx + event_track_offset * 2;
    int* g_unsorted = parameters.dev_pvfinder_track_idx_unsorted + event_track_offset * 2;
    const int n_entries = s_start[N_INTERVALS];
    const bool in_shared = cached && n_entries <= CACHE;
    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        int ivals[2]; int n = 0;
        float z_poca = 0.0f;
        if (cached) {
            const int packed = s_trk[i];
            if (packed < 0) continue;
            n = packed & 0xff;
            ivals[0] = (packed >> 8) & 0xff;
            ivals[1] = (packed >> 16) & 0xff;
            z_poca = s_trk_z[i];
        }
        else {
            const float* feat = parameters.dev_pvfinder_track_features + (event_track_offset + i) * 9;
            if (!pvfinder_track_selected(feat)) continue;
            z_poca = feat[2];
            assign_intervals(z_poca, ivals, &n);
        }
        for (int j = 0; j < n; ++j) {
            const int pos = atomicAdd(&s_cursor[ivals[j]], 1);
            if (in_shared) {
                s_ent[pos] = (int)i | (ivals[j] << 10);   // local track index (< CACHE) and interval
                s_ent_z[pos] = z_poca;
            }
            else {
                g_unsorted[pos] = (int)i;   // local track index within event
            }
        }
    }
    __syncthreads();

    // The atomic cursors leave each interval's tracks in scheduling order, and
    // the VELO reconstruction's own track order also changes from run to run.
    // Put each interval's tracks in a canonical order, by their features (z,
    // x, y, A..F, lexicographic; the track index only breaks exact ties, whose
    // tracks contribute identically), so the CSR and every sum over an
    // interval's tracks are the same from run to run. Each thread ranks one
    // entry against the others of its interval (z from shared memory, the
    // full features only on equal z) and writes it to its place. (A bitonic
    // sort of the whole event was slower: its many block-wide barriers cost
    // more than the ranking loops.) An event too large for the shared-memory
    // scatter is ranked the same way from global memory.
    const float* ev_feat = parameters.dev_pvfinder_track_features + (size_t)event_track_offset * 9;
    // Only reached on equal z (the entry itself is skipped below).
    const auto before = [ev_feat](int p, int q) {
        const float* fp = ev_feat + (size_t)p * 9;
        const float* fq = ev_feat + (size_t)q * 9;
#pragma unroll
        for (int k = 0; k < 9; ++k) {
            const int f = k == 0 ? 2 : k < 3 ? k - 1 : k;   // z, x, y, A..F
            const float a = fp[f], b = fq[f];
            if (a < b) return true;
            if (a > b) return false;
        }
        return p < q;
    };
    if (!in_shared) {
        for (int pos = thread_id; pos < n_entries; pos += blockDim.x) {
            int iv = 0;
            while (s_start[iv + 1] <= pos) ++iv;   // this entry's interval
            const int me = g_unsorted[pos];
            const float zme = ev_feat[(size_t)me * 9 + 2];
            int rank = 0;
            for (int q = s_start[iv]; q < s_start[iv + 1]; ++q) {
                const int other = g_unsorted[q];
                const float zq = ev_feat[(size_t)other * 9 + 2];
                rank += (zq < zme || (zq == zme && q != pos && before(other, me))) ? 1 : 0;
            }
            g_idx[s_start[iv] + rank] = me;
        }
    }
    if (in_shared) {
        for (int pos = thread_id; pos < n_entries; pos += blockDim.x) {
            const int packed = s_ent[pos], me = packed & 1023;
            const int iv = packed >> 10;   // this entry's interval (packed by the scatter)
            const float zme = s_ent_z[pos];
            int rank = 0;
            for (int q = s_start[iv]; q < s_start[iv + 1]; ++q) {
                const float zq = s_ent_z[q];
                rank += (zq < zme || (zq == zme && q != pos && before(s_ent[q] & 1023, me))) ? 1 : 0;
            }
            g_idx[s_start[iv] + rank] = me;
        }
    }
}

// Validation dump only: each track's VELO Kalman state at the beamline, in the
// order of dev_pvfinder_track_features (x, y, z, tx, ty, c00), and the
// beamline (pos x, y, z, tx x, y), so weights/scripts/validate_features.py can
// recompute the features independently. One block per event.
__global__ void pvfinder_dump_states_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters, float* states, float* beamline)
{
    const unsigned event_number = blockIdx.x;
    const auto tracks = parameters.dev_velo_tracks_view[event_number];
    const auto kalman = parameters.dev_velo_states_view[event_number];
    for (unsigned i = threadIdx.x; i < tracks.size(); i += blockDim.x) {
        const auto st = kalman.state(tracks.track(i).track_index());
        float* o = states + (size_t) (tracks.offset() + i) * 6;
        o[0] = st.x(); o[1] = st.y(); o[2] = st.z(); o[3] = st.tx(); o[4] = st.ty(); o[5] = st.c00();
    }
    if (event_number == 0 && threadIdx.x == 0) {
        beamline[0] = dev_beamline.pos.x; beamline[1] = dev_beamline.pos.y; beamline[2] = dev_beamline.pos.z;
        beamline[3] = dev_beamline.tx.x;  beamline[4] = dev_beamline.tx.y;
    }
}

constexpr unsigned FC_L15_FLOATS = 1880u;   // layers 1-5 weights and biases

__device__ __forceinline__ unsigned pvfinder_pack_bf16x2(float lo, float hi)
{
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);   // lo in the low half
    return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ void pvfinder_mma_bf16_16816(float d[4], const unsigned a[4], const unsigned b[2])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();   // init() refuses precision = bfloat16 below sm_80
#endif
}

// ---------------------------------------------------------------------------
// FC stage, FP32 (precision = float32): L1-L5, L6A, bias + LeakyReLU and the
// sum over each interval's tracks in one kernel, one warp per work item, so
// nothing waits on a block-wide barrier and nothing but the sums goes to
// memory. Lane t takes track t of a 32-track tile through L1-L5 in registers
// (weights in shared memory, read as broadcasts); the layer-5 outputs go
// through the warp's staging area to L6A, one lane per output column with W6A
// transposed in shared memory. Tracks are summed in the CSR's canonical order,
// so the result is reproducible. The BF16 variant is pvfinder_fused_fc_tc_kernel.
// ---------------------------------------------------------------------------
constexpr unsigned FW_WARPS = 8u;
constexpr unsigned FW_THREADS = FW_WARPS * 32u;
constexpr unsigned fw_w6_bytes() { return L6A_WIDTH * 20u * 4u; }
// Per warp: layer-5 outputs [32][20] and the slot's sums [L6A_WIDTH].
constexpr unsigned FW_WARP_FLOATS = 32u * 20u + L6A_WIDTH;
constexpr unsigned fw_smem_bytes()
{
    return (FC_L15_FLOATS + L6A_WIDTH) * 4u + fw_w6_bytes() + FW_WARPS * FW_WARP_FLOATS * 4u;
}

__global__ void __launch_bounds__(FW_THREADS) pvfinder_fused_fc_warp_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights,
    unsigned n_events,
    unsigned* work_counter,
    const int* __restrict__ slot_row,   // feature row of each slot, -1 when the UNet skips it
    int features_format,                // 0 float32, 1 bfloat16, 2 bfloat16 channels last
    const FCWorkItem* __restrict__ slot_items,   // the work list, largest first
    unsigned n_items,
    bool write_histogram)               // validation dump only
{
    extern __shared__ __align__(16) unsigned char fw_smem[];
    float* s_w = reinterpret_cast<float*>(fw_smem);             // L1-L5 weights and biases
    float* s_b6 = s_w + FC_L15_FLOATS;                           // L6A bias
    unsigned char* s_w6 = fw_smem + (FC_L15_FLOATS + L6A_WIDTH) * 4u;
    const unsigned warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    float* s_h = reinterpret_cast<float*>(s_w6 + fw_w6_bytes()) + warp * FW_WARP_FLOATS;
    float* s_feat = s_h + 32u * 20u;

    const float* w6A = dev_weights + FC_L15_FLOATS;   // row-major [L6A_WIDTH][20]
    const float* b6A = w6A + L6A_WEIGHT_FLOATS;
    for (unsigned i = threadIdx.x; i < FC_L15_FLOATS; i += FW_THREADS) s_w[i] = dev_weights[i];
    for (unsigned i = threadIdx.x; i < L6A_WIDTH; i += FW_THREADS) s_b6[i] = b6A[i];
    {
        float* w = reinterpret_cast<float*>(s_w6);   // transposed: [20][L6A_WIDTH]
        for (unsigned i = threadIdx.x; i < L6A_WEIGHT_FLOATS; i += FW_THREADS) {
            const unsigned n = i / 20u, m = i % 20u;
            w[m * L6A_WIDTH + n] = w6A[i];
        }
    }
    __syncthreads();
    const float* w1 = s_w;        const float* b1 = w1 + 180;
    const float* w2 = b1 + 20;    const float* b2 = w2 + 400;
    const float* w3 = b2 + 20;    const float* b3 = w3 + 400;
    const float* w4 = b3 + 20;    const float* b4 = w4 + 400;
    const float* w5 = b4 + 20;    const float* b5 = w5 + 400;

    const unsigned total = n_items;
    while (true) {
        unsigned item = 0;
        if (lane == 0) item = atomicAdd(work_counter, 1u);
        item = __shfl_sync(0xffffffffu, item, 0);
        if (item >= total) break;
        // Everything about the slot in one item (built on the host from the CSR).
        const FCWorkItem it = slot_items[item];
        const unsigned slot = it.slot;
        const int a = (int) it.first;
        const int n_local = (int) it.entries;
        const long long row = it.row;
        const unsigned ev = slot / N_INTERVALS, iv = slot % N_INTERVALS;
        float* g_hist = write_histogram ? parameters.dev_pvfinder_output_histogram + ev * KDE_BINS + iv * N_BINS_PER_CHANNEL : nullptr;

        if (n_local == 0) {
            if (row >= 0) {
                if (features_format != 0) {
                    unsigned* g = reinterpret_cast<unsigned*>(static_cast<float*>(parameters.dev_pvfinder_interval_features))
                                  + (unsigned long long)row * (L6A_WIDTH / 2);
                    for (unsigned i = lane; i < L6A_WIDTH / 2; i += 32u) g[i] = 0u;
                } else {
                    float* g = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
                    for (unsigned i = lane; i < L6A_WIDTH; i += 32u) g[i] = 0.0f;
                }
            }
            if (write_histogram) for (unsigned i = lane; i < N_BINS_PER_CHANNEL; i += 32u) g_hist[i] = 0.0f;
            continue;
        }

        const auto tracks_view = parameters.dev_velo_tracks_view[ev];
        const unsigned track_offset = tracks_view.offset();
        const int* g_idx = parameters.dev_pvfinder_track_idx + track_offset * 2 + a;
        for (unsigned i = lane; i < L6A_WIDTH; i += 32u) s_feat[i] = 0.0f;

        for (int t0 = 0; t0 < n_local; t0 += 32) {
            const unsigned nt = min((unsigned) (n_local - t0), 32u);
            // L1-L5 for this lane's track (the same sums as the block kernel).
            float x1[20], x2[20];
            if (lane < nt) {
                float in[9];
                pvfinder_interval_input(parameters.dev_pvfinder_track_features
                                            + (track_offset + (unsigned) g_idx[t0 + (int) lane]) * 9, (int) iv, in);
#pragma unroll
                for (unsigned j = 0; j < 20; ++j) {
                    float v = b1[j];
#pragma unroll
                    for (unsigned i = 0; i < 9; ++i) v += w1[j * 9 + i] * in[i];
                    x1[j] = pvfinder_leaky_relu(v);
                }
                const float* lw[4] = {w2, w3, w4, w5};
                const float* lb[4] = {b2, b3, b4, b5};
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    float* hin = (l & 1) ? x2 : x1;
                    float* hout = (l & 1) ? x1 : x2;
#pragma unroll
                    for (unsigned j = 0; j < 20; ++j) {
                        float v = lb[l][j];
#pragma unroll
                        for (unsigned i = 0; i < 20; ++i) v += lw[l][j * 20 + i] * hin[i];
                        hout[j] = pvfinder_leaky_relu(v);
                    }
                }
            }
            else {
#pragma unroll
                for (unsigned j = 0; j < 20; ++j) x1[j] = 0.0f;
            }
            // Layer 5's output is in x1 after four ping-pongs.
#pragma unroll
            for (unsigned j = 0; j < 20; ++j) s_h[lane * 20u + j] = x1[j];
            __syncwarp();

            {
                const float* w6 = reinterpret_cast<const float*>(s_w6);
                for (unsigned col = lane; col < L6A_WIDTH; col += 32u) {
                    float acc = s_feat[col];
                    for (unsigned t = 0; t < nt; ++t) {
                        float v = s_b6[col];
#pragma unroll
                        for (unsigned m = 0; m < 20; ++m) v += w6[m * L6A_WIDTH + col] * s_h[t * 20u + m];
                        acc += pvfinder_leaky_relu(v);
                    }
                    s_feat[col] = acc;
                }
            }
            __syncwarp();   // s_h is rewritten by the next tile, s_feat is read below
        }

        if (row >= 0) {
            if (lane == 0) parameters.dev_pvfinder_row_slot[row] = (int) slot;
            if (features_format != 0) {
                __nv_bfloat16* g = reinterpret_cast<__nv_bfloat16*>(static_cast<float*>(parameters.dev_pvfinder_interval_features)) + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = lane; i < L6A_WIDTH; i += 32u) {
                    const unsigned dst = features_format == 2 ? (i % N_BINS_PER_CHANNEL) * N_LATENT_CHANNELS + i / N_BINS_PER_CHANNEL : i;
                    g[dst] = __float2bfloat16(s_feat[i]);
                }
            } else {
                float* g = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = lane; i < L6A_WIDTH; i += 32u) g[i] = s_feat[i];
            }
        }
        if (write_histogram) {
            const float weight = 1.0f / n_local;
            for (unsigned bin = lane; bin < N_BINS_PER_CHANNEL; bin += 32u) {
                float chan_sum = 0.0f;
                for (unsigned c = 0; c < N_LATENT_CHANNELS; ++c) chan_sum += s_feat[c * 100 + bin];
                g_hist[bin] = pvfinder_softplus(chan_sum) * weight;
            }
        }
        __syncwarp();   // s_feat is reset by the next slot
    }
}

// ---------------------------------------------------------------------------
// Tensor-core fused FC (fc_fused, fc_fused_per_warp, bfloat16 L6A): one warp
// per slot, slots taken from the host-built work list (largest first).
//  - Layer 1 (raw track features, large range): FP32, one lane per track.
//  - Layers 2-5: a chain of
//    m16n8k16 + m16n8k8 mma in registers, tracks as rows; a layer's output
//    fragments, activated and rounded to BF16, are the next layer's input
//    fragments. Their weights are in shared memory as per-lane fragments.
//  - L6A with the operands swapped: A = W6A (16 output columns per m-tile,
//    ldmatrix from BF16 rows of FT_K6), B = the layer-5 fragments, whose
//    rows (tracks) become the 8-wide n dimension. Tracks are padded to 8
//    rather than 16 (1.20x the entries on the 2024 sample instead of 1.43x),
//    and each lane keeps the slot's sums for its columns in registers across
//    tiles; the 4 lanes sharing a column combine them once per slot.
//  - The histogram (FC's own output, not read by the UNet) only when asked.
// FT_K6: K = 20 as a 16-deep and an 8-deep step, rows of 24 BF16 (48 bytes:
// an 8-row ldmatrix phase hits 8 different 16-byte bank groups).
// ---------------------------------------------------------------------------
constexpr unsigned FT_WARPS = 8u;
constexpr unsigned FT_THREADS = FT_WARPS * 32u;
constexpr unsigned FT_K6 = 24u;              // BF16 per W6A row: 20 weights, 4 zeros
constexpr unsigned FT_H = 12u;               // words per staged hidden row (24 BF16)
static_assert(L6A_WIDTH % 16u == 0, "L6A columns in 16-row m-tiles");
constexpr unsigned FT_CT = L6A_WIDTH / 16u;  // L6A column tiles
// Slots with more entries are split into work items of FT_CHUNK entries,
// done by different warps; each writes its partial sums and the last one to
// finish adds them up in chunk order (deterministic); see FCWorkItem.
constexpr unsigned FT_CHUNK = 64u;
// Per warp: the staged layer-1 outputs [32][FT_H] words, reused for the
// slot's sums [L6A_WIDTH] once its tiles are done.
constexpr unsigned FT_WARP_WORDS = (32u * FT_H > L6A_WIDTH) ? 32u * FT_H : L6A_WIDTH;
// Layers 2-5 as per-lane fragments: [layer][n-tile][3 words][32 lanes] and
// their biases [layer][n-tile][2][32 lanes].
constexpr unsigned FT_HB_WORDS = 4u * 3u * 3u * 32u;
constexpr unsigned FT_HBIAS_FLOATS = 4u * 3u * 2u * 32u;
constexpr unsigned ft_smem_bytes()
{
    return (200u + L6A_WIDTH) * 4u + L6A_WIDTH * FT_K6 * 2u + (FT_HB_WORDS + FT_HBIAS_FLOATS) * 4u
           + FT_WARPS * FT_WARP_WORDS * 4u;
}

__device__ __forceinline__ void pvfinder_mma_bf16_1688(float d[4], const unsigned a[2], unsigned b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
#else
    __trap();
#endif
}

__device__ __forceinline__ void pvfinder_ldsm_x4(unsigned r[4], const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#else
    __trap();
#endif
}

__device__ __forceinline__ void pvfinder_ldsm_x2(unsigned r[2], const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
#else
    __trap();
#endif
}

__global__ void __launch_bounds__(FT_THREADS, 2) pvfinder_fused_fc_tc_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights,
    unsigned n_events,
    unsigned* work_counter,
    const int* __restrict__ slot_row,   // feature row of each slot, -1 when the UNet skips it
    int features_format,                // 0 float32, 1 bfloat16, 2 bfloat16 channels last
    const FCWorkItem* __restrict__ slot_items,   // the work list, largest first
    unsigned n_items,
    bool write_histogram,               // validation dump only
    float* __restrict__ partial,        // [rows][L6A_WIDTH] partial sums of split slots
    unsigned* __restrict__ arrive)      // per split slot: chunks done
{
    extern __shared__ __align__(16) unsigned char ft_smem[];
    float* s_w1 = reinterpret_cast<float*>(ft_smem);   // layer 1: 180 weights, 20 biases
    float* s_b6 = s_w1 + 200;
    __nv_bfloat16* s_w6 = reinterpret_cast<__nv_bfloat16*>(s_b6 + L6A_WIDTH);   // [L6A_WIDTH][FT_K6]
    unsigned* s_hb = reinterpret_cast<unsigned*>(s_w6 + L6A_WIDTH * FT_K6);
    float* s_hbias = reinterpret_cast<float*>(s_hb + FT_HB_WORDS);
    const unsigned warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    const unsigned grp = lane >> 2, quad = lane & 3u;
    unsigned* s_h = reinterpret_cast<unsigned*>(s_hbias + FT_HBIAS_FLOATS) + warp * FT_WARP_WORDS;
    float* s_feat = reinterpret_cast<float*>(s_h);   // after the slot's tiles

    const float* w6A = dev_weights + FC_L15_FLOATS;   // row-major [L6A_WIDTH][20]
    const float* b6A = w6A + L6A_WEIGHT_FLOATS;
    for (unsigned i = threadIdx.x; i < 200u; i += FT_THREADS) s_w1[i] = dev_weights[i];
    for (unsigned i = threadIdx.x; i < L6A_WIDTH; i += FT_THREADS) s_b6[i] = b6A[i];
    for (unsigned i = threadIdx.x; i < L6A_WIDTH * FT_K6; i += FT_THREADS) {
        const unsigned n = i / FT_K6, k = i % FT_K6;
        s_w6[i] = __float2bfloat16(k < 20u ? w6A[n * 20u + k] : 0.0f);
    }
    // Layers 2-5: B[k][n] = W[n][k], n = nt * 8 + g; k-steps {2q, 2q+1}, {2q+8, 2q+9}
    // (16-deep) and {16+2q, 17+2q} (8-deep); stored per lane.
    for (unsigned i = threadIdx.x; i < 4u * 3u * 32u; i += FT_THREADS) {
        const unsigned l = i / 96u, nt = (i / 32u) % 3u, ln = i % 32u;
        const unsigned g = ln >> 2, q = ln & 3u;
        const float* w = dev_weights + 200u + l * 420u;
        const float* b = w + 400u;
        const unsigned n = nt * 8u + g;
        const unsigned ks[3] = {2u * q, 2u * q + 8u, 16u + 2u * q};
        for (unsigned f = 0; f < 3; ++f) {
            const unsigned k = ks[f];
            const float lo = (n < 20u && k < 20u) ? w[n * 20u + k] : 0.0f;
            const float hi = (n < 20u && k + 1u < 20u) ? w[n * 20u + k + 1u] : 0.0f;
            s_hb[((l * 3u + nt) * 3u + f) * 32u + ln] = pvfinder_pack_bf16x2(lo, hi);
        }
        const unsigned c = nt * 8u + 2u * q;
        s_hbias[((l * 3u + nt) * 2u + 0u) * 32u + ln] = c < 20u ? b[c] : 0.0f;
        s_hbias[((l * 3u + nt) * 2u + 1u) * 32u + ln] = c + 1u < 20u ? b[c + 1u] : 0.0f;
    }
    __syncthreads();
    // ldmatrix rows of this lane for W6A: row (l % 8) + 8 ((l / 8) % 2) of the
    // m-tile, k half l / 16 (16-deep step); the 8-deep step uses lanes 0-15.
    const unsigned a_row = (lane & 7u) + ((lane >> 3) & 1u) * 8u, a_half = lane >> 4;

    const unsigned total = n_items;
    while (true) {
        unsigned item = 0;
        if (lane == 0) item = atomicAdd(work_counter, 1u);
        item = __shfl_sync(0xffffffffu, item, 0);
        if (item >= total) break;
        // Everything about the work item in one item (built on the host from the CSR).
        const FCWorkItem it = slot_items[item];
        const unsigned slot = it.slot;
        const int a = (int) it.first;
        const unsigned pbase = it.partial;
        int n_local = (int) it.entries;   // a split slot's total, for the histogram, below
        const unsigned chunk = it.chunk;
        const unsigned n_chunks = it.n_chunks;
        const long long row = it.row;
        const unsigned ev = slot / N_INTERVALS, iv = slot % N_INTERVALS;
        float* g_hist = write_histogram ? parameters.dev_pvfinder_output_histogram + ev * KDE_BINS + iv * N_BINS_PER_CHANNEL : nullptr;

        if (n_local == 0) {
            if (row >= 0) {
                if (features_format != 0) {
                    unsigned* g = reinterpret_cast<unsigned*>(static_cast<float*>(parameters.dev_pvfinder_interval_features))
                                  + (unsigned long long)row * (L6A_WIDTH / 2);
                    for (unsigned i = lane; i < L6A_WIDTH / 2; i += 32u) g[i] = 0u;
                } else {
                    float* g = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
                    for (unsigned i = lane; i < L6A_WIDTH; i += 32u) g[i] = 0.0f;
                }
            }
            if (write_histogram) for (unsigned i = lane; i < N_BINS_PER_CHANNEL; i += 32u) g_hist[i] = 0.0f;
            continue;
        }

        const auto tracks_view = parameters.dev_velo_tracks_view[ev];
        const unsigned track_offset = tracks_view.offset();
        const int* g_idx = parameters.dev_pvfinder_track_idx + track_offset * 2 + a;
        // This lane's sums: columns ct * 16 + grp (acc[ct][0]) and + 8 (acc[ct][1]),
        // over its tracks 2 quad and 2 quad + 1 of every 8-track tile.
        float acc[FT_CT][2];
#pragma unroll
        for (unsigned ct = 0; ct < FT_CT; ++ct) acc[ct][0] = acc[ct][1] = 0.0f;

        for (int t0 = 0; t0 < n_local; t0 += 32) {
            const unsigned nt = min((unsigned) (n_local - t0), 32u);
            const unsigned n_mt = (nt + 15u) / 16u;
            const unsigned n_t8 = (nt + 7u) / 8u;
            // Layer 1 for this lane's track.
            float x1[20];
            if (lane < nt) {
                float in[9];
                pvfinder_interval_input(parameters.dev_pvfinder_track_features
                                            + (track_offset + (unsigned) g_idx[t0 + (int) lane]) * 9, (int) iv, in);
#pragma unroll
                for (unsigned j = 0; j < 20; ++j) {
                    float v = s_w1[180 + j];
#pragma unroll
                    for (unsigned i = 0; i < 9; ++i) v += s_w1[j * 9 + i] * in[i];
                    x1[j] = pvfinder_leaky_relu(v);
                }
            }
            else {
#pragma unroll
                for (unsigned j = 0; j < 20; ++j) x1[j] = 0.0f;
            }
#pragma unroll
            for (unsigned j = 0; j < 10; ++j) s_h[lane * FT_H + j] = pvfinder_pack_bf16x2(x1[2 * j], x1[2 * j + 1]);
            s_h[lane * FT_H + 10] = 0u;
            s_h[lane * FT_H + 11] = 0u;
            __syncwarp();

            // Fragments of the hidden state, tracks as rows: a16 (k 0-15), a8 (k 16-23).
            unsigned a16[2][4], a8[2][2];
#pragma unroll
            for (unsigned mt = 0; mt < 2; ++mt) {
                const unsigned r0 = (mt * 16u + grp) * FT_H, r1 = r0 + 8u * FT_H;
                a16[mt][0] = s_h[r0 + quad];      a16[mt][1] = s_h[r1 + quad];
                a16[mt][2] = s_h[r0 + quad + 4u]; a16[mt][3] = s_h[r1 + quad + 4u];
                a8[mt][0] = s_h[r0 + quad + 8u];  a8[mt][1] = s_h[r1 + quad + 8u];
            }
            __syncwarp();   // s_h is free again (it becomes s_feat after the last tile)
            {   // layers 2-5 on tensor cores
#pragma unroll
                for (unsigned l = 0; l < 4; ++l) {
#pragma unroll
                    for (unsigned mt = 0; mt < 2; ++mt) {
                        if (mt >= n_mt) break;   // warp-uniform
                        unsigned o[6];
#pragma unroll
                        for (unsigned n3 = 0; n3 < 3; ++n3) {
                            const unsigned* hb = s_hb + ((l * 3u + n3) * 3u) * 32u + lane;
                            const float* hbias = s_hbias + ((l * 3u + n3) * 2u) * 32u + lane;
                            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                            const unsigned b16[2] = {hb[0], hb[32]};
                            pvfinder_mma_bf16_16816(d, a16[mt], b16);
                            pvfinder_mma_bf16_1688(d, a8[mt], hb[64]);
                            const float bias0 = hbias[0], bias1 = hbias[32];
                            o[2 * n3] = pvfinder_pack_bf16x2(pvfinder_leaky_relu(d[0] + bias0),
                                                             pvfinder_leaky_relu(d[1] + bias1));
                            o[2 * n3 + 1] = pvfinder_pack_bf16x2(pvfinder_leaky_relu(d[2] + bias0),
                                                                 pvfinder_leaky_relu(d[3] + bias1));
                        }
                        a16[mt][0] = o[0]; a16[mt][1] = o[1]; a16[mt][2] = o[2]; a16[mt][3] = o[3];
                        a8[mt][0] = o[4];  a8[mt][1] = o[5];
                    }
                }
            }

            // L6A, swapped: C[col][track] = sum_k W6A[col][k] h[track][k]. For
            // 8-track tile j = 2 mt + h: B = {a16[mt][h], a16[mt][2 + h]}, a8[mt][h].
#pragma unroll
            for (unsigned ct = 0; ct < FT_CT; ++ct) {
                unsigned wa16[4], wa8[2];
                pvfinder_ldsm_x4(wa16, s_w6 + (ct * 16u + a_row) * FT_K6 + a_half * 8u);
                pvfinder_ldsm_x2(wa8, s_w6 + (ct * 16u + a_row) * FT_K6 + 16u);
                const float bias0 = s_b6[ct * 16u + grp], bias1 = s_b6[ct * 16u + grp + 8u];
#pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    if (j >= n_t8) break;   // warp-uniform
                    const unsigned mt = j >> 1, h = j & 1u;
                    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    const unsigned b16[2] = {a16[mt][h], a16[mt][2 + h]};
                    pvfinder_mma_bf16_16816(d, wa16, b16);
                    pvfinder_mma_bf16_1688(d, wa8, a8[mt][h]);
                    const unsigned t = j * 8u + 2u * quad;   // tracks t and t + 1
                    if (t < nt) { acc[ct][0] += pvfinder_leaky_relu(d[0] + bias0); acc[ct][1] += pvfinder_leaky_relu(d[2] + bias1); }
                    if (t + 1u < nt) { acc[ct][0] += pvfinder_leaky_relu(d[1] + bias0); acc[ct][1] += pvfinder_leaky_relu(d[3] + bias1); }
                }
            }
        }

        // Combine the 4 lanes of each column (fixed order), stage the sums:
        // in s_feat, or for a split slot in this chunk's partial row.
        float* dst = n_chunks > 1 ? partial + (size_t) (pbase + chunk) * L6A_WIDTH : s_feat;
#pragma unroll
        for (unsigned ct = 0; ct < FT_CT; ++ct) {
            float v0 = acc[ct][0], v1 = acc[ct][1];
            v0 += __shfl_xor_sync(0xffffffffu, v0, 1);
            v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
            v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
            v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
            if (quad == 0) {
                dst[ct * 16u + grp] = v0;
                dst[ct * 16u + grp + 8u] = v1;
            }
        }
        if (n_chunks > 1) {
            // The last chunk to finish adds the partial rows up, in chunk order.
            __threadfence();
            __syncwarp();
            unsigned done = 0;
            if (lane == 0) done = atomicAdd(arrive + pbase, 1u);
            done = __shfl_sync(0xffffffffu, done, 0);
            if (done != n_chunks - 1u) continue;
            __threadfence();
            for (unsigned i = lane; i < L6A_WIDTH; i += 32u) {
                float sum = 0.0f;
                for (unsigned c = 0; c < n_chunks; ++c) sum += __ldcg(partial + (size_t) (pbase + c) * L6A_WIDTH + i);
                s_feat[i] = sum;
            }
            if (write_histogram) {   // the whole slot's entries, for the histogram's weight
                const int* g_start = parameters.dev_pvfinder_interval_start + ev * CSR_STRIDE;
                n_local = g_start[iv + 1] - g_start[iv];
            }
        }
        __syncwarp();

        if (row >= 0) {
            if (lane == 0) parameters.dev_pvfinder_row_slot[row] = (int) slot;
            if (features_format != 0) {
                __nv_bfloat16* g = reinterpret_cast<__nv_bfloat16*>(static_cast<float*>(parameters.dev_pvfinder_interval_features)) + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = lane; i < L6A_WIDTH; i += 32u) {
                    const unsigned dst = features_format == 2 ? (i % N_BINS_PER_CHANNEL) * N_LATENT_CHANNELS + i / N_BINS_PER_CHANNEL : i;
                    g[dst] = __float2bfloat16(s_feat[i]);
                }
            } else {
                float* g = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = lane; i < L6A_WIDTH; i += 32u) g[i] = s_feat[i];
            }
        }
        if (write_histogram) {
            const float weight = 1.0f / n_local;
            for (unsigned bin = lane; bin < N_BINS_PER_CHANNEL; bin += 32u) {
                float chan_sum = 0.0f;
                for (unsigned c = 0; c < N_LATENT_CHANNELS; ++c) chan_sum += s_feat[c * 100 + bin];
                g_hist[bin] = pvfinder_softplus(chan_sum) * weight;
            }
        }
        __syncwarp();   // s_feat (s_h) is rewritten by the next slot
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
void pvfinder_fc_aggregation_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    const unsigned n_tracks = first<host_number_of_reconstructed_velo_tracks_t>(arguments);
    const unsigned batch = m_unet_batch_events.value();
    if (batch == 0) {
        throw StrException("pvfinder_fc_aggregation: unet_batch_events must be >= 1");
    }
    const unsigned padded_events = (n_events + batch - 1) / batch * batch;
    const unsigned n_slots = n_events * N_INTERVALS;
    // A track feeds one or two intervals.
    const unsigned max_entries = n_tracks * 2u;
    // The BF16 kernel splits a slot of n > FT_CHUNK entries into ceil(n /
    // FT_CHUNK) <= 2 n / FT_CHUNK items: at most max_entries / FT_CHUNK items
    // more than slots, and max_entries * 2 / FT_CHUNK partial sums.
    const unsigned max_items = n_slots + max_entries / FT_CHUNK + 1u;
    const unsigned max_partial = max_entries * 2u / FT_CHUNK + 1u;

    set_size<dev_pvfinder_track_features_t>(arguments, n_tracks * 9u);
    set_size<dev_pvfinder_interval_start_t>(arguments, n_events * CSR_STRIDE);
    set_size<host_pvfinder_interval_start_t>(arguments, n_events * CSR_STRIDE);
    set_size<dev_pvfinder_track_idx_t>(arguments, max_entries);
    set_size<dev_pvfinder_track_idx_unsorted_t>(arguments, max_entries);
    set_size<dev_pvfinder_interval_features_t>(arguments, padded_events * INTERVAL_FEATURES_STRIDE);
    set_size<host_pvfinder_unet_rows_t>(arguments, 4u);
    set_size<dev_pvfinder_slot_row_t>(arguments, n_slots);
    set_size<host_pvfinder_slot_row_t>(arguments, n_slots);
    set_size<dev_pvfinder_row_slot_t>(arguments, n_slots);
    set_size<dev_pvfinder_slot_order_t>(arguments, max_items * FC_WORK_ITEM_WORDS);
    set_size<host_pvfinder_slot_order_t>(arguments, max_items * FC_WORK_ITEM_WORDS);
    set_size<dev_pvfinder_work_counter_t>(arguments, 1u);
    set_size<dev_pvfinder_fc_partial_t>(arguments, m_bf16 ? max_partial * L6A_WIDTH : 0u);
    set_size<dev_pvfinder_fc_arrive_t>(arguments, m_bf16 ? max_partial : 0u);
    set_size<dev_pvfinder_output_histogram_t>(arguments, m_dump_dir.value().empty() ? 0u : n_events * KDE_BINS);
}

void pvfinder_fc_aggregation_t::update(const Constants& constants) const { updateCommon(constants); }

// Loads this instance's weights, keys namespaced by weight file as in
// pvfinder_unet (instances with the same file share one device copy), and
// sizes the FC kernel's grid for this device.
void pvfinder_fc_aggregation_t::init()
{
    const std::string& precision = m_precision.value();
    if (precision != "float32" && precision != "bfloat16") {
        throw StrException("pvfinder_fc_aggregation: precision must be float32 or bfloat16, got '" + precision + "'");
    }
    m_bf16 = precision == "bfloat16";

    const std::string path = m_weight_file.value();
    if (path.empty()) {
        throw StrException(
            "pvfinder_fc_aggregation: weight_file is not set. Produce weights with the repository's "
            "weights/ pipeline (make -C weights verify MODEL=<name>) and generate the sequence "
            "configuration with PVFINDER_WEIGHTS_DIR pointing at them (make -C weights env MODEL=<name>).");
    }
    auto& registry = PVFinder::WeightRegistry::instance();
    const std::string key = "pvfinder_fc:" + path + ":weights";
    if (!registry.contains(key)) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            throw StrException("pvfinder_fc_aggregation: cannot open " + path);
        }
        const size_t bytes = static_cast<size_t>(f.tellg());
        f.seekg(0);
        std::vector<char> host_buf(bytes);
        f.read(host_buf.data(), bytes);
        // Layers 1-5 (FC_L15_FLOATS: 9x20 + 20, then 4 x (20x20 + 20)), then
        // W6A row-major [L6A_WIDTH][20] and b6A [L6A_WIDTH]. L6A_WIDTH is fixed
        // by the build (--unet-batch-channels); the file must match it.
        constexpr size_t expected_floats = FC_L15_FLOATS + L6A_WEIGHT_FLOATS + L6A_WIDTH;
        if (bytes != expected_floats * sizeof(float)) {
            throw StrException(
                "pvfinder_fc_aggregation: " + path + " has " + std::to_string(bytes / sizeof(float)) +
                " floats, this build expects " + std::to_string(expected_floats) + " (N_LATENT_CHANNELS = " +
                std::to_string(N_LATENT_CHANNELS) + "); the model's latentChannels and the build's "
                "--unet-batch-channels must match");
        }
        registry.load_from_buffer(key, host_buf.data(), bytes);
    }
    m_dev_weights = registry.get<float>(key);

    int device = 0, sm_count = 0, cc_major = 0, per_sm = 0;
    cudaCheck(cudaGetDevice(&device));
    cudaCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
    cudaCheck(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
    if (m_bf16) {
        if (cc_major < 8) {
            throw StrException("pvfinder_fc_aggregation: precision = bfloat16 needs bfloat16 tensor cores "
                               "(compute capability 8.0 or newer)");
        }
        cudaCheck(cudaFuncSetAttribute(
            pvfinder_fused_fc_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) ft_smem_bytes()));
        cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, pvfinder_fused_fc_tc_kernel, (int) FT_THREADS, ft_smem_bytes()));
        // See m_fused_grid_fraction: leave room for other streams' kernels.
        m_grid = std::max(1u, (unsigned) (sm_count * std::max(per_sm, 1) * m_fused_grid_fraction.value()));
    }
    else {
        cudaCheck(cudaFuncSetAttribute(
            pvfinder_fused_fc_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) fw_smem_bytes()));
        cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, pvfinder_fused_fc_warp_kernel, (int) FW_THREADS, fw_smem_bytes()));
        m_grid = (unsigned) (sm_count * std::max(per_sm, 1));
    }
}

void pvfinder_fc_aggregation_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    const unsigned n_slots = n_events * N_INTERVALS;
    const bool write_histogram = !m_dump_dir.value().empty();

    unsigned* host_unet_rows = data<host_pvfinder_unet_rows_t>(arguments);
    host_unet_rows[0] = 1u;
    host_unet_rows[1] = 0u;
    host_unet_rows[2] = m_bf16 ? 1u : 0u;
    host_unet_rows[3] = m_bf16 ? 1u : 0u;
    if (n_events == 0) return;

    // 1. The CSR, for the events in the event list; every other event keeps an
    //    all-zero CSR row (no tracks, empty intervals).
    const unsigned n_selected = size<dev_event_list_t>(arguments);
    if (n_selected < n_events) {
        Allen::memset_async<dev_pvfinder_interval_start_t>(arguments, 0, context);
    }
    if (n_selected > 0) {
        global_function(pvfinder_build_csr_kernel)(dim3(n_selected), m_block_dim, context)(arguments);
    }

    // 2. Its host copy, on this sequence's stream into Allen's (pinned) host
    //    memory. (A plain cudaMemcpy runs on the legacy default stream, which
    //    Allen's blocking streams all synchronise with: every slice would then
    //    drain the whole device, a cost that grows with the number of streams.)
    Allen::copy<host_pvfinder_interval_start_t, dev_pvfinder_interval_start_t>(arguments, context);
    const int* host_csr = data<host_pvfinder_interval_start_t>(arguments);

    // 3. The UNet's rows: one per interval with tracks, in (event, interval)
    //    order; the rows past the last one, up to a whole UNet batch, zeroed.
    int* host_slot_row = data<host_pvfinder_slot_row_t>(arguments);
    int n_rows = 0;
    for (unsigned ev = 0; ev < n_events; ++ev) {
        const int* start = host_csr + ev * CSR_STRIDE;
        for (unsigned iv = 0; iv < N_INTERVALS; ++iv) {
            host_slot_row[ev * N_INTERVALS + iv] = start[iv + 1] > start[iv] ? n_rows++ : -1;
        }
    }
    host_unet_rows[1] = (unsigned) n_rows;
    Allen::copy_async<dev_pvfinder_slot_row_t, host_pvfinder_slot_row_t>(arguments, context, n_slots);
    const unsigned row_floats = m_bf16 ? L6A_WIDTH / 2u : L6A_WIDTH;   // bfloat16 rows take half
    const unsigned batch_rows = m_unet_batch_events.value() * N_INTERVALS;
    const unsigned padded_rows = ((unsigned) n_rows + batch_rows - 1) / batch_rows * batch_rows;
    if (padded_rows > (unsigned) n_rows) {
        Allen::memset_async<dev_pvfinder_interval_features_t>(
            arguments, 0, context, (size_t) (padded_rows - n_rows) * row_floats, (size_t) n_rows * row_floats);
    }

    // 4. The work list: FCWorkItem by decreasing entries (a counting sort, ties
    //    in slot and chunk order), so the largest slots do not start last and
    //    run alone at the end of the kernel. For the BF16 kernel, slots with
    //    more than FT_CHUNK entries become several items (see FT_CHUNK).
    //    Slots without tracks need no output (no row), except for the
    //    validation's histogram.
    const int chunk = m_bf16 ? (int) FT_CHUNK : std::numeric_limits<int>::max();
    size_t total_entries = 0;
    for (unsigned ev = 0; ev < n_events; ++ev) total_entries += (size_t) host_csr[ev * CSR_STRIDE + CSR_TOTAL];
    // Guaranteed by set_arguments_size (total_entries <= twice the tracks);
    // checked because a violation would write past the buffer.
    if ((n_slots + total_entries / FT_CHUNK + 1) * FC_WORK_ITEM_WORDS > size<dev_pvfinder_slot_order_t>(arguments)) {
        throw StrException("pvfinder_fc_aggregation: FC work list larger than its buffer (sizing bug)");
    }
    int max_size = 0;
    for (unsigned ev = 0; ev < n_events; ++ev) {
        const int* start = host_csr + ev * CSR_STRIDE;
        for (unsigned iv = 0; iv < N_INTERVALS; ++iv) max_size = std::max(max_size, std::min(chunk, start[iv + 1] - start[iv]));
    }
    const auto for_each_item = [&](auto&& f) {
        unsigned next_partial = 0;
        for (unsigned ev = 0; ev < n_events; ++ev) {
            for (unsigned iv = 0; iv < N_INTERVALS; ++iv) {
                const int first = host_csr[ev * CSR_STRIDE + iv], n = host_csr[ev * CSR_STRIDE + iv + 1] - first;
                if (n == 0 && !write_histogram) continue;
                const unsigned n_chunks = n > chunk ? (unsigned) ((n + chunk - 1) / chunk) : 1u;
                const unsigned pbase = n_chunks > 1 ? next_partial : 0u;
                next_partial += n_chunks > 1 ? n_chunks : 0u;
                for (unsigned c = 0; c < n_chunks; ++c) {
                    const int size = n_chunks > 1 ? std::min(chunk, n - (int) c * chunk) : n;
                    f(ev * N_INTERVALS + iv, first + (int) c * chunk, size, c, n_chunks, pbase);
                }
            }
        }
        return next_partial;
    };
    std::vector<unsigned> bucket((size_t) max_size + 2, 0u);   // items by max_size - size
    for_each_item([&](unsigned, int, int size, unsigned, unsigned, unsigned) { ++bucket[max_size - size + 1]; });
    for (size_t i = 1; i < bucket.size(); ++i) bucket[i] += bucket[i - 1];
    const unsigned n_items = bucket.back();
    FCWorkItem* items = reinterpret_cast<FCWorkItem*>(data<host_pvfinder_slot_order_t>(arguments));
    const unsigned n_partial = for_each_item(
        [&](unsigned slot, int first, int size, unsigned c, unsigned n_chunks, unsigned pbase) {
            items[bucket[max_size - size]++] =
                FCWorkItem {slot, (unsigned) first, (unsigned) size, host_slot_row[slot], pbase, c, n_chunks, 0u};
        });
    // Guaranteed by set_arguments_size; checked because a violation would
    // write past the partial-sum buffers.
    if ((size_t) n_partial > size<dev_pvfinder_fc_arrive_t>(arguments)) {
        throw StrException("pvfinder_fc_aggregation: split FC slots exceed the partial-sum buffer (sizing bug)");
    }
    if (n_items > 0) {
        Allen::copy_async<dev_pvfinder_slot_order_t, host_pvfinder_slot_order_t>(
            arguments, context, n_items * FC_WORK_ITEM_WORDS);
    }
    if (n_partial > 0) {
        Allen::memset_async<dev_pvfinder_fc_arrive_t>(arguments, 0, context, n_partial);
    }

    // 5. The FC: all work items, one kernel.
    Allen::memset_async<dev_pvfinder_work_counter_t>(arguments, 0, context);
    const auto* slot_items = reinterpret_cast<const FCWorkItem*>(data<dev_pvfinder_slot_order_t>(arguments));
    if (m_bf16) {
        global_function(pvfinder_fused_fc_tc_kernel)(dim3(m_grid), dim3(FT_THREADS), context, ft_smem_bytes())(
            arguments,
            m_dev_weights,
            n_events,
            data<dev_pvfinder_work_counter_t>(arguments),
            data<dev_pvfinder_slot_row_t>(arguments),
            2,   // bfloat16, channels last
            slot_items,
            n_items,
            write_histogram,
            data<dev_pvfinder_fc_partial_t>(arguments),
            data<dev_pvfinder_fc_arrive_t>(arguments));
    }
    else {
        global_function(pvfinder_fused_fc_warp_kernel)(dim3(m_grid), dim3(FW_THREADS), context, fw_smem_bytes())(
            arguments,
            m_dev_weights,
            n_events,
            data<dev_pvfinder_work_counter_t>(arguments),
            data<dev_pvfinder_slot_row_t>(arguments),
            0,   // float32
            slot_items,
            n_items,
            write_histogram);
    }

    if (write_histogram && !m_dump_done) {
        dump(arguments, context);
        m_dump_done = true;
    }
}

// Validation dump (first call, when dump_validation is set): the raw
// buffers, so the validation can recompute this stage from the checkpoint with
// exactly Allen's own track-to-interval assignment. Every file: uint32 magic
// 0xFC01, n_events, n_tracks, N_LATENT_CHANNELS, then the array. Interval
// features in the dense [event][interval] float32 [channel][bin] layout,
// intervals without a row as zeros (what the UNet is given for them).
void pvfinder_fc_aggregation_t::dump(const ArgumentReferences<Parameters>& arguments, const Allen::Context& context) const
{
    const std::string& dump_dir = m_dump_dir.value();
    const unsigned n_ev = first<host_number_of_events_t>(arguments);
    const unsigned n_trk = first<host_number_of_reconstructed_velo_tracks_t>(arguments);
    const unsigned n_rows = data<host_pvfinder_unet_rows_t>(arguments)[1];

    const auto csr = make_host_buffer<dev_pvfinder_interval_start_t>(arguments, context);
    const auto track_idx = make_host_buffer<dev_pvfinder_track_idx_t>(arguments, context);
    const auto track_features = make_host_buffer<dev_pvfinder_track_features_t>(arguments, context);
    const auto histogram = make_host_buffer<dev_pvfinder_output_histogram_t>(arguments, context);
    const auto features = make_host_buffer<dev_pvfinder_interval_features_t>(arguments, context);
    const auto views = make_host_buffer<dev_velo_tracks_view_t>(arguments, context);
    const int* slot_row = data<host_pvfinder_slot_row_t>(arguments);

    std::vector<float> interval_features((size_t) n_ev * INTERVAL_FEATURES_STRIDE, 0.0f);
    const auto* bf16 = reinterpret_cast<const uint16_t*>(features.data());
    for (size_t slot = 0; slot < (size_t) n_ev * N_INTERVALS; ++slot) {
        const int row = slot_row[slot];
        if (row < 0 || row >= (int) n_rows) continue;
        float* dst = interval_features.data() + slot * L6A_WIDTH;
        for (unsigned e = 0; e < L6A_WIDTH; ++e) {   // e = channel * 100 + bin
            if (m_bf16) {
                // channels last; a bfloat16 is the upper half of the float with its value
                const uint32_t bits = (uint32_t) bf16[(size_t) row * L6A_WIDTH + (e % N_BINS_PER_CHANNEL) * N_LATENT_CHANNELS
                                                      + e / N_BINS_PER_CHANNEL] << 16;
                std::memcpy(dst + e, &bits, sizeof(float));
            }
            else {
                dst[e] = features.data()[(size_t) row * L6A_WIDTH + e];
            }
        }
    }
    std::vector<unsigned> offsets(n_ev);
    for (unsigned e = 0; e < n_ev; ++e) offsets[e] = views.data()[e].offset();

    // Track states and beamline, for the feature validation.
    auto dev_states = make_device_buffer<float>(arguments, std::max<size_t>(n_trk, 1) * 6);
    auto dev_beamline = make_device_buffer<float>(arguments, 5);
    global_function(pvfinder_dump_states_kernel)(dim3(n_ev), dim3(128), context)(
        arguments, dev_states.data(), dev_beamline.data());
    std::vector<float> states((size_t) n_trk * 6), beamline(5);
    Allen::copy(std::span<float> {states}, dev_states.get(), context, Allen::memcpyDeviceToHost, states.size());
    Allen::copy(std::span<float> {beamline}, dev_beamline.get(), context, Allen::memcpyDeviceToHost, 5);

    const uint32_t header[4] = {0xFC01u, n_ev, n_trk, N_LATENT_CHANNELS};
    auto write = [&](const char* name, const void* d, size_t bytes) {
        std::ofstream out(dump_dir + "/" + name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(header), sizeof(header));
        out.write(reinterpret_cast<const char*>(d), bytes);
        if (!out) throw StrException("pvfinder_fc_aggregation: cannot write " + dump_dir + "/" + name);
    };
    write("allen_fc_csr.bin", csr.data(), (size_t) n_ev * CSR_STRIDE * sizeof(int));
    write("allen_fc_track_idx.bin", track_idx.data(), (size_t) n_trk * 2 * sizeof(int));
    write("allen_fc_track_offsets.bin", offsets.data(), offsets.size() * sizeof(unsigned));
    write("allen_fc_track_features.bin", track_features.data(), (size_t) n_trk * 9 * sizeof(float));
    write("allen_fc_interval_features.bin", interval_features.data(), interval_features.size() * sizeof(float));
    write("allen_fc_histogram.bin", histogram.data(), (size_t) n_ev * KDE_BINS * sizeof(float));
    write("allen_fc_track_states.bin", states.data(), states.size() * sizeof(float));
    write("allen_fc_beamline.bin", beamline.data(), beamline.size() * sizeof(float));
    info_cout << "[pvfinder_fc_aggregation] validation dump written to " << dump_dir << " (" << n_ev << " events, "
              << n_trk << " tracks)\n";
}

} // namespace pvfinder_fc_aggregation

#else

void pvfinder_fc_aggregation::pvfinder_fc_aggregation_t::set_arguments_size(
  ArgumentReferences<Parameters>,
  const RuntimeOptions&,
  const Constants&) const
{
  throw StrException("pvfinder_fc_aggregation is only available in CUDA builds");
}

void pvfinder_fc_aggregation::pvfinder_fc_aggregation_t::operator()(
  const ArgumentReferences<Parameters>&,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context&) const
{
  throw StrException("pvfinder_fc_aggregation is only available in CUDA builds");
}

void pvfinder_fc_aggregation::pvfinder_fc_aggregation_t::update(const Constants&) const {}

void pvfinder_fc_aggregation::pvfinder_fc_aggregation_t::init()
{
  throw StrException("pvfinder_fc_aggregation is only available in CUDA builds");
}

void pvfinder_fc_aggregation::pvfinder_fc_aggregation_t::dump(
  const ArgumentReferences<Parameters>&,
  const Allen::Context&) const
{}

#endif // TARGET_DEVICE_CUDA
