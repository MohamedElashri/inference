#include "PVFinderFCAggregation.cuh"
#include "PVFinderWeightRegistry.h"
#include "PVFinderTrackFeatures.cuh"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <mutex>
#include <cuda_bf16.h>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#ifdef ALLEN_WITH_CUBLAS
#include <cublas_v2.h>
#include <limits>
// Thread-local cuBLAS handle — one per Allen pipeline thread, never shared.
static thread_local cublasHandle_t s_cublas_handle = nullptr;
static thread_local bool s_cublas_inited = false;
static inline cublasHandle_t get_cublas_handle() {
    if (!s_cublas_inited) {
        cublasCreate(&s_cublas_handle);
        s_cublas_inited = true;
    }
    return s_cublas_handle;
}
#endif  // ALLEN_WITH_CUBLAS

INSTANTIATE_ALGORITHM(pvfinder_fc_aggregation::pvfinder_fc_aggregation_t)

namespace pvfinder_fc_aggregation {

// Track-to-interval assignment and selection, as in the training data
// (pv-finder_v2 tools/split_data_intervals.py, checked against the t2hists
// arrays): interval i covers z in [-100 + 10 i, -90 + 10 i), extended by
// 2.5 mm on both sides, so a track within 2.5 mm of an edge also feeds the
// neighbouring interval (at most two intervals per track). Only tracks with
// sigma_z < 2 mm and |x| / sigma_x, |y| / sigma_y < 4 are used, with
// sigma = 1 / sqrt(|A|), 1 / sqrt(|B|), 1 / sqrt(|C|).
constexpr float PVF_Z_MIN = -100.0f;
constexpr float PVF_INTERVAL_WIDTH = 10.0f;
constexpr float PVF_INTERVAL_EXTENSION = 2.5f;

__device__ bool pvfinder_track_selected(const float* feat) {
    const float x = feat[0], y = feat[1], z = feat[2];
    const float A = feat[3], B = feat[4], C = feat[5];
    if (!(z > PVF_Z_MIN - PVF_INTERVAL_EXTENSION && z < PVF_Z_MIN + 40.0f * PVF_INTERVAL_WIDTH + PVF_INTERVAL_EXTENSION))
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
        if (i < 0 || i >= 40) continue;
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

// Exact softplus log(1 + exp(x)), branchless and overflow-safe. It used to
// return x for x > 0, dropping log(1 + exp(-x)) (up to log 2 near 0), so the
// FC-only histogram did not match PyTorch's softplus.
__device__ float pvfinder_softplus(float x) {
    return fmaxf(x, 0.0f) + logf(1.0f + expf(-fabsf(x)));
}

// Inline LeakyReLU and linear layer — runs in registers, no global writes.
__device__ __forceinline__ float pvfinder_leaky_relu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}

// w_stride defaults to 0, meaning "use in_f as the row stride" (every
// pre-existing call site's original behavior, unchanged). Passing a
// nonzero w_stride lets a caller read only the first in_f columns/out_f
// rows of a matrix whose REAL stored stride is wider than in_f -- the same
// "valid cuBLAS/kernel sub-block read into a wider real buffer" trick
// m_l6a_m already uses for L6A's GEMM, applied here to L1-L5's weight
// matrices for the m_l1_l5_hidden_width throughput probe.
__device__ __forceinline__ void pvfinder_linear_layer_reg(
    const float* __restrict__ x, float* __restrict__ y,
    const float* __restrict__ w, const float* __restrict__ b,
    int in_f, int out_f, int w_stride = 0)
{
    const int stride = (w_stride > 0) ? w_stride : in_f;
    for (int i = 0; i < out_f; ++i) {
        float sum = b[i];
        for (int j = 0; j < in_f; ++j) sum += w[i * stride + j] * x[j];
        y[i] = pvfinder_leaky_relu(sum);
    }
}

// ---------------------------------------------------------------------------
// CSR index builder kernel.
//
// Grid: (n_events)  blockDim: 256
//
// For each event, builds a CSR (compressed sparse row) representation that
// maps each interval to a contiguous range of track indices:
//
//   interval_start[ev * 42 + i]          = start offset in track_idx[]
//   interval_start[ev * 42 + 41]         = total entries (sentinel)
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
__global__ void pvfinder_build_csr_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    bool canonical_order)   // see m_canonical_track_order
{
    const unsigned event_number      = blockIdx.x;
    const unsigned thread_id         = threadIdx.x;
    const auto     velo_tracks_view  = parameters.dev_velo_tracks_view[event_number];
    const unsigned num_tracks        = velo_tracks_view.size();
    const unsigned event_track_offset = velo_tracks_view.offset();

    __shared__ int s_counts[40];   // histogram
    __shared__ int s_start[41];    // exclusive prefix sum → CSR start offsets
    __shared__ int s_cursor[40];   // per-interval fill cursors (advanced atomically)

    // Up to CACHE tracks and entries, pass 1 keeps each track's interval
    // assignment and z in shared memory, so the features are read once; the
    // scatter then goes to shared memory and, with canonical_order, each
    // interval's tracks are ranked there and written to global memory once.
    // Larger events (none in the 2024 minimum-bias sample: at most 661 tracks,
    // 756 entries) take the uncached path.
    constexpr int CACHE = 1024;
    __shared__ int s_trk[CACHE];     // -1 not selected, else n | iv0 << 8 | iv1 << 16
    __shared__ float s_trk_z[CACHE];
    __shared__ int s_ent[CACHE];     // scatter order: local track index per entry
    __shared__ float s_ent_z[CACHE]; // and its z
    const bool cached = num_tracks <= (unsigned) CACHE;

    for (int i = thread_id; i < 40; i += blockDim.x) s_counts[i] = 0;
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
        for (int i = 0; i < 40; ++i) {
            s_start[i]  = acc;
            s_cursor[i] = acc;
            acc += s_counts[i];
        }
        s_start[40] = acc;  // sentinel
    }
    __syncthreads();

    // Write interval_start[] to global memory
    int* g_start = parameters.dev_pvfinder_interval_start + event_number * 42;
    for (int i = thread_id; i <= 40; i += blockDim.x)
        g_start[i] = s_start[i];
    // index 41 = total track_idx entries for this event (= s_start[40])
    if (thread_id == 0) g_start[41] = s_start[40];

    // Pass 3 — scatter: local track indices per interval, in shared memory
    // when cached (the order is fixed below), else straight to track_idx[]
    int* g_idx = parameters.dev_pvfinder_track_idx + event_track_offset * 2;
    const int n_entries = s_start[40];
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
                g_idx[pos] = (int)i;   // local track index within event
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
    // scatter keeps the scatter order.
    if (in_shared) {
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
        for (int pos = thread_id; pos < n_entries; pos += blockDim.x) {
            const int packed = s_ent[pos], me = packed & 1023;
            if (!canonical_order) {
                g_idx[pos] = me;
                continue;
            }
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

// ---------------------------------------------------------------------------
// Interval-parallel FC+Aggregation kernel — CSR edition.
//
// Grid: (n_events, N_INTERVALS=40)  blockDim: 256
//
// Each block exclusively owns one interval. Uses the CSR index built by
// pvfinder_build_csr_kernel so the inner loop iterates over only the
// ~T/40 tracks belonging to this interval.
//
// Static shared mem: s_feat[L6A_WIDTH] + s_hist[100] (3.6 KB by default,
// L6A_WIDTH=800). This keeps occupancy high (many blocks resident per SM)
// which is the dominant factor on both SM 7.5 and SM 8.6.
//
// NOTE: L6A weight caching in shared memory was tried (69.2 KB dynamic
// smem) but regressed on SM 8.6 — the 69 KB smem drops blocks-per-SM
// from ~16 to 1, killing occupancy. The RTX 3090 L2 ($936 GB/s) handles
// the 64 KB weight matrix well enough without smem caching.
// ---------------------------------------------------------------------------
__global__ void pvfinder_fused_fc_aggregation_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights)
{
    // Standard 2D dispatch: blockIdx.x = event, blockIdx.y = interval.
    // Empty intervals (n_local == 0) early-return after reading the CSR count,
    // avoiding the shared-memory init and MLP loop entirely.
    const unsigned event_number = blockIdx.x;
    const unsigned interval     = blockIdx.y;
    const unsigned thread_id    = threadIdx.x;
    const unsigned warp_id      = thread_id / warpSize;
    const unsigned n_warps      = blockDim.x / warpSize;

    const auto velo_tracks_view       = parameters.dev_velo_tracks_view[event_number];
    const unsigned event_track_offset = velo_tracks_view.offset();

    // CSR pointers for this event
    const int* g_start  = parameters.dev_pvfinder_interval_start + event_number * 42;
    const int  iv_begin = g_start[interval];
    const int  iv_end   = g_start[interval + 1];
    const int  n_local  = iv_end - iv_begin;
    const int* g_idx    = parameters.dev_pvfinder_track_idx + event_track_offset * 2;

    // Early return for empty intervals: output is already zeroed by cudaMemsetAsync.
    // This saves the shared-memory init (~3 µs per block) and the MLP loop.
    if (n_local == 0) return;

    const float* w1  = dev_weights;
    const float* b1  = w1  + 180;
    const float* w2  = b1  + 20;
    const float* b2  = w2  + 400;
    const float* w3  = b2  + 20;
    const float* b3  = w3  + 400;
    const float* w4  = b3  + 20;
    const float* b4  = w4  + 400;
    const float* w5  = b4  + 20;
    const float* b5  = w5  + 400;
    const float* w6A = b5  + 20;
    const float* b6A = w6A + L6A_WEIGHT_FLOATS;

    __shared__ float s_feat[L6A_WIDTH];
    __shared__ float s_hist[100];

    for (int i = thread_id; i < (int)L6A_WIDTH; i += blockDim.x) s_feat[i] = 0.0f;
    for (int i = thread_id; i < 100; i += blockDim.x) s_hist[i] = 0.0f;
    __syncthreads();

    // Process tracks in batches of size warpSize (32) per warp.
    for (int i = warp_id * warpSize; i < n_local; i += n_warps * warpSize) {
        const int lane_id = thread_id % warpSize;
        const int track_idx_in_batch = i + lane_id;
        const bool valid_track = track_idx_in_batch < n_local;

        float x1[20], x2[20];
        
        // Thread-parallel L1-L5: each thread processes L1-L5 for a UNIQUE track.
        if (valid_track) {
            const int local_idx = g_idx[iv_begin + track_idx_in_batch];
            const unsigned gtidx = event_track_offset + (unsigned)local_idx;
            const float* feat = parameters.dev_pvfinder_track_features + gtidx * 9;
            float in[9];
            pvfinder_interval_input(feat, (int)interval, in);

            pvfinder_linear_layer_reg(in, x1, w1, b1, 9,  20);
            pvfinder_linear_layer_reg(x1,  x2, w2, b2, 20, 20);
            pvfinder_linear_layer_reg(x2,  x1, w3, b3, 20, 20);
            pvfinder_linear_layer_reg(x1,  x2, w4, b4, 20, 20);
            pvfinder_linear_layer_reg(x2,  x1, w5, b5, 20, 20);
        }

        // Warp-collaborative L6A: loop over the tracks in this warp's current batch.
        const int batch_size = min(warpSize, n_local - i);
        for (int t = 0; t < batch_size; ++t) {
            
            // Broadcast the target track's x1 array to all lanes in the warp
            float broadcasted_x1[20];
            for (int m = 0; m < 20; ++m) {
                broadcasted_x1[m] = __shfl_sync(0xffffffff, x1[m], t);
            }

            // All lanes collaboratively process L6A for track t
            for (int k = lane_id; k < 100; k += warpSize) {
                float chan_sum = 0.0f;
                for (int c = 0; c < (int)N_LATENT_CHANNELS; ++c) {
                    const int neuron = c * 100 + k;
                    float val = b6A[neuron];
                    for (int m = 0; m < 20; ++m) {
                        val += w6A[m * L6A_WIDTH + neuron] * broadcasted_x1[m];
                    }
                    val = pvfinder_leaky_relu(val);
                    chan_sum += val;
                    atomicAdd(&s_feat[neuron], val); // neuron is c*100 + k
                }
                atomicAdd(&s_hist[k], pvfinder_softplus(chan_sum));
            }
        }
    }
    __syncthreads();

    const float weight = n_local > 0 ? 1.0f / n_local : 1.0f;

    float* g_feat = parameters.dev_pvfinder_interval_features
                    + event_number * INTERVAL_FEATURES_STRIDE + interval * L6A_WIDTH;
    // The UNet input is the SUM over the interval's tracks, as in the
    // trained model (TrackIntervalsToKDE: y0 = sum over tracks); only the
    // FC-only histogram is averaged.
    for (int i = thread_id; i < (int)L6A_WIDTH; i += blockDim.x)
        g_feat[i] = s_feat[i];

    float* g_hist = parameters.dev_pvfinder_output_histogram
                    + event_number * 4000 + interval * 100;
    for (int i = thread_id; i < 100; i += blockDim.x)
        g_hist[i] = s_hist[i] * weight;
}

// ===========================================================================
// cuBLAS kernels (compiled only when ALLEN_WITH_CUBLAS is defined).
// ===========================================================================
#ifdef ALLEN_WITH_CUBLAS

// ---------------------------------------------------------------------------
// Kernel 1 — L1-L5 per track.
//
// Grid: ceil(T_chunk / 256)  blockDim: 256
//
// Each thread processes one CSR track entry for events in [chunk_start, chunk_end).
// Evaluates L1-L5 MLP in registers and writes the 20-float hidden state to
// dev_pvfinder_l5_output at the corresponding row.
//
// T_chunk is the total number of CSR track entries (boundary tracks counted
// twice) for the current chunk. Entry t corresponds to the t-th slot in the
// CSR track_idx array when scanned consecutively across the chunk's events.
//
// Strategy: binary-search the CSR interval_start array across the chunk
// to find (event, absolute track_idx_entry) for linear slot t.
//
// Simpler fast path: walk the CSR sentinel (index 41) for each event to
// find which event owns slot t, then look up the local track index. A
// linear scan rather than a binary search: neighboring threads' consecutive
// t values usually land in the same or an adjacent event, so a warp's
// iteration counts stay nearly uniform and the scan's simple, cached,
// sequential access pattern outperforms a binary search's data-dependent
// branching for the chunk sizes in use here.
// ---------------------------------------------------------------------------
__global__ void pvfinder_l1_to_l5_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights,
    unsigned chunk_start,       // first event index in this chunk
    unsigned chunk_end,         // exclusive: last event index + 1
    unsigned csr_offset,        // starting position in track_idx[] for chunk_start event
    unsigned T_chunk,           // total CSR entries in this chunk
    bool single_hidden_layer,   // throughput-ceiling probe: skip layers 2-5
                                 // (see m_fc_single_hidden_layer doc comment) -- NOT
                                 // physics-valid when true, timing only
    unsigned hidden_width,      // throughput-ceiling probe: use only the first
                                 // hidden_width of each layer's 20 real neurons (see
                                 // m_l1_l5_hidden_width doc comment) -- NOT physics-valid
                                 // when < 20, timing only
    const unsigned* __restrict__ chunk_col_offset,  // this chunk's cumulative per-event entry
                                 // counts (n_events_in_chunk + 1, starting at 0), or nullptr
    bool l5_bf16)               // write the hidden states as bfloat16 (see m_l6a_dtype)
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T_chunk) return;

    // Which event, and which of its CSR entries, this thread owns.
    unsigned ev, local_t;
    if (chunk_col_offset != nullptr) {
        // Binary search: the last event whose first entry is <= t.
        unsigned lo = 0, hi = chunk_end - chunk_start;   // chunk_col_offset[hi] == T_chunk > t
        while (hi - lo > 1) {
            const unsigned mid = (lo + hi) / 2;
            if (chunk_col_offset[mid] <= t) lo = mid; else hi = mid;
        }
        ev = chunk_start + lo;
        local_t = t - chunk_col_offset[lo];
    }
    else {
        // Without the offsets: walk the events' entry counts.
        ev = chunk_start;
        local_t = t;
        while (ev < chunk_end) {
            const unsigned n_entries = (unsigned)parameters.dev_pvfinder_interval_start[ev * 42 + 41];
            if (local_t < n_entries) break;
            local_t -= n_entries;
            ++ev;
        }
        if (ev >= chunk_end) return;
    }

    // Recover the actual track index within this event
    const int* g_start = parameters.dev_pvfinder_interval_start + ev * 42;
    const auto  velo_tracks_view  = parameters.dev_velo_tracks_view[ev];
    const unsigned event_track_offset = velo_tracks_view.offset();
    const int* g_idx = parameters.dev_pvfinder_track_idx + event_track_offset * 2;
    const int local_track = g_idx[local_t];
    const unsigned gtidx = event_track_offset + (unsigned)local_track;

    // The entry's interval: the last iv with g_start[iv] <= local_t (binary
    // search; empty intervals share their start with the next one, and the
    // last such start is the non-empty interval that owns the entry).
    int iv = 0;
    {
        int lo = 0, hi = 40;   // g_start[40] = entries in the event > local_t
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (g_start[mid] <= (int)local_t) lo = mid; else hi = mid;
        }
        iv = lo;
    }
    float in[9];
    pvfinder_interval_input(parameters.dev_pvfinder_track_features + gtidx * 9, iv, in);

    // Evaluate L1-L5 in registers
    const float* w1 = dev_weights;
    const float* b1 = w1 + 180;
    const float* w2 = b1 + 20;
    const float* b2 = w2 + 400;
    const float* w3 = b2 + 20;
    const float* b3 = w3 + 400;
    const float* w4 = b3 + 20;
    const float* b4 = w4 + 400;
    const float* w5 = b4 + 20;
    const float* b5 = w5 + 400;

    // hw <= 20 bounds how many of each layer's real neurons get
    // computed. Layer 1's stride is already in_f=9 (unaffected by hw, no
    // w_stride override needed); layers 2-5's real stored stride is 20
    // regardless of hw, so w_stride=20 is passed explicitly to avoid
    // misreading the weight matrices as if they were hw-wide (see
    // pvfinder_linear_layer_reg's doc comment). x1/x2 are zero-initialized
    // so neurons >= hw read back as 0 in the final output write below,
    // matching l6a_m's "untouched neurons are zero downstream" convention.
    const int hw = (int)hidden_width;
    float x1[20] = {0.0f}, x2[20] = {0.0f};
    pvfinder_linear_layer_reg(in, x1, w1, b1, 9,  hw);
    if (!single_hidden_layer) {
        pvfinder_linear_layer_reg(x1,  x2, w2, b2, hw, hw, 20);
        pvfinder_linear_layer_reg(x2,  x1, w3, b3, hw, hw, 20);
        pvfinder_linear_layer_reg(x1,  x2, w4, b4, hw, hw, 20);
        pvfinder_linear_layer_reg(x2,  x1, w5, b5, hw, hw, 20);
    }

    // Write x1[20] as row t of dev_pvfinder_l5_output [T_chunk × 20] row-major
    if (l5_bf16) {
        __nv_bfloat162* out = reinterpret_cast<__nv_bfloat162*>(static_cast<float*>(parameters.dev_pvfinder_l5_output))
                              + (unsigned long long)t * 10;
        for (int m = 0; m < 10; ++m) out[m] = __floats2bfloat162_rn(x1[2 * m], x1[2 * m + 1]);
        return;
    }
    float* out = parameters.dev_pvfinder_l5_output + (unsigned long long)t * 20;
    for (int m = 0; m < 20; ++m) out[m] = x1[m];
}

// ---------------------------------------------------------------------------
// Kernel 2 — Apply L6A bias + LeakyReLU in-place after cuBLAS GEMM.
//
// cuBLAS writes        dev_l6a_output [L6A_WIDTH × T_chunk]  (column-major)
// i.e. element (n, t) is at offset n + t*L6A_WIDTH  (n ∈ [0,L6A_WIDTH), t ∈ [0,T_chunk))
//
// Grid: ceil(T_chunk * L6A_WIDTH / 512)  blockDim: 512
// ---------------------------------------------------------------------------
// l6a_m: how many of L6A_WIDTH's neurons to actually process (see m_l6a_m
// doc comment in the header -- throughput testing only; L6A_WIDTH is the
// physics-valid default). Neurons >= l6a_m are left untouched (stale data
// from a prior chunk's GEMM, never read downstream since
// pvfinder_reduce_l6a_kernel is bounded the same way). idx is linear over
// [0, T_chunk*l6a_m) and must be decomposed into (n, t) and re-mapped to the
// buffer's true stride-L6A_WIDTH layout -- the physical buffer is always
// [L6A_WIDTH x T_chunk] regardless of l6a_m, so idx itself is NOT a valid
// flat offset once l6a_m != L6A_WIDTH.
__global__ void pvfinder_l6a_bias_relu_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ b6A,  // bias[L6A_WIDTH]
    unsigned T_chunk,
    unsigned l6a_m)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T_chunk * l6a_m) return;
    const unsigned n = idx % l6a_m;  // neuron index
    const unsigned t = idx / l6a_m;  // track/column index
    const unsigned long long off = (unsigned long long)n + (unsigned long long)t * (unsigned long long)L6A_WIDTH;
    float val = parameters.dev_pvfinder_l6a_output[off] + b6A[n];
    parameters.dev_pvfinder_l6a_output[off] = pvfinder_leaky_relu(val);
}

// ---------------------------------------------------------------------------
// Kernel 3 — Reduce L6A outputs into interval features and histogram.
//
// Grid: (B_chunk * 40)  blockDim: 128
//
// Each block handles one (event, interval) slot. The kernel maps blockIdx.x
// to (event, interval) via the CSR bounds — no DtoH copy needed. Blocks
// that correspond to empty intervals or out-of-range events early-return.
//
// dev_l6a_output is column-major [L6A_WIDTH × T_chunk]. For track at column t:
//   neuron n → dev_l6a_output[n + t*L6A_WIDTH]
// ---------------------------------------------------------------------------
// l6a_m: how many of L6A_WIDTH's neurons to actually accumulate (see m_l6a_m
// doc comment in the header -- throughput testing only). Bounding the
// per-track accumulation loop below is what actually removes work here,
// unlike the GEMM-only l6a_m test: this reduction (atomics-heavy, one
// iteration per track per neuron) is the dominant cost in the L6A block, not
// the GEMM. s_feat/s_hist and the output writes below stay sized at the full
// L6A_WIDTH/100 (downstream buffers are always that shape); neurons >= l6a_m
// simply never get a nonzero contribution.
//
// UseAtomic (see m_use_nonatomic_l6a_reduce doc comment in the header):
// the thread<->n mapping below (n = thread_id, thread_id+blockDim.x, ...) is
// identical on every iteration of the track loop, so a given s_feat[n] slot
// is written by exactly one thread for the block's entire lifetime -- no
// cross-thread race. UseAtomic=false trades atomicAdd for a plain +=, which
// should be equivalent given that invariant; kept as a compile-time template
// parameter (not a runtime branch) so the untested variant carries zero
// overhead relative to a hand-written non-atomic kernel. Ignored when
// WarpParallelTracks=true (that path always uses its own small, bounded
// combine step -- see below).
//
// WarpParallelTracks (see m_use_warp_parallel_reduce doc comment): when true,
// tracks are split round-robin across the block's warps instead of every
// thread processing one track at a time serially. Each warp accumulates its
// assigned tracks' contributions into private per-lane registers (no shared
// memory traffic during the track loop itself), then all warps combine their
// partial sums into s_feat via a bounded atomicAdd -- exactly N_WARPS=4
// atomics per neuron per block, independent of n_local, unlike the O(n_local)
// atomics UseAtomic controls in the serial path.
//
// FuseBiasRelu (see m_use_fused_bias_relu_reduce doc comment): when
// true, this kernel reads the RAW GEMM output (pvfinder_l6a_bias_relu_kernel
// is not launched at all in this mode) and applies bias+LeakyReLU inline,
// identical math to what that separate kernel used to write back in-place --
// b6A must be non-null in this mode.
//
// BUG FIX: this kernel used to take a csr_offset parameter ("offset into
// the global CSR that corresponds to chunk_start") and add it into the
// dev_pvfinder_l6a_output column index (col = csr_offset + ev_col_offset +
// t). dev_pvfinder_l6a_output is a chunk-relative buffer, reused across
// chunks -- cuBLAS always writes each chunk's GEMM output starting at
// column 0 (see the operator() call site; nothing offsets cublasSgemm's
// output pointer), and pvfinder_l6a_bias_relu_kernel (the epilogue kernel
// this one can replace via FuseBiasRelu) indexes the exact same buffer
// using only its own chunk-relative t in [0, T_chunk) -- no csr_offset at
// all. Adding a cumulative whole-batch csr_offset here was simply wrong:
// for any chunk after the first (i.e. any batch spanning more than one
// chunk -- the normal production case), this kernel was reading from the
// wrong column, silently returning incorrect physics results whenever the
// erroneous column still happened to land inside the buffer's bounds
// (which T_chunk_max's safety margin usually provided), and crashing with
// an illegal memory access once the cumulative offset grew large enough to
// exceed it (found via a large multi-chunk batch, n=500 at chunk_size=100
// -- 5 chunks -- compute-sanitizer pinpointed the exact out-of-bounds
// read). Every earlier correctness check used a small enough event count
// to never exceed a single chunk, so this bug went unexercised for a long
// time. Fixed by removing csr_offset from the column computation entirely
// (col = ev_col_offset + t, matching pvfinder_l6a_bias_relu_kernel's own
// indexing) and dropping the now-unused parameter.
//
// The per-(event,interval) processing logic below is factored into this
// helper so both the original one-block-per-slot dispatch and the
// grid-stride work-stealing dispatch (see the UseGridStride branch in
// pvfinder_reduce_l6a_kernel below) share identical accumulation logic and
// can't drift apart.
//
// ev_col_offset (the CSR column offset for this slot's event within the
// chunk) is computed only once n_local>0 is confirmed below, rather than
// unconditionally by every thread before the check -- avoids wasted work
// on empty slots and redundant (thread_id-independent) work on non-empty
// ones. PrecomputedOffset sources the value from an O(1) lookup into a
// host-precomputed per-chunk array instead of an O(events-in-chunk) serial
// CSR-sentinel walk (see m_use_precomputed_csr_offset's doc comment in
// PVFinderFCAggregation.cuh); a measured, real win under production-scale
// contention.
//
// active_channels bounds a DIFFERENT dimension than l6a_m. l6a_m bounds how many of L6A_WIDTH's flat
// neurons the GEMM/accumulation step touches; it never bounded the
// shared-memory zero-init, the softplus reduction's channel loop, or the
// output write-back below -- all three are hardcoded to the full
// L6A_WIDTH/N_LATENT_CHANNELS/L6A_WIDTH regardless of l6a_m, because they
// operate on the buffer's real shape (N_LATENT_CHANNELS channels x 100
// bins), not on "how many neurons are nonzero". This is still a
// within-this-build throughput probe: N_LATENT_CHANNELS itself is fixed at
// compile time (see its definition in PVFinderFCAggregation.cuh) to
// actually run a different latentChannels architecture -- active_channels
// only lets you simulate something narrower than that, still inside the
// same physical buffer. active_channels (default N_LATENT_CHANNELS =
// physics-valid) bounds exactly these three, in channel units (not raw
// neuron units): set active_channels = l6a_m/100 to represent the SAME
// hypothetical narrower architecture consistently across both throughput
// probes -- NOT physics-valid when active_channels < N_LATENT_CHANNELS,
// timing only.
template <bool UseAtomic, bool WarpParallelTracks, bool FuseBiasRelu,
          bool PrecomputedOffset>
__device__ __forceinline__ void pvfinder_reduce_l6a_process_slot(
    pvfinder_fc_aggregation_t::Parameters& parameters,
    unsigned chunk_start,
    unsigned event_number,
    unsigned interval,
    unsigned l6a_m,
    unsigned active_channels,
    const float* __restrict__ b6A,   // bias[L6A_WIDTH], only read when FuseBiasRelu
    const unsigned* __restrict__ event_col_offset,  // only read when PrecomputedOffset
    const int* __restrict__ slot_row,               // nullptr: dense rows (see m_skip_empty_intervals)
    int features_format,                            // 0 float32, 1 bfloat16, 2 bfloat16 channels last (see m_unet_input_dtype/layout)
    bool l6a_bf16)                                  // the GEMM output is bfloat16 (see m_l6a_dtype)
{
    const unsigned thread_id = threadIdx.x;
    const float* l6a_f32 = parameters.dev_pvfinder_l6a_output;
    const __nv_bfloat16* l6a_b16 = reinterpret_cast<const __nv_bfloat16*>(l6a_f32);
    auto load_l6a = [&](unsigned long long off) {
        return l6a_bf16 ? __bfloat162float(l6a_b16[off]) : l6a_f32[off];
    };
    const unsigned active_neurons = active_channels * 100u;
    const int* g_start  = parameters.dev_pvfinder_interval_start + event_number * 42;
    const int  iv_begin = g_start[interval];
    const int  iv_end   = g_start[interval + 1];
    const int  n_local  = iv_end - iv_begin;
    // Row of dev_pvfinder_interval_features this interval's features go to:
    // event * 40 + interval when dense, the host-assigned compact row
    // otherwise, where -1 means the UNet skips the interval and no features
    // are written (its histogram still is: that is FC's own output).
    long long row = (long long)event_number * N_INTERVALS + interval;
    if (slot_row != nullptr) row = slot_row[row];
    if (n_local == 0) {
        // This block still gets launched even for an empty slot, so rather
        // than rely on a separate whole-buffer cudaMemsetAsync to leave
        // correct zeros here, write them directly -- cudaMemsetAsync is a
        // real cost under production-scale multi-thread contention that a
        // single-thread profile understates.
        if (row >= 0 && features_format != 0) {   // zeros: same in either bfloat16 layout
            __nv_bfloat16* g_feat = reinterpret_cast<__nv_bfloat16*>(static_cast<float*>(parameters.dev_pvfinder_interval_features))
                                    + (unsigned long long)row * L6A_WIDTH;
            for (unsigned i = thread_id; i < active_neurons; i += blockDim.x) g_feat[i] = __float2bfloat16(0.0f);
        } else if (row >= 0) {
            float* g_feat = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
            for (unsigned i = thread_id; i < active_neurons; i += blockDim.x) g_feat[i] = 0.0f;
        }
        float* g_hist = parameters.dev_pvfinder_output_histogram
                        + event_number * 4000u + interval * 100u;
        for (int i = thread_id; i < 100; i += blockDim.x) g_hist[i] = 0.0f;
        return;
    }

    // Determine the CSR column offset for this event within the chunk. Every
    // thread in the block agrees on n_local (same event_number/interval per
    // block), so the whole block either returned together above or reaches
    // here together -- the __syncthreads() calls below are safe.
    unsigned ev_col_offset;
    if constexpr (PrecomputedOffset) {
        ev_col_offset = event_col_offset[event_number - chunk_start];
    } else {
        ev_col_offset = 0;
        for (unsigned e = chunk_start; e < event_number; ++e) {
            const int* g = parameters.dev_pvfinder_interval_start + e * 42;
            ev_col_offset += (unsigned)g[41];
        }
    }

    // Shared memory accumulators: s_feat[L6A_WIDTH] + s_hist[100]. Zero-init
    // must cover whatever the l6a_m-bounded accumulation step below touches,
    // so this is bounded by max(l6a_m, active_neurons), not active_neurons
    // alone -- if l6a_m were ever set wider than active_channels*100, entries
    // between the two would accumulate into (correctly zeroed) memory rather
    // than uninitialized shared memory. The documented, intended usage is
    // l6a_m == active_neurons; this bound is just a safety margin against
    // that invariant being violated, not a normal operating mode.
    const unsigned zero_init_bound = l6a_m > active_neurons ? l6a_m : active_neurons;
    __shared__ float s_feat[L6A_WIDTH];
    __shared__ float s_hist[100];
    for (unsigned i = thread_id; i < zero_init_bound; i += blockDim.x) s_feat[i] = 0.0f;
    for (int i = thread_id; i < 100; i += blockDim.x) s_hist[i] = 0.0f;
    __syncthreads();

    if constexpr (WarpParallelTracks) {
        // N_WARPS matches KERNEL3_BLOCK=128 (the only block size this kernel
        // is ever launched with) / warpSize=32. A wider block (more warps
        // cooperating per slot) was tried and measured as a net regression:
        // doubling block size roughly halves blocks resident per SM (same
        // total concurrent warp count either way), so there is no net
        // parallelism gain to fund the extra per-block sync/zero-init
        // overhead.
        constexpr unsigned N_WARPS = 4;
        constexpr unsigned MAX_PER_LANE = (L6A_WIDTH + 31u) / 32u;  // 25 by default

        const unsigned warp_id = thread_id / warpSize;
        const unsigned lane_id = thread_id % warpSize;

        float acc[MAX_PER_LANE];
#pragma unroll
        for (unsigned i = 0; i < MAX_PER_LANE; ++i) acc[i] = 0.0f;

        // Round-robin tracks across warps: warp w handles tracks
        // iv_begin+w, iv_begin+w+N_WARPS, ... -- concurrently with the other
        // warps in this block, unlike the serial-over-tracks path below.
        for (int t = iv_begin + (int)warp_id; t < iv_end; t += (int)N_WARPS) {
            const unsigned col = ev_col_offset + (unsigned)t;
#pragma unroll
            for (unsigned i = 0; i < MAX_PER_LANE; ++i) {
                const unsigned n = lane_id + i * warpSize;
                if (n < l6a_m) {
                    float val = load_l6a(
                        (unsigned long long)n + (unsigned long long)col * (unsigned long long)L6A_WIDTH);
                    if constexpr (FuseBiasRelu) {
                        val = pvfinder_leaky_relu(val + b6A[n]);
                    }
                    acc[i] += val;
                }
            }
        }

        // Combine: each lane's MAX_PER_LANE partial sums go into s_feat via a
        // bounded atomicAdd -- exactly N_WARPS contributions per neuron,
        // regardless of n_local (unlike the serial path's O(n_local) atomics).
#pragma unroll
        for (unsigned i = 0; i < MAX_PER_LANE; ++i) {
            const unsigned n = lane_id + i * warpSize;
            if (n < l6a_m) atomicAdd(&s_feat[n], acc[i]);
        }
    } else {
        // Accumulate: for each track t in [iv_begin, iv_end)
        // Column index in dev_l6a_output = ev_col_offset + t (chunk-relative --
        // see the BUG FIX note on the kernel's doc comment above)
        for (int t = iv_begin; t < iv_end; ++t) {
            const unsigned col = ev_col_offset + (unsigned)t;
            // Each thread sums a strided subset of the l6a_m active neurons.
            for (int n = thread_id; n < (int)l6a_m; n += blockDim.x) {
                float val = load_l6a((unsigned long long)n + (unsigned long long)col * (unsigned long long)L6A_WIDTH);
                if constexpr (FuseBiasRelu) {
                    val = pvfinder_leaky_relu(val + b6A[n]);
                }
                if constexpr (UseAtomic) {
                    atomicAdd(&s_feat[n], val);
                } else {
                    s_feat[n] += val;
                }
                // Accumulate softplus-reduced histogram bin (n / 8 maps 800 → 100)
                // s_hist[n % 100] is updated after the full s_feat loop below.
            }
        }
    }
    __syncthreads();

    // Reduce s_feat[L6A_WIDTH] → s_hist[100] via softplus of per-bin channel sums.
    // Bounded by active_channels, not N_LATENT_CHANNELS -- channels
    // >= active_channels are guaranteed zero (never accumulated into, per
    // the intended l6a_m == active_neurons usage), so skipping them is
    // exact, not approximate.
    for (int k = thread_id; k < 100; k += blockDim.x) {
        float chan_sum = 0.0f;
        for (unsigned c = 0; c < active_channels; ++c) chan_sum += s_feat[c * 100 + k];
        s_hist[k] = pvfinder_softplus(chan_sum);
    }
    __syncthreads();

    const float weight = 1.0f / n_local;

    // Only the active_neurons portion is written -- the buffer's
    // tail (positions >= active_neurons) is left whatever it already was
    // (stale/uninitialized), same as l6a_m's own "untouched neurons" design.
    // This is a pure throughput probe (FC-alone benchmarking never re-reads
    // this buffer through UNet), not something that would be valid if the
    // output were actually consumed downstream.
    if (row >= 0 && slot_row != nullptr && thread_id == 0)
        parameters.dev_pvfinder_row_slot[row] = (int) (event_number * N_INTERVALS + interval);
    if (row >= 0 && features_format != 0) {
        // Rounded once, here, for the UNet's BF16 path, which then reads the
        // features without a conversion pass; format 2 stores the row
        // channels last ([bin][channel]), the BF16 path's NWC layout.
        __nv_bfloat16* g_feat = reinterpret_cast<__nv_bfloat16*>(static_cast<float*>(parameters.dev_pvfinder_interval_features))
                                + (unsigned long long)row * L6A_WIDTH;
        for (unsigned i = thread_id; i < active_neurons; i += blockDim.x) {
            const unsigned dst = features_format == 2 ? (i % 100u) * N_LATENT_CHANNELS + i / 100u : i;
            g_feat[dst] = __float2bfloat16(s_feat[i]);
        }
    } else if (row >= 0) {
        float* g_feat = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
        // The UNet input is the SUM over the interval's tracks, as in the
        // trained model (TrackIntervalsToKDE: y0 = sum over tracks); only
        // the FC-only histogram below is averaged.
        for (unsigned i = thread_id; i < active_neurons; i += blockDim.x)
            g_feat[i] = s_feat[i];
    }

    float* g_hist = parameters.dev_pvfinder_output_histogram
                    + event_number * 4000u + interval * 100u;
    for (int i = thread_id; i < 100; i += blockDim.x)
        g_hist[i] = s_hist[i] * weight;
}

// ---------------------------------------------------------------------------
// Kernel 3 wrapper. UseGridStride (see m_use_grid_stride_reduce doc
// comment): when false, one block per (event, interval) slot (blockIdx.x
// decoded directly). When true, a FIXED number of blocks (sized to this
// GPU's actual occupancy ceiling for this kernel, computed once via the
// CUDA occupancy API -- see the dispatch site in operator()) work-steal
// over the same dense slot space via work_counter, an atomicAdd-claimed
// index broadcast through shared memory to the rest of each block. Every
// claimed slot (empty or not) is processed identically to the static-grid
// path via the same pvfinder_reduce_l6a_process_slot helper -- empty slots
// still self-zero exactly as in the static-grid path, so this doesn't need
// a full-buffer memset either. The __syncthreads() after each
// process_slot() call is required before the next iteration's shared-memory
// reuse (s_feat/s_hist zero-init, and s_work_item's next broadcast) --
// without it, threads that finish a slot's write-back loops earlier than
// others could race the next slot's shared-memory init.
//
// (A warp-scoped variant -- one warp handling an entire slot alone so a
// block's warps could each work a different slot concurrently instead of
// jointly waiting on one shared slot -- was tried and measured as a large
// regression: it trades away the dominant win of splitting one busy slot's
// track loop across all warps, for a track-count distribution where busy
// slots dominate, so the tradeoff loses badly.)
// ---------------------------------------------------------------------------
template <bool UseAtomic, bool WarpParallelTracks, bool FuseBiasRelu, bool UseGridStride,
          bool PrecomputedOffset = false>
__global__ void pvfinder_reduce_l6a_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    unsigned chunk_start,
    unsigned chunk_end,
    unsigned T_chunk,
    unsigned l6a_m,
    unsigned active_channels,        // throughput probe, see m_l6a_active_channels
    const float* __restrict__ b6A,   // bias[L6A_WIDTH], only read when FuseBiasRelu
    unsigned* work_counter,          // only used when UseGridStride
    // only used when PrecomputedOffset -- no default: this project's
    // global_function()/invoke_device_function() plumbing builds its
    // argument tuple explicitly and bypasses normal C++ default-argument
    // substitution, so every call site must pass this explicitly (nullptr
    // where unused).
    const unsigned* __restrict__ event_col_offset,
    const int* __restrict__ slot_row,   // nullptr: dense rows (see m_skip_empty_intervals)
    int features_format,                // 0 float32, 1 bfloat16, 2 bfloat16 channels last
    bool l6a_bf16)                      // the GEMM output is bfloat16 (see m_l6a_dtype)
{
    if constexpr (UseGridStride) {
        const unsigned total_work = (chunk_end - chunk_start) * 40u;
        __shared__ unsigned s_work_item;
        while (true) {
            if (threadIdx.x == 0) s_work_item = atomicAdd(work_counter, 1u);
            __syncthreads();
            const unsigned work_item = s_work_item;
            if (work_item >= total_work) break;
            const unsigned rel_ev    = work_item / 40u;
            const unsigned interval  = work_item % 40u;
            const unsigned event_number = chunk_start + rel_ev;
            pvfinder_reduce_l6a_process_slot<UseAtomic, WarpParallelTracks, FuseBiasRelu,
                PrecomputedOffset>(
                parameters, chunk_start, event_number, interval, l6a_m, active_channels, b6A,
                event_col_offset, slot_row, features_format, l6a_bf16);
            __syncthreads();
        }
    } else {
        // blockIdx.x indexes over (relative_event, interval) in this chunk.
        const unsigned n_chunk_events = chunk_end - chunk_start;
        const unsigned rel_ev = blockIdx.x / 40u;
        const unsigned interval = blockIdx.x % 40u;
        if (rel_ev >= n_chunk_events) return;
        const unsigned event_number = chunk_start + rel_ev;
        pvfinder_reduce_l6a_process_slot<UseAtomic, WarpParallelTracks, FuseBiasRelu,
            PrecomputedOffset>(
            parameters, chunk_start, event_number, interval, l6a_m, active_channels, b6A,
            event_col_offset, slot_row, features_format, l6a_bf16);
    }
}


// ---------------------------------------------------------------------------
// Fused FC aggregation (m_fc_fused): L1-L5, L6A, bias + LeakyReLU and the sum
// over each interval's tracks in one kernel, without the per-entry L6A output
// ever leaving the SM. Replaces pvfinder_l1_to_l5_kernel, the cuBLAS SGEMM and
// pvfinder_reduce_l6a_kernel, the whole slice in one launch.
//
// Blocks claim (event, interval) slots through work_counter. For a slot's
// tracks, in tiles of FUSED_TILE: the network inputs are staged in shared
// memory, L1-L5 run cooperatively (thread per (track, neuron), weights in
// shared memory), then each thread takes FUSED_NPT of the L6A_WIDTH output
// neurons, with their weights in registers, and adds leaky(W6A h + b6A) over
// the tile's tracks into its accumulators. The sum over tracks runs in a fixed
// order, so the result is deterministic (the atomic reduction is not).
// FP32 throughout: same arithmetic as the unfused path up to summation order.
// ---------------------------------------------------------------------------
constexpr unsigned FUSED_BLOCK = 128u;
constexpr unsigned FUSED_TILE = 32u;
constexpr unsigned FUSED_NPT = (L6A_WIDTH + FUSED_BLOCK - 1) / FUSED_BLOCK;
constexpr unsigned FC_L15_FLOATS = 1880u;   // layers 1-5 weights and biases

// Tensor-core L6A (l6a_dtype = bfloat16, sm_80 and newer): per tile, the
// product [tracks x 20] x [20 x L6A_WIDTH] as mma.sync m16n8k16 with bfloat16
// operands (K padded to 32) and float32 accumulation. Warp w owns output
// columns [w * FUSED_TC_TILES * 8, (w + 1) * FUSED_TC_TILES * 8); its W6A
// fragments stay in registers.
constexpr unsigned FUSED_WARPS = FUSED_BLOCK / 32u;
static_assert(L6A_WIDTH % 8u == 0, "tensor-core L6A works on 8-column tiles");
constexpr unsigned FUSED_TC_TILES = (L6A_WIDTH / 8u + FUSED_WARPS - 1) / FUSED_WARPS;
static_assert(FUSED_TILE == 32u, "tensor-core L6A takes two 16-row tiles of tracks");

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
    __trap();   // the host refuses l6a_dtype = bfloat16 below sm_80
#endif
}

template <bool TensorCores>
__global__ void __launch_bounds__(FUSED_BLOCK) pvfinder_fused_fc_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights,
    unsigned n_events,
    unsigned* work_counter,
    const int* __restrict__ slot_row,   // nullptr: dense rows
    int features_format)                // 0 float32, 1 bfloat16, 2 bfloat16 channels last
{
    __shared__ float s_w[FC_L15_FLOATS];
    __shared__ float s_in[FUSED_TILE][9];
    __shared__ float s_h[2][FUSED_TILE][20];
    __shared__ float s_feat[L6A_WIDTH];
    __shared__ float s_b6[TensorCores ? L6A_WIDTH : 1];
    __shared__ unsigned s_item;
    const unsigned tid = threadIdx.x;
    const unsigned warp = tid / 32u, lane = tid % 32u;
    const unsigned grp = lane >> 2, quad = lane & 3u;   // mma fragment coordinates

    for (unsigned i = tid; i < FC_L15_FLOATS; i += FUSED_BLOCK) s_w[i] = dev_weights[i];
    const float* w6A = dev_weights + FC_L15_FLOATS;
    const float* b6A = w6A + L6A_WEIGHT_FLOATS;
    // FP32: this thread's L6A neurons n = tid + k * FUSED_BLOCK, weights in
    // registers. Tensor cores: this warp's W6A fragments, B[k][n] = W6A[n][k],
    // k = ks * 16 + 2 * quad + 8 * h (+1), n = column tile * 8 + grp.
    constexpr unsigned NPT = TensorCores ? 1u : FUSED_NPT;
    constexpr unsigned TC_TILES = TensorCores ? FUSED_TC_TILES : 1u;
    float w6[NPT][20], b6[NPT];
    unsigned bfrag[TC_TILES][2][2];
    if constexpr (TensorCores) {
        for (unsigned i = tid; i < L6A_WIDTH; i += FUSED_BLOCK) s_b6[i] = b6A[i];
#pragma unroll
        for (unsigned j = 0; j < TC_TILES; ++j) {
            const unsigned n = (warp * TC_TILES + j) * 8u + grp;
#pragma unroll
            for (unsigned ks = 0; ks < 2; ++ks) {
#pragma unroll
                for (unsigned h = 0; h < 2; ++h) {
                    const unsigned k = ks * 16u + 2u * quad + 8u * h;
                    const float lo = (n < L6A_WIDTH && k < 20u) ? w6A[n * 20 + k] : 0.0f;
                    const float hi = (n < L6A_WIDTH && k + 1u < 20u) ? w6A[n * 20 + k + 1] : 0.0f;
                    bfrag[j][ks][h] = pvfinder_pack_bf16x2(lo, hi);
                }
            }
        }
    }
    else {
#pragma unroll
        for (unsigned k = 0; k < NPT; ++k) {
            const unsigned n = tid + k * FUSED_BLOCK;
#pragma unroll
            for (unsigned m = 0; m < 20; ++m) w6[k][m] = n < L6A_WIDTH ? w6A[n * 20 + m] : 0.0f;
            b6[k] = n < L6A_WIDTH ? b6A[n] : 0.0f;
        }
    }
    __syncthreads();
    const float* w1 = s_w;        const float* b1 = w1 + 180;
    const float* w2 = b1 + 20;    const float* b2 = w2 + 400;
    const float* w3 = b2 + 20;    const float* b3 = w3 + 400;
    const float* w4 = b3 + 20;    const float* b4 = w4 + 400;
    const float* w5 = b4 + 20;    const float* b5 = w5 + 400;
    const float* lw[4] = {w2, w3, w4, w5};
    const float* lb[4] = {b2, b3, b4, b5};

    const unsigned total = n_events * N_INTERVALS;
    while (true) {
        if (tid == 0) s_item = atomicAdd(work_counter, 1u);
        __syncthreads();
        const unsigned slot = s_item;
        __syncthreads();   // everyone has read s_item before thread 0 may overwrite it
        if (slot >= total) break;
        const unsigned ev = slot / N_INTERVALS, iv = slot % N_INTERVALS;
        const int* g_start = parameters.dev_pvfinder_interval_start + ev * 42;
        const int a = g_start[iv], n_local = g_start[iv + 1] - a;
        long long row = slot;
        if (slot_row != nullptr) row = slot_row[slot];
        float* g_hist = parameters.dev_pvfinder_output_histogram + ev * 4000u + iv * 100u;

        if (n_local == 0) {
            if (row >= 0) {
                if (features_format != 0) {
                    __nv_bfloat16* g = reinterpret_cast<__nv_bfloat16*>(static_cast<float*>(parameters.dev_pvfinder_interval_features)) + (unsigned long long)row * L6A_WIDTH;
                    for (unsigned i = tid; i < L6A_WIDTH; i += FUSED_BLOCK) g[i] = __float2bfloat16(0.0f);
                } else {
                    float* g = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
                    for (unsigned i = tid; i < L6A_WIDTH; i += FUSED_BLOCK) g[i] = 0.0f;
                }
            }
            for (unsigned i = tid; i < 100; i += FUSED_BLOCK) g_hist[i] = 0.0f;
            continue;
        }

        const auto tracks_view = parameters.dev_velo_tracks_view[ev];
        const unsigned track_offset = tracks_view.offset();
        const int* g_idx = parameters.dev_pvfinder_track_idx + track_offset * 2 + a;
        float acc[NPT];
#pragma unroll
        for (unsigned k = 0; k < NPT; ++k) acc[k] = 0.0f;
        if constexpr (TensorCores) {
            // Tensor-core tiles add straight into s_feat, each column owned by one lane.
            for (unsigned i = tid; i < L6A_WIDTH; i += FUSED_BLOCK) s_feat[i] = 0.0f;
        }

        for (int t0 = 0; t0 < n_local; t0 += (int) FUSED_TILE) {
            const unsigned nt = min((unsigned) (n_local - t0), FUSED_TILE);
            // Network inputs in the training order (see pvfinder_interval_input).
            for (unsigned i = tid; i < nt; i += FUSED_BLOCK) {
                const float* f = parameters.dev_pvfinder_track_features + (track_offset + (unsigned) g_idx[t0 + i]) * 9;
                pvfinder_interval_input(f, (int) iv, s_in[i]);
            }
            __syncthreads();
            // Layer 1 (9 -> 20)
            for (unsigned q = tid; q < nt * 20; q += FUSED_BLOCK) {
                const unsigned t = q / 20, j = q % 20;
                float v = b1[j];
#pragma unroll
                for (unsigned i = 0; i < 9; ++i) v += w1[j * 9 + i] * s_in[t][i];
                s_h[0][t][j] = pvfinder_leaky_relu(v);
            }
            __syncthreads();
            // Layers 2-5 (20 -> 20), ping-pong
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const float* hin = &s_h[l & 1][0][0];
                float* hout = &s_h[(l + 1) & 1][0][0];
                for (unsigned q = tid; q < nt * 20; q += FUSED_BLOCK) {
                    const unsigned t = q / 20, j = q % 20;
                    float v = lb[l][j];
#pragma unroll
                    for (unsigned i = 0; i < 20; ++i) v += lw[l][j * 20 + i] * hin[t * 20 + i];
                    hout[t * 20 + j] = pvfinder_leaky_relu(v);
                }
                __syncthreads();
            }
            // L6A + bias + LeakyReLU, summed over the tile's tracks (layer 5
            // output is in s_h[0] after four ping-pongs).
            if constexpr (TensorCores) {
                // A fragments (rows = tracks, k = layer-5 neuron), rows past
                // nt as zeros; row r's result depends on row r only.
                const unsigned n_mt = (nt + 15u) / 16u;
                unsigned afrag[2][2][4];
#pragma unroll
                for (unsigned mt = 0; mt < 2; ++mt) {
#pragma unroll
                    for (unsigned ks = 0; ks < 2; ++ks) {
#pragma unroll
                        for (unsigned q = 0; q < 4; ++q) {
                            const unsigned r = mt * 16u + grp + 8u * (q & 1u);
                            const unsigned k = ks * 16u + 2u * quad + 8u * (q >> 1);
                            const float lo = (r < nt && k < 20u) ? s_h[0][r][k] : 0.0f;
                            const float hi = (r < nt && k + 1u < 20u) ? s_h[0][r][k + 1] : 0.0f;
                            afrag[mt][ks][q] = pvfinder_pack_bf16x2(lo, hi);
                        }
                    }
                }
#pragma unroll
                for (unsigned j = 0; j < TC_TILES; ++j) {
                    const unsigned col = (warp * TC_TILES + j) * 8u + 2u * quad;
                    if ((warp * TC_TILES + j) * 8u >= L6A_WIDTH) break;   // warp-uniform
                    const float bias0 = s_b6[col], bias1 = s_b6[col + 1];
                    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
                    for (unsigned mt = 0; mt < 2; ++mt) {
                        if (mt >= n_mt) break;   // block-uniform
                        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        pvfinder_mma_bf16_16816(d, afrag[mt][0], bfrag[j][0]);
                        pvfinder_mma_bf16_16816(d, afrag[mt][1], bfrag[j][1]);
                        const unsigned r0 = mt * 16u + grp, r1 = r0 + 8u;
                        if (r0 < nt) { s0 += pvfinder_leaky_relu(d[0] + bias0); s1 += pvfinder_leaky_relu(d[1] + bias1); }
                        if (r1 < nt) { s0 += pvfinder_leaky_relu(d[2] + bias0); s1 += pvfinder_leaky_relu(d[3] + bias1); }
                    }
                    // Sum over the 8 row groups (fixed order: deterministic).
#pragma unroll
                    for (unsigned off = 4; off < 32; off <<= 1) {
                        s0 += __shfl_xor_sync(0xffffffffu, s0, off);
                        s1 += __shfl_xor_sync(0xffffffffu, s1, off);
                    }
                    if (grp == 0) { s_feat[col] += s0; s_feat[col + 1] += s1; }
                }
            }
            else {
                for (unsigned t = 0; t < nt; ++t) {
                    float h[20];
#pragma unroll
                    for (unsigned m = 0; m < 20; ++m) h[m] = s_h[0][t][m];
#pragma unroll
                    for (unsigned k = 0; k < NPT; ++k) {
                        float v = b6[k];
#pragma unroll
                        for (unsigned m = 0; m < 20; ++m) v += w6[k][m] * h[m];
                        acc[k] += pvfinder_leaky_relu(v);
                    }
                }
            }
            __syncthreads();   // s_in / s_h are reused by the next tile
        }

        // Interval features (the sum over tracks) and the FC histogram.
        if constexpr (!TensorCores) {
#pragma unroll
            for (unsigned k = 0; k < NPT; ++k) {
                const unsigned n = tid + k * FUSED_BLOCK;
                if (n < L6A_WIDTH) s_feat[n] = acc[k];
            }
        }
        __syncthreads();
        if (row >= 0) {
            if (slot_row != nullptr && tid == 0) parameters.dev_pvfinder_row_slot[row] = (int) slot;
            if (features_format != 0) {
                __nv_bfloat16* g = reinterpret_cast<__nv_bfloat16*>(static_cast<float*>(parameters.dev_pvfinder_interval_features)) + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = tid; i < L6A_WIDTH; i += FUSED_BLOCK) {
                    const unsigned dst = features_format == 2 ? (i % 100u) * N_LATENT_CHANNELS + i / 100u : i;
                    g[dst] = __float2bfloat16(s_feat[i]);
                }
            } else {
                float* g = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = tid; i < L6A_WIDTH; i += FUSED_BLOCK) g[i] = s_feat[i];
            }
        }
        const float weight = 1.0f / n_local;
        for (unsigned bin = tid; bin < 100; bin += FUSED_BLOCK) {
            float chan_sum = 0.0f;
            for (unsigned c = 0; c < N_LATENT_CHANNELS; ++c) chan_sum += s_feat[c * 100 + bin];
            g_hist[bin] = pvfinder_softplus(chan_sum) * weight;
        }
        __syncthreads();   // s_feat is reused by the next slot
    }
}

// ---------------------------------------------------------------------------
// Warp-per-slot fused FC, FP32 (fc_fused with fc_fused_per_warp, the
// default): the same computation as pvfinder_fused_fc_kernel<false>, but each
// warp owns a slot, so nothing waits on a block-wide barrier. Lane t takes
// track t of a 32-track tile through L1-L5 in registers (weights in shared
// memory, read as broadcasts); the layer-5 outputs go through the warp's
// staging area to L6A, one lane per output column with W6A transposed in
// shared memory. Bit-identical to the block kernel: same sums in the same
// order. The BF16 tensor-core variant is pvfinder_fused_fc_tc_kernel.
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
    const int* __restrict__ slot_row,   // nullptr: dense rows
    int features_format,                // 0 float32, 1 bfloat16, 2 bfloat16 channels last
    const uint4* __restrict__ slot_items,   // {slot, first entry, entries, row} per slot to do, or nullptr (see m_fc_largest_first)
    unsigned n_items,
    bool write_histogram)               // see m_write_histogram
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

    const unsigned total = slot_items != nullptr ? n_items : n_events * N_INTERVALS;
    while (true) {
        unsigned item = 0;
        if (lane == 0) item = atomicAdd(work_counter, 1u);
        item = __shfl_sync(0xffffffffu, item, 0);
        if (item >= total) break;
        unsigned slot;
        int a, n_local;
        long long row;
        if (slot_items != nullptr) {
            // Everything about the slot in one load (built on the host from the CSR).
            const uint4 it = slot_items[item];
            slot = it.x;
            a = (int) (it.y & 0x7ffu);
            n_local = (int) (it.z & 0xfffu);
            row = (int) it.w;
        }
        else {
            slot = item;
            const int* g_start = parameters.dev_pvfinder_interval_start + (slot / N_INTERVALS) * 42;
            a = g_start[slot % N_INTERVALS];
            n_local = g_start[slot % N_INTERVALS + 1] - a;
            row = slot;
            if (slot_row != nullptr) row = slot_row[slot];
        }
        const unsigned ev = slot / N_INTERVALS, iv = slot % N_INTERVALS;
        float* g_hist = parameters.dev_pvfinder_output_histogram + ev * 4000u + iv * 100u;

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
            if (write_histogram) for (unsigned i = lane; i < 100u; i += 32u) g_hist[i] = 0.0f;
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
            if (slot_row != nullptr && lane == 0) parameters.dev_pvfinder_row_slot[row] = (int) slot;
            if (features_format != 0) {
                __nv_bfloat16* g = reinterpret_cast<__nv_bfloat16*>(static_cast<float*>(parameters.dev_pvfinder_interval_features)) + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = lane; i < L6A_WIDTH; i += 32u) {
                    const unsigned dst = features_format == 2 ? (i % 100u) * N_LATENT_CHANNELS + i / 100u : i;
                    g[dst] = __float2bfloat16(s_feat[i]);
                }
            } else {
                float* g = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = lane; i < L6A_WIDTH; i += 32u) g[i] = s_feat[i];
            }
        }
        if (write_histogram) {
            const float weight = 1.0f / n_local;
            for (unsigned bin = lane; bin < 100u; bin += 32u) {
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
//  - Layers 2-5 with HiddenTC (fc_hidden_dtype = bfloat16): a chain of
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
// finish adds them up in chunk order (deterministic). Work item fields:
// y = first CSR entry (11 bits) | first partial row << 11,
// z = entries (12 bits) | chunk (6) << 12 | chunks (6) << 18.
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

template <bool HiddenTC>
__global__ void __launch_bounds__(FT_THREADS, 2) pvfinder_fused_fc_tc_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights,
    unsigned n_events,
    unsigned* work_counter,
    const int* __restrict__ slot_row,   // nullptr: dense rows
    int features_format,                // 0 float32, 1 bfloat16, 2 bfloat16 channels last
    const uint4* __restrict__ slot_items,   // {slot, first entry, entries, row} per slot to do, or nullptr (see m_fc_largest_first)
    unsigned n_items,
    bool write_histogram,               // see m_write_histogram
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

    const unsigned total = slot_items != nullptr ? n_items : n_events * N_INTERVALS;
    while (true) {
        unsigned item = 0;
        if (lane == 0) item = atomicAdd(work_counter, 1u);
        item = __shfl_sync(0xffffffffu, item, 0);
        if (item >= total) break;
        unsigned slot, n_chunks = 1, chunk = 0, pbase = 0;
        int a, n_local;
        long long row;
        if (slot_items != nullptr) {
            // Everything about the work item in one load (built on the host from the CSR).
            const uint4 it = slot_items[item];
            slot = it.x;
            a = (int) (it.y & 0x7ffu);
            pbase = it.y >> 11;
            n_local = (int) (it.z & 0xfffu);
            chunk = (it.z >> 12) & 63u;
            n_chunks = (it.z >> 18) & 63u;
            row = (int) it.w;
        }
        else {
            slot = item;
            const int* g_start = parameters.dev_pvfinder_interval_start + (slot / N_INTERVALS) * 42;
            a = g_start[slot % N_INTERVALS];
            n_local = g_start[slot % N_INTERVALS + 1] - a;
            row = slot;
            if (slot_row != nullptr) row = slot_row[slot];
        }
        const unsigned ev = slot / N_INTERVALS, iv = slot % N_INTERVALS;
        float* g_hist = parameters.dev_pvfinder_output_histogram + ev * 4000u + iv * 100u;

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
            if (write_histogram) for (unsigned i = lane; i < 100u; i += 32u) g_hist[i] = 0.0f;
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
            // Layer 1 (and, without HiddenTC, layers 2-5) for this lane's track.
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
                if constexpr (!HiddenTC) {
                    float x2[20];
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const float* w = dev_weights + 200u + l * 420u;
                        const float* b = w + 400u;
                        float* hin = (l & 1) ? x2 : x1;
                        float* hout = (l & 1) ? x1 : x2;
#pragma unroll
                        for (unsigned j = 0; j < 20; ++j) {
                            float v = __ldg(b + j);
#pragma unroll
                            for (unsigned i = 0; i < 20; ++i) v += __ldg(w + j * 20 + i) * hin[i];
                            hout[j] = pvfinder_leaky_relu(v);
                        }
                    }
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
            if constexpr (HiddenTC) {
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
                const int* g_start = parameters.dev_pvfinder_interval_start + ev * 42;
                n_local = g_start[iv + 1] - g_start[iv];
            }
        }
        __syncwarp();

        if (row >= 0) {
            if (slot_row != nullptr && lane == 0) parameters.dev_pvfinder_row_slot[row] = (int) slot;
            if (features_format != 0) {
                __nv_bfloat16* g = reinterpret_cast<__nv_bfloat16*>(static_cast<float*>(parameters.dev_pvfinder_interval_features)) + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = lane; i < L6A_WIDTH; i += 32u) {
                    const unsigned dst = features_format == 2 ? (i % 100u) * N_LATENT_CHANNELS + i / 100u : i;
                    g[dst] = __float2bfloat16(s_feat[i]);
                }
            } else {
                float* g = parameters.dev_pvfinder_interval_features + (unsigned long long)row * L6A_WIDTH;
                for (unsigned i = lane; i < L6A_WIDTH; i += 32u) g[i] = s_feat[i];
            }
        }
        if (write_histogram) {
            const float weight = 1.0f / n_local;
            for (unsigned bin = lane; bin < 100u; bin += 32u) {
                float chan_sum = 0.0f;
                for (unsigned c = 0; c < N_LATENT_CHANNELS; ++c) chan_sum += s_feat[c * 100 + bin];
                g_hist[bin] = pvfinder_softplus(chan_sum) * weight;
            }
        }
        __syncwarp();   // s_feat (s_h) is rewritten by the next slot
    }
}

#endif  // ALLEN_WITH_CUBLAS

void pvfinder_fc_aggregation_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned total_events = first<host_number_of_events_t>(arguments);
    const unsigned unet_batch_events = m_unet_batch_events.value();
    if (unet_batch_events == 0) {
        throw std::runtime_error("pvfinder_fc_aggregation: unet_batch_events must be >= 1");
    }
    const unsigned padded_events = ((total_events + unet_batch_events - 1) / unet_batch_events) * unet_batch_events;
    const unsigned total_tracks = first<host_number_of_reconstructed_velo_tracks_t>(arguments);
    set_size<dev_pvfinder_output_histogram_t>  (arguments, total_events * 4000);
    set_size<dev_pvfinder_interval_features_t> (arguments, padded_events * INTERVAL_FEATURES_STRIDE);
    set_size<host_pvfinder_unet_rows_t>(arguments, 4);
    #ifdef ALLEN_WITH_CUBLAS
    set_size<dev_pvfinder_slot_row_t>(arguments, m_skip_empty_intervals.value() ? total_events * N_INTERVALS : 0u);
    set_size<dev_pvfinder_row_slot_t>(arguments, m_skip_empty_intervals.value() ? total_events * N_INTERVALS : 0u);
    // Work list (uint4 per item) and, for split slots, partial sums and arrival counters.
    const bool work_list = m_fc_fused.value() && m_fc_fused_per_warp.value() && m_fc_largest_first.value();
    const unsigned max_entries = total_tracks * 2u;
    const unsigned max_partial = max_entries * 2u / FT_CHUNK + 1u;
    set_size<dev_pvfinder_slot_order_t>(arguments,
        work_list ? (total_events * N_INTERVALS + max_entries / FT_CHUNK + 1u) * 4u : 0u);
    set_size<dev_pvfinder_fc_partial_t>(arguments, work_list ? max_partial * L6A_WIDTH : 0u);
    set_size<dev_pvfinder_fc_arrive_t>(arguments, work_list ? max_partial : 0u);
#else
    set_size<dev_pvfinder_slot_row_t>(arguments, 0u);
    set_size<dev_pvfinder_row_slot_t>(arguments, 0u);
    set_size<dev_pvfinder_slot_order_t>(arguments, 0u);
    set_size<dev_pvfinder_fc_partial_t>(arguments, 0u);
    set_size<dev_pvfinder_fc_arrive_t>(arguments, 0u);
#endif
    // CSR index buffers
    set_size<dev_pvfinder_interval_start_t>(arguments, total_events * 42);
    set_size<dev_pvfinder_track_idx_t>     (arguments, total_tracks  * 2);
    set_size<dev_pvfinder_track_features_t>(arguments, total_tracks * 9);
#ifdef ALLEN_WITH_CUBLAS
    // intermediate buffers: sized for one B_CHUNK=20-event chunk, reused per chunk.
    //
    // IMPORTANT: Allen calls set_arguments_size() during Scheduler::Scheduler() init
    // BEFORE any events are loaded, so total_events may be 0.  We must never divide by
    // total_events directly.  Use a safe upper-bound of MAX_TRACKS_PER_EVENT instead.
    // fc_chunk_size is a runtime property, not a hardcoded constant, so
    // buffer sizing must track whatever value it's set to (same value
    // operator() chunks by, below).
    const unsigned B_CHUNK = m_fc_chunk_size.value();
    // T_chunk_max sizing: sizing this buffer from *average* tracks/event
    // across the whole batch with zero safety margin is NOT safe for any
    // *individual* chunk's actual entry count -- a real illegal-memory-access
    // crash under exactly that sizing motivated the current approach.
    // Assuming every event in a chunk simultaneously hits the absolute
    // per-event worst case is safe but wastefully over-allocates (real
    // per-chunk entry totals concentrate tightly around their mean, as
    // expected for a sum of ~independent per-event contributions, so that
    // worst case essentially never occurs across a whole chunk at once).
    // m_safe_avg_entries_per_event is an empirically calibrated margin
    // (against a large real-event sample) exposed as a runtime property
    // (rather than a compile-time constant) specifically so a tighter
    // candidate value can be tested without a rebuild -- see its own doc
    // comment before changing it from the default.
    const unsigned T_chunk_max = m_safe_avg_entries_per_event.value() * B_CHUNK;
    // dev_l5_output: [T_chunk_max × 20]  row-major — L1-L5 hidden states
    // The fused FC keeps these on chip: no buffers.
    const bool fused = m_fc_fused.value();
    // bfloat16 GEMM operands and output (m_l6a_dtype) take half the floats.
    const unsigned l6a_div = m_l6a_dtype.value() == "bfloat16" ? 2u : 1u;   // "auto" keeps the unfused path float32
    set_size<dev_pvfinder_l5_output_t> (arguments, fused ? 0u : T_chunk_max * 20u / l6a_div);
    // dev_l6a_output: [L6A_WIDTH × T_chunk_max]  column-major — raw L6A GEMM output (~24 MB at L6A_WIDTH=800)
    set_size<dev_pvfinder_l6a_output_t>(arguments, fused ? 0u : L6A_WIDTH * T_chunk_max / l6a_div);
    // Single-element atomic work counter for grid-stride reduce.
    set_size<dev_pvfinder_reduce_work_counter_t>(arguments, 1u);
    // Per-chunk cumulative CSR column offsets, B_CHUNK+1 entries.
    // One (B_CHUNK + 1)-entry block per chunk, all uploaded at once (see operator()).
    const unsigned n_chunks = total_events > 0 ? (total_events + B_CHUNK - 1) / B_CHUNK : 1u;
    set_size<dev_pvfinder_event_col_offset_t>(arguments, n_chunks * (B_CHUNK + 1u));
#else
    set_size<dev_pvfinder_l5_output_t> (arguments, 0u);
    set_size<dev_pvfinder_l6a_output_t>(arguments, 0u);
    set_size<dev_pvfinder_reduce_work_counter_t>(arguments, 0u);
    set_size<dev_pvfinder_event_col_offset_t>(arguments, 0u);
#endif  // ALLEN_WITH_CUBLAS
}


void pvfinder_fc_aggregation_t::update(const Constants& constants) const { updateCommon(constants); }

void pvfinder_fc_aggregation_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    static std::once_flag flag;
    const std::string weight_file_path = m_weight_file.value();
    if (weight_file_path.empty()) {
        throw std::runtime_error(
            "pvfinder_fc_aggregation: weight_file is not set. Produce weights with the repository's "
            "weights/ pipeline (make -C weights verify MODEL=<name>) and generate the sequence "
            "configuration with PVFINDER_WEIGHTS_DIR pointing at them (make -C weights env MODEL=<name>).");
    }
    std::call_once(flag, [&weight_file_path]() {
        if (!PVFinder::WeightRegistry::instance().contains("fc_weights")) {
            std::string path = weight_file_path;
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f.is_open()) {
                throw std::runtime_error("Cannot open " + path);
            }
            const size_t bytes = static_cast<size_t>(f.tellg());
            f.seekg(0);
            std::vector<char> host_buf(bytes);
            f.read(host_buf.data(), bytes);

            // Validate the file size, then (non-cuBLAS builds only, see below)
            // transpose L6A weights from [l6a_rows][20] to [20][l6a_rows].
            // Offset to w6A is: 180+20 + 400+20 + 400+20 + 400+20 + 400+20 = 1880 floats
            // (layer1: 9*20+20=200; layer2-5: (20*20+20)*4=1680; fixed regardless
            // of latentChannels, only layer6A's own size varies with it).
            //
            // l6a_rows is intentionally NOT inferred from the file's own byte
            // size: an inferred value could silently disagree with L6A_WIDTH,
            // which every other kernel/buffer in this file derives at compile
            // time from N_LATENT_CHANNELS -- reading/writing as if the file
            // had a different row count than the build expects would
            // corrupt/misalign everything downstream, even with a
            // byte-count-correct transpose (an earlier version of this loader
            // hardcoded the row count, which heap-corrupted on a
            // differently-sized weight file: the transpose below overflowed
            // host_buf, corrupting the heap and crashing later at an unrelated
            // free() with a symptom that looked unrelated to its actual
            // cause). Instead, this loader validates that the loaded file's
            // size matches this build's L6A_WIDTH exactly, and throws a clear
            // error naming the mismatch otherwise -- a mismatch here means
            // this build's --unet-batch-channels doesn't match the weight
            // file's latentChannels; rebuild to match, or use a matching
            // weight file, rather than silently running an inconsistent pair.
            constexpr size_t kFixedFloats = 1880;      // layers 1-5, always this size
            constexpr size_t kFloatsPerL6ARow = 21;    // 20 weight + 1 bias, per row
            constexpr size_t kExpectedL6ARows = L6A_WIDTH;
            constexpr size_t kExpectedTotalFloats = kFixedFloats + kExpectedL6ARows * kFloatsPerL6ARow;
            const size_t total_floats = bytes / sizeof(float);
            if (total_floats != kExpectedTotalFloats) {
                throw std::runtime_error(
                    "fc_weights file " + path + " has " + std::to_string(total_floats) +
                    " floats, but this build expects " + std::to_string(kExpectedTotalFloats) +
                    " (fixed layer1-5 block of " + std::to_string(kFixedFloats) +
                    " floats + " + std::to_string(kExpectedL6ARows) + " L6A rows of " +
                    std::to_string(kFloatsPerL6ARow) + " floats each, i.e. N_LATENT_CHANNELS=" +
                    std::to_string(N_LATENT_CHANNELS) + "). This usually means the weight "
                    "file's latentChannels doesn't match this build's "
                    "PVFINDER_UNET_N_BATCH_CHANNELS (--unet-batch-channels) -- rebuild to "
                    "match the weight file, or use a weight file matching this build.");
            }
#ifndef ALLEN_WITH_CUBLAS
            // The file stores W6A row-major [L6A_WIDTH x 20] (PyTorch's own
            // layout, as written by weights/scripts/convert.py). Only the
            // non-cuBLAS fallback kernel wants it transposed: it indexes
            // w6A[m * L6A_WIDTH + neuron]. The cuBLAS SGEMM below
            // (CUBLAS_OP_T, lda=20) reads the row-major file layout directly;
            // transposing for it as well scrambles layer 6A without any size
            // error (weights/scripts/validate_fc.py detects this).
            const size_t l6a_rows = kExpectedL6ARows;
            const size_t l6a_weight_floats = l6a_rows * 20;

            float* floats = reinterpret_cast<float*>(host_buf.data());
            std::vector<float> w6A_transposed(l6a_weight_floats);
            for (size_t r = 0; r < l6a_rows; ++r) {
                for (int c = 0; c < 20; ++c) {
                    w6A_transposed[c * l6a_rows + r] = floats[kFixedFloats + r * 20 + c];
                }
            }
            std::memcpy(floats + kFixedFloats, w6A_transposed.data(), l6a_weight_floats * sizeof(float));
#endif

            PVFinder::WeightRegistry::instance().load_from_buffer(
                "fc_weights", host_buf.data(), bytes);
#ifdef ALLEN_WITH_CUBLAS
            // Rounded copy of W6A (row-major, as in the file) for the
            // bfloat16 GEMM (see m_l6a_dtype); 32 KB, unused otherwise.
            {
                const float* w6A_host = reinterpret_cast<const float*>(host_buf.data()) + kFixedFloats;
                std::vector<__nv_bfloat16> w6A_bf16(L6A_WEIGHT_FLOATS);
                for (size_t i = 0; i < L6A_WEIGHT_FLOATS; ++i) w6A_bf16[i] = __float2bfloat16(w6A_host[i]);
                PVFinder::WeightRegistry::instance().load_from_buffer(
                    "fc_w6a_bf16", w6A_bf16.data(), w6A_bf16.size() * sizeof(__nv_bfloat16));
            }
#endif
        }
    });
    const float* dev_weights = PVFinder::WeightRegistry::instance().get<float>("fc_weights");

    const unsigned n_events = first<host_number_of_events_t>(arguments);

    // -----------------------------------------------------------------------
    // Step 1: Build CSR index (same as before).
    // -----------------------------------------------------------------------
    global_function(pvfinder_build_csr_kernel)(
        dim3(n_events), m_block_dim, context)(
        arguments, m_canonical_track_order.value());

    // -----------------------------------------------------------------------
    // Step 2: Zero the interval feature and histogram output buffers.
    // (Same in both cuBLAS and non-cuBLAS paths — correctness guarantee for
    // empty intervals, free on-stream cost.)
    // -----------------------------------------------------------------------
    const unsigned unet_batch_events = m_unet_batch_events.value();
    const unsigned padded_events = ((n_events + unet_batch_events - 1) / unet_batch_events) * unet_batch_events;
#ifdef ALLEN_WITH_CUBLAS
    const bool compact = m_skip_empty_intervals.value();
#else
    // Compact rows are built from the cuBLAS path's host CSR readback; the
    // fallback kernel always writes dense rows.
    const bool compact = false;
#endif
    // Storage type of the interval features for the UNet (see m_unet_input_dtype).
    const std::string& input_dtype = m_unet_input_dtype.value();
    if (input_dtype != "float32" && input_dtype != "bfloat16") {
        throw std::runtime_error("pvfinder_fc_aggregation: unet_input_dtype must be float32 or bfloat16, got '" +
                                 input_dtype + "'");
    }
    const std::string& input_layout = m_unet_input_layout.value();
    if (input_layout != "ncw" && input_layout != "nwc") {
        throw std::runtime_error("pvfinder_fc_aggregation: unet_input_layout must be ncw or nwc, got '" +
                                 input_layout + "'");
    }
    if (input_layout == "nwc" && input_dtype != "bfloat16") {
        throw std::runtime_error("pvfinder_fc_aggregation: unet_input_layout = nwc needs unet_input_dtype = bfloat16");
    }
#ifdef ALLEN_WITH_CUBLAS
    const bool features_bf16 = input_dtype == "bfloat16";
    const bool features_nwc = input_layout == "nwc";
#else
    const bool features_bf16 = false;   // the fallback kernel writes float32
    const bool features_nwc = false;
#endif
    const int features_format = !features_bf16 ? 0 : features_nwc ? 2 : 1;
    const size_t feature_bytes = features_bf16 ? sizeof(__nv_bfloat16) : sizeof(float);
    char* features_base = reinterpret_cast<char*>(data<dev_pvfinder_interval_features_t>(arguments));
    unsigned* host_unet_rows = data<host_pvfinder_unet_rows_t>(arguments);
    host_unet_rows[0] = compact ? 1u : 0u;
    host_unet_rows[1] = 0u;
    host_unet_rows[2] = features_bf16 ? 1u : 0u;
    host_unet_rows[3] = features_nwc ? 1u : 0u;
#ifdef ALLEN_WITH_CUBLAS
    const bool skip_redundant_memset = m_skip_redundant_memset.value();
#else
    // The non-cuBLAS fallback kernel (pvfinder_fused_fc_aggregation_kernel)
    // still early-returns and relies on the full memset -- never skip it here.
    const bool skip_redundant_memset = false;
#endif
    if (compact) {
        // Compact rows are all written by the reduce kernel, and the tail of
        // the UNet's last batch is zeroed below once the row count is known.
        if (!skip_redundant_memset) {
            cudaMemsetAsync(
                data<dev_pvfinder_output_histogram_t>(arguments),
                0,
                n_events * 4000u * sizeof(float),
                context.stream());
        }
    } else if (skip_redundant_memset) {
        // pvfinder_reduce_l6a_kernel writes explicit zeros for every empty
        // (event, interval) slot itself now (see its doc comment) -- only
        // the padding tail of dev_pvfinder_interval_features, which no FC
        // kernel ever writes to (real events only go up to n_events), still
        // needs zeroing. dev_pvfinder_output_histogram needs no memset at
        // all in this mode.
        if (padded_events > n_events) {
            cudaMemsetAsync(
                features_base + (unsigned long long)n_events * INTERVAL_FEATURES_STRIDE * feature_bytes,
                0,
                (unsigned long long)(padded_events - n_events) * INTERVAL_FEATURES_STRIDE * feature_bytes,
                context.stream());
        }
    } else {
        cudaMemsetAsync(
            data<dev_pvfinder_interval_features_t>(arguments),
            0,
            padded_events * INTERVAL_FEATURES_STRIDE * sizeof(float),
            context.stream());
        cudaMemsetAsync(
            data<dev_pvfinder_output_histogram_t>(arguments),
            0,
            n_events * 4000u * sizeof(float),
            context.stream());
    }

#ifdef ALLEN_WITH_CUBLAS
    // -----------------------------------------------------------------------
    // 3-kernel + cuBLAS pipeline, chunked over B_CHUNK events.
    //
    // T_chunk computation strategy: ONE batch DtoH of the full CSR sentinel column
    // (index 41 of each event's interval_start array = total CSR entries for that event).
    // Total transfer: n_events * 42 * sizeof(int) ≈ 16 KB for 100 events — negligible.
    // This replaces the previous approach of n_events individual cudaMemcpy DtoH calls
    // (100 blocking syncs per slice), which caused the 77% overhead regression.
    //
    // Flow:
    //   1. cudaStreamSynchronize  — wait for CSR kernel (once per slice)
    //   2. cudaMemcpy DtoH        — copy all CSR offsets at once (~16 KB)
    //   3. Host arithmetic        — compute T_chunk[i] for each chunk
    //   4. Per-chunk kernel loop  — no further DtoH on the hot path
    // -----------------------------------------------------------------------
    // Single batch DtoH: copy all event CSR offset arrays to host, on this
    // sequence's stream into pinned memory, then wait for that stream only.
    // (A plain cudaMemcpy runs on the legacy default stream, which Allen's
    // blocking streams all synchronise with: every slice then drained the
    // whole device, a cost that grows with the number of streams.)
    const unsigned csr_words = n_events * 42u;
    thread_local int* tl_host_csr = nullptr;
    thread_local size_t tl_host_csr_words = 0;
    if (csr_words > tl_host_csr_words) {
        if (tl_host_csr != nullptr) cudaFreeHost(tl_host_csr);
        tl_host_csr_words = std::max<size_t>(csr_words, 2 * tl_host_csr_words);
        cudaCheck(cudaMallocHost(&tl_host_csr, tl_host_csr_words * sizeof(int)));
    }
    cudaMemcpyAsync(tl_host_csr,
                    data<dev_pvfinder_interval_start_t>(arguments),
                    csr_words * sizeof(int),
                    cudaMemcpyDeviceToHost,
                    context.stream());
    cudaStreamSynchronize(context.stream());
    const int* host_csr = tl_host_csr;

    // Compact rows for the UNet (see m_skip_empty_intervals): intervals with
    // at least min_interval_tracks tracks get consecutive rows, in (event,
    // interval) order, everything else -1. Uploaded while the stream is idle
    // (it was just synchronised), so the pageable copy costs no stall.
    // The FC histogram is only read by the validation dump (see m_write_histogram).
    const bool write_histogram = m_write_histogram.value() || !m_dump_dir.value().empty();
    const int* slot_row = nullptr;
    const int* host_slot_row_ptr = nullptr;
    if (compact) {
        const unsigned min_tracks = m_min_interval_tracks.value();
        if (min_tracks == 0) {
            throw std::runtime_error("pvfinder_fc_aggregation: min_interval_tracks must be >= 1");
        }
        const unsigned n_slots = n_events * N_INTERVALS;
        thread_local std::vector<int> host_slot_row;
        host_slot_row.resize(n_slots);
        int next_row = 0;
        for (unsigned ev = 0; ev < n_events; ++ev) {
            const int* start = host_csr + ev * 42u;
            for (unsigned iv = 0; iv < N_INTERVALS; ++iv) {
                const bool keep = (unsigned)(start[iv + 1] - start[iv]) >= min_tracks;
                host_slot_row[ev * N_INTERVALS + iv] = keep ? next_row++ : -1;
            }
        }
        host_unet_rows[1] = (unsigned)next_row;
        cudaMemcpyAsync(data<dev_pvfinder_slot_row_t>(arguments), host_slot_row.data(),
                        n_slots * sizeof(int), cudaMemcpyHostToDevice, context.stream());
        slot_row = data<dev_pvfinder_slot_row_t>(arguments);
        host_slot_row_ptr = host_slot_row.data();

        // The UNet reads whole batches of unet_batch_events * 40 rows: zero
        // the rows past the last one in use, so it never reads stale data.
        const unsigned batch_rows = unet_batch_events * N_INTERVALS;
        const unsigned padded_rows = ((unsigned)next_row + batch_rows - 1) / batch_rows * batch_rows;
        if (padded_rows > (unsigned)next_row) {
            cudaMemsetAsync(
                features_base + (unsigned long long)next_row * L6A_WIDTH * feature_bytes,
                0,
                (unsigned long long)(padded_rows - next_row) * L6A_WIDTH * feature_bytes,
                context.stream());
        }
    }


    const float* w1  = dev_weights;
    const float* b1  = w1  + 180;
    const float* w2  = b1  + 20;
    const float* b2  = w2  + 400;
    const float* w3  = b2  + 20;
    const float* b3  = w3  + 400;
    const float* w4  = b3  + 20;
    const float* b4  = w4  + 400;
    const float* w5  = b4  + 20;
    const float* b5  = w5  + 400;
    const float* w6A = b5  + 20;
    const float* b6A = w6A + L6A_WEIGHT_FLOATS;

    // Must match set_arguments_size's buffer sizing.
    const unsigned B_CHUNK = m_fc_chunk_size.value();
    constexpr unsigned KERNEL1_BLOCK = 256u;
    constexpr unsigned KERNEL2_BLOCK = 512u;
    constexpr unsigned KERNEL3_BLOCK = 128u;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasHandle_t cublas = get_cublas_handle();
    cublasSetStream(cublas, context.stream());

    const bool use_warp_parallel   = m_use_warp_parallel_reduce.value();
    const bool use_nonatomic       = m_use_nonatomic_l6a_reduce.value();
    const bool use_fused_bias_relu = m_use_fused_bias_relu_reduce.value();
    const bool use_grid_stride_reduce = m_use_grid_stride_reduce.value();
    const bool fc_single_hidden_layer = m_fc_single_hidden_layer.value();
    const unsigned l1_l5_hidden_width = m_l1_l5_hidden_width.value();
    const bool use_precomputed_csr_offset = m_use_precomputed_csr_offset.value();
    // Precision of L6A (see m_l6a_dtype). "auto": the fused FC's tensor cores
    // when it writes bfloat16 features for the UNet and the device has them.
    const std::string& l6a_dtype = m_l6a_dtype.value();
    if (l6a_dtype != "float32" && l6a_dtype != "bfloat16" && l6a_dtype != "auto") {
        throw std::runtime_error("pvfinder_fc_aggregation: l6a_dtype must be float32, bfloat16 or auto, got '" +
                                 l6a_dtype + "'");
    }
    thread_local int tl_cc_major = -1;
    if (tl_cc_major < 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&tl_cc_major, cudaDevAttrComputeCapabilityMajor, device_id);
    }
    const bool l6a_bf16 = !m_fc_fused.value() && l6a_dtype == "bfloat16";
    // With fc_fused, a bfloat16 L6A runs on tensor cores instead.
    const bool fused_tensor_cores = m_fc_fused.value() &&
        (l6a_dtype == "bfloat16" || (l6a_dtype == "auto" && features_bf16 && tl_cc_major >= 8));
    // Layers 2-5 on tensor cores too (see m_fc_hidden_dtype); "auto" follows L6A.
    const std::string& hidden_dtype = m_fc_hidden_dtype.value();
    if (hidden_dtype != "float32" && hidden_dtype != "bfloat16" && hidden_dtype != "auto") {
        throw std::runtime_error("pvfinder_fc_aggregation: fc_hidden_dtype must be float32, bfloat16 or auto, got '" +
                                 hidden_dtype + "'");
    }
    const bool hidden_tensor_cores = fused_tensor_cores && m_fc_fused_per_warp.value() && hidden_dtype != "float32";
    if (l6a_bf16 && !use_fused_bias_relu) {
        throw std::runtime_error("pvfinder_fc_aggregation: l6a_dtype = bfloat16 needs use_fused_bias_relu_reduce = true");
    }
    const __nv_bfloat16* w6A_bf16 =
        l6a_bf16 ? PVFinder::WeightRegistry::instance().get<__nv_bfloat16>("fc_w6a_bf16") : nullptr;
    // Every chunk's cumulative per-event column offsets, one (B_CHUNK + 1)
    // block per chunk, uploaded in one copy now, while the stream is idle
    // after the CSR readback. (Uploading each chunk's block from pageable
    // memory just before its kernels made every chunk wait for the previous
    // one: a pageable host-to-device copy synchronises the stream first.)
    const unsigned col_offset_stride = B_CHUNK + 1u;
    if (use_precomputed_csr_offset) {
        thread_local std::vector<unsigned> host_col_offset;
        const unsigned n_chunks = (n_events + B_CHUNK - 1) / B_CHUNK;
        host_col_offset.assign((size_t)n_chunks * col_offset_stride, 0u);
        for (unsigned c = 0; c < n_chunks; ++c) {
            unsigned running = 0;
            const unsigned first = c * B_CHUNK, last = std::min(first + B_CHUNK, n_events);
            for (unsigned ev = first; ev < last; ++ev) {
                running += (unsigned)host_csr[ev * 42 + 41];
                host_col_offset[(size_t)c * col_offset_stride + ev - first + 1] = running;
            }
        }
        if (n_chunks > 0) {
            cudaMemcpyAsync(data<dev_pvfinder_event_col_offset_t>(arguments), host_col_offset.data(),
                            host_col_offset.size() * sizeof(unsigned), cudaMemcpyHostToDevice, context.stream());
        }
    }

    // See m_use_grid_stride_reduce doc comment: query this GPU's actual
    // occupancy ceiling for the grid-stride
    // instantiation once per thread (cached; SM count and per-SM occupancy
    // don't change during a run) rather than hardcoding a specific GPU's SM
    // count -- portable across devices.
    thread_local int tl_grid_stride_blocks = 0;
    if (use_grid_stride_reduce && tl_grid_stride_blocks == 0) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        int sm_count = 0;
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            pvfinder_reduce_l6a_kernel<true, true, true, true>,
            (int)KERNEL3_BLOCK, 0);
        tl_grid_stride_blocks = sm_count * max_blocks_per_sm;
    }

    // Work list for the per-warp fused FC kernels (see m_fc_largest_first):
    // uint4 {slot, first CSR entry, entries | chunk fields, feature row} by
    // decreasing entries (a counting sort, ties in slot and chunk order), from
    // pinned memory. For the tensor-core kernel, slots with more than FT_CHUNK
    // entries become several items (see FT_CHUNK). Empty slots are left out
    // when they need no output: compact rows (no feature row) and no histogram.
    const uint4* slot_items = nullptr;
    unsigned n_items = 0;
    float* fc_partial = nullptr;
    unsigned* fc_arrive = nullptr;
    if (m_fc_fused.value() && m_fc_fused_per_warp.value() && m_fc_largest_first.value() && n_events > 0) {
        const unsigned n_slots = n_events * N_INTERVALS;
        const bool with_empty = !compact || write_histogram;
        const int chunk = fused_tensor_cores ? (int) FT_CHUNK : std::numeric_limits<int>::max();
        size_t total_entries = 0;
        for (unsigned ev = 0; ev < n_events; ++ev) total_entries += (size_t) host_csr[ev * 42u + 41];
        const size_t capacity = n_slots + total_entries / FT_CHUNK + 1;
        if (capacity * 4u > size<dev_pvfinder_slot_order_t>(arguments)) {
            throw std::runtime_error("pvfinder_fc_aggregation: FC work list larger than its buffer");
        }
        thread_local uint4* tl_items = nullptr;
        thread_local size_t tl_items_size = 0;
        if (capacity > tl_items_size) {
            if (tl_items != nullptr) cudaFreeHost(tl_items);
            tl_items_size = std::max<size_t>(capacity, 2 * tl_items_size);
            cudaCheck(cudaMallocHost(&tl_items, tl_items_size * sizeof(uint4)));
        }
        // Item sizes are at most max_size; bucket by max_size - size.
        int max_size = 0;
        for (unsigned ev = 0; ev < n_events; ++ev)
            for (unsigned iv = 0; iv < N_INTERVALS; ++iv)
                max_size = std::max(max_size, std::min(chunk, host_csr[ev * 42u + iv + 1] - host_csr[ev * 42u + iv]));
        thread_local std::vector<unsigned> bucket;
        bucket.assign((size_t) max_size + 2, 0u);
        const auto for_each_item = [&](auto&& f) {
            unsigned next_partial = 0;
            for (unsigned ev = 0; ev < n_events; ++ev) {
                for (unsigned iv = 0; iv < N_INTERVALS; ++iv) {
                    const int first = host_csr[ev * 42u + iv], n = host_csr[ev * 42u + iv + 1] - first;
                    if (n == 0 && !with_empty) continue;
                    const unsigned n_chunks = n > chunk ? (unsigned) ((n + chunk - 1) / chunk) : 1u;   // <= 32 (2047 / 64)
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
        for_each_item([&](unsigned, int, int size, unsigned, unsigned, unsigned) { ++bucket[max_size - size + 1]; });
        for (size_t i = 1; i < bucket.size(); ++i) bucket[i] += bucket[i - 1];
        n_items = bucket.back();
        const unsigned n_partial = for_each_item([&](unsigned slot, int first, int size, unsigned c, unsigned n_chunks,
                                                     unsigned pbase) {
            const int row = host_slot_row_ptr != nullptr ? host_slot_row_ptr[slot] : (int) slot;
            tl_items[bucket[max_size - size]++] = make_uint4(
                slot, (unsigned) first | (pbase << 11), (unsigned) size | (c << 12) | (n_chunks << 18), (unsigned) row);
        });
        for (unsigned ev = 0; ev < n_events; ++ev) {   // entries, chunks and first entry fit their fields
            if (host_csr[ev * 42u + 41] > 0x7ff) {
                throw std::runtime_error("pvfinder_fc_aggregation: more than 2047 CSR entries in an event");
            }
        }
        if (n_partial > (1u << 21) || (size_t) n_partial > size<dev_pvfinder_fc_arrive_t>(arguments)) {
            throw std::runtime_error("pvfinder_fc_aggregation: too many split FC slots for the partial-sum buffer");
        }
        if (n_items > 0) {
            cudaMemcpyAsync(data<dev_pvfinder_slot_order_t>(arguments), tl_items, n_items * sizeof(uint4),
                            cudaMemcpyHostToDevice, context.stream());
        }
        if (n_partial > 0) {
            cudaMemsetAsync(data<dev_pvfinder_fc_arrive_t>(arguments), 0, n_partial * sizeof(unsigned), context.stream());
        }
        slot_items = reinterpret_cast<const uint4*>(data<dev_pvfinder_slot_order_t>(arguments));
        fc_partial = data<dev_pvfinder_fc_partial_t>(arguments);
        fc_arrive = data<dev_pvfinder_fc_arrive_t>(arguments);
    }

    // Fused FC (m_fc_fused): the whole slice in one kernel, no chunks.
    const bool fc_fused = m_fc_fused.value();
    if (fc_fused && n_events > 0) {
        thread_local int tl_fused_blocks[2] = {0, 0};
        int& fused_blocks = tl_fused_blocks[fused_tensor_cores ? 1 : 0];
        if (fused_blocks == 0) {
            int device_id = 0, sm_count = 0, per_sm = 0, cc_major = 0;
            cudaGetDevice(&device_id);
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
            cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device_id);
            if (fused_tensor_cores && cc_major < 8) {
                throw std::runtime_error("pvfinder_fc_aggregation: l6a_dtype = bfloat16 with fc_fused needs "
                                         "bfloat16 tensor cores (compute capability 8.0 or newer)");
            }
            if (fused_tensor_cores)
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pvfinder_fused_fc_kernel<true>, (int) FUSED_BLOCK, 0);
            else
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pvfinder_fused_fc_kernel<false>, (int) FUSED_BLOCK, 0);
            fused_blocks = sm_count * std::max(per_sm, 1);
        }
        cudaMemsetAsync(data<dev_pvfinder_reduce_work_counter_t>(arguments), 0, sizeof(unsigned), context.stream());
        if (m_fc_fused_per_warp.value() && fused_tensor_cores) {
            thread_local int tl_tc_blocks[2] = {0, 0};
            int& tc_blocks = tl_tc_blocks[hidden_tensor_cores ? 1 : 0];
            if (tc_blocks == 0) {
                int device_id = 0, sm_count = 0, per_sm = 0;
                cudaGetDevice(&device_id);
                cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
                if (hidden_tensor_cores) {
                    cudaFuncSetAttribute(pvfinder_fused_fc_tc_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int) ft_smem_bytes());
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pvfinder_fused_fc_tc_kernel<true>,
                                                                  (int) FT_THREADS, ft_smem_bytes());
                }
                else {
                    cudaFuncSetAttribute(pvfinder_fused_fc_tc_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int) ft_smem_bytes());
                    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pvfinder_fused_fc_tc_kernel<false>,
                                                                  (int) FT_THREADS, ft_smem_bytes());
                }
                tc_blocks = sm_count * std::max(per_sm, 1);
            }
            // See m_fused_grid_fraction: leave room for other streams' kernels.
            const unsigned tc_grid = std::max(1u, (unsigned) (tc_blocks * m_fused_grid_fraction.value()));
            if (hidden_tensor_cores) {
                global_function(pvfinder_fused_fc_tc_kernel<true>)(
                    dim3(tc_grid), dim3(FT_THREADS), context, ft_smem_bytes())(
                    arguments, dev_weights, n_events, data<dev_pvfinder_reduce_work_counter_t>(arguments), slot_row,
                    features_format, slot_items, n_items, write_histogram, fc_partial, fc_arrive);
            }
            else {
                global_function(pvfinder_fused_fc_tc_kernel<false>)(
                    dim3(tc_grid), dim3(FT_THREADS), context, ft_smem_bytes())(
                    arguments, dev_weights, n_events, data<dev_pvfinder_reduce_work_counter_t>(arguments), slot_row,
                    features_format, slot_items, n_items, write_histogram, fc_partial, fc_arrive);
            }
        }
        else if (m_fc_fused_per_warp.value()) {
            thread_local int tl_warp_blocks = 0;
            if (tl_warp_blocks == 0) {
                int device_id = 0, sm_count = 0, per_sm = 0;
                cudaGetDevice(&device_id);
                cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
                cudaFuncSetAttribute(pvfinder_fused_fc_warp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int) fw_smem_bytes());
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pvfinder_fused_fc_warp_kernel,
                                                              (int) FW_THREADS, fw_smem_bytes());
                tl_warp_blocks = sm_count * std::max(per_sm, 1);
            }
            global_function(pvfinder_fused_fc_warp_kernel)(
                dim3((unsigned) tl_warp_blocks), dim3(FW_THREADS), context, fw_smem_bytes())(
                arguments, dev_weights, n_events, data<dev_pvfinder_reduce_work_counter_t>(arguments), slot_row,
                features_format, slot_items, n_items, write_histogram);
        }
        else if (fused_tensor_cores) {
            global_function(pvfinder_fused_fc_kernel<true>)(dim3((unsigned) fused_blocks), dim3(FUSED_BLOCK), context)(
                arguments, dev_weights, n_events, data<dev_pvfinder_reduce_work_counter_t>(arguments), slot_row,
                features_format);
        }
        else {
            global_function(pvfinder_fused_fc_kernel<false>)(dim3((unsigned) fused_blocks), dim3(FUSED_BLOCK), context)(
                arguments, dev_weights, n_events, data<dev_pvfinder_reduce_work_counter_t>(arguments), slot_row,
                features_format);
        }
    }

    unsigned csr_col_offset = 0;  // running total of CSR entries before chunk_start
    for (unsigned chunk_start = 0; !fc_fused && chunk_start < n_events; chunk_start += B_CHUNK) {
        const unsigned chunk_end = std::min(chunk_start + B_CHUNK, n_events);

        // Compute T_chunk from the already-copied host array — NO device reads here.
        unsigned T_chunk = 0;
        for (unsigned ev = chunk_start; ev < chunk_end; ++ev) {
            T_chunk += (unsigned)host_csr[ev * 42 + 41];  // sentinel = total CSR entries
        }
        if (T_chunk == 0) { csr_col_offset += T_chunk; continue; }

        // See m_use_precomputed_csr_offset doc comment: precompute this
        // chunk's cumulative per-event column offsets on the host -- reusing
        // the same host_csr data T_chunk just
        // walked above, so this costs one more cheap host-side pass, not a
        // new device round-trip for the source data -- and upload once per
        // chunk (at most (B_CHUNK+1)*4 bytes, e.g. 404 bytes at
        // fc_chunk_size=100). pvfinder_reduce_l6a_kernel then looks this up
        // in O(1) instead of walking host_csr's device-side mirror
        // (dev_pvfinder_interval_start) in O(events-in-chunk) per slot.
        const unsigned* chunk_col_offset =
            data<dev_pvfinder_event_col_offset_t>(arguments) + (chunk_start / B_CHUNK) * col_offset_stride;

        // --- Kernel 1: L1-L5 per track in this chunk ---
        const unsigned k1_blocks = (T_chunk + KERNEL1_BLOCK - 1) / KERNEL1_BLOCK;
        global_function(pvfinder_l1_to_l5_kernel)(
            dim3(k1_blocks), dim3(KERNEL1_BLOCK), context)(
            arguments, dev_weights, chunk_start, chunk_end, csr_col_offset, T_chunk,
            fc_single_hidden_layer, l1_l5_hidden_width,
            use_precomputed_csr_offset ? chunk_col_offset : nullptr, l6a_bf16);

        // --- cuBLAS SGEMM: L6A ---
        // W6A is stored row-major [L6A_WIDTH×20] (no load-time transpose in
        // cuBLAS builds), i.e. column-major [20×L6A_WIDTH] with lda=20, so
        // CUBLAS_OP_T gives the [L6A_WIDTH×20] matrix PyTorch uses.
        // X [T_chunk×20] row-major = [20×T_chunk] col-major, op=N.
        // Y = W6A^T × X → [l6a_m×T_chunk] col-major → dev_l6a_output.
        // The M argument (rows computed) is overridable via m_l6a_m, and
        // the K argument (reduction depth) via m_l1_l5_hidden_width -- lda/ldb/ldc
        // stay fixed at the real buffer strides (20, 20, L6A_WIDTH) regardless, since a
        // smaller M or K is a valid cuBLAS sub-block read/write into the same
        // wider-strided real buffers (see m_l6a_m/m_l1_l5_hidden_width doc
        // comments for why this is throughput-only).
        const int l6a_m = (int)m_l6a_m.value();
        const int l1_l5_hidden_width_i = (int)l1_l5_hidden_width;
        if (l6a_bf16) {
            // Same product with bfloat16 operands and output, accumulated in
            // float32 (see m_l6a_dtype): half the memory traffic of the
            // output, which the reduce kernel reads back.
            cublasGemmEx(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                l6a_m, (int)T_chunk, l1_l5_hidden_width_i,
                &alpha,
                w6A_bf16, CUDA_R_16BF, 20,
                data<dev_pvfinder_l5_output_t>(arguments), CUDA_R_16BF, 20,
                &beta,
                data<dev_pvfinder_l6a_output_t>(arguments), CUDA_R_16BF, (int)L6A_WIDTH,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
        else {
            cublasSgemm(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                l6a_m, (int)T_chunk, l1_l5_hidden_width_i,
                &alpha,
                w6A, 20,
                data<dev_pvfinder_l5_output_t>(arguments), 20,
                &beta,
                data<dev_pvfinder_l6a_output_t>(arguments), (int)L6A_WIDTH);
        }

        // --- Kernel 2: bias + LeakyReLU in-place on L6A output ---
        // Skipped entirely when use_fused_bias_relu -- pvfinder_reduce_l6a_kernel
        // applies the same bias+LeakyReLU inline on its own read of the raw
        // GEMM output instead (see m_use_fused_bias_relu_reduce doc comment).
        const unsigned l6a_m_u = (unsigned)l6a_m;
        const unsigned active_channels_u = m_l6a_active_channels.value();
        if (!use_fused_bias_relu) {
            const unsigned k2_blocks = (T_chunk * l6a_m_u + KERNEL2_BLOCK - 1) / KERNEL2_BLOCK;
            global_function(pvfinder_l6a_bias_relu_kernel)(
                dim3(k2_blocks), dim3(KERNEL2_BLOCK), context)(
                arguments, b6A, T_chunk, l6a_m_u);
        }

        // --- Kernel 3: reduce L6A → interval features + histogram ---
        const unsigned n_chunk_events = chunk_end - chunk_start;
        const unsigned grid_blocks = n_chunk_events * 40u;

        if (use_grid_stride_reduce) {
            // Only supported combined with warp_parallel_reduce +
            // fused_bias_relu_reduce. Reset the work counter before each
            // chunk's launch, then launch the fixed, occupancy-sized grid.
            // use_precomputed_csr_offset is also only wired into this path.
            cudaMemsetAsync(
                data<dev_pvfinder_reduce_work_counter_t>(arguments), 0, sizeof(unsigned), context.stream());
            if (use_precomputed_csr_offset) {
                global_function(pvfinder_reduce_l6a_kernel<true, true, true, true, true>)(
                    dim3((unsigned)tl_grid_stride_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A,
                    data<dev_pvfinder_reduce_work_counter_t>(arguments),
                    chunk_col_offset, slot_row, features_format, l6a_bf16);
            } else {
                global_function(pvfinder_reduce_l6a_kernel<true, true, true, true>)(
                    dim3((unsigned)tl_grid_stride_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A,
                    data<dev_pvfinder_reduce_work_counter_t>(arguments), nullptr, slot_row, features_format, l6a_bf16);
            }
        } else if (use_warp_parallel) {
            // UseAtomic is irrelevant on this path (its own bounded combine
            // step is always used regardless) -- fixed to true as a no-op
            // placeholder to avoid instantiating a redundant variant.
            if (use_fused_bias_relu) {
                global_function(pvfinder_reduce_l6a_kernel<true, true, true, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr, slot_row, features_format, l6a_bf16);
            } else {
                global_function(pvfinder_reduce_l6a_kernel<true, true, false, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr, slot_row, features_format, l6a_bf16);
            }
        } else if (use_nonatomic) {
            if (use_fused_bias_relu) {
                global_function(pvfinder_reduce_l6a_kernel<false, false, true, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr, slot_row, features_format, l6a_bf16);
            } else {
                global_function(pvfinder_reduce_l6a_kernel<false, false, false, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr, slot_row, features_format, l6a_bf16);
            }
        } else {
            if (use_fused_bias_relu) {
                global_function(pvfinder_reduce_l6a_kernel<true, false, true, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr, slot_row, features_format, l6a_bf16);
            } else {
                global_function(pvfinder_reduce_l6a_kernel<true, false, false, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr, slot_row, features_format, l6a_bf16);
            }
        }

        csr_col_offset += T_chunk;
    }

#else  // ---- non-cuBLAS fallback: original fused kernel ----

    // -----------------------------------------------------------------------
    // Step 3: Launch aggregation over all (event, interval) pairs.
    // -----------------------------------------------------------------------
    global_function(pvfinder_fused_fc_aggregation_kernel)(
        dim3(n_events, 40), m_block_dim, context)(
        arguments, dev_weights);

#endif  // ALLEN_WITH_CUBLAS

    // -----------------------------------------------------------------------
    // Validation dump (first call only, when dump_validation is set). Raw
    // buffers, so weights/scripts/validate_fc.py can recompute this stage from the
    // checkpoint using exactly Allen's own track-to-interval assignment.
    // Every file: uint32 magic 0xFC01, n_events, n_tracks, N_LATENT_CHANNELS,
    // then the array.
    // -----------------------------------------------------------------------
    const std::string& dump_dir = m_dump_dir.value();
    if (!dump_dir.empty() && !m_dump_done) {
        cudaStreamSynchronize(context.stream());
        const unsigned n_ev  = first<host_number_of_events_t>(arguments);
        const unsigned n_trk = first<host_number_of_reconstructed_velo_tracks_t>(arguments);

        std::vector<int>      h_csr(n_ev * 42u);
        std::vector<int>      h_idx(n_trk * 2u);
        std::vector<float>    h_feat(n_trk * 9u);
        std::vector<float>    h_ifeat((size_t)n_ev * INTERVAL_FEATURES_STRIDE);
        std::vector<float>    h_hist(n_ev * 4000u);
        std::vector<Allen::Views::Velo::Consolidated::Tracks> h_views(n_ev);
        cudaMemcpy(h_csr.data(), data<dev_pvfinder_interval_start_t>(arguments),
                   h_csr.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_idx.data(), data<dev_pvfinder_track_idx_t>(arguments),
                   h_idx.size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_feat.data(), data<dev_pvfinder_track_features_t>(arguments),
                   h_feat.size() * sizeof(float), cudaMemcpyDeviceToHost);
        // Interval features as float, whatever their storage type (a BF16
        // value is the upper half of the float with the same value).
        auto copy_features = [&](float* dst, size_t count) {
            if (features_bf16) {
                std::vector<uint16_t> raw(count);
                cudaMemcpy(raw.data(), features_base, count * sizeof(uint16_t), cudaMemcpyDeviceToHost);
                for (size_t i = 0; i < count; ++i) {
                    // channels-last rows back to [channel][bin]
                    const size_t r = i / L6A_WIDTH, e = i % L6A_WIDTH;
                    const size_t src = features_nwc ? r * L6A_WIDTH + (e % 100) * N_LATENT_CHANNELS + e / 100 : i;
                    const uint32_t bits = (uint32_t)raw[src] << 16;
                    std::memcpy(dst + i, &bits, sizeof(float));
                }
            } else {
                cudaMemcpy(dst, features_base, count * sizeof(float), cudaMemcpyDeviceToHost);
            }
        };
        if (compact) {
            // Compact rows back to [event][interval] order; skipped intervals
            // read as zero, which is what the UNet is given for them.
            std::vector<int> h_slot_row((size_t)n_ev * N_INTERVALS);
            std::vector<float> h_rows((size_t)host_unet_rows[1] * L6A_WIDTH);
            cudaMemcpy(h_slot_row.data(), data<dev_pvfinder_slot_row_t>(arguments),
                       h_slot_row.size() * sizeof(int), cudaMemcpyDeviceToHost);
            copy_features(h_rows.data(), h_rows.size());
            std::fill(h_ifeat.begin(), h_ifeat.end(), 0.0f);
            for (size_t slot = 0; slot < h_slot_row.size(); ++slot) {
                if (h_slot_row[slot] < 0) continue;
                std::copy_n(h_rows.data() + (size_t)h_slot_row[slot] * L6A_WIDTH, L6A_WIDTH,
                            h_ifeat.data() + slot * L6A_WIDTH);
            }
        } else {
            copy_features(h_ifeat.data(), h_ifeat.size());
        }
        cudaMemcpy(h_hist.data(), data<dev_pvfinder_output_histogram_t>(arguments),
                   h_hist.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_views.data(), data<dev_velo_tracks_view_t>(arguments),
                   h_views.size() * sizeof(Allen::Views::Velo::Consolidated::Tracks), cudaMemcpyDeviceToHost);
        std::vector<unsigned> h_offsets(n_ev);
        for (unsigned e = 0; e < n_ev; ++e) h_offsets[e] = h_views[e].offset();

        const uint32_t header[4] = {0xFC01u, n_ev, n_trk, N_LATENT_CHANNELS};
        auto write_dump = [&](const char* name, const void* d, size_t bytes) {
            std::ofstream out(dump_dir + "/" + name, std::ios::binary);
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            out.write(reinterpret_cast<const char*>(d), bytes);
        };
        write_dump("allen_fc_csr.bin", h_csr.data(), h_csr.size() * sizeof(int));
        write_dump("allen_fc_track_idx.bin", h_idx.data(), h_idx.size() * sizeof(int));
        write_dump("allen_fc_track_offsets.bin", h_offsets.data(), h_offsets.size() * sizeof(unsigned));
        write_dump("allen_fc_track_features.bin", h_feat.data(), h_feat.size() * sizeof(float));
        write_dump("allen_fc_interval_features.bin", h_ifeat.data(), h_ifeat.size() * sizeof(float));
        write_dump("allen_fc_histogram.bin", h_hist.data(), h_hist.size() * sizeof(float));
        printf("[pvfinder_fc_aggregation] validation dump written to %s (%u events, %u tracks)\n",
               dump_dir.c_str(), n_ev, n_trk);
        m_dump_done = true;
    }
}

} // namespace pvfinder_fc_aggregation
