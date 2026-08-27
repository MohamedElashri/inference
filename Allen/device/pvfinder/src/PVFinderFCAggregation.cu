#include "PVFinderFCAggregation.cuh"
#include "PVFinderWeightRegistry.h"
#include <mutex>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#ifdef ALLEN_WITH_CUBLAS
#include <cublas_v2.h>
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

__device__ void assign_intervals(float z_poca, int* intervals, int* num_intervals) {
    z_poca += 100.0f;
    *num_intervals = (z_poca >= 0.0f && z_poca <= 400.0f) ? 1 + (z_poca <= 10.0f || z_poca >= 390.0f) : 1;
    int base_interval = min(39, max(0, int(z_poca / 10.0f)));
    intervals[0] = base_interval;
    if (*num_intervals > 1) {
        intervals[1] = (z_poca <= 10.0f) ? base_interval + 1 : base_interval - 1;
    }
}

__device__ float pvfinder_softplus(float x) {
    return x > 0.0f ? x : logf(1.0f + expf(x));
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
// Invalid tracks (z_poca <= -98) are omitted entirely.
//
// Three shared-memory passes:
//   1. Histogram: count tracks per interval (pass1 atomic into s_counts[40])
//   2. Exclusive prefix sum: compute s_start[41] from s_counts
//   3. Scatter:   fill track_idx[] advancing s_cursor[40] atomically
// ---------------------------------------------------------------------------
__global__ void pvfinder_build_csr_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters)
{
    const unsigned event_number      = blockIdx.x;
    const unsigned thread_id         = threadIdx.x;
    const auto     velo_tracks_view  = parameters.dev_velo_tracks_view[event_number];
    const unsigned num_tracks        = velo_tracks_view.size();
    const unsigned event_track_offset = velo_tracks_view.offset();

    __shared__ int s_counts[40];   // histogram
    __shared__ int s_start[41];    // exclusive prefix sum → CSR start offsets
    __shared__ int s_cursor[40];   // per-interval fill cursors (advanced atomically)

    for (int i = thread_id; i < 40; i += blockDim.x) s_counts[i] = 0;
    __syncthreads();

    // Pass 1 — count how many (track, interval) entries per interval
    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        const unsigned gtidx  = event_track_offset + i;
        const float    z_poca = parameters.dev_pvfinder_track_features[gtidx * 9 + 2];
        if (z_poca <= -98.0f) continue;
        int ivals[2]; int n = 0;
        assign_intervals(z_poca, ivals, &n);
        for (int j = 0; j < n; ++j)
            atomicAdd(&s_counts[ivals[j]], 1);
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
    __syncthreads();

    // Pass 3 — scatter: write local track indices into track_idx[]
    int* g_idx = parameters.dev_pvfinder_track_idx + event_track_offset * 2;
    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        const unsigned gtidx  = event_track_offset + i;
        const float    z_poca = parameters.dev_pvfinder_track_features[gtidx * 9 + 2];
        if (z_poca <= -98.0f) continue;
        int ivals[2]; int n = 0;
        assign_intervals(z_poca, ivals, &n);
        for (int j = 0; j < n; ++j) {
            const int pos = atomicAdd(&s_cursor[ivals[j]], 1);
            g_idx[pos] = (int)i;   // local track index within event
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
            
            pvfinder_linear_layer_reg(feat, x1, w1, b1, 9,  20);
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
    for (int i = thread_id; i < (int)L6A_WIDTH; i += blockDim.x)
        g_feat[i] = s_feat[i] * weight;

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
    unsigned hidden_width)      // throughput-ceiling probe: use only the first
                                 // hidden_width of each layer's 20 real neurons (see
                                 // m_l1_l5_hidden_width doc comment) -- NOT physics-valid
                                 // when < 20, timing only
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T_chunk) return;

    // Determine which event and which local interval entry this thread owns.
    // The CSR track_idx array is indexed via per-event offsets.
    // We recover the global track index from the pre-built CSR structure.
    unsigned ev = chunk_start;
    unsigned local_t = t;
    while (ev < chunk_end) {
        const int* g_start_ev = parameters.dev_pvfinder_interval_start + ev * 42;
        const unsigned n_entries = (unsigned)g_start_ev[41]; // total CSR entries for this event
        if (local_t < n_entries) break;
        local_t -= n_entries;
        ++ev;
    }
    if (ev >= chunk_end) return;

    // Recover the actual track index within this event
    const int* g_start = parameters.dev_pvfinder_interval_start + ev * 42;
    const auto  velo_tracks_view  = parameters.dev_velo_tracks_view[ev];
    const unsigned event_track_offset = velo_tracks_view.offset();
    const int* g_idx = parameters.dev_pvfinder_track_idx + event_track_offset * 2;
    const int local_track = g_idx[local_t];
    const unsigned gtidx = event_track_offset + (unsigned)local_track;

    // Evaluate L1-L5 in registers
    const float* feat = parameters.dev_pvfinder_track_features + gtidx * 9;
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
    pvfinder_linear_layer_reg(feat, x1, w1, b1, 9,  hw);
    if (!single_hidden_layer) {
        pvfinder_linear_layer_reg(x1,  x2, w2, b2, hw, hw, 20);
        pvfinder_linear_layer_reg(x2,  x1, w3, b3, hw, hw, 20);
        pvfinder_linear_layer_reg(x1,  x2, w4, b4, hw, hw, 20);
        pvfinder_linear_layer_reg(x2,  x1, w5, b5, hw, hw, 20);
    }

    // Write x1[20] as row t of dev_pvfinder_l5_output [T_chunk × 20] row-major
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
    const unsigned* __restrict__ event_col_offset)  // only read when PrecomputedOffset
{
    const unsigned thread_id = threadIdx.x;
    const unsigned active_neurons = active_channels * 100u;
    const int* g_start  = parameters.dev_pvfinder_interval_start + event_number * 42;
    const int  iv_begin = g_start[interval];
    const int  iv_end   = g_start[interval + 1];
    const int  n_local  = iv_end - iv_begin;
    if (n_local == 0) {
        // This block still gets launched even for an empty slot, so rather
        // than rely on a separate whole-buffer cudaMemsetAsync to leave
        // correct zeros here, write them directly -- cudaMemsetAsync is a
        // real cost under production-scale multi-thread contention that a
        // single-thread profile understates.
        float* g_feat = parameters.dev_pvfinder_interval_features
                        + (unsigned long long)event_number * INTERVAL_FEATURES_STRIDE + interval * L6A_WIDTH;
        for (unsigned i = thread_id; i < active_neurons; i += blockDim.x) g_feat[i] = 0.0f;
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
                    float val = parameters.dev_pvfinder_l6a_output[
                        (unsigned long long)n + (unsigned long long)col * (unsigned long long)L6A_WIDTH];
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
                float val = parameters.dev_pvfinder_l6a_output[(unsigned long long)n + (unsigned long long)col * (unsigned long long)L6A_WIDTH];
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
    float* g_feat = parameters.dev_pvfinder_interval_features
                    + (unsigned long long)event_number * INTERVAL_FEATURES_STRIDE + interval * L6A_WIDTH;
    for (unsigned i = thread_id; i < active_neurons; i += blockDim.x)
        g_feat[i] = s_feat[i] * weight;

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
    const unsigned* __restrict__ event_col_offset)
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
                event_col_offset);
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
            event_col_offset);
    }
}

#endif  // ALLEN_WITH_CUBLAS

void pvfinder_fc_aggregation_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned total_events = first<host_number_of_events_t>(arguments);
    const unsigned padded_events = ((total_events + 19) / 20) * 20;
    const unsigned total_tracks = first<host_number_of_reconstructed_velo_tracks_t>(arguments);
    set_size<dev_pvfinder_output_histogram_t>  (arguments, total_events * 4000);
    set_size<dev_pvfinder_interval_features_t> (arguments, padded_events * INTERVAL_FEATURES_STRIDE);
    // CSR index buffers
    set_size<dev_pvfinder_interval_start_t>(arguments, total_events * 42);
    set_size<dev_pvfinder_track_idx_t>     (arguments, total_tracks  * 2);
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
    set_size<dev_pvfinder_l5_output_t> (arguments, T_chunk_max * 20u);
    // dev_l6a_output: [L6A_WIDTH × T_chunk_max]  column-major — raw L6A GEMM output (~24 MB at L6A_WIDTH=800)
    set_size<dev_pvfinder_l6a_output_t>(arguments, L6A_WIDTH * T_chunk_max);
    // Single-element atomic work counter for grid-stride reduce.
    set_size<dev_pvfinder_reduce_work_counter_t>(arguments, 1u);
    // Per-chunk cumulative CSR column offsets, B_CHUNK+1 entries.
    set_size<dev_pvfinder_event_col_offset_t>(arguments, B_CHUNK + 1u);
#else
    set_size<dev_pvfinder_l5_output_t> (arguments, 0u);
    set_size<dev_pvfinder_l6a_output_t>(arguments, 0u);
    set_size<dev_pvfinder_reduce_work_counter_t>(arguments, 0u);
    set_size<dev_pvfinder_event_col_offset_t>(arguments, 0u);
#endif  // ALLEN_WITH_CUBLAS
}


void pvfinder_fc_aggregation_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    static std::once_flag flag;
    const std::string weight_file_path = m_weight_file.value();
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

            // Transpose L6A weights from [l6a_rows][20] to [20][l6a_rows].
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

            PVFinder::WeightRegistry::instance().load_from_buffer(
                "fc_weights", host_buf.data(), bytes);
        }
    });
    const float* dev_weights = PVFinder::WeightRegistry::instance().get<float>("fc_weights");

    const unsigned n_events = first<host_number_of_events_t>(arguments);

    // -----------------------------------------------------------------------
    // Step 1: Build CSR index (same as before).
    // -----------------------------------------------------------------------
    global_function(pvfinder_build_csr_kernel)(
        dim3(n_events), m_block_dim, context)(
        arguments);

    // -----------------------------------------------------------------------
    // Step 2: Zero the interval feature and histogram output buffers.
    // (Same in both cuBLAS and non-cuBLAS paths — correctness guarantee for
    // empty intervals, free on-stream cost.)
    // -----------------------------------------------------------------------
    const unsigned padded_events = ((n_events + 19) / 20) * 20;
#ifdef ALLEN_WITH_CUBLAS
    const bool skip_redundant_memset = m_skip_redundant_memset.value();
#else
    // The non-cuBLAS fallback kernel (pvfinder_fused_fc_aggregation_kernel)
    // still early-returns and relies on the full memset -- never skip it here.
    const bool skip_redundant_memset = false;
#endif
    if (skip_redundant_memset) {
        // pvfinder_reduce_l6a_kernel writes explicit zeros for every empty
        // (event, interval) slot itself now (see its doc comment) -- only
        // the padding tail of dev_pvfinder_interval_features, which no FC
        // kernel ever writes to (real events only go up to n_events), still
        // needs zeroing. dev_pvfinder_output_histogram needs no memset at
        // all in this mode.
        if (padded_events > n_events) {
            cudaMemsetAsync(
                data<dev_pvfinder_interval_features_t>(arguments)
                    + (unsigned long long)n_events * INTERVAL_FEATURES_STRIDE,
                0,
                (unsigned long long)(padded_events - n_events) * INTERVAL_FEATURES_STRIDE * sizeof(float),
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
    cudaStreamSynchronize(context.stream());

    // Single batch DtoH: copy all event CSR offset arrays to host.
    const unsigned csr_words = n_events * 42u;
    std::vector<int> host_csr(csr_words);
    cudaMemcpy(host_csr.data(),
               data<dev_pvfinder_interval_start_t>(arguments),
               csr_words * sizeof(int),
               cudaMemcpyDeviceToHost);

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
    // Reused across chunks, sized once to this call's B_CHUNK+1.
    std::vector<unsigned> host_col_offset;
    if (use_precomputed_csr_offset) host_col_offset.resize(B_CHUNK + 1u);

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

    unsigned csr_col_offset = 0;  // running total of CSR entries before chunk_start
    for (unsigned chunk_start = 0; chunk_start < n_events; chunk_start += B_CHUNK) {
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
        if (use_precomputed_csr_offset) {
            unsigned running = 0;
            host_col_offset[0] = 0;
            for (unsigned ev = chunk_start; ev < chunk_end; ++ev) {
                running += (unsigned)host_csr[ev * 42 + 41];
                host_col_offset[ev - chunk_start + 1] = running;
            }
            cudaMemcpyAsync(
                data<dev_pvfinder_event_col_offset_t>(arguments),
                host_col_offset.data(),
                (chunk_end - chunk_start + 1u) * sizeof(unsigned),
                cudaMemcpyHostToDevice,
                context.stream());
        }

        // --- Kernel 1: L1-L5 per track in this chunk ---
        const unsigned k1_blocks = (T_chunk + KERNEL1_BLOCK - 1) / KERNEL1_BLOCK;
        global_function(pvfinder_l1_to_l5_kernel)(
            dim3(k1_blocks), dim3(KERNEL1_BLOCK), context)(
            arguments, dev_weights, chunk_start, chunk_end, csr_col_offset, T_chunk,
            fc_single_hidden_layer, l1_l5_hidden_width);

        // --- cuBLAS SGEMM: L6A ---
        // W6A [20×L6A_WIDTH] row-major = [L6A_WIDTH×20] col-major → CUBLAS_OP_T gives [L6A_WIDTH×20].
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
        cublasSgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            l6a_m, (int)T_chunk, l1_l5_hidden_width_i,
            &alpha,
            w6A, 20,
            data<dev_pvfinder_l5_output_t>(arguments), 20,
            &beta,
            data<dev_pvfinder_l6a_output_t>(arguments), (int)L6A_WIDTH);

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
                    data<dev_pvfinder_event_col_offset_t>(arguments));
            } else {
                global_function(pvfinder_reduce_l6a_kernel<true, true, true, true>)(
                    dim3((unsigned)tl_grid_stride_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A,
                    data<dev_pvfinder_reduce_work_counter_t>(arguments), nullptr);
            }
        } else if (use_warp_parallel) {
            // UseAtomic is irrelevant on this path (its own bounded combine
            // step is always used regardless) -- fixed to true as a no-op
            // placeholder to avoid instantiating a redundant variant.
            if (use_fused_bias_relu) {
                global_function(pvfinder_reduce_l6a_kernel<true, true, true, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr);
            } else {
                global_function(pvfinder_reduce_l6a_kernel<true, true, false, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr);
            }
        } else if (use_nonatomic) {
            if (use_fused_bias_relu) {
                global_function(pvfinder_reduce_l6a_kernel<false, false, true, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr);
            } else {
                global_function(pvfinder_reduce_l6a_kernel<false, false, false, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr);
            }
        } else {
            if (use_fused_bias_relu) {
                global_function(pvfinder_reduce_l6a_kernel<true, false, true, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr);
            } else {
                global_function(pvfinder_reduce_l6a_kernel<true, false, false, false>)(
                    dim3(grid_blocks), dim3(KERNEL3_BLOCK), context)(
                    arguments, chunk_start, chunk_end, T_chunk, l6a_m_u, active_channels_u, b6A, nullptr, nullptr);
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
}

} // namespace pvfinder_fc_aggregation
