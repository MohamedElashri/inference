#include "PVFinderFCAggregation.cuh"
#include "PVFinderWeightRegistry.h"
#include <mutex>
#include <fstream>
#include <vector>
#include <string>
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

__device__ __forceinline__ void pvfinder_linear_layer_reg(
    const float* __restrict__ x, float* __restrict__ y,
    const float* __restrict__ w, const float* __restrict__ b,
    int in_f, int out_f)
{
    for (int i = 0; i < out_f; ++i) {
        float sum = b[i];
        for (int j = 0; j < in_f; ++j) sum += w[i * in_f + j] * x[j];
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
// Static shared mem: s_feat[800] + s_hist[100] = 3.6 KB.
// This keeps occupancy high (many blocks resident per SM) which is the
// dominant factor on both SM 7.5 and SM 8.6.
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
    const float* b6A = w6A + 16000;

    __shared__ float s_feat[8 * 100];  // 3.2 KB
    __shared__ float s_hist[100];      // 0.4 KB

    for (int i = thread_id; i < 800; i += blockDim.x) s_feat[i] = 0.0f;
    for (int i = thread_id; i < 100; i += blockDim.x) s_hist[i] = 0.0f;
    __syncthreads();

    // Process tracks in batches of size warpSize (32) per warp.
    for (int i = warp_id * warpSize; i < n_local; i += n_warps * warpSize) {
        const int lane_id = thread_id % warpSize;
        const int track_idx_in_batch = i + lane_id;
        const bool valid_track = track_idx_in_batch < n_local;

        float x1[20], x2[20];
        
        // Phase 1: Thread-parallel L1-L5
        // Each thread processes L1-L5 for a UNIQUE track.
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

        // Phase 2: Warp-collaborative L6A
        // Loop over the tracks in this warp's current batch.
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
                for (int c = 0; c < 8; ++c) {
                    const int neuron = c * 100 + k;
                    float val = b6A[neuron];
                    for (int m = 0; m < 20; ++m) {
                        val += w6A[m * 800 + neuron] * broadcasted_x1[m];
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
                    + event_number * 32000 + interval * 800;
    for (int i = thread_id; i < 800; i += blockDim.x)
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
// ---------------------------------------------------------------------------
__global__ void pvfinder_l1_to_l5_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights,
    unsigned chunk_start,       // first event index in this chunk
    unsigned chunk_end,         // exclusive: last event index + 1
    unsigned csr_offset,        // starting position in track_idx[] for chunk_start event
    unsigned T_chunk)           // total CSR entries in this chunk
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T_chunk) return;

    // Determine which event and which local interval entry this thread owns.
    // The CSR track_idx array is indexed via per-event offsets.
    // We recover the global track index from the pre-built CSR structure.
    // Strategy: binary-search the CSR interval_start array across the chunk
    // to find (event, absolute track_idx_entry) for linear slot t.
    //
    // Simpler fast path: walk the CSR sentinel (index 41) for each event to
    // find which event owns slot t, then look up the local track index.
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

    float x1[20], x2[20];
    pvfinder_linear_layer_reg(feat, x1, w1, b1, 9,  20);
    pvfinder_linear_layer_reg(x1,  x2, w2, b2, 20, 20);
    pvfinder_linear_layer_reg(x2,  x1, w3, b3, 20, 20);
    pvfinder_linear_layer_reg(x1,  x2, w4, b4, 20, 20);
    pvfinder_linear_layer_reg(x2,  x1, w5, b5, 20, 20);

    // Write x1[20] as row t of dev_pvfinder_l5_output [T_chunk × 20] row-major
    float* out = parameters.dev_pvfinder_l5_output + (unsigned long long)t * 20;
    for (int m = 0; m < 20; ++m) out[m] = x1[m];
}

// ---------------------------------------------------------------------------
// Kernel 2 — Apply L6A bias + LeakyReLU in-place after cuBLAS GEMM.
//
// cuBLAS writes        dev_l6a_output [800 × T_chunk]  (column-major)
// i.e. element (n, t) is at offset n + t*800  (n ∈ [0,800), t ∈ [0,T_chunk))
//
// Grid: ceil(T_chunk * 800 / 512)  blockDim: 512
// ---------------------------------------------------------------------------
// l6a_m: how many of the 800 neurons to actually process (see m_l6a_m doc
// comment in the header -- throughput testing only; 800 is the physics-valid
// default). Neurons >= l6a_m are left untouched (stale data from a prior
// chunk's GEMM, never read downstream since pvfinder_reduce_l6a_kernel is
// bounded the same way). idx is linear over [0, T_chunk*l6a_m) and must be
// decomposed into (n, t) and re-mapped to the buffer's true stride-800
// layout -- the physical buffer is always [800 x T_chunk] regardless of
// l6a_m, so idx itself is NOT a valid flat offset once l6a_m != 800.
__global__ void pvfinder_l6a_bias_relu_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ b6A,  // bias[800]
    unsigned T_chunk,
    unsigned l6a_m)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T_chunk * l6a_m) return;
    const unsigned n = idx % l6a_m;  // neuron index
    const unsigned t = idx / l6a_m;  // track/column index
    const unsigned long long off = (unsigned long long)n + (unsigned long long)t * 800u;
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
// dev_l6a_output is column-major [800 × T_chunk]. For track at column t:
//   neuron n → dev_l6a_output[n + t*800]
// ---------------------------------------------------------------------------
// l6a_m: how many of the 800 neurons to actually accumulate (see m_l6a_m doc
// comment in the header -- throughput testing only). Bounding the per-track
// accumulation loop below is what actually removes work here, unlike the
// GEMM-only l6a_m test: this reduction (atomics-heavy, one iteration per
// track per neuron) is the dominant cost in the L6A block, not the GEMM.
// s_feat/s_hist and the output writes below stay sized at the full 800/100
// (downstream buffers are always that shape); neurons >= l6a_m simply never
// get a nonzero contribution.
//
// UseAtomic (see m_use_nonatomic_l6a_reduce doc comment in the header):
// the thread<->n mapping below (n = thread_id, thread_id+blockDim.x, ...) is
// identical on every iteration of the track loop, so a given s_feat[n] slot
// is written by exactly one thread for the block's entire lifetime -- no
// cross-thread race. UseAtomic=false trades atomicAdd for a plain +=, which
// should be equivalent given that invariant; kept as a compile-time template
// parameter (not a runtime branch) so the untested variant carries zero
// overhead relative to a hand-written non-atomic kernel.
template <bool UseAtomic>
__global__ void pvfinder_reduce_l6a_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    unsigned chunk_start,
    unsigned chunk_end,
    unsigned csr_offset,        // offset into the global CSR that corresponds to chunk_start
    unsigned T_chunk,
    unsigned l6a_m)
{
    // blockIdx.x indexes over (relative_event, interval) in this chunk.
    const unsigned n_chunk_events = chunk_end - chunk_start;
    const unsigned rel_ev = blockIdx.x / 40u;
    const unsigned interval  = blockIdx.x % 40u;
    if (rel_ev >= n_chunk_events) return;
    const unsigned event_number = chunk_start + rel_ev;

    // Determine the CSR column offset for this event within the chunk.
    // Walk the per-event CSR sentinels from chunk_start to this event.
    unsigned ev_col_offset = 0;
    for (unsigned e = chunk_start; e < event_number; ++e) {
        const int* g = parameters.dev_pvfinder_interval_start + e * 42;
        ev_col_offset += (unsigned)g[41];
    }

    const int* g_start  = parameters.dev_pvfinder_interval_start + event_number * 42;
    const int  iv_begin = g_start[interval];
    const int  iv_end   = g_start[interval + 1];
    const int  n_local  = iv_end - iv_begin;
    if (n_local == 0) return;  // no tracks — output already zeroed by cudaMemsetAsync

    // Shared memory accumulators: s_feat[800] + s_hist[100]
    __shared__ float s_feat[800];
    __shared__ float s_hist[100];
    const unsigned thread_id = threadIdx.x;
    for (int i = thread_id; i < 800; i += blockDim.x) s_feat[i] = 0.0f;
    for (int i = thread_id; i < 100; i += blockDim.x) s_hist[i] = 0.0f;
    __syncthreads();

    // Accumulate: for each track t in [iv_begin, iv_end)
    // Column index in dev_l6a_output = csr_offset + ev_col_offset + t
    for (int t = iv_begin; t < iv_end; ++t) {
        const unsigned col = csr_offset + ev_col_offset + (unsigned)t;
        // Each thread sums a strided subset of the l6a_m active neurons.
        for (int n = thread_id; n < (int)l6a_m; n += blockDim.x) {
            const float val = parameters.dev_pvfinder_l6a_output[(unsigned long long)n + (unsigned long long)col * 800u];
            if constexpr (UseAtomic) {
                atomicAdd(&s_feat[n], val);
            } else {
                s_feat[n] += val;
            }
            // Accumulate softplus-reduced histogram bin (n / 8 maps 800 → 100)
            // s_hist[n % 100] is updated after the full s_feat loop below.
        }
    }
    __syncthreads();

    // Reduce s_feat[800] → s_hist[100] via softplus of per-bin channel sums
    for (int k = thread_id; k < 100; k += blockDim.x) {
        float chan_sum = 0.0f;
        for (int c = 0; c < 8; ++c) chan_sum += s_feat[c * 100 + k];
        s_hist[k] = pvfinder_softplus(chan_sum);
    }
    __syncthreads();

    const float weight = 1.0f / n_local;

    float* g_feat = parameters.dev_pvfinder_interval_features
                    + (unsigned long long)event_number * 32000u + interval * 800u;
    for (int i = thread_id; i < 800; i += blockDim.x)
        g_feat[i] = s_feat[i] * weight;

    float* g_hist = parameters.dev_pvfinder_output_histogram
                    + event_number * 4000u + interval * 100u;
    for (int i = thread_id; i < 100; i += blockDim.x)
        g_hist[i] = s_hist[i] * weight;
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
    set_size<dev_pvfinder_interval_features_t> (arguments, padded_events * 32000);
    // CSR index buffers
    set_size<dev_pvfinder_interval_start_t>(arguments, total_events * 42);
    set_size<dev_pvfinder_track_idx_t>     (arguments, total_tracks  * 2);
#ifdef ALLEN_WITH_CUBLAS
    // intermediate buffers: sized for one B_CHUNK=20-event chunk, reused per chunk.
    //
    // IMPORTANT: Allen calls set_arguments_size() during Scheduler::Scheduler() init
    // BEFORE any events are loaded, so total_events may be 0.  We must never divide by
    // total_events directly.  Use a safe upper-bound of MAX_TRACKS_PER_EVENT instead.
    constexpr unsigned B_CHUNK              = 20u;
    constexpr unsigned MAX_TRACKS_PER_EVENT = 600u;  // generous upper bound for MinBias nu=7.6
    const unsigned avg_tracks_per_event =
        (total_events > 0u)
            ? ((total_tracks + total_events - 1u) / total_events)
            : MAX_TRACKS_PER_EVENT;
    // T_chunk_max: tracks per chunk × 2 (boundary-track double-counting in CSR).
    // Clamp to MAX_TRACKS_PER_EVENT * B_CHUNK * 2 so the L6A buffer stays ≤ 25.6 MB.
    const unsigned T_chunk_max =
        std::min(avg_tracks_per_event, MAX_TRACKS_PER_EVENT) * B_CHUNK * 2u;
    // dev_l5_output: [T_chunk_max × 20]  row-major — L1-L5 hidden states (~2.4 MB)
    set_size<dev_pvfinder_l5_output_t> (arguments, T_chunk_max * 20u);
    // dev_l6a_output: [800 × T_chunk_max]  column-major — raw L6A GEMM output (~24 MB)
    set_size<dev_pvfinder_l6a_output_t>(arguments, 800u * T_chunk_max);
#else
    set_size<dev_pvfinder_l5_output_t> (arguments, 0u);
    set_size<dev_pvfinder_l6a_output_t>(arguments, 0u);
#endif  // ALLEN_WITH_CUBLAS
}


void pvfinder_fc_aggregation_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        if (!PVFinder::WeightRegistry::instance().contains("fc_weights")) {
            std::string path = "/data/home/melashri/iris/inference/fc_weights.bin";
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f.is_open()) {
                throw std::runtime_error("Cannot open " + path);
            }
            const size_t bytes = static_cast<size_t>(f.tellg());
            f.seekg(0);
            std::vector<char> host_buf(bytes);
            f.read(host_buf.data(), bytes);

            // Transpose L6A weights from [800][20] to [20][800]
            // Offset to w6A is: 180+20 + 400+20 + 400+20 + 400+20 + 400+20 = 1880 floats
            float* floats = reinterpret_cast<float*>(host_buf.data());
            std::vector<float> w6A_transposed(16000);
            for (int r = 0; r < 800; ++r) {
                for (int c = 0; c < 20; ++c) {
                    w6A_transposed[c * 800 + r] = floats[1880 + r * 20 + c];
                }
            }
            std::memcpy(floats + 1880, w6A_transposed.data(), 16000 * sizeof(float));

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
    cudaMemsetAsync(
        data<dev_pvfinder_interval_features_t>(arguments),
        0,
        padded_events * 32000u * sizeof(float),
        context.stream());
    cudaMemsetAsync(
        data<dev_pvfinder_output_histogram_t>(arguments),
        0,
        n_events * 4000u * sizeof(float),
        context.stream());

#ifdef ALLEN_WITH_CUBLAS
    // -----------------------------------------------------------------------
    // Phase 2: 3-kernel + cuBLAS pipeline, chunked over B=20 events.
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
    const float* b6A = w6A + 16000;

    constexpr unsigned B_CHUNK = 20u;
    constexpr unsigned KERNEL1_BLOCK = 256u;
    constexpr unsigned KERNEL2_BLOCK = 512u;
    constexpr unsigned KERNEL3_BLOCK = 128u;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasHandle_t cublas = get_cublas_handle();
    cublasSetStream(cublas, context.stream());

    unsigned csr_col_offset = 0;  // running total of CSR entries before chunk_start
    for (unsigned chunk_start = 0; chunk_start < n_events; chunk_start += B_CHUNK) {
        const unsigned chunk_end = std::min(chunk_start + B_CHUNK, n_events);

        // Compute T_chunk from the already-copied host array — NO device reads here.
        unsigned T_chunk = 0;
        for (unsigned ev = chunk_start; ev < chunk_end; ++ev) {
            T_chunk += (unsigned)host_csr[ev * 42 + 41];  // sentinel = total CSR entries
        }
        if (T_chunk == 0) { csr_col_offset += T_chunk; continue; }

        // --- Kernel 1: L1-L5 per track in this chunk ---
        const unsigned k1_blocks = (T_chunk + KERNEL1_BLOCK - 1) / KERNEL1_BLOCK;
        global_function(pvfinder_l1_to_l5_kernel)(
            dim3(k1_blocks), dim3(KERNEL1_BLOCK), context)(
            arguments, dev_weights, chunk_start, chunk_end, csr_col_offset, T_chunk);

        // --- cuBLAS SGEMM: L6A ---
        // W6A [20×800] row-major = [800×20] col-major → CUBLAS_OP_T gives [800×20].
        // X [T_chunk×20] row-major = [20×T_chunk] col-major, op=N.
        // Y = W6A^T × X → [l6a_m×T_chunk] col-major → dev_l6a_output.
        // Only the M argument (rows computed) is overridable via m_l6a_m -- lda/ldb/ldc
        // stay fixed at the real buffer strides (20, 20, 800) regardless, since a
        // smaller M is a valid cuBLAS sub-block write into the same 800-row-stride
        // output buffer (see m_l6a_m doc comment for why this is throughput-only).
        const int l6a_m = (int)m_l6a_m.value();
        cublasSgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            l6a_m, (int)T_chunk, 20,
            &alpha,
            w6A, 20,
            data<dev_pvfinder_l5_output_t>(arguments), 20,
            &beta,
            data<dev_pvfinder_l6a_output_t>(arguments), 800);

        // --- Kernel 2: bias + LeakyReLU in-place on L6A output ---
        const unsigned l6a_m_u = (unsigned)l6a_m;
        const unsigned k2_blocks = (T_chunk * l6a_m_u + KERNEL2_BLOCK - 1) / KERNEL2_BLOCK;
        global_function(pvfinder_l6a_bias_relu_kernel)(
            dim3(k2_blocks), dim3(KERNEL2_BLOCK), context)(
            arguments, b6A, T_chunk, l6a_m_u);

        // --- Kernel 3: reduce L6A → interval features + histogram ---
        const unsigned n_chunk_events = chunk_end - chunk_start;
        if (m_use_nonatomic_l6a_reduce.value()) {
            global_function(pvfinder_reduce_l6a_kernel<false>)(
                dim3(n_chunk_events * 40u), dim3(KERNEL3_BLOCK), context)(
                arguments, chunk_start, chunk_end, csr_col_offset, T_chunk, l6a_m_u);
        } else {
            global_function(pvfinder_reduce_l6a_kernel<true>)(
                dim3(n_chunk_events * 40u), dim3(KERNEL3_BLOCK), context)(
                arguments, chunk_start, chunk_end, csr_col_offset, T_chunk, l6a_m_u);
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
