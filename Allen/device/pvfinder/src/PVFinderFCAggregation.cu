#include "PVFinderFCAggregation.cuh"
#include "PVFinderWeightRegistry.h"
#include <mutex>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
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

static constexpr int FC_FIRST_W_OFFSET = 0;
static constexpr int FC_FIRST_B_OFFSET = FC_FIRST_W_OFFSET + FC_INPUT_DIM * FC_HIDDEN_DIM;
static constexpr int FC_HIDDEN_BLOCK_FLOATS = FC_HIDDEN_DIM * FC_HIDDEN_DIM + FC_HIDDEN_DIM;
static constexpr int FC_HIDDEN_BASE_OFFSET = FC_FIRST_B_OFFSET + FC_HIDDEN_DIM;
static constexpr int FC_FINAL_W_OFFSET =
    FC_HIDDEN_BASE_OFFSET + (FC_HIDDEN_LAYERS - 1) * FC_HIDDEN_BLOCK_FLOATS;
static constexpr int FC_FINAL_B_OFFSET = FC_FINAL_W_OFFSET + FC_HIDDEN_DIM * FC_OUTPUT_DIM;
static constexpr int FC_TOTAL_FLOATS = FC_FINAL_B_OFFSET + FC_OUTPUT_DIM;
static constexpr int FC_INTERVAL_FEATURES_PER_EVENT = FC_INTERVALS * FC_OUTPUT_DIM;

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

__device__ __forceinline__ void pvfinder_hidden_stack_reg(
    const float* __restrict__ feat,
    float* __restrict__ x1,
    float* __restrict__ x2,
    const float* __restrict__ dev_weights)
{
    const float* w = dev_weights + FC_FIRST_W_OFFSET;
    const float* b = dev_weights + FC_FIRST_B_OFFSET;
    pvfinder_linear_layer_reg(feat, x1, w, b, FC_INPUT_DIM, FC_HIDDEN_DIM);

    for (int layer = 1; layer < FC_HIDDEN_LAYERS; ++layer) {
        w = dev_weights + FC_HIDDEN_BASE_OFFSET + (layer - 1) * FC_HIDDEN_BLOCK_FLOATS;
        b = w + FC_HIDDEN_DIM * FC_HIDDEN_DIM;
        if (layer & 1) {
            pvfinder_linear_layer_reg(x1, x2, w, b, FC_HIDDEN_DIM, FC_HIDDEN_DIM);
        }
        else {
            pvfinder_linear_layer_reg(x2, x1, w, b, FC_HIDDEN_DIM, FC_HIDDEN_DIM);
        }
    }
}

__device__ __forceinline__ const float* pvfinder_final_hidden(
    float* __restrict__ x1,
    float* __restrict__ x2)
{
    return (FC_HIDDEN_LAYERS & 1) ? x1 : x2;
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
// Compact work-list builder.
//
// Grid: (n_events)  blockDim: 64
//
// Each block scans all 40 intervals for its event and appends a packed entry
//   (event_number << 6) | interval
// to dev_pvfinder_nonempty_slots[] for every interval that has >=1 track.
// A global atomic counter in dev_pvfinder_nonempty_count[0] gives the final
// list length, which the host then uses as the grid dimension for the
// aggregation kernel.
// ---------------------------------------------------------------------------
__global__ void pvfinder_compact_nonempty_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    unsigned n_events)
{
    const unsigned event_number = blockIdx.x;
    if (event_number >= n_events) return;

    const int* g_start = parameters.dev_pvfinder_interval_start + event_number * 42;

    // Each thread claims one or more intervals.
    for (int iv = (int)threadIdx.x; iv < 40; iv += (int)blockDim.x) {
        const int count = g_start[iv + 1] - g_start[iv];
        if (count > 0) {
            // Atomically reserve a slot in the output list.
            int pos = atomicAdd(parameters.dev_pvfinder_nonempty_count, 1);
            // Pack: upper bits = event, lower 6 bits = interval (0-39).
            parameters.dev_pvfinder_nonempty_slots[pos] =
                (event_number << 6u) | (unsigned)iv;
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
// Static shared mem: s_feat[FC_OUTPUT_DIM] + s_hist[100].
// This keeps occupancy high (many blocks resident per SM) which is the
// dominant factor on both SM 7.5 and SM 8.6.
//
// NOTE: final-layer weight caching in shared memory was tried for the old 800-output model
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

    const float* w6A = dev_weights + FC_FINAL_W_OFFSET;
    const float* b6A = dev_weights + FC_FINAL_B_OFFSET;

    __shared__ float s_feat[FC_OUTPUT_DIM];
    __shared__ float s_hist[100];      // 0.4 KB

    for (int i = thread_id; i < FC_OUTPUT_DIM; i += blockDim.x) s_feat[i] = 0.0f;
    for (int i = thread_id; i < 100; i += blockDim.x) s_hist[i] = 0.0f;
    __syncthreads();

    // Process tracks in batches of size warpSize (32) per warp.
    for (int i = warp_id * warpSize; i < n_local; i += n_warps * warpSize) {
        const int lane_id = thread_id % warpSize;
        const int track_idx_in_batch = i + lane_id;
        const bool valid_track = track_idx_in_batch < n_local;

        float x1[FC_HIDDEN_DIM], x2[FC_HIDDEN_DIM];
        
        // Phase 1: thread-parallel hidden stack.
        // Each thread processes the hidden stack for a unique track.
        if (valid_track) {
            const int local_idx = g_idx[iv_begin + track_idx_in_batch];
            const unsigned gtidx = event_track_offset + (unsigned)local_idx;
            const float* feat = parameters.dev_pvfinder_track_features + gtidx * 9;
            
            pvfinder_hidden_stack_reg(feat, x1, x2, dev_weights);
        }

        // Phase 2: warp-collaborative final layer.
        // Loop over the tracks in this warp's current batch.
        const int batch_size = min(warpSize, n_local - i);
        for (int t = 0; t < batch_size; ++t) {
            
            // Broadcast the target track's x1 array to all lanes in the warp
            const float* hidden = pvfinder_final_hidden(x1, x2);
            float broadcasted_x1[FC_HIDDEN_DIM];
            for (int m = 0; m < FC_HIDDEN_DIM; ++m) {
                broadcasted_x1[m] = __shfl_sync(0xffffffff, hidden[m], t);
            }

            // All lanes collaboratively process the final layer for track t.
            for (int k = lane_id; k < 100; k += warpSize) {
                float chan_sum = 0.0f;
                for (int c = 0; c < FC_LATENT_CHANNELS; ++c) {
                    const int neuron = c * 100 + k;
                    float val = b6A[neuron];
                    for (int m = 0; m < FC_HIDDEN_DIM; ++m) {
                        val += w6A[m * FC_OUTPUT_DIM + neuron] * broadcasted_x1[m];
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
                    + event_number * FC_INTERVAL_FEATURES_PER_EVENT + interval * FC_OUTPUT_DIM;
    for (int i = thread_id; i < FC_OUTPUT_DIM; i += blockDim.x)
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
// Kernel 1a — gather CSR-ordered track inputs for the all-cuBLAS experiment.
//
// Output layout is column-major [FC_INPUT_DIM x T_chunk], equivalent to a
// row-major [T_chunk x FC_INPUT_DIM] view for simple writes.
// ---------------------------------------------------------------------------
__global__ void pvfinder_gather_fc_input_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    unsigned chunk_start,
    unsigned chunk_end,
    unsigned T_chunk)
{
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T_chunk) return;

    unsigned ev = chunk_start;
    unsigned local_t = t;
    while (ev < chunk_end) {
        const int* g_start_ev = parameters.dev_pvfinder_interval_start + ev * 42;
        const unsigned n_entries = (unsigned)g_start_ev[41];
        if (local_t < n_entries) break;
        local_t -= n_entries;
        ++ev;
    }
    if (ev >= chunk_end) return;

    const auto velo_tracks_view = parameters.dev_velo_tracks_view[ev];
    const unsigned event_track_offset = velo_tracks_view.offset();
    const int* g_idx = parameters.dev_pvfinder_track_idx + event_track_offset * 2;
    const int local_track = g_idx[local_t];
    const unsigned gtidx = event_track_offset + (unsigned)local_track;
    const float* feat = parameters.dev_pvfinder_track_features + gtidx * FC_INPUT_DIM;
    float* out = parameters.dev_pvfinder_fc_input + (unsigned long long)t * FC_INPUT_DIM;
    for (int i = 0; i < FC_INPUT_DIM; ++i) out[i] = feat[i];
}

// ---------------------------------------------------------------------------
// Kernel 1b — hidden stack per track.
//
// Grid: ceil(T_chunk / 256)  blockDim: 256
//
// Each thread processes one CSR track entry for events in [chunk_start, chunk_end).
// Evaluates the hidden MLP in registers and writes the FC_HIDDEN_DIM-float
// hidden state to dev_pvfinder_l5_output at the corresponding row.
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

    // Evaluate hidden stack in registers
    const float* feat = parameters.dev_pvfinder_track_features + gtidx * 9;
    float x1[FC_HIDDEN_DIM], x2[FC_HIDDEN_DIM];
    pvfinder_hidden_stack_reg(feat, x1, x2, dev_weights);
    const float* hidden = pvfinder_final_hidden(x1, x2);

    // Write hidden state as row t of dev_pvfinder_l5_output [T_chunk x FC_HIDDEN_DIM] row-major
    float* out = parameters.dev_pvfinder_l5_output + (unsigned long long)t * FC_HIDDEN_DIM;
    for (int m = 0; m < FC_HIDDEN_DIM; ++m) out[m] = hidden[m];
}

// ---------------------------------------------------------------------------
// Kernel 2 — Apply bias + LeakyReLU in-place after cuBLAS GEMM.
//
// The buffer is column-major [features x T_chunk].
//
// Grid: ceil(T_chunk * features / 512)  blockDim: 512
// ---------------------------------------------------------------------------
__global__ void pvfinder_bias_relu_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    unsigned features,
    unsigned T_chunk)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T_chunk * features) return;
    const unsigned n = idx % features;
    output[idx] = pvfinder_leaky_relu(output[idx] + bias[n]);
}

// ---------------------------------------------------------------------------
// Kernel 3 — Reduce final-layer outputs into interval features and histogram.
//
// Grid: (B_chunk * 40)  blockDim: 128
//
// Each block handles one (event, interval) slot. The kernel maps blockIdx.x
// to (event, interval) via the CSR bounds — no DtoH copy needed. Blocks
// that correspond to empty intervals or out-of-range events early-return.
//
// dev_l6a_output is column-major [FC_OUTPUT_DIM x T_chunk]. For track at column t:
//   neuron n -> dev_l6a_output[n + t*FC_OUTPUT_DIM]
// ---------------------------------------------------------------------------
__global__ void pvfinder_reduce_l6a_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    unsigned chunk_start,
    unsigned chunk_end,
    unsigned csr_offset,        // offset into the global CSR that corresponds to chunk_start
    unsigned T_chunk)
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

    // Shared memory accumulators: s_feat[FC_OUTPUT_DIM] + s_hist[100]
    __shared__ float s_feat[FC_OUTPUT_DIM];
    __shared__ float s_hist[100];
    const unsigned thread_id = threadIdx.x;
    for (int i = thread_id; i < FC_OUTPUT_DIM; i += blockDim.x) s_feat[i] = 0.0f;
    for (int i = thread_id; i < 100; i += blockDim.x) s_hist[i] = 0.0f;
    __syncthreads();

    // Accumulate: for each track t in [iv_begin, iv_end)
    // Column index in dev_l6a_output = csr_offset + ev_col_offset + t
    for (int t = iv_begin; t < iv_end; ++t) {
        const unsigned col = csr_offset + ev_col_offset + (unsigned)t;
        // Each thread sums a strided subset of the final-layer neurons.
        for (int n = thread_id; n < FC_OUTPUT_DIM; n += blockDim.x) {
            const float val = parameters.dev_pvfinder_l6a_output[
                (unsigned long long)n + (unsigned long long)col * FC_OUTPUT_DIM];
            atomicAdd(&s_feat[n], val);
            // Accumulate softplus-reduced histogram bin after the full s_feat loop below.
            // s_hist[n % 100] is updated after the full s_feat loop below.
        }
    }
    __syncthreads();

    // Reduce s_feat[FC_OUTPUT_DIM] -> s_hist[100] via softplus of per-bin channel sums.
    for (int k = thread_id; k < 100; k += blockDim.x) {
        float chan_sum = 0.0f;
        for (int c = 0; c < FC_LATENT_CHANNELS; ++c) chan_sum += s_feat[c * 100 + k];
        s_hist[k] = pvfinder_softplus(chan_sum);
    }
    __syncthreads();

    const float weight = 1.0f / n_local;

    float* g_feat = parameters.dev_pvfinder_interval_features
                    + (unsigned long long)event_number * FC_INTERVAL_FEATURES_PER_EVENT +
                      interval * FC_OUTPUT_DIM;
    for (int i = thread_id; i < FC_OUTPUT_DIM; i += blockDim.x)
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
    set_size<dev_pvfinder_interval_features_t> (arguments, padded_events * FC_INTERVAL_FEATURES_PER_EVENT);
    // CSR index buffers
    set_size<dev_pvfinder_interval_start_t>(arguments, total_events * 42);
    set_size<dev_pvfinder_track_idx_t>     (arguments, total_tracks  * 2);
    // buffers (compact work-list — allocated, currently unused)
    set_size<dev_pvfinder_nonempty_slots_t>(arguments, total_events * 40);
    set_size<dev_pvfinder_nonempty_count_t>(arguments, 1);
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
    // dev_fc_input: [T_chunk_max x FC_INPUT_DIM] row-major CSR-ordered inputs
    set_size<dev_pvfinder_fc_input_t>  (arguments, T_chunk_max * FC_INPUT_DIM);
    // dev_l5_output: [T_chunk_max x FC_HIDDEN_DIM] row-major hidden states
    set_size<dev_pvfinder_l5_output_t> (arguments, T_chunk_max * FC_HIDDEN_DIM);
    // dev_hidden_tmp: [T_chunk_max x FC_HIDDEN_DIM] row-major hidden-state ping-pong buffer
    set_size<dev_pvfinder_hidden_tmp_t>(arguments, T_chunk_max * FC_HIDDEN_DIM);
    // dev_l6a_output: [FC_OUTPUT_DIM x T_chunk_max] column-major raw final-layer output
    set_size<dev_pvfinder_l6a_output_t>(arguments, FC_OUTPUT_DIM * T_chunk_max);
#else
    set_size<dev_pvfinder_fc_input_t>  (arguments, 0u);
    set_size<dev_pvfinder_l5_output_t> (arguments, 0u);
    set_size<dev_pvfinder_hidden_tmp_t>(arguments, 0u);
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
    std::call_once(flag, [this]() {
        if (!PVFinder::WeightRegistry::instance().contains("fc_weights")) {
            std::string path = m_weight_file.value();
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f.is_open()) {
                throw std::runtime_error("Cannot open " + path);
            }
            const size_t bytes = static_cast<size_t>(f.tellg());
            const size_t expected_bytes = static_cast<size_t>(FC_TOTAL_FLOATS) * sizeof(float);
            if (bytes != expected_bytes) {
                throw std::runtime_error(
                    "PVFinder FC weight file has " + std::to_string(bytes) +
                    " bytes, expected " + std::to_string(expected_bytes) +
                    " for architecture 9->" + std::to_string(FC_HIDDEN_DIM) +
                    "x" + std::to_string(FC_HIDDEN_LAYERS) +
                    "->" + std::to_string(FC_OUTPUT_DIM) + ": " + path);
            }
            f.seekg(0);
            std::vector<char> host_buf(bytes);
            f.read(host_buf.data(), bytes);

            // Transpose final-layer weights from [FC_OUTPUT_DIM][FC_HIDDEN_DIM]
            // to [FC_HIDDEN_DIM][FC_OUTPUT_DIM] for coalesced kernel/cuBLAS access.
            float* floats = reinterpret_cast<float*>(host_buf.data());
            std::vector<float> w6A_transposed(FC_HIDDEN_DIM * FC_OUTPUT_DIM);
            for (int r = 0; r < FC_OUTPUT_DIM; ++r) {
                for (int c = 0; c < FC_HIDDEN_DIM; ++c) {
                    w6A_transposed[c * FC_OUTPUT_DIM + r] =
                        floats[FC_FINAL_W_OFFSET + r * FC_HIDDEN_DIM + c];
                }
            }
            std::memcpy(
                floats + FC_FINAL_W_OFFSET,
                w6A_transposed.data(),
                w6A_transposed.size() * sizeof(float));

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
        padded_events * FC_INTERVAL_FEATURES_PER_EVENT * sizeof(float),
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

    const float* w6A = dev_weights + FC_FINAL_W_OFFSET;
    const float* b6A = dev_weights + FC_FINAL_B_OFFSET;

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

        const unsigned k1_blocks = (T_chunk + KERNEL1_BLOCK - 1) / KERNEL1_BLOCK;
        const float* final_hidden = data<dev_pvfinder_l5_output_t>(arguments);

        if (m_use_cublas_hidden_stack.value()) {
            // --- Kernel 1a: gather FC inputs in CSR order ---
            global_function(pvfinder_gather_fc_input_kernel)(
                dim3(k1_blocks), dim3(KERNEL1_BLOCK), context)(
                arguments, chunk_start, chunk_end, T_chunk);

            // --- cuBLAS hidden stack: 9->32 and 32->32 layers ---
            const float* layer_w = dev_weights + FC_FIRST_W_OFFSET;
            const float* layer_b = dev_weights + FC_FIRST_B_OFFSET;
            float* hidden_a = data<dev_pvfinder_l5_output_t>(arguments);
            float* hidden_b = data<dev_pvfinder_hidden_tmp_t>(arguments);

            cublasSgemm(cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                FC_HIDDEN_DIM, (int)T_chunk, FC_INPUT_DIM,
                &alpha,
                layer_w, FC_INPUT_DIM,
                data<dev_pvfinder_fc_input_t>(arguments), FC_INPUT_DIM,
                &beta,
                hidden_a, FC_HIDDEN_DIM);

            const unsigned hidden_bias_blocks =
                (T_chunk * FC_HIDDEN_DIM + KERNEL2_BLOCK - 1) / KERNEL2_BLOCK;
            global_function(pvfinder_bias_relu_kernel)(
                dim3(hidden_bias_blocks), dim3(KERNEL2_BLOCK), context)(
                hidden_a, layer_b, FC_HIDDEN_DIM, T_chunk);

            for (int layer = 1; layer < FC_HIDDEN_LAYERS; ++layer) {
                layer_w = dev_weights + FC_HIDDEN_BASE_OFFSET + (layer - 1) * FC_HIDDEN_BLOCK_FLOATS;
                layer_b = layer_w + FC_HIDDEN_DIM * FC_HIDDEN_DIM;
                float* input_hidden = (layer & 1) ? hidden_a : hidden_b;
                float* output_hidden = (layer & 1) ? hidden_b : hidden_a;

                cublasSgemm(cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    FC_HIDDEN_DIM, (int)T_chunk, FC_HIDDEN_DIM,
                    &alpha,
                    layer_w, FC_HIDDEN_DIM,
                    input_hidden, FC_HIDDEN_DIM,
                    &beta,
                    output_hidden, FC_HIDDEN_DIM);

                global_function(pvfinder_bias_relu_kernel)(
                    dim3(hidden_bias_blocks), dim3(KERNEL2_BLOCK), context)(
                    output_hidden, layer_b, FC_HIDDEN_DIM, T_chunk);
            }

            final_hidden =
                (FC_HIDDEN_LAYERS & 1) ?
                    data<dev_pvfinder_l5_output_t>(arguments) :
                    data<dev_pvfinder_hidden_tmp_t>(arguments);
        }
        else {
            // --- Kernel 1b: register hidden stack per track in this chunk ---
            global_function(pvfinder_l1_to_l5_kernel)(
                dim3(k1_blocks), dim3(KERNEL1_BLOCK), context)(
                arguments, dev_weights, chunk_start, chunk_end, csr_col_offset, T_chunk);
        }

        // --- cuBLAS SGEMM: final layer ---
        // W6A [FC_HIDDEN_DIM x FC_OUTPUT_DIM] row-major =
        // [FC_OUTPUT_DIM x FC_HIDDEN_DIM] col-major with CUBLAS_OP_T.
        // X [T_chunk x FC_HIDDEN_DIM] row-major =
        // [FC_HIDDEN_DIM x T_chunk] col-major, op=N.
        // Y = W6A^T x X -> [FC_OUTPUT_DIM x T_chunk] col-major.
        cublasSgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            FC_OUTPUT_DIM, (int)T_chunk, FC_HIDDEN_DIM,
            &alpha,
            w6A, FC_HIDDEN_DIM,
            final_hidden, FC_HIDDEN_DIM,
            &beta,
            data<dev_pvfinder_l6a_output_t>(arguments), FC_OUTPUT_DIM);

        // --- Kernel 2: bias + LeakyReLU in-place on final-layer output ---
        const unsigned k2_blocks = (T_chunk * FC_OUTPUT_DIM + KERNEL2_BLOCK - 1) / KERNEL2_BLOCK;
        global_function(pvfinder_bias_relu_kernel)(
            dim3(k2_blocks), dim3(KERNEL2_BLOCK), context)(
            data<dev_pvfinder_l6a_output_t>(arguments), b6A, FC_OUTPUT_DIM, T_chunk);

        // --- Kernel 3: reduce final-layer output -> interval features + histogram ---
        const unsigned n_chunk_events = chunk_end - chunk_start;
        global_function(pvfinder_reduce_l6a_kernel)(
            dim3(n_chunk_events * 40u), dim3(KERNEL3_BLOCK), context)(
            arguments, chunk_start, chunk_end, csr_col_offset, T_chunk);

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
