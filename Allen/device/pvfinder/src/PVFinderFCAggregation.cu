#include "PVFinderFCAggregation.cuh"
#include "PVFinderWeightRegistry.h"
#include <mutex>
#include <fstream>
#include <vector>
#include <string>

INSTANTIATE_ALGORITHM(pvfinder_fc_aggregation::pvfinder_fc_aggregation_t)

namespace pvfinder_fc_aggregation {

__device__ void assign_intervals(float z_poca, int* intervals, int* num_intervals) {
    z_poca += 100.0f;
    int in_range   = (z_poca >= 0.0f) & (z_poca <= 400.0f);
    int at_lo_edge = (z_poca <= 10.0f);
    int at_hi_edge = (z_poca >= 390.0f);
    *num_intervals = 1 + (in_range & (at_lo_edge | at_hi_edge));
    int base_interval = min(39, max(0, int(z_poca / 10.0f)));
    intervals[0] = base_interval;
    intervals[1] = base_interval + (2 * at_lo_edge - 1);
}

__device__ float pvfinder_softplus(float x) {
    return fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
}

// Inline LeakyReLU and linear layer — runs in registers, no global writes.
__device__ __forceinline__ float pvfinder_leaky_relu(float x) {
    return fmaxf(x, 0.0f) + 0.01f * fminf(x, 0.0f);
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
// ---------------------------------------------------------------------------
__global__ void pvfinder_fused_fc_aggregation_kernel(
    pvfinder_fc_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights)
{
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
    // CSR: 42 ints per event (40 interval starts + total sentinel + 1 extra for g_start[41])
    // track_idx: at most 2 entries per track (boundary tracks assigned to 2 intervals)
    set_size<dev_pvfinder_interval_start_t>(arguments, total_events * 42);
    set_size<dev_pvfinder_track_idx_t>     (arguments, total_tracks  * 2);
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
            // Because w1: 180, b1: 20
            // w2: 400, b2: 20
            // w3: 400, b3: 20
            // w4: 400, b4: 20
            // w5: 400, b5: 20
            // Total before w6A: 200 + 420 + 420 + 420 + 420 = 1880 floats
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

    global_function(pvfinder_build_csr_kernel)(
        dim3(first<host_number_of_events_t>(arguments)), m_block_dim, context)(
        arguments);

    global_function(pvfinder_fused_fc_aggregation_kernel)(
        dim3(first<host_number_of_events_t>(arguments), 40), m_block_dim, context)(
        arguments, dev_weights);
}

} // namespace pvfinder_fc_aggregation
