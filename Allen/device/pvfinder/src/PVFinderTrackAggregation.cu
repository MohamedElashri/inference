#include "PVFinderTrackAggregation.cuh"
#include "PVFinderWeightRegistry.h"
#include <mutex>

INSTANTIATE_ALGORITHM(pvfinder_track_aggregation::pvfinder_track_aggregation_t)

namespace pvfinder_track_aggregation {

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
// Interval-parallel FC+Aggregation kernel.
//
// Grid: (n_events, N_INTERVALS=40) — each block exclusively owns one interval.
// This gives 100*40=4000 blocks, saturating all 68 SMs on RTX 2080 Ti.
//
// Properties:
//  - No atomics into dev_pvfinder_interval_features (exclusive block ownership).
//  - Atomics for s_feat/s_hist go into shared memory (fast, single SM).
//  - Shared mem per block: s_feat[800] + s_hist[100] + s_count = ~3.6 KB.
// ---------------------------------------------------------------------------
__global__ void pvfinder_fused_fc_aggregation_kernel(
    pvfinder_track_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights)
{
    const unsigned event_number = blockIdx.x;
    const unsigned interval     = blockIdx.y;
    const unsigned thread_id    = threadIdx.x;
    const unsigned warp_id      = thread_id / warpSize;
    const unsigned n_warps      = blockDim.x / warpSize;

    const auto velo_tracks_view      = parameters.dev_velo_tracks_view[event_number];
    const unsigned num_tracks        = velo_tracks_view.size();
    const unsigned event_track_offset = velo_tracks_view.offset();

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
    __shared__ int   s_count;

    for (int i = thread_id; i < 800; i += blockDim.x) s_feat[i] = 0.0f;
    for (int i = thread_id; i < 100; i += blockDim.x) s_hist[i] = 0.0f;
    if (thread_id == 0) s_count = 0;
    __syncthreads();

    // Pass 1: count tracks in this interval
    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        const unsigned gtidx  = event_track_offset + i;
        const float    z_poca = parameters.dev_pvfinder_track_features[gtidx * 9 + 2];
        if (z_poca <= -98.0f) continue;
        int ivals[2]; int n = 0;
        assign_intervals(z_poca, ivals, &n);
        for (int j = 0; j < n; ++j)
            if ((unsigned)ivals[j] == interval) { atomicAdd(&s_count, 1); break; }
    }
    __syncthreads();

    // Pass 2: fused FC + accumulation into shared s_feat / s_hist
    for (unsigned i = warp_id; i < num_tracks; i += n_warps) {
        const unsigned gtidx  = event_track_offset + i;
        const float    z_poca = parameters.dev_pvfinder_track_features[gtidx * 9 + 2];
        if (z_poca <= -98.0f) continue;

        int ivals[2]; int ni = 0;
        assign_intervals(z_poca, ivals, &ni);

        bool mine = false;
        for (int j = 0; j < ni; ++j)
            if ((unsigned)ivals[j] == interval) { mine = true; break; }
        if (!mine) continue;

        const float* feat = parameters.dev_pvfinder_track_features + gtidx * 9;
        float x1[20], x2[20];
        pvfinder_linear_layer_reg(feat, x1, w1, b1, 9,  20);
        pvfinder_linear_layer_reg(x1,  x2, w2, b2, 20, 20);
        pvfinder_linear_layer_reg(x2,  x1, w3, b3, 20, 20);
        pvfinder_linear_layer_reg(x1,  x2, w4, b4, 20, 20);
        pvfinder_linear_layer_reg(x2,  x1, w5, b5, 20, 20);

        const int lane_id = thread_id % warpSize;
        for (int k = lane_id; k < 100; k += warpSize) {
            float chan_sum = 0.0f;
            for (int c = 0; c < 8; ++c) {
                const int    neuron = c * 100 + k;
                const float* row    = w6A + neuron * 20;
                float val = b6A[neuron];
                for (int m = 0; m < 20; ++m) val += row[m] * x1[m];
                val = pvfinder_leaky_relu(val);
                chan_sum += val;
                atomicAdd(&s_feat[c * 100 + k], val);
            }
            atomicAdd(&s_hist[k], pvfinder_softplus(chan_sum));
        }
    }
    __syncthreads();

    const float weight = s_count > 0 ? 1.0f / s_count : 1.0f;

    float* g_feat = parameters.dev_pvfinder_interval_features
                    + event_number * 32000 + interval * 800;
    for (int i = thread_id; i < 800; i += blockDim.x)
        g_feat[i] = s_feat[i] * weight;

    float* g_hist = parameters.dev_pvfinder_output_histogram
                    + event_number * 4000 + interval * 100;
    for (int i = thread_id; i < 100; i += blockDim.x)
        g_hist[i] = s_hist[i] * weight;
}

void pvfinder_track_aggregation_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned total_events = first<host_number_of_events_t>(arguments);
    set_size<dev_pvfinder_output_histogram_t>(arguments,  total_events * 4000);
    set_size<dev_pvfinder_interval_features_t>(arguments, total_events * 32000);
}

void pvfinder_track_aggregation_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        if (!PVFinder::WeightRegistry::instance().contains("fc_weights")) {
            PVFinder::WeightRegistry::instance().load(
                "fc_weights", "/data/home/melashri/iris/inference/fc_weights.bin");
        }
    });
    const float* dev_weights = PVFinder::WeightRegistry::instance().get<float>("fc_weights");

    global_function(pvfinder_fused_fc_aggregation_kernel)(
        dim3(first<host_number_of_events_t>(arguments), 40), m_block_dim, context)(
        arguments, dev_weights);
}

} // namespace pvfinder_track_aggregation
