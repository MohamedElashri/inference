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

// Fused FC+Aggregation kernel.
// Computes the full 6-layer MLP per track entirely in registers (800 floats
// only live as a local array — never written to global memory), then
// accumulates the latent values directly into the interval feature and
// histogram buffers.  The 391 MB global latent buffer is eliminated.
__global__ void pvfinder_fused_fc_aggregation_kernel(
    pvfinder_track_aggregation_t::Parameters parameters,
    const float* __restrict__ dev_weights)
{
    const unsigned event_number = blockIdx.x;
    const unsigned thread_id    = threadIdx.x;
    const unsigned warp_id      = thread_id / warpSize;

    const auto velo_tracks_view  = parameters.dev_velo_tracks_view[event_number];
    const unsigned num_tracks    = velo_tracks_view.size();
    const unsigned event_track_offset = velo_tracks_view.offset();

    // FC weight layout (matches PVFinderFCEngine.cu exactly):
    //   L1: W(20x9)=180  B(20)
    //   L2: W(20x20)=400 B(20)  x4
    //   L6A: W(800x20)=16000 B(800)
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

    // Shared memory: interval counts + collapsed histogram
    __shared__ int   s_interval_counts[40];
    __shared__ float s_output_histograms[4000];

    for (int i = thread_id; i < 40;   i += blockDim.x) s_interval_counts[i]  = 0;
    for (int i = thread_id; i < 4000; i += blockDim.x) s_output_histograms[i] = 0.0f;
    // Zero the global interval-features buffer for this event
    for (int i = thread_id; i < 32000; i += blockDim.x)
        parameters.dev_pvfinder_interval_features[event_number * 32000 + i] = 0.0f;

    __syncthreads();

    // Pass 1: count tracks per interval (needed for normalisation weight)
    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        const unsigned gtidx   = event_track_offset + i;
        const float    z_poca  = parameters.dev_pvfinder_track_features[gtidx * 9 + 2];
        if (z_poca > -98.0f) {
            int intervals[2]; int n = 0;
            assign_intervals(z_poca, intervals, &n);
            for (int j = 0; j < n; ++j) atomicAdd(&s_interval_counts[intervals[j]], 1);
        }
    }

    __syncthreads();

    // Pass 2: fused FC inference + aggregation (one track per warp)
    for (unsigned i = warp_id; i < num_tracks; i += (blockDim.x / warpSize)) {
        const unsigned gtidx  = event_track_offset + i;
        const float    z_poca = parameters.dev_pvfinder_track_features[gtidx * 9 + 2];
        if (z_poca <= -98.0f) continue;

        int intervals[2]; int n = 0;
        assign_intervals(z_poca, intervals, &n);

        // Run the 6-layer MLP entirely in registers — no global writes
        const float* feat = parameters.dev_pvfinder_track_features + gtidx * 9;
        float x1[20], x2[20];
        pvfinder_linear_layer_reg(feat, x1, w1, b1, 9,  20);
        pvfinder_linear_layer_reg(x1,  x2, w2, b2, 20, 20);
        pvfinder_linear_layer_reg(x2,  x1, w3, b3, 20, 20);
        pvfinder_linear_layer_reg(x1,  x2, w4, b4, 20, 20);
        pvfinder_linear_layer_reg(x2,  x1, w5, b5, 20, 20);
        // Layer 6A: 20 -> 800, stored in a register-local array
        float latent[800];
        for (int o = 0; o < 800; ++o) {
            float sum = b6A[o];
            for (int k = 0; k < 20; ++k) sum += w6A[o * 20 + k] * x1[k];
            latent[o] = pvfinder_leaky_relu(sum);
        }

        // Accumulate into interval features and histogram (lane-parallel within warp)
        const int lane_id = thread_id % warpSize;
        for (int j = 0; j < n; ++j) {
            const int interval = intervals[j];
            for (int k = lane_id; k < 100; k += warpSize) {
                float sum = 0.0f;
                for (int c = 0; c < 8; ++c) {
                    const float val = latent[c * 100 + k];
                    sum += val;
                    atomicAdd(
                        &parameters.dev_pvfinder_interval_features[
                            event_number * 32000 + (interval * 8 + c) * 100 + k],
                        val);
                }
                atomicAdd(&s_output_histograms[interval * 100 + k], pvfinder_softplus(sum));
            }
        }
    }

    __syncthreads();

    // Normalise and write outputs
    for (int i = thread_id; i < 4000; i += blockDim.x) {
        const int   interval = i / 100;
        const float weight   = s_interval_counts[interval] > 0
                               ? 1.0f / s_interval_counts[interval] : 1.0f;
        parameters.dev_pvfinder_output_histogram[event_number * 4000 + i] =
            s_output_histograms[i] * weight;
    }
    for (int i = thread_id; i < 32000; i += blockDim.x) {
        const int   interval = i / 800;
        const float weight   = s_interval_counts[interval] > 0
                               ? 1.0f / s_interval_counts[interval] : 1.0f;
        parameters.dev_pvfinder_interval_features[event_number * 32000 + i] *= weight;
    }
}

void pvfinder_track_aggregation_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned total_events = first<host_number_of_events_t>(arguments);
    set_size<dev_pvfinder_output_histogram_t>(arguments,   total_events * 4000);
    set_size<dev_pvfinder_interval_features_t>(arguments,  total_events * 32000);
    // FC weights are a persistent device pointer — zero pool allocation needed
    set_size<dev_pvfinder_fc_weights_t>(arguments, 0);
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
        dim3(first<host_number_of_events_t>(arguments)), m_block_dim, context)(
        arguments, dev_weights);
}

} // namespace pvfinder_track_aggregation
