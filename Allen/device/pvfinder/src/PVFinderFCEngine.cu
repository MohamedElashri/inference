#include "PVFinderFCEngine.cuh"
#include "PVFinderWeightRegistry.h"
#include <mutex>

INSTANTIATE_ALGORITHM(pvfinder_fc_engine::pvfinder_fc_engine_t)

namespace pvfinder_fc_engine {

__device__ float pvfinder_leaky_relu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}

__device__ void pvfinder_linear_layer(const float* x, float* y, const float* w, const float* b, int in_f, int out_f) {
    for (int i = 0; i < out_f; ++i) {
        float sum = b[i];
        for (int j = 0; j < in_f; ++j) {
            sum += w[i * in_f + j] * x[j];
        }
        y[i] = pvfinder_leaky_relu(sum);
    }
}

__global__ void pvfinder_fc_engine_kernel(
    pvfinder_fc_engine_t::Parameters parameters,
    const float* dev_weights)
{
    const unsigned event_number = blockIdx.x;
    const unsigned thread_id = threadIdx.x;

    const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];
    const unsigned num_tracks = velo_tracks_view.size();
    unsigned event_track_offset = velo_tracks_view.offset();

    // Setup pointers to the contiguous device weights memory loaded statically from binary format.
    // Layer 1: W (20x9) = 180, B (20)
    // Layer 2: W (20x20) = 400, B (20)
    // Layer 3: W (20x20) = 400, B (20)
    // Layer 4: W (20x20) = 400, B (20)
    // Layer 5: W (20x20) = 400, B (20)
    // Layer 6A: W (800x20) = 16000, B (800)
    
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
    const float* w6A = b5 + 20;                  
    const float* b6A = w6A + 16000;              

    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        unsigned global_track_idx = event_track_offset + i;
        const float* track_features = ((const float*)parameters.dev_pvfinder_track_features) + (global_track_idx * 9);
        float* latent_features = parameters.dev_pvfinder_latent_features + (global_track_idx * 800);

        // Process linear layers.
        // Array references x1 and x2 alternate handling sequential memory representations avoiding allocation bugs.
        float x1[20];
        float x2[20];

        pvfinder_linear_layer(track_features, x1, w1, b1, 9, 20);
        pvfinder_linear_layer(x1, x2, w2, b2, 20, 20);
        pvfinder_linear_layer(x2, x1, w3, b3, 20, 20);
        pvfinder_linear_layer(x1, x2, w4, b4, 20, 20);
        pvfinder_linear_layer(x2, x1, w5, b5, 20, 20);
        pvfinder_linear_layer(x1, latent_features, w6A, b6A, 20, 800);
        // if (event_number == 0 && i == 0) {
        //     printf("DEBUG PVFinderFCEngine [Event 0, Track 0]: Latent[0..5] = %f, %f, %f, %f, %f, %f\n",
        //            latent_features[0], latent_features[1], latent_features[2], latent_features[3], latent_features[4], latent_features[5]);
        // }
    }
}

void pvfinder_fc_engine_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    unsigned total_tracks = first<host_number_of_reconstructed_velo_tracks_t>(arguments);
    set_size<dev_pvfinder_latent_features_t>(arguments, total_tracks * 800);
}

void pvfinder_fc_engine_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    // Make sure we load the weights ONCE upon execution launch onto persistent device memory
    static std::once_flag flag;
    std::call_once(flag, []() {
        if (!PVFinder::WeightRegistry::instance().contains("fc_weights")) {
            PVFinder::WeightRegistry::instance().load("fc_weights", "/data/home/melashri/iris/inference/fc_weights.bin");
        }
    });

    // Pass the active device cache mapping.
    const float* dev_weights = PVFinder::WeightRegistry::instance().get<float>("fc_weights");

    global_function(pvfinder_fc_engine_kernel)(
        dim3(first<host_number_of_events_t>(arguments)), m_block_dim, context)(arguments, dev_weights);
}

} // namespace pvfinder_fc_engine
