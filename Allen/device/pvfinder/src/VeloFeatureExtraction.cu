#include "VeloFeatureExtraction.cuh"
#include <math_constants.h>

INSTANTIATE_ALGORITHM(pvfinder_velo_feature_extraction::pvfinder_velo_feature_extraction_t)

namespace pvfinder_velo_feature_extraction {

__device__ float3 normalize(float3 v) {
    float mag = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (mag > 0.0f) {
        return make_float3(v.x / mag, v.y / mag, v.z / mag);
    } else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

// NOTE: cross(float3, float3) is now provided by Allen's backend/include/BackendCommon.h
// (added upstream after v7r9). The previous local definition here was numerically
// identical and has been removed to avoid an overload ambiguity.

__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ bool state_poca(
    const Allen::Views::Physics::KalmanState& state,
    float& poca_x, float& poca_y, float& poca_z)
{
    // Extrapolate state to z=0 to get intercept parameters x0, y0
    float x0 = state.x() - state.z() * state.tx();
    float y0 = state.y() - state.z() * state.ty();

    float tx = state.tx();
    float ty = state.ty();
    float t_sq = tx * tx + ty * ty;

    // Evaluate Z_POCA geometrically minimizing trajectory distance D(z) to the z-axis (0,0)
    if (t_sq > 1e-8f) {
        poca_z = -(x0 * tx + y0 * ty) / t_sq;
    } else {
        poca_z = 0.0f; // Track is entirely parallel to beamline
    }

    poca_x = x0 + tx * poca_z;
    poca_y = y0 + ty * poca_z;

    float distance = sqrtf(poca_x * poca_x + poca_y * poca_y);

    // Keep distance logic check ensuring tracks are within sensible bounds 
    return distance < 1000.f;
}

__device__ void calculate_ellipsoid_params(
    const Allen::Views::Physics::KalmanState& state,
    float* ellipsoid_params,
    float& poca_x, float& poca_y, float& poca_z)
{
    bool poca_success = state_poca(state, poca_x, poca_y, poca_z);
    if (!poca_success) {
        poca_x = poca_y = poca_z = 0.0f;
        for (int i = 0; i < 6; ++i) ellipsoid_params[i] = 0.0f;
        return;
    }

    float3 center = make_float3(poca_x, poca_y, poca_z);
    float3 track_dir = normalize(make_float3(state.tx(), state.ty(), 1.0f));
    float3 zhat = normalize(center);
    float3 xhat = track_dir;
    float3 yhat = normalize(cross(zhat, xhat));
    float road_error = sqrtf(state.c00());
    float3 u1 = make_float3(road_error * zhat.x, road_error * zhat.y, road_error * zhat.z);
    float3 u2 = make_float3(road_error * yhat.x, road_error * yhat.y, road_error * yhat.z);
    float arg = dot(xhat, track_dir);
    arg = fminf(arg, 0.9999f);
    float u3_scale = (road_error * arg) / sqrtf(1.0f - arg * arg);
    float3 u3 = make_float3(u3_scale * xhat.x, u3_scale * xhat.y, u3_scale * xhat.z);

    ellipsoid_params[0] = u1.x * u1.x + u2.x * u2.x + u3.x * u3.x;
    ellipsoid_params[1] = u1.y * u1.y + u2.y * u2.y + u3.y * u3.y;
    ellipsoid_params[2] = u1.z * u1.z + u2.z * u2.z + u3.z * u3.z;
    ellipsoid_params[3] = u1.x * u1.y + u2.x * u2.y + u3.x * u3.y;
    ellipsoid_params[4] = u1.x * u1.z + u2.x * u2.z + u3.x * u3.z;
    ellipsoid_params[5] = u1.y * u1.z + u2.y * u2.z + u3.y * u3.z;
}

__global__ void pvfinder_velo_feature_extraction_kernel(
    pvfinder_velo_feature_extraction_t::Parameters parameters)
{
    const unsigned event_number = blockIdx.x;
    const unsigned thread_id = threadIdx.x;

    const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];
    const auto velo_states_view = parameters.dev_velo_states_view[event_number];
    const unsigned num_tracks = velo_tracks_view.size();
    
    // Each track has 9 features. We write them linearly to global memory.
    // We need to calculate the global offset for this event's tracks.
    unsigned event_track_offset = velo_tracks_view.offset();

    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        const auto track = velo_tracks_view.track(i);
        const auto state = velo_states_view.state(track.track_index());

        float ellipsoid_params[6];
        float poca_x, poca_y, poca_z;
        calculate_ellipsoid_params(state, ellipsoid_params, poca_x, poca_y, poca_z);

        unsigned global_track_idx = event_track_offset + i;
        float* track_features = &parameters.dev_pvfinder_track_features[global_track_idx * 9];

        // Ensure ordering identically matches the offline PyTorch definitions (x,y,z,A,B,C,D,E,F)
        track_features[0] = poca_x;
        track_features[1] = poca_y;
        track_features[2] = poca_z;
        track_features[3] = ellipsoid_params[0]; // A
        track_features[4] = ellipsoid_params[1]; // B
        track_features[5] = ellipsoid_params[2]; // C
        track_features[6] = ellipsoid_params[3]; // D
        track_features[7] = ellipsoid_params[4]; // E
        track_features[8] = ellipsoid_params[5]; // F
        // if (event_number == 0 && i == 0) {
        //     printf("DEBUG VeloFeatureExtraction [Event 0, Track 0]: POCA=(%f, %f, %f), A=%f, B=%f, C=%f\n",
        //            track_features[0], track_features[1], track_features[2], track_features[3], track_features[4], track_features[5]);
        // }
    }
}

void pvfinder_velo_feature_extraction_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    // dev_pvfinder_track_features size is 9 * total_number_of_reconstructed_velo_tracks
    unsigned total_tracks = first<host_number_of_reconstructed_velo_tracks_t>(arguments);
    set_size<dev_pvfinder_track_features_t>(arguments, total_tracks * 9);
}

void pvfinder_velo_feature_extraction_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
    global_function(pvfinder_velo_feature_extraction_kernel)(
        dim3(first<host_number_of_events_t>(arguments)), m_block_dim, context)(arguments);
}

} // namespace pvfinder_velo_feature_extraction
