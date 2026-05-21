#include "PVFinderUNet.cuh"
#include "PVFinderUNetKernels.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_unet::pvfinder_unet_t)

namespace pvfinder_unet {

static constexpr int B_EVENTS_MAX = 20;
static constexpr int N_CHUNK_INTERVALS = B_EVENTS_MAX * N_INTERVALS;
static constexpr size_t CUDNN_WORKSPACE_LIMIT_BYTES = 64ul * 1024ul * 1024ul;

#ifdef ALLEN_CUDNN_BACKEND_CUDA
// ---------------------------------------------------------------------------
// Process-level global descriptor set.
// Created exactly once via s_init_flag — shared read-only across all threads.
// Shapes are compile-time constants so no synchronisation is needed after init.
// ---------------------------------------------------------------------------
struct GlobalDescriptors {
    // CBR layers: cuDNN conv + PVFinder bias/ReLU, with BN folded into weights/bias at init.
    // BN folded into weights/bias at init time (fold_bn lambda).
    Allen::CuDNN::ForwardConvPlan rcbn1;    // Conv(8→N_FEAT,      k=25, pad=12)
    Allen::CuDNN::ForwardConvPlan rcbn2;    // Conv(N_FEAT→N_FEAT, k=7,  pad=3)
    Allen::CuDNN::ForwardConvPlan rcbn3;    // Conv(N_FEAT→N_FEAT, k=5,  pad=2)
    Allen::CuDNN::ForwardConvPlan up1_c;    // Conv(N_FEAT→N_FEAT, k=5,  pad=2) after ConvTranspose
    Allen::CuDNN::ForwardConvPlan up2_c;    // Conv(N_FEAT→N_FEAT, k=5,  pad=2)
    // V2.4 opt-in generic fused CBR plans. These are created only when the
    // feature flag is enabled so the default PVFinder path keeps its baseline
    // workspace and algorithm-selection behavior.
    Allen::CuDNN::FusedConvPlan rcbn1_fused;
    Allen::CuDNN::FusedConvPlan rcbn2_fused;
    Allen::CuDNN::FusedConvPlan rcbn3_fused;
    Allen::CuDNN::FusedConvPlan up1c_fused;
    Allen::CuDNN::FusedConvPlan up2c_fused;
    // Non-CBR paths: plain conv (no BN/ReLU fusion).
    Allen::CuDNN::ForwardConvPlan oint_half;// Conv(N_FEAT→N_FEAT, k=5, pad=2) — two halves
    Allen::CuDNN::ForwardConvPlan outc;     // Conv(N_FEAT→1,      k=5, pad=2)

    // BN-folded weights and biases for each CBR layer (device pointers, owned here).
    float* rcbn1_w_f = nullptr; float* rcbn1_b_f = nullptr;
    float* rcbn2_w_f = nullptr; float* rcbn2_b_f = nullptr;
    float* rcbn3_w_f = nullptr; float* rcbn3_b_f = nullptr;
    float* up1c_w_f  = nullptr; float* up1c_b_f  = nullptr;
    float* up2c_w_f  = nullptr; float* up2c_b_f  = nullptr;

    // Phase M: FP16 CBR descriptors and weights (CUDNN_DATA_HALF, BN-folded at init).
    Allen::CuDNN::ForwardConvPlan rcbn1_h, rcbn2_h, rcbn3_h, up1c_h, up2c_h;
    __half* rcbn1_w_h = nullptr; __half* rcbn1_b_h = nullptr;
    __half* rcbn2_w_h = nullptr; __half* rcbn2_b_h = nullptr;
    __half* rcbn3_w_h = nullptr; __half* rcbn3_b_h = nullptr;
    __half* up1c_w_h  = nullptr; __half* up1c_b_h  = nullptr;
    __half* up2c_w_h  = nullptr; __half* up2c_b_h  = nullptr;

    // FP16 activation pool: contiguous allocation, partitioned per-layer.
    // Offsets: ncw, x1, x2, x3, up1, cat2, up2 (cat2 also reused for up2_c output).
    __half* fp16_pool = nullptr;
    __half* fp16_ncw  = nullptr;  // [N, N_BATCH_CHANNELS, W_IN]
    __half* fp16_x1   = nullptr;  // [N, N_FEAT, W_IN]        — rcbn1 skip preserved
    __half* fp16_x2   = nullptr;  // [N, N_FEAT, W_HALF]
    __half* fp16_x3   = nullptr;  // [N, N_FEAT, W_QTR]
    __half* fp16_up1  = nullptr;  // [N, N_FEAT, W_HALF]
    __half* fp16_cat2 = nullptr;  // [N, 2*N_FEAT, W_HALF]    — also up2_c output
    __half* fp16_up2  = nullptr;  // [N, N_FEAT, W_IN]

    Allen::CuDNN::BackwardDataConvPlan up1_t;
    Allen::CuDNN::BackwardDataConvPlan up2_t;
};

static GlobalDescriptors s_desc;
static std::once_flag    s_init_flag;
static std::once_flag    s_desc_init_flag;
static bool              s_desc_use_generic_fused_cbr = false;
static bool              s_desc_use_allen_external_workspace = false;
static bool              s_desc_workspace_plan_ready = false;
static Allen::CuDNN::WorkspacePlan s_desc_workspace_plan {};

// Weight blob: device pointers per layer (filled in init(), used in operator()).
struct WeightBlob {
    const float* w_rcbn1_w;  const float* w_rcbn1_b;
    const float* w_rcbn1_gamma; const float* w_rcbn1_beta;
    const float* w_rcbn1_mean;  const float* w_rcbn1_var;
    float rcbn1_eps;

    const float* w_rcbn2_w;  const float* w_rcbn2_b;
    const float* w_rcbn2_gamma; const float* w_rcbn2_beta;
    const float* w_rcbn2_mean;  const float* w_rcbn2_var;
    float rcbn2_eps;

    const float* w_rcbn3_w;  const float* w_rcbn3_b;
    const float* w_rcbn3_gamma; const float* w_rcbn3_beta;
    const float* w_rcbn3_mean;  const float* w_rcbn3_var;
    float rcbn3_eps;

    const float* w_up1t_w;   const float* w_up1t_b;
    const float* w_up1c_w;   const float* w_up1c_b;
    const float* w_up1c_gamma; const float* w_up1c_beta;
    const float* w_up1c_mean;  const float* w_up1c_var;
    float up1c_eps;

    const float* w_up2t_w;   const float* w_up2t_b;
    const float* w_up2c_w;   const float* w_up2c_b;
    const float* w_up2c_gamma; const float* w_up2c_beta;
    const float* w_up2c_mean;  const float* w_up2c_var;
    float up2c_eps;

    const float* w_oint_a_w;
    const float* w_oint_b_w;
    const float* w_oint_b;
    const float* w_outc_w;   const float* w_outc_b;
};
static WeightBlob s_wb {};
static bool s_wb_loaded = false;

static Allen::CuDNN::DeviceWeights& unet_weights()
{
    static Allen::CuDNN::DeviceWeights weights {"pvfinder_unet"};
    return weights;
}

static bool generic_fused_cbr_env_enabled()
{
    const char* value = std::getenv("PVFINDER_USE_GENERIC_FUSED_CBR");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static bool allen_external_workspace_env_enabled()
{
    const char* value = std::getenv("PVFINDER_USE_ALLEN_EXTERNAL_WORKSPACE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static size_t workspace_bytes_to_floats(size_t bytes)
{
    return (bytes + sizeof(float) - 1) / sizeof(float);
}

static const char* precision_mode_name(bool use_fp16)
{
    return use_fp16 ? "fp16-experimental" : "fp32";
}

static void write_precision_record(
    const std::string& dump_dir,
    bool use_fp16,
    bool use_generic_fused_cbr,
    bool use_allen_external_workspace)
{
    std::ofstream record(dump_dir + "/allen_precision_record.json");
    if (!record) return;

    record << "{\n";
    record << "  \"record_version\": 1,\n";
    record << "  \"client\": \"pvfinder_unet\",\n";
    record << "  \"model_or_algorithm\": \"PVFinder UNet\",\n";
    record << "  \"unet_channels\": " << N_FEAT << ",\n";
    record << "  \"dtype_mode\": \"" << precision_mode_name(use_fp16) << "\",\n";
    record << "  \"reference_output_source\": \"PyTorch FP32 UNet validation script\",\n";
    record << "  \"recommended_max_abs_tolerance\": " << (use_fp16 ? "1e-2" : "1e-3") << ",\n";
    record << "  \"compute_contract\": {\n";
    if (use_fp16) {
        record << "    \"cbr_layers\": {\n";
        record << "      \"input_output_type\": \"CUDNN_DATA_HALF\",\n";
        record << "      \"filter_type\": \"CUDNN_DATA_HALF\",\n";
        record << "      \"compute_type\": \"CUDNN_DATA_FLOAT\",\n";
        record << "      \"math_type\": \"CUDNN_TENSOR_OP_MATH\",\n";
        record << "      \"tensor_ops_enabled\": true,\n";
        record << "      \"allow_tf32\": false,\n";
        record << "      \"fp16_experimental\": true\n";
        record << "    },\n";
        record << "    \"transpose_and_output_layers\": \"FP32\",\n";
        record << "    \"conversion_ownership\": \"PVFinder explicit f32_to_f16/f16_to_f32 kernels\"\n";
    }
    else {
        record << "    \"all_cudnn_plans\": {\n";
        record << "      \"input_output_type\": \"CUDNN_DATA_FLOAT\",\n";
        record << "      \"filter_type\": \"CUDNN_DATA_FLOAT\",\n";
        record << "      \"compute_type\": \"CUDNN_DATA_FLOAT\",\n";
        record << "      \"math_type\": \"CUDNN_TENSOR_OP_MATH\",\n";
        record << "      \"tensor_ops_enabled\": true,\n";
        record << "      \"allow_tf32\": true,\n";
        record << "      \"fp16_experimental\": false\n";
        record << "    }\n";
    }
    record << "  },\n";
    record << "  \"runtime_flags\": {\n";
    record << "    \"use_fp16\": " << (use_fp16 ? "true" : "false") << ",\n";
    record << "    \"use_generic_fused_cbr\": " << (use_generic_fused_cbr ? "true" : "false") << ",\n";
    record << "    \"use_allen_external_workspace\": " << (use_allen_external_workspace ? "true" : "false") << "\n";
    record << "  },\n";
    record << "  \"workspace\": {\n";
    record << "    \"policy\": \"" << (use_allen_external_workspace ? "AllenExternal" : "OwnedInitTime") << "\",\n";
    record << "    \"non_overlapping_total_bytes\": " << (s_desc_workspace_plan_ready ? s_desc_workspace_plan.total_bytes : 0) << ",\n";
    record << "    \"max_required_bytes\": " << (s_desc_workspace_plan_ready ? s_desc_workspace_plan.max_required_bytes : 0) << "\n";
    record << "  },\n";
    record << "  \"status\": \"" << (use_fp16 ? "experimental" : "required-baseline") << "\",\n";
    record << "  \"known_physics_caveats\": \"FP16 mode quantizes BN-folded CBR weights and intermediate CBR activations; keep opt-in until physics acceptance is recorded.\"\n";
    record << "}\n";
}

template<typename Plan>
static void add_workspace_requirement(
    Allen::CuDNN::WorkspacePlanner& planner,
    const char* name,
    const Plan& plan)
{
    if (plan.is_created()) {
        planner.add(name, plan.workspace_bytes());
    }
}

static Allen::CuDNN::WorkspacePlan build_workspace_plan(const GlobalDescriptors& desc)
{
    Allen::CuDNN::WorkspacePlanner planner;
    add_workspace_requirement(planner, "rcbn1", desc.rcbn1);
    add_workspace_requirement(planner, "rcbn2", desc.rcbn2);
    add_workspace_requirement(planner, "rcbn3", desc.rcbn3);
    add_workspace_requirement(planner, "up1_c", desc.up1_c);
    add_workspace_requirement(planner, "up2_c", desc.up2_c);
    add_workspace_requirement(planner, "rcbn1_fused", desc.rcbn1_fused);
    add_workspace_requirement(planner, "rcbn2_fused", desc.rcbn2_fused);
    add_workspace_requirement(planner, "rcbn3_fused", desc.rcbn3_fused);
    add_workspace_requirement(planner, "up1c_fused", desc.up1c_fused);
    add_workspace_requirement(planner, "up2c_fused", desc.up2c_fused);
    add_workspace_requirement(planner, "rcbn1_h", desc.rcbn1_h);
    add_workspace_requirement(planner, "rcbn2_h", desc.rcbn2_h);
    add_workspace_requirement(planner, "rcbn3_h", desc.rcbn3_h);
    add_workspace_requirement(planner, "up1c_h", desc.up1c_h);
    add_workspace_requirement(planner, "up2c_h", desc.up2c_h);
    add_workspace_requirement(planner, "oint_half", desc.oint_half);
    add_workspace_requirement(planner, "outc", desc.outc);
    add_workspace_requirement(planner, "up1_t", desc.up1_t);
    add_workspace_requirement(planner, "up2_t", desc.up2_t);
    return planner.non_overlapping_plan();
}

static void init_global_descriptors(
    cudnnHandle_t handle,
    const WeightBlob& wb,
    bool use_generic_fused_cbr,
    bool use_allen_external_workspace)
{
    constexpr int N = N_CHUNK_INTERVALS;

    Allen::CuDNN::ConvPlanOptions conv_options {};
    conv_options.workspace_limit_bytes = CUDNN_WORKSPACE_LIMIT_BYTES;
    if (use_allen_external_workspace) {
        conv_options.workspace_policy = Allen::CuDNN::WorkspacePolicy::AllenExternal;
    }

    Allen::CuDNN::FusedConvPlanOptions fused_cbr_options {};
    fused_cbr_options.conv = conv_options;
    fused_cbr_options.backend_preference =
        Allen::CuDNN::FusedConvBackendPreference::ForceLegacyConvPlusCudaPostOp;
    fused_cbr_options.fallback_policy = Allen::CuDNN::FusedConvFallbackPolicy::RequireRequestedBackend;
    fused_cbr_options.post_ops =
        Allen::CuDNN::PostOpSequence::bias_activation(Allen::CuDNN::ActivationMode::Relu);

    auto create_cbr_plan = [handle, use_generic_fused_cbr, &conv_options, &fused_cbr_options](
                               Allen::CuDNN::ForwardConvPlan& legacy_plan,
                               Allen::CuDNN::FusedConvPlan& fused_plan,
                               Allen::CuDNN::Conv2DShape shape) {
        if (use_generic_fused_cbr) {
            fused_plan.create(handle, shape, fused_cbr_options);
        }
        else {
            legacy_plan.create(handle, shape, conv_options);
        }
    };

    // Helper: allocate device buffer for fused weights/bias and launch the
    // BN-folding kernel.
    // scale[k] = gamma[k]/sqrt(var[k]+eps), w_f[k,...]=scale[k]*w[k,...],
    // b_f[k] = scale[k]*(b[k]-mean[k])+beta[k].
    auto fold_bn = [](const float* w, const float* b,
                      const float* gamma, const float* beta,
                      const float* mean,  const float* var, float eps,
                      int K, int CxHxW,
                      float*& w_f, float*& b_f,
                      cudaStream_t stream)
    {
        cudaMalloc(&w_f, (size_t)K * CxHxW * sizeof(float));
        cudaMalloc(&b_f, (size_t)K * sizeof(float));
        fold_bn_into_conv_kernel<<<K, dim3(256), 0, stream>>>(
            w_f, b_f, w, b, gamma, beta, mean, var, eps, K, CxHxW);
    };

    // Helper: convert FP32 BN-folded weights to FP16 (for Phase M Tensor Core path).
    auto to_half = [](const float* w_f, const float* b_f, int K, int CxHxW,
                      __half*& w_h, __half*& b_h, cudaStream_t stream) {
        size_t wn = (size_t)K * CxHxW;
        cudaMalloc(&w_h, wn * sizeof(__half));
        cudaMalloc(&b_h, (size_t)K * sizeof(__half));
        int threads = 256;
        f32_to_f16_kernel<<<((wn + threads - 1) / threads), threads, 0, stream>>>(w_h, w_f, (int)wn);
        f32_to_f16_kernel<<<((K  + threads - 1) / threads), threads, 0, stream>>>(b_h, b_f, K);
    };

    // Fold BN into conv weights for each CBR layer, then build the fused graph.
    // All fold kernels run on stream 0 (init is single-threaded here).
    fold_bn(wb.w_rcbn1_w, wb.w_rcbn1_b,
            wb.w_rcbn1_gamma, wb.w_rcbn1_beta,
            wb.w_rcbn1_mean,  wb.w_rcbn1_var, wb.rcbn1_eps,
            N_FEAT, N_BATCH_CHANNELS * 25,
            s_desc.rcbn1_w_f, s_desc.rcbn1_b_f, 0);
    cudaDeviceSynchronize();
    create_cbr_plan(
        s_desc.rcbn1,
        s_desc.rcbn1_fused,
        Allen::CuDNN::Conv1DShape::forward(N, N_BATCH_CHANNELS, W_IN, N_FEAT, 25, 12));
    to_half(s_desc.rcbn1_w_f, s_desc.rcbn1_b_f, N_FEAT, N_BATCH_CHANNELS * 25,
            s_desc.rcbn1_w_h, s_desc.rcbn1_b_h, 0);
    cudaDeviceSynchronize();
    Allen::CuDNN::ConvPlanOptions fp16_options {};
    fp16_options.workspace_limit_bytes = CUDNN_WORKSPACE_LIMIT_BYTES;
    if (use_allen_external_workspace) {
        fp16_options.workspace_policy = Allen::CuDNN::WorkspacePolicy::AllenExternal;
    }
    fp16_options.precision = Allen::CuDNN::fp16_precision_policy();
    s_desc.rcbn1_h.create(handle, Allen::CuDNN::Conv1DShape::forward(N, N_BATCH_CHANNELS, W_IN, N_FEAT, 25, 12),
                          fp16_options);

    fold_bn(wb.w_rcbn2_w, wb.w_rcbn2_b,
            wb.w_rcbn2_gamma, wb.w_rcbn2_beta,
            wb.w_rcbn2_mean,  wb.w_rcbn2_var, wb.rcbn2_eps,
            N_FEAT, N_FEAT * 7,
            s_desc.rcbn2_w_f, s_desc.rcbn2_b_f, 0);
    cudaDeviceSynchronize();
    create_cbr_plan(
        s_desc.rcbn2,
        s_desc.rcbn2_fused,
        Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_IN, N_FEAT, 7, 3));
    to_half(s_desc.rcbn2_w_f, s_desc.rcbn2_b_f, N_FEAT, N_FEAT * 7,
            s_desc.rcbn2_w_h, s_desc.rcbn2_b_h, 0);
    cudaDeviceSynchronize();
    s_desc.rcbn2_h.create(handle, Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_IN, N_FEAT, 7, 3),
                          fp16_options);

    fold_bn(wb.w_rcbn3_w, wb.w_rcbn3_b,
            wb.w_rcbn3_gamma, wb.w_rcbn3_beta,
            wb.w_rcbn3_mean,  wb.w_rcbn3_var, wb.rcbn3_eps,
            N_FEAT, N_FEAT * 5,
            s_desc.rcbn3_w_f, s_desc.rcbn3_b_f, 0);
    cudaDeviceSynchronize();
    create_cbr_plan(
        s_desc.rcbn3,
        s_desc.rcbn3_fused,
        Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_HALF, N_FEAT, 5, 2));
    to_half(s_desc.rcbn3_w_f, s_desc.rcbn3_b_f, N_FEAT, N_FEAT * 5,
            s_desc.rcbn3_w_h, s_desc.rcbn3_b_h, 0);
    cudaDeviceSynchronize();
    s_desc.rcbn3_h.create(handle, Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_HALF, N_FEAT, 5, 2),
                          fp16_options);

    fold_bn(wb.w_up1c_w, wb.w_up1c_b,
            wb.w_up1c_gamma, wb.w_up1c_beta,
            wb.w_up1c_mean,  wb.w_up1c_var, wb.up1c_eps,
            N_FEAT, N_FEAT * 5,
            s_desc.up1c_w_f, s_desc.up1c_b_f, 0);
    cudaDeviceSynchronize();
    create_cbr_plan(
        s_desc.up1_c,
        s_desc.up1c_fused,
        Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_HALF, N_FEAT, 5, 2));
    to_half(s_desc.up1c_w_f, s_desc.up1c_b_f, N_FEAT, N_FEAT * 5,
            s_desc.up1c_w_h, s_desc.up1c_b_h, 0);
    cudaDeviceSynchronize();
    s_desc.up1c_h.create(handle, Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_HALF, N_FEAT, 5, 2),
                         fp16_options);

    fold_bn(wb.w_up2c_w, wb.w_up2c_b,
            wb.w_up2c_gamma, wb.w_up2c_beta,
            wb.w_up2c_mean,  wb.w_up2c_var, wb.up2c_eps,
            N_FEAT, N_FEAT * 5,
            s_desc.up2c_w_f, s_desc.up2c_b_f, 0);
    cudaDeviceSynchronize();
    create_cbr_plan(
        s_desc.up2_c,
        s_desc.up2c_fused,
        Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_IN, N_FEAT, 5, 2));
    to_half(s_desc.up2c_w_f, s_desc.up2c_b_f, N_FEAT, N_FEAT * 5,
            s_desc.up2c_w_h, s_desc.up2c_b_h, 0);
    cudaDeviceSynchronize();
    s_desc.up2c_h.create(handle, Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_IN, N_FEAT, 5, 2),
                         fp16_options);

    // Allocate dedicated FP16 activation pool for Phase M benchmark.
    // Layout: ncw | x1 | x2 | x3 | up1 | cat2 | up2
    // cat2 slot is also reused as up2_c output (same element count).
    {
        size_t sz_ncw  = (size_t)N * N_BATCH_CHANNELS * W_IN;
        size_t sz_x1   = (size_t)N * N_FEAT * W_IN;
        size_t sz_x2   = (size_t)N * N_FEAT * W_HALF;
        size_t sz_x3   = (size_t)N * N_FEAT * W_QTR;
        size_t sz_up1  = (size_t)N * N_FEAT * W_HALF;
        size_t sz_cat2 = (size_t)N * N_FEAT * 2 * W_HALF;
        size_t sz_up2  = (size_t)N * N_FEAT * W_IN;
        size_t total   = sz_ncw + sz_x1 + sz_x2 + sz_x3 + sz_up1 + sz_cat2 + sz_up2;
        cudaMalloc(&s_desc.fp16_pool, total * sizeof(__half));
        __half* p = s_desc.fp16_pool;
        s_desc.fp16_ncw  = p; p += sz_ncw;
        s_desc.fp16_x1   = p; p += sz_x1;
        s_desc.fp16_x2   = p; p += sz_x2;
        s_desc.fp16_x3   = p; p += sz_x3;
        s_desc.fp16_up1  = p; p += sz_up1;
        s_desc.fp16_cat2 = p; p += sz_cat2;
        s_desc.fp16_up2  = p;
    }

    // Non-CBR paths keep the existing timed forward-conv plan selection.
    s_desc.oint_half.create(handle, Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_IN, N_FEAT, 5, 2), conv_options);
    s_desc.outc.create(     handle, Allen::CuDNN::Conv1DShape::forward(N, N_FEAT, W_IN, 1,      5, 2), conv_options);

    Allen::CuDNN::ConvPlanOptions bwd_options {};
    bwd_options.algorithm_policy = Allen::CuDNN::AlgorithmSelectionPolicy::Heuristic;
    bwd_options.workspace_policy = use_allen_external_workspace ?
        Allen::CuDNN::WorkspacePolicy::AllenExternal :
        Allen::CuDNN::WorkspacePolicy::OwnedInitTime;
    bwd_options.workspace_limit_bytes = CUDNN_WORKSPACE_LIMIT_BYTES;

    s_desc.up1_t.create(
        handle,
        Allen::CuDNN::Conv1DShape::backward_data(N, N_FEAT, W_QTR, N_FEAT, W_HALF, 2, 0, 2),
        bwd_options);
    s_desc.up2_t.create(
        handle,
        Allen::CuDNN::Conv1DShape::backward_data(N, N_FEAT * 2, W_HALF, N_FEAT, W_IN, 2, 0, 2),
        bwd_options);

    s_desc_workspace_plan = build_workspace_plan(s_desc);
    s_desc_workspace_plan_ready = true;
    if (use_allen_external_workspace && std::getenv("ALLEN_CUDNN_VERBOSE") != nullptr) {
        std::fprintf(
            stderr,
            "PVFinderUNet: AllenExternal cuDNN workspace plan mode=%s total_bytes=%zu max_required_bytes=%zu\n",
            Allen::CuDNN::to_string(Allen::CuDNN::WorkspacePolicy::AllenExternal),
            s_desc_workspace_plan.total_bytes,
            s_desc_workspace_plan.max_required_bytes);
    }
}

// ---------------------------------------------------------------------------
// Binary weight file parser
// Layout (from convert_cnn_weights.py):
//   uint32  magic = 0xCAFE0001
//   conv(8→64,k=25):  int32 in,out,k | float[out*in*k] weights | float[out] bias
//   bn(64):           int32 features | float eps | float[f] gamma,beta,mean,var
//   ... repeated for rcbn2, rcbn3
//   convT(64→64,k=2,s=2): int32 in,out,k,stride | float[in*out*k] | float[out]
//   conv+bn for up1.convbnrelu, up2.convbnrelu
//   conv(128→64,k=5): out_intermediate
//   conv(64→1,k=5):   outc
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Read a block of floats from a host buffer at a given byte offset.
// Returns updated offset.
// ---------------------------------------------------------------------------
static size_t read_float_block(
    const std::vector<char>& buf, size_t offset,
    float* dst, size_t count)
{
    std::memcpy(dst, buf.data() + offset, count * sizeof(float));
    return offset + count * sizeof(float);
}

static size_t read_int32(const std::vector<char>& buf, size_t offset, int& v)
{
    std::memcpy(&v, buf.data() + offset, 4);
    return offset + 4;
}

static size_t read_float32(const std::vector<char>& buf, size_t offset, float& v)
{
    std::memcpy(&v, buf.data() + offset, 4);
    return offset + 4;
}

// ---------------------------------------------------------------------------
// Load weights from binary file into AllenCuDNN's process-lifetime DeviceWeights
// facade and fill WeightBlob.
// ---------------------------------------------------------------------------
static WeightBlob load_weights(const std::string& path)
{
    // Read entire file into host buffer
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        throw std::runtime_error("PVFinderUNet: cannot open weight file: " + path);
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::vector<char> buf(file_size);
    fread(buf.data(), 1, file_size, fp);
    fclose(fp);

    size_t off = 0;

    // Magic
    uint32_t magic = 0;
    std::memcpy(&magic, buf.data(), 4);
    off += 4;
    if (magic != 0xCAFE0001u) {
        throw std::runtime_error("PVFinderUNet: bad magic in weight file");
    }

    WeightBlob wb {};
    auto& weights = unet_weights();

    // Helper lambdas
    auto load_conv = [&](const std::string& key_w, const std::string& key_b,
                          const float*& out_w, const float*& out_b) {
        int in_c, out_c, k;
        off = read_int32(buf, off, in_c);
        off = read_int32(buf, off, out_c);
        off = read_int32(buf, off, k);
        size_t wcount = (size_t)out_c * in_c * k;
        // Load weight block
        std::vector<float> w_host(wcount);
        off = read_float_block(buf, off, w_host.data(), wcount);
        if (!weights.contains(key_w)) weights.load_from_buffer(key_w, w_host.data(), wcount * sizeof(float));
        out_w = weights.get<float>(key_w);
        // Load bias block
        std::vector<float> b_host(out_c);
        off = read_float_block(buf, off, b_host.data(), out_c);
        if (!weights.contains(key_b)) weights.load_from_buffer(key_b, b_host.data(), out_c * sizeof(float));
        out_b = weights.get<float>(key_b);
    };

    auto load_bn = [&](const std::string& prefix,
                        const float*& gamma, const float*& beta,
                        const float*& mean,  const float*& var, float& eps) {
        int features;
        off = read_int32(buf, off, features);
        off = read_float32(buf, off, eps);
        std::vector<float> g(features), b(features), m(features), v(features);
        off = read_float_block(buf, off, g.data(), features);
        off = read_float_block(buf, off, b.data(), features);
        off = read_float_block(buf, off, m.data(), features);
        off = read_float_block(buf, off, v.data(), features);
        auto ld = [&](const std::string& k, const std::vector<float>& d, const float*& ptr) {
            if (!weights.contains(k)) weights.load_from_buffer(k, d.data(), d.size() * sizeof(float));
            ptr = weights.get<float>(k);
        };
        ld(prefix + ".gamma", g, gamma);
        ld(prefix + ".beta",  b, beta);
        ld(prefix + ".mean",  m, mean);
        ld(prefix + ".var",   v, var);
    };

    auto load_convt = [&](const std::string& key_w, const std::string& key_b,
                           const float*& out_w, const float*& out_b) {
        int in_c, out_c, k, stride;
        off = read_int32(buf, off, in_c);
        off = read_int32(buf, off, out_c);
        off = read_int32(buf, off, k);
        off = read_int32(buf, off, stride);
        size_t wcount = (size_t)in_c * out_c * k;
        std::vector<float> w_host(wcount);
        off = read_float_block(buf, off, w_host.data(), wcount);
        if (!weights.contains(key_w)) weights.load_from_buffer(key_w, w_host.data(), wcount * sizeof(float));
        out_w = weights.get<float>(key_w);
        std::vector<float> b_host(out_c);
        off = read_float_block(buf, off, b_host.data(), out_c);
        if (!weights.contains(key_b)) weights.load_from_buffer(key_b, b_host.data(), out_c * sizeof(float));
        out_b = weights.get<float>(key_b);
    };

    // rcbn1
    load_conv("rcbn1.w", "rcbn1.b", wb.w_rcbn1_w, wb.w_rcbn1_b);
    load_bn("rcbn1.bn", wb.w_rcbn1_gamma, wb.w_rcbn1_beta, wb.w_rcbn1_mean, wb.w_rcbn1_var, wb.rcbn1_eps);
    // rcbn2
    load_conv("rcbn2.w", "rcbn2.b", wb.w_rcbn2_w, wb.w_rcbn2_b);
    load_bn("rcbn2.bn", wb.w_rcbn2_gamma, wb.w_rcbn2_beta, wb.w_rcbn2_mean, wb.w_rcbn2_var, wb.rcbn2_eps);
    // rcbn3
    load_conv("rcbn3.w", "rcbn3.b", wb.w_rcbn3_w, wb.w_rcbn3_b);
    load_bn("rcbn3.bn", wb.w_rcbn3_gamma, wb.w_rcbn3_beta, wb.w_rcbn3_mean, wb.w_rcbn3_var, wb.rcbn3_eps);
    // up1: ConvTranspose + ConvBNrelu
    load_convt("up1t.w", "up1t.b", wb.w_up1t_w, wb.w_up1t_b);
    load_conv("up1c.w", "up1c.b", wb.w_up1c_w, wb.w_up1c_b);
    load_bn("up1c.bn", wb.w_up1c_gamma, wb.w_up1c_beta, wb.w_up1c_mean, wb.w_up1c_var, wb.up1c_eps);
    // up2: ConvTranspose + ConvBNrelu
    load_convt("up2t.w", "up2t.b", wb.w_up2t_w, wb.w_up2t_b);
    load_conv("up2c.w", "up2c.b", wb.w_up2c_w, wb.w_up2c_b);
    load_bn("up2c.bn", wb.w_up2c_gamma, wb.w_up2c_beta, wb.w_up2c_mean, wb.w_up2c_var, wb.up2c_eps);
    // out_intermediate: Conv(128→64, k=5).  Load full weight [64,128,5] then split into
    // two halves [64,64,5] for channels 0:64 and 64:128 so we can avoid cat_out.
    {
        int in_c, out_c, k;
        off = read_int32(buf, off, in_c);   // 128
        off = read_int32(buf, off, out_c);  // 64
        off = read_int32(buf, off, k);      // 5
        // Full weight: [out_c, in_c, k] = [64, 128, 5]
        size_t full = (size_t)out_c * in_c * k;
        std::vector<float> w_full(full);
        off = read_float_block(buf, off, w_full.data(), full);
        // Split: each output filter has in_c=128 weights per kernel position.
        // Layout (NCHW flattened): [out_c][in_c][k] — split on in_c dimension.
        size_t half_elems = (size_t)out_c * (in_c / 2) * k;  // 64*64*5
        std::vector<float> w_a(half_elems), w_b(half_elems);
        for (int oc = 0; oc < out_c; ++oc) {
            for (int ic = 0; ic < in_c; ++ic) {
                for (int ki = 0; ki < k; ++ki) {
                    float val = w_full[((size_t)oc * in_c + ic) * k + ki];
                    size_t dst_idx = ((size_t)oc * (in_c/2) + (ic % (in_c/2))) * k + ki;
                    if (ic < in_c / 2) w_a[dst_idx] = val;
                    else               w_b[dst_idx] = val;
                }
            }
        }
        if (!weights.contains("oint.a.w")) weights.load_from_buffer("oint.a.w", w_a.data(), half_elems * sizeof(float));
        if (!weights.contains("oint.b.w")) weights.load_from_buffer("oint.b.w", w_b.data(), half_elems * sizeof(float));
        wb.w_oint_a_w = weights.get<float>("oint.a.w");
        wb.w_oint_b_w = weights.get<float>("oint.b.w");
        // Bias [out_c]
        std::vector<float> bias(out_c);
        off = read_float_block(buf, off, bias.data(), out_c);
        if (!weights.contains("oint.b")) weights.load_from_buffer("oint.b", bias.data(), out_c * sizeof(float));
        wb.w_oint_b = weights.get<float>("oint.b");
    }
    // outc
    load_conv("outc.w", "outc.b", wb.w_outc_w, wb.w_outc_b);

    return wb;
}

#endif // ALLEN_CUDNN_BACKEND_CUDA

// ---------------------------------------------------------------------------
// init(): load weights + init global descriptors — both done exactly once.
// ---------------------------------------------------------------------------
void pvfinder_unet_t::init()
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    if (m_init_done) return;
    std::call_once(s_init_flag, [this]() {
        s_wb = load_weights(m_weight_file.value());
        s_wb_loaded = true;
    });
    m_init_done = true;
#endif
}

// ---------------------------------------------------------------------------
// set_arguments_size
// Workspace for cuDNN is owned per plan by default. In the V2.5 opt-in
// AllenExternal mode, dev_unet_conv_ws_t is a single reusable Allen buffer.
// The first sizing pass reserves the plan workspace limit because algorithm
// selection happens later with a live cudnnHandle_t; later passes may use the
// exact selected non-overlapping max once descriptors exist.
// ---------------------------------------------------------------------------
void pvfinder_unet_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    const unsigned padded_events = ((n_events + B_EVENTS_MAX - 1) / B_EVENTS_MAX) * B_EVENTS_MAX;
    constexpr unsigned N_batch = N_CHUNK_INTERVALS;

    set_size<dev_unet_x1_t>   (arguments, N_batch * N_FEAT * W_IN);
    set_size<dev_unet_x2_t>   (arguments, N_batch * N_FEAT * W_HALF);
    set_size<dev_unet_x3_t>   (arguments, N_batch * N_FEAT * W_IN);
    set_size<dev_unet_up1_t>  (arguments, N_batch * N_FEAT * W_HALF);
    set_size<dev_unet_cat2_t> (arguments, N_batch * N_FEAT * 2 * W_HALF);
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    const bool use_allen_external_workspace =
        m_use_allen_external_workspace.value() || allen_external_workspace_env_enabled();
    if (use_allen_external_workspace) {
        const size_t workspace_bytes = s_desc_workspace_plan_ready ?
            s_desc_workspace_plan.total_bytes :
            CUDNN_WORKSPACE_LIMIT_BYTES;
        set_size<dev_unet_conv_ws_t>(
            arguments,
            static_cast<unsigned>(std::max<size_t>(1, workspace_bytes_to_floats(workspace_bytes))));
    }
    else
#endif
    {
        set_size<dev_unet_conv_ws_t>(arguments, 1u);
    }
    set_size<dev_pvfinder_kde_output_t>(arguments, padded_events * N_INTERVALS * W_IN);
}

// ---------------------------------------------------------------------------
// Per-layer helpers
// ---------------------------------------------------------------------------

#ifdef ALLEN_CUDNN_BACKEND_CUDA
// Conv1d + bias + ReLU (BN folded into w_fused/b_fused at init).
// Uses cudnnConvolutionForward with the Phase K timed algorithm, then
// launches bias_relu_kernel (no BN math — BN already in w_fused/b_fused).
void pvfinder_unet_t::run_convbnrelu(
    const Allen::CuDNN::ForwardConvPlan& desc,
    const float* input, float* output,
    const float* w_fused, const float* b_fused,
    int K, int W_out, int N,
    cudnnHandle_t handle,
    const dim3& block, const Allen::Context& ctx,
    Allen::CuDNN::Workspace workspace) const
{
    if (desc.workspace_policy() == Allen::CuDNN::WorkspacePolicy::AllenExternal) {
        desc.forward(handle, 1.f, 0.f, input, w_fused, output, workspace);
    }
    else {
        desc.forward(handle, 1.f, 0.f, input, w_fused, output);
    }
    launch_bias_relu(output, b_fused, K, W_out, N, block, ctx);
}

// FP16 variant: Tensor Core conv (CUDNN_DATA_HALF desc) + FP16 bias+relu kernel.
void pvfinder_unet_t::run_convbnrelu_half(
    const Allen::CuDNN::ForwardConvPlan& desc,
    const __half* input, __half* output,
    const __half* w_fused, const __half* b_fused,
    int K, int W_out, int N,
    cudnnHandle_t handle,
    const dim3& block, const Allen::Context& ctx,
    Allen::CuDNN::Workspace workspace) const
{
    if (desc.workspace_policy() == Allen::CuDNN::WorkspacePolicy::AllenExternal) {
        desc.forward_half(handle, 1.f, 0.f, input, w_fused, output, workspace);
    }
    else {
        desc.forward_half(handle, 1.f, 0.f, input, w_fused, output);
    }
    launch_bias_relu_half(output, b_fused, K, W_out, N, block, ctx);
}

// Generic V2.4 FP32 CBR path: FusedConvPlan wraps cudnnConvolutionForward and
// one backend-owned CUDA post-op kernel for channel bias + ReLU.
void pvfinder_unet_t::run_fused_convbnrelu(
    const Allen::CuDNN::FusedConvPlan& plan,
    const float* input, float* output,
    const float* w_fused, const float* b_fused,
    cudnnHandle_t handle,
    Allen::CuDNN::Workspace workspace) const
{
    if (plan.workspace_policy() == Allen::CuDNN::WorkspacePolicy::AllenExternal) {
        plan.execute(handle, 1.f, 0.f, input, w_fused, b_fused, output, workspace);
    }
    else {
        plan.execute(handle, 1.f, 0.f, input, w_fused, b_fused, output);
    }
}

// Conv1d only (no BN/ReLU).
void pvfinder_unet_t::run_conv(
    const Allen::CuDNN::ForwardConvPlan& desc,
    const float* input,  float* output,
    const float* w_ptr,  const float* bias_ptr,
    int N, int C_out, int W,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle,
    float beta_val,
    Allen::CuDNN::Workspace workspace) const
{
    const float alpha = 1.f;
    if (desc.workspace_policy() == Allen::CuDNN::WorkspacePolicy::AllenExternal) {
        desc.forward(handle, alpha, beta_val, input, w_ptr, output, workspace);
    }
    else {
        desc.forward(handle, alpha, beta_val, input, w_ptr, output);
    }
    if (bias_ptr && beta_val == 0.f)
        launch_bias_add(output, bias_ptr, C_out, W, N, block, ctx);
}

// ConvTranspose1d via AllenCuDNN backward-data plan. Algorithm selected at init time.
void pvfinder_unet_t::run_conv_transpose(
    const float* input, float* output,
    const Allen::CuDNN::BackwardDataConvPlan& plan,
    const float* w_ptr, const float* bias_ptr,
    int N, int C_out, int W_out,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle,
    Allen::CuDNN::Workspace workspace) const
{
    const float alpha = 1.f, beta = 0.f;
    if (plan.workspace_policy() == Allen::CuDNN::WorkspacePolicy::AllenExternal) {
        plan.backward_data(handle, alpha, beta, w_ptr, input, output, workspace);
    }
    else {
        plan.backward_data(handle, alpha, beta, w_ptr, input, output);
    }
    launch_bias_add(output, bias_ptr, C_out, W_out, N, block, ctx);
}
#endif // ALLEN_CUDNN_BACKEND_CUDA

// ---------------------------------------------------------------------------
// operator(): full UNet forward pass
// ---------------------------------------------------------------------------
void pvfinder_unet_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    if (!m_init_done || !s_wb_loaded) return;

    const unsigned n_events = first<host_number_of_events_t>(arguments);

    // One thread_local handle per OS thread — created lazily, routed to this stream.
    cudnnHandle_t handle = Allen::CuDNN::get_thread_local_handle(context.stream());

    // Descriptor creation needs a live handle (for algorithm selection), so it runs
    // here on first operator() call rather than in init().
    const bool use_fp16 = m_use_fp16.value();
    const bool requested_generic_fused_cbr =
        (m_use_generic_fused_cbr.value() || generic_fused_cbr_env_enabled()) && !use_fp16;
    const bool requested_allen_external_workspace =
        m_use_allen_external_workspace.value() || allen_external_workspace_env_enabled();

    std::call_once(s_desc_init_flag, [handle, requested_generic_fused_cbr, requested_allen_external_workspace]() {
        s_desc_use_generic_fused_cbr = requested_generic_fused_cbr;
        s_desc_use_allen_external_workspace = requested_allen_external_workspace;
        init_global_descriptors(handle, s_wb, requested_generic_fused_cbr, requested_allen_external_workspace);
    });
    const bool use_generic_fused_cbr = s_desc_use_generic_fused_cbr;
    const bool use_allen_external_workspace = s_desc_use_allen_external_workspace;

    const dim3 block = m_block_dim;
    constexpr int N = N_CHUNK_INTERVALS;  // batch size = 20 * 40 = 800

    // Scratch buffers (fixed size, reused each event iteration)
    float* x1   = data<dev_unet_x1_t>(arguments);
    float* x2   = data<dev_unet_x2_t>(arguments);
    float* x3   = data<dev_unet_x3_t>(arguments);
    float* up1  = data<dev_unet_up1_t>(arguments);
    float* cat2 = data<dev_unet_cat2_t>(arguments);
    Allen::CuDNN::Workspace cudnn_workspace {};
    if (use_allen_external_workspace) {
        cudnn_workspace = {
            data<dev_unet_conv_ws_t>(arguments),
            size<dev_unet_conv_ws_t>(arguments) * sizeof(float)};
        cudnn_workspace.require(s_desc_workspace_plan.total_bytes, "PVFinderUNet");
    }

    // Buffer aliases (liveness-proven safe)
    float* up2   = cat2;  // cat2[N,128,50] and up2[N,64,100] have same element count
    float* oint  = x1;    // x1 skip consumed before oint written
    float* logits = x3;   // x3 consumed after maxpool; reused as logits

    constexpr unsigned ncw_stride = N_INTERVALS * N_BATCH_CHANNELS * W_IN;
    constexpr unsigned kde_stride = N_INTERVALS * W_IN;

    const float* ncw_base = data<dev_pvfinder_interval_features_t>(arguments);
    float*       kde_base = data<dev_pvfinder_kde_output_t>(arguments);

    const unsigned padded_events = ((n_events + B_EVENTS_MAX - 1) / B_EVENTS_MAX) * B_EVENTS_MAX;

    // FP16 pool pointers (only used when use_fp16 is true)
    __half* fp16_ncw  = s_desc.fp16_ncw;
    __half* fp16_x1   = s_desc.fp16_x1;
    __half* fp16_x2   = s_desc.fp16_x2;
    __half* fp16_x3   = s_desc.fp16_x3;
    __half* fp16_up1  = s_desc.fp16_up1;
    __half* fp16_cat2 = s_desc.fp16_cat2;
    __half* fp16_up2  = s_desc.fp16_up2;

    for (unsigned chunk_start = 0; chunk_start < padded_events; chunk_start += B_EVENTS_MAX) {
        const float* ncw = ncw_base + chunk_start * ncw_stride;
        float*       kde = kde_base + chunk_start * kde_stride;

        if (!use_fp16) {
            // ---- FP32 path (Phase L baseline) ----
            if (use_generic_fused_cbr) {
                run_fused_convbnrelu(s_desc.rcbn1_fused, ncw, x1, s_desc.rcbn1_w_f, s_desc.rcbn1_b_f, handle, cudnn_workspace);
                run_fused_convbnrelu(s_desc.rcbn2_fused, x1, up2, s_desc.rcbn2_w_f, s_desc.rcbn2_b_f, handle, cudnn_workspace);
            }
            else {
                run_convbnrelu(s_desc.rcbn1, ncw, x1, s_desc.rcbn1_w_f, s_desc.rcbn1_b_f, N_FEAT, W_IN, N, handle, block, context, cudnn_workspace);
                run_convbnrelu(s_desc.rcbn2, x1, up2, s_desc.rcbn2_w_f, s_desc.rcbn2_b_f, N_FEAT, W_IN, N, handle, block, context, cudnn_workspace);
            }
            launch_maxpool(up2, x2, N, N_FEAT, W_IN, block, context);

            if (use_generic_fused_cbr) {
                run_fused_convbnrelu(s_desc.rcbn3_fused, x2, up2, s_desc.rcbn3_w_f, s_desc.rcbn3_b_f, handle, cudnn_workspace);
            }
            else {
                run_convbnrelu(s_desc.rcbn3, x2, up2, s_desc.rcbn3_w_f, s_desc.rcbn3_b_f, N_FEAT, W_HALF, N, handle, block, context, cudnn_workspace);
            }
            launch_maxpool(up2, x3, N, N_FEAT, W_HALF, block, context);

            run_conv_transpose(x3, up2,
                s_desc.up1_t,
                s_wb.w_up1t_w, s_wb.w_up1t_b,
                N, N_FEAT, W_HALF, block, context, handle, cudnn_workspace);
            if (use_generic_fused_cbr) {
                run_fused_convbnrelu(s_desc.up1c_fused, up2, up1, s_desc.up1c_w_f, s_desc.up1c_b_f, handle, cudnn_workspace);
            }
            else {
                run_convbnrelu(s_desc.up1_c, up2, up1, s_desc.up1c_w_f, s_desc.up1c_b_f, N_FEAT, W_HALF, N, handle, block, context, cudnn_workspace);
            }

            launch_concat(up1, x2, cat2, N, N_FEAT, N_FEAT, W_HALF, block, context);
            run_conv_transpose(cat2, logits,
                s_desc.up2_t,
                s_wb.w_up2t_w, s_wb.w_up2t_b,
                N, N_FEAT, W_IN, block, context, handle, cudnn_workspace);
            if (use_generic_fused_cbr) {
                run_fused_convbnrelu(s_desc.up2c_fused, logits, up2, s_desc.up2c_w_f, s_desc.up2c_b_f, handle, cudnn_workspace);
            }
            else {
                run_convbnrelu(s_desc.up2_c, logits, up2, s_desc.up2c_w_f, s_desc.up2c_b_f, N_FEAT, W_IN, N, handle, block, context, cudnn_workspace);
            }

            run_conv(s_desc.oint_half, x1, logits,
                s_wb.w_oint_b_w, nullptr,
                N, N_FEAT, W_IN, block, context, handle, 0.f, cudnn_workspace);
            run_conv(s_desc.oint_half, up2, logits,
                s_wb.w_oint_a_w, nullptr,
                N, N_FEAT, W_IN, block, context, handle, 1.f, cudnn_workspace);
            launch_bias_add(logits, s_wb.w_oint_b, N_FEAT, W_IN, N, block, context);
            run_conv(s_desc.outc, logits, oint,
                s_wb.w_outc_w, s_wb.w_outc_b,
                N, 1, W_IN, block, context, handle, 0.f, cudnn_workspace);
        } else {
            // ---- FP16 path (Phase M benchmark) ----
            // CBR layers run as Tensor Core FP16 convs; ConvTranspose and output
            // layers stay FP32. Explicit F32↔F16 conversions at the boundaries.

            // Encoder
            launch_f32_to_f16(fp16_ncw, ncw, N * N_BATCH_CHANNELS * W_IN, block, context);
            run_convbnrelu_half(s_desc.rcbn1_h, fp16_ncw, fp16_x1,
                s_desc.rcbn1_w_h, s_desc.rcbn1_b_h, N_FEAT, W_IN, N, handle, block, context, cudnn_workspace);
            run_convbnrelu_half(s_desc.rcbn2_h, fp16_x1, fp16_up2,
                s_desc.rcbn2_w_h, s_desc.rcbn2_b_h, N_FEAT, W_IN, N, handle, block, context, cudnn_workspace);
            launch_maxpool_half(fp16_up2, fp16_x2, N, N_FEAT, W_IN, block, context);

            run_convbnrelu_half(s_desc.rcbn3_h, fp16_x2, fp16_up2,
                s_desc.rcbn3_w_h, s_desc.rcbn3_b_h, N_FEAT, W_HALF, N, handle, block, context, cudnn_workspace);
            launch_maxpool_half(fp16_up2, fp16_x3, N, N_FEAT, W_HALF, block, context);

            // ConvTranspose1: needs FP32. Convert fp16_x3 → x3.
            launch_f16_to_f32(x3, fp16_x3, N * N_FEAT * W_QTR, block, context);
            run_conv_transpose(x3, up2,
                s_desc.up1_t,
                s_wb.w_up1t_w, s_wb.w_up1t_b,
                N, N_FEAT, W_HALF, block, context, handle, cudnn_workspace);

            // up1_c FP16: convert FP32 up2 → fp16_up2, then conv.
            launch_f32_to_f16(fp16_up2, up2, N * N_FEAT * W_HALF, block, context);
            run_convbnrelu_half(s_desc.up1c_h, fp16_up2, fp16_up1,
                s_desc.up1c_w_h, s_desc.up1c_b_h, N_FEAT, W_HALF, N, handle, block, context, cudnn_workspace);

            // Concat FP16: fp16_up1 + fp16_x2 → fp16_cat2.
            launch_concat_half(fp16_up1, fp16_x2, fp16_cat2, N, N_FEAT, N_FEAT, W_HALF, block, context);

            // ConvTranspose2: needs FP32. Convert fp16_cat2 → cat2.
            launch_f16_to_f32(cat2, fp16_cat2, N * N_FEAT * 2 * W_HALF, block, context);
            run_conv_transpose(cat2, logits,
                s_desc.up2_t,
                s_wb.w_up2t_w, s_wb.w_up2t_b,
                N, N_FEAT, W_IN, block, context, handle, cudnn_workspace);

            // up2_c FP16: convert FP32 logits → fp16_up2, conv → fp16_cat2 (reused).
            launch_f32_to_f16(fp16_up2, logits, N * N_FEAT * W_IN, block, context);
            run_convbnrelu_half(s_desc.up2c_h, fp16_up2, fp16_cat2,
                s_desc.up2c_w_h, s_desc.up2c_b_h, N_FEAT, W_IN, N, handle, block, context, cudnn_workspace);

            // Output stage: FP32. Convert fp16_x1 (rcbn1 skip) → x1, fp16_cat2 → up2.
            launch_f16_to_f32(x1, fp16_x1, N * N_FEAT * W_IN, block, context);
            launch_f16_to_f32(up2, fp16_cat2, N * N_FEAT * W_IN, block, context);

            run_conv(s_desc.oint_half, x1, logits,
                s_wb.w_oint_b_w, nullptr,
                N, N_FEAT, W_IN, block, context, handle, 0.f, cudnn_workspace);
            run_conv(s_desc.oint_half, up2, logits,
                s_wb.w_oint_a_w, nullptr,
                N, N_FEAT, W_IN, block, context, handle, 1.f, cudnn_workspace);
            launch_bias_add(logits, s_wb.w_oint_b, N_FEAT, W_IN, N, block, context);
            run_conv(s_desc.outc, logits, oint,
                s_wb.w_outc_w, s_wb.w_outc_b,
                N, 1, W_IN, block, context, handle, 0.f, cudnn_workspace);
        }

        launch_softplus_scale(oint, KDE_SCALE, N * W_IN, block, context);
        squeeze_copy_kernel<<<
            ((unsigned)(N * W_IN) + block.x - 1) / block.x, block,
            0, context.stream()>>>(oint, kde, N * W_IN);
    }

    // Validation dump (first slice only, when dump_dir property is set)
    const std::string& dump_dir = m_dump_dir.value();
    if (!dump_dir.empty() && !m_dump_done) {
        cudaStreamSynchronize(context.stream());
        const unsigned ncw_elems = n_events * N_INTERVALS * N_BATCH_CHANNELS * W_IN;
        const unsigned kde_elems = n_events * N_INTERVALS * W_IN;
        std::vector<float> h_ncw(ncw_elems), h_kde(kde_elems);
        cudaMemcpy(h_ncw.data(), data<dev_pvfinder_interval_features_t>(arguments),
                   ncw_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_kde.data(), data<dev_pvfinder_kde_output_t>(arguments),
                   kde_elems * sizeof(float), cudaMemcpyDeviceToHost);
        const uint32_t magic = 0xAB1EU;
        auto write_bin = [&](const std::string& path, const float* d, unsigned n) {
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(&magic),    sizeof(magic));
            f.write(reinterpret_cast<const char*>(&n_events), sizeof(n_events));
            f.write(reinterpret_cast<const char*>(d),         n * sizeof(float));
        };
        write_bin(dump_dir + "/allen_ncw_input.bin",  h_ncw.data(), ncw_elems);
        write_bin(dump_dir + "/allen_kde_output.bin", h_kde.data(), kde_elems);
        write_precision_record(dump_dir, use_fp16, use_generic_fused_cbr, use_allen_external_workspace);
        printf("[pvfinder_unet] Validation dump written to %s (%u events)\n",
               dump_dir.c_str(), n_events);
        m_dump_done = true;
    }
#endif
}

} // namespace pvfinder_unet
