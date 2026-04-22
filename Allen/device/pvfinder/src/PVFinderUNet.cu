#include "PVFinderUNet.cuh"
#include "PVFinderUNetKernels.cuh"

#include <cstdio>
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

#ifdef ALLEN_CUDNN_BACKEND_CUDA
// ---------------------------------------------------------------------------
// Process-level global descriptor set.
// Created exactly once via s_init_flag — shared read-only across all threads.
// Shapes are compile-time constants so no synchronisation is needed after init.
// ---------------------------------------------------------------------------
struct GlobalDescriptors {
    Allen::CuDNN::ConvDescriptors rcbn1;    // Conv(8→64,  k=25, pad=12)
    Allen::CuDNN::ConvDescriptors rcbn2;    // Conv(64→64, k=7,  pad=3)
    Allen::CuDNN::ConvDescriptors rcbn3;    // Conv(64→64, k=5,  pad=2)
    Allen::CuDNN::ConvDescriptors up1_c;    // Conv(64→64, k=5,  pad=2) after ConvTranspose
    Allen::CuDNN::ConvDescriptors up2_c;    // Conv(64→64, k=5,  pad=2)
    Allen::CuDNN::ConvDescriptors oint_half;// Conv(64→64, k=5,  pad=2) — two halves
    Allen::CuDNN::ConvDescriptors outc;     // Conv(64→1,  k=5,  pad=2)

    // ConvTranspose descriptors (filter + conv only; tensor descs are local in operator())
    cudnnFilterDescriptor_t       filter_up1_t = nullptr;
    cudnnConvolutionDescriptor_t  conv_up1_t   = nullptr;
    cudnnFilterDescriptor_t       filter_up2_t = nullptr;
    cudnnConvolutionDescriptor_t  conv_up2_t   = nullptr;

    // ConvTranspose algorithm + workspace (selected by cudnnGetConvolutionBackwardDataAlgorithm_v7)
    cudnnConvolutionBwdDataAlgo_t  algo_up1_t   = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    cudnnConvolutionBwdDataAlgo_t  algo_up2_t   = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    void*                          ws_up1_t     = nullptr;
    void*                          ws_up2_t     = nullptr;
    size_t                         ws_up1_bytes = 0;
    size_t                         ws_up2_bytes = 0;
};

static GlobalDescriptors s_desc;
static std::once_flag    s_init_flag;
static std::once_flag    s_desc_init_flag;

static void init_global_descriptors(cudnnHandle_t handle)
{
    constexpr int N = N_CHUNK_INTERVALS;
    // Forward convs: algorithm selected at create() time via cudnnGetConvolutionForwardAlgorithm_v7.
    s_desc.rcbn1.create(    handle, {N, N_BATCH_CHANNELS, 1, W_IN},  {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0,12});
    s_desc.rcbn2.create(    handle, {N, N_FEAT,            1, W_IN},  {N_FEAT, N_FEAT,            1,  7}, {0, 3});
    s_desc.rcbn3.create(    handle, {N, N_FEAT,            1, W_HALF},{N_FEAT, N_FEAT,            1,  5}, {0, 2});
    s_desc.up1_c.create(    handle, {N, N_FEAT,            1, W_HALF},{N_FEAT, N_FEAT,            1,  5}, {0, 2});
    s_desc.up2_c.create(    handle, {N, N_FEAT,            1, W_IN},  {N_FEAT, N_FEAT,            1,  5}, {0, 2});
    s_desc.oint_half.create(handle, {N, N_FEAT,            1, W_IN},  {N_FEAT, N_FEAT,            1,  5}, {0, 2});
    s_desc.outc.create(     handle, {N, N_FEAT,            1, W_IN},  {1,      N_FEAT,            1,  5}, {0, 2});

    // ConvTranspose: filter + conv descriptors only (shared, read-only after init).
    // Tensor descriptors for in/out are created as locals in operator() per-call.
    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&s_desc.filter_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        s_desc.filter_up1_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&s_desc.conv_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        s_desc.conv_up1_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&s_desc.filter_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        s_desc.filter_up2_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT*2, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&s_desc.conv_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        s_desc.conv_up2_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Algorithm sweep for ConvTranspose (cudnnConvolutionBackwardData)
    // Uses temporary tensor descriptors for the query — not stored, as operator()
    // creates per-call thread-local descriptors.
    static constexpr size_t kBwdBudget  = 64ul * 1024 * 1024;
    static constexpr int    kBwdMaxAlgo = 8;

    // up1_t: dy=[N, N_FEAT, 1, W_QTR], dx=[N, N_FEAT, 1, W_HALF]
    {
        cudnnTensorDescriptor_t dy_desc, dx_desc;
        cudnnCreateTensorDescriptor(&dy_desc);
        cudnnCreateTensorDescriptor(&dx_desc);
        cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_QTR);
        cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_HALF);
        int returned = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perf[kBwdMaxAlgo];
        if (cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle, s_desc.filter_up1_t, dy_desc, s_desc.conv_up1_t, dx_desc,
                kBwdMaxAlgo, &returned, perf) == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= kBwdBudget) {
                    s_desc.algo_up1_t   = perf[i].algo;
                    s_desc.ws_up1_bytes = perf[i].memory;
                    if (s_desc.ws_up1_bytes > 0) cudaMalloc(&s_desc.ws_up1_t, s_desc.ws_up1_bytes);
                    break;
                }
            }
        }
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
    }

    // up2_t: dy=[N, N_FEAT*2, 1, W_HALF], dx=[N, N_FEAT, 1, W_IN]
    {
        cudnnTensorDescriptor_t dy_desc, dx_desc;
        cudnnCreateTensorDescriptor(&dy_desc);
        cudnnCreateTensorDescriptor(&dx_desc);
        cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT*2, 1, W_HALF);
        cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_IN);
        int returned = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perf[kBwdMaxAlgo];
        if (cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle, s_desc.filter_up2_t, dy_desc, s_desc.conv_up2_t, dx_desc,
                kBwdMaxAlgo, &returned, perf) == CUDNN_STATUS_SUCCESS) {
            for (int i = 0; i < returned; ++i) {
                if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory <= kBwdBudget) {
                    s_desc.algo_up2_t   = perf[i].algo;
                    s_desc.ws_up2_bytes = perf[i].memory;
                    if (s_desc.ws_up2_bytes > 0) cudaMalloc(&s_desc.ws_up2_t, s_desc.ws_up2_bytes);
                    break;
                }
            }
        }
        cudnnDestroyTensorDescriptor(dy_desc);
        cudnnDestroyTensorDescriptor(dx_desc);
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
struct WeightBlob {
    // Per-layer device pointers into WeightRegistry
    // Naming: w_<layer>_<weight/bias/gamma/beta/mean/var>
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

    const float* w_up1t_w;   const float* w_up1t_b;     // ConvTranspose
    const float* w_up1c_w;   const float* w_up1c_b;     // Conv after transpose
    const float* w_up1c_gamma; const float* w_up1c_beta;
    const float* w_up1c_mean;  const float* w_up1c_var;
    float up1c_eps;

    const float* w_up2t_w;   const float* w_up2t_b;
    const float* w_up2c_w;   const float* w_up2c_b;
    const float* w_up2c_gamma; const float* w_up2c_beta;
    const float* w_up2c_mean;  const float* w_up2c_var;
    float up2c_eps;

    // out_intermediate: split into two 64-channel halves to avoid cat_out[N,128,100] buffer
    const float* w_oint_a_w; // [64, 64, 1, 5] — input channels 0:64  (from up2)
    const float* w_oint_b_w; // [64, 64, 1, 5] — input channels 64:128 (from x1)
    const float* w_oint_b;   // bias [64] — added after both halves
    const float* w_outc_w;   const float* w_outc_b;
};

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
// Load weights from binary file into WeightRegistry and fill WeightBlob.
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

    auto& reg = Allen::CuDNN::WeightRegistry::instance();
    WeightBlob wb {};

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
        if (!reg.contains(key_w)) reg.load_from_buffer(key_w, w_host.data(), wcount * sizeof(float));
        out_w = reg.get<float>(key_w);
        // Load bias block
        std::vector<float> b_host(out_c);
        off = read_float_block(buf, off, b_host.data(), out_c);
        if (!reg.contains(key_b)) reg.load_from_buffer(key_b, b_host.data(), out_c * sizeof(float));
        out_b = reg.get<float>(key_b);
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
            if (!reg.contains(k)) reg.load_from_buffer(k, d.data(), d.size() * sizeof(float));
            ptr = reg.get<float>(k);
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
        if (!reg.contains(key_w)) reg.load_from_buffer(key_w, w_host.data(), wcount * sizeof(float));
        out_w = reg.get<float>(key_w);
        std::vector<float> b_host(out_c);
        off = read_float_block(buf, off, b_host.data(), out_c);
        if (!reg.contains(key_b)) reg.load_from_buffer(key_b, b_host.data(), out_c * sizeof(float));
        out_b = reg.get<float>(key_b);
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
        if (!reg.contains("oint.a.w")) reg.load_from_buffer("oint.a.w", w_a.data(), half_elems * sizeof(float));
        if (!reg.contains("oint.b.w")) reg.load_from_buffer("oint.b.w", w_b.data(), half_elems * sizeof(float));
        wb.w_oint_a_w = reg.get<float>("oint.a.w");
        wb.w_oint_b_w = reg.get<float>("oint.b.w");
        // Bias [out_c]
        std::vector<float> bias(out_c);
        off = read_float_block(buf, off, bias.data(), out_c);
        if (!reg.contains("oint.b")) reg.load_from_buffer("oint.b", bias.data(), out_c * sizeof(float));
        wb.w_oint_b = reg.get<float>("oint.b");
    }
    // outc
    load_conv("outc.w", "outc.b", wb.w_outc_w, wb.w_outc_b);

    return wb;
}

// ---------------------------------------------------------------------------
// Static weight blob (filled once at init())
// ---------------------------------------------------------------------------
static WeightBlob s_wb {};
static bool s_wb_loaded = false;
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
// Workspace for cuDNN is owned per ConvDescriptors (cudaMalloc'd at init).
// dev_unet_conv_ws_t is kept at 1 float as Allen requires a non-zero allocation.
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
    set_size<dev_unet_conv_ws_t>(arguments, 1u);                    
    set_size<dev_pvfinder_kde_output_t>(arguments, padded_events * N_INTERVALS * W_IN);
}

// ---------------------------------------------------------------------------
// Per-layer helpers
// ---------------------------------------------------------------------------

#ifdef ALLEN_CUDNN_BACKEND_CUDA
// Conv1d + bias + BN + ReLU.
void pvfinder_unet_t::run_convbnrelu(
    const Allen::CuDNN::ConvDescriptors& desc,
    const float* input, float* output,
    const float* w_ptr,  const float* bias_ptr,
    const float* bn_gamma, const float* bn_beta,
    const float* bn_mean,  const float* bn_var, float bn_eps,
    int N, int C_out, int W,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle) const
{
    const float alpha = 1.f, beta = 0.f;
    desc.forward(handle, alpha, beta, input, w_ptr, output);
    launch_bias_bn_relu(output, bias_ptr, bn_gamma, bn_beta, bn_mean, bn_var, bn_eps,
                        C_out, W, N, block, ctx);
}

// Conv1d only (no BN/ReLU).
void pvfinder_unet_t::run_conv(
    const Allen::CuDNN::ConvDescriptors& desc,
    const float* input,  float* output,
    const float* w_ptr,  const float* bias_ptr,
    int N, int C_out, int W,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle,
    float beta_val) const
{
    const float alpha = 1.f;
    desc.forward(handle, alpha, beta_val, input, w_ptr, output);
    if (bias_ptr && beta_val == 0.f)
        launch_bias_add(output, bias_ptr, C_out, W, N, block, ctx);
}

// ConvTranspose1d via cudnnConvolutionBackwardData. Algorithm selected at init time.
void pvfinder_unet_t::run_conv_transpose(
    const float* input, float* output,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t in_desc,
    cudnnTensorDescriptor_t out_desc,
    const float* w_ptr, const float* bias_ptr,
    int N, int C_out, int W_out,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle,
    cudnnConvolutionBwdDataAlgo_t algo,
    void* workspace, size_t ws_bytes) const
{
    const float alpha = 1.f, beta = 0.f;
    ALLEN_CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle, &alpha,
        filter_desc, w_ptr,
        in_desc,     input,
        conv_desc,
        algo,
        workspace, ws_bytes,
        &beta,
        out_desc, output));
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
    std::call_once(s_desc_init_flag, init_global_descriptors, handle);

    const dim3 block = m_block_dim;
    constexpr int N = N_CHUNK_INTERVALS;  // batch size = 20 * 40 = 800

    // Scratch buffers (fixed size, reused each event iteration)
    float* x1   = data<dev_unet_x1_t>(arguments);
    float* x2   = data<dev_unet_x2_t>(arguments);
    float* x3   = data<dev_unet_x3_t>(arguments);
    float* up1  = data<dev_unet_up1_t>(arguments);
    float* cat2 = data<dev_unet_cat2_t>(arguments);

    // Buffer aliases (liveness-proven safe)
    float* up2   = cat2;  // cat2[N,128,50] and up2[N,64,100] have same element count
    float* oint  = x1;    // x1 skip consumed before oint written
    float* logits = x3;   // x3 consumed after maxpool; reused as logits

    constexpr unsigned ncw_stride = N_INTERVALS * N_BATCH_CHANNELS * W_IN;
    constexpr unsigned kde_stride = N_INTERVALS * W_IN;

    // ConvTranspose tensor descriptors — local per operator() call so concurrent
    // threads each have their own (cudnnSetTensor4dDescriptor is not thread-safe on shared).
    cudnnTensorDescriptor_t td_up1_in = nullptr, td_up1_out = nullptr;
    cudnnTensorDescriptor_t td_up2_in = nullptr, td_up2_out = nullptr;
    ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&td_up1_in));
    ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&td_up1_out));
    ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&td_up2_in));
    ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&td_up2_out));
    ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        td_up1_in,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_QTR));
    ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        td_up1_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_HALF));
    ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        td_up2_in,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT*2, 1, W_HALF));
    ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        td_up2_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT,   1, W_IN));

    const float* ncw_base = data<dev_pvfinder_interval_features_t>(arguments);
    float*       kde_base = data<dev_pvfinder_kde_output_t>(arguments);

    const unsigned padded_events = ((n_events + B_EVENTS_MAX - 1) / B_EVENTS_MAX) * B_EVENTS_MAX;

    for (unsigned chunk_start = 0; chunk_start < padded_events; chunk_start += B_EVENTS_MAX) {
        const float* ncw = ncw_base + chunk_start * ncw_stride;
        float*       kde = kde_base + chunk_start * kde_stride;

        // ---- Downsampling ----
        run_convbnrelu(s_desc.rcbn1, ncw, x1,
            s_wb.w_rcbn1_w, s_wb.w_rcbn1_b,
            s_wb.w_rcbn1_gamma, s_wb.w_rcbn1_beta,
            s_wb.w_rcbn1_mean,  s_wb.w_rcbn1_var, s_wb.rcbn1_eps,
            N, N_FEAT, W_IN, block, context, handle);

        run_convbnrelu(s_desc.rcbn2, x1, up2,
            s_wb.w_rcbn2_w, s_wb.w_rcbn2_b,
            s_wb.w_rcbn2_gamma, s_wb.w_rcbn2_beta,
            s_wb.w_rcbn2_mean,  s_wb.w_rcbn2_var, s_wb.rcbn2_eps,
            N, N_FEAT, W_IN, block, context, handle);
        launch_maxpool(up2, x2, N, N_FEAT, W_IN, block, context);

        run_convbnrelu(s_desc.rcbn3, x2, up2,
            s_wb.w_rcbn3_w, s_wb.w_rcbn3_b,
            s_wb.w_rcbn3_gamma, s_wb.w_rcbn3_beta,
            s_wb.w_rcbn3_mean,  s_wb.w_rcbn3_var, s_wb.rcbn3_eps,
            N, N_FEAT, W_HALF, block, context, handle);
        launch_maxpool(up2, x3, N, N_FEAT, W_HALF, block, context);

        // ---- Upsampling ----
        run_conv_transpose(x3, up2,
            s_desc.filter_up1_t, s_desc.conv_up1_t, td_up1_in, td_up1_out,
            s_wb.w_up1t_w, s_wb.w_up1t_b,
            N, N_FEAT, W_HALF, block, context, handle,
            s_desc.algo_up1_t, s_desc.ws_up1_t, s_desc.ws_up1_bytes);
        run_convbnrelu(s_desc.up1_c, up2, up1,
            s_wb.w_up1c_w, s_wb.w_up1c_b,
            s_wb.w_up1c_gamma, s_wb.w_up1c_beta,
            s_wb.w_up1c_mean,  s_wb.w_up1c_var, s_wb.up1c_eps,
            N, N_FEAT, W_HALF, block, context, handle);

        launch_concat(up1, x2, cat2, N, N_FEAT, N_FEAT, W_HALF, block, context);
        run_conv_transpose(cat2, logits,
            s_desc.filter_up2_t, s_desc.conv_up2_t, td_up2_in, td_up2_out,
            s_wb.w_up2t_w, s_wb.w_up2t_b,
            N, N_FEAT, W_IN, block, context, handle,
            s_desc.algo_up2_t, s_desc.ws_up2_t, s_desc.ws_up2_bytes);
        run_convbnrelu(s_desc.up2_c, logits, up2,
            s_wb.w_up2c_w, s_wb.w_up2c_b,
            s_wb.w_up2c_gamma, s_wb.w_up2c_beta,
            s_wb.w_up2c_mean,  s_wb.w_up2c_var, s_wb.up2c_eps,
            N, N_FEAT, W_IN, block, context, handle);

        // ---- Output ----
        // Safe sequence: read x1 skip → logits first, then overwrite x1 (=oint) with half-a.
        run_conv(s_desc.oint_half, x1, logits,
            s_wb.w_oint_b_w, nullptr,
            N, N_FEAT, W_IN, block, context, handle, 0.f);
        run_conv(s_desc.oint_half, up2, oint,
            s_wb.w_oint_a_w, nullptr,
            N, N_FEAT, W_IN, block, context, handle, 0.f);
        {
            const unsigned total = (unsigned)(N * N_FEAT * W_IN);
            const unsigned grid  = (total + block.x - 1) / block.x;
            accumulate_add_kernel<<<grid, block, 0, context.stream()>>>(oint, logits, total);
            launch_bias_add(oint, s_wb.w_oint_b, N_FEAT, W_IN, N, block, context);
        }
        run_conv(s_desc.outc, oint, logits,
            s_wb.w_outc_w, s_wb.w_outc_b,
            N, 1, W_IN, block, context, handle, 0.f);

        launch_softplus_scale(logits, KDE_SCALE, N * W_IN, block, context);
        squeeze_copy_kernel<<<
            ((unsigned)(N * W_IN) + block.x - 1) / block.x, block,
            0, context.stream()>>>(logits, kde, N * W_IN);
    }

    cudnnDestroyTensorDescriptor(td_up1_in);
    cudnnDestroyTensorDescriptor(td_up1_out);
    cudnnDestroyTensorDescriptor(td_up2_in);
    cudnnDestroyTensorDescriptor(td_up2_out);

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
        printf("[pvfinder_unet] Validation dump written to %s (%u events)\n",
               dump_dir.c_str(), n_events);
        m_dump_done = true;
    }
#endif
}

} // namespace pvfinder_unet
