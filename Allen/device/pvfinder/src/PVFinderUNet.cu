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

// ---------------------------------------------------------------------------
// init(): load weights, create cuDNN handle & descriptors
// ---------------------------------------------------------------------------
void pvfinder_unet_t::init()
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    if (m_init_done) return;
    // Weight loading touches shared statics — protect with call_once.
    // m_handle and descriptors are per-instance (one per thread/stream) and need no lock.
    static std::once_flag s_load_flag;
    std::call_once(s_load_flag, [this]() {
        s_wb = load_weights(m_weight_file.value());
        s_wb_loaded = true;
    });
    m_handle.create();
    init_descriptors();
    m_init_done = true;
#endif
}

void pvfinder_unet_t::init_descriptors() const
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    // All 1D convolutions modelled as 4D with H=1
    // format: {K, C, R=1, S=kernel}
    m_desc_rcbn1.create(m_handle, {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0,12},{1,1},{1,1});
    m_desc_rcbn2.create(m_handle, {N_FEAT, N_FEAT,            1,  7}, {0, 3},{1,1},{1,1});
    m_desc_rcbn3.create(m_handle, {N_FEAT, N_FEAT,            1,  5}, {0, 2},{1,1},{1,1});
    m_desc_up1_c.create(m_handle, {N_FEAT, N_FEAT,            1,  5}, {0, 2},{1,1},{1,1});
    m_desc_up2_c.create   (m_handle, {N_FEAT, N_FEAT,   1, 5}, {0,2},{1,1},{1,1});
    m_desc_oint_half.create(m_handle, {N_FEAT, N_FEAT,   1, 5}, {0,2},{1,1},{1,1}); // 64→64 half
    m_desc_outc.create     (m_handle, {1,      N_FEAT,   1, 5}, {0,2},{1,1},{1,1});

    // Transpose conv filter+conv descriptors (read-only after init, thread-safe to share)
    // up1t: filter [N_FEAT, N_FEAT, 1, 2], stride=2
    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_filter_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_filter_up1_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_conv_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_conv_up1_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // up2t: filter [N_FEAT*2, N_FEAT, 1, 2], stride=2
    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_filter_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_filter_up2_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT*2, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_conv_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_conv_up2_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Pre-warm ConvDescriptors workspace cache at N=40 so operator() only reads the cache
    // (workspace_bytes() is not safe to call concurrently as it mutates m_cache_valid)
    constexpr unsigned N = N_INTERVALS;
    m_desc_rcbn1.workspace_bytes(m_handle, {(int)N, N_BATCH_CHANNELS, 1, W_IN});
    m_desc_rcbn2.workspace_bytes(m_handle, {(int)N, N_FEAT, 1, W_IN});
    m_desc_rcbn3.workspace_bytes(m_handle, {(int)N, N_FEAT, 1, W_HALF});
    m_desc_up1_c.workspace_bytes(m_handle, {(int)N, N_FEAT, 1, W_HALF});
    m_desc_up2_c.workspace_bytes(m_handle, {(int)N, N_FEAT, 1, W_IN});
    m_desc_oint_half.workspace_bytes(m_handle, {(int)N, N_FEAT, 1, W_IN});
    m_desc_outc.workspace_bytes(m_handle, {(int)N, N_FEAT, 1, W_IN});
    // Tensor descriptors for transpose-conv are local in operator() — not stored here.
#endif
}

// ---------------------------------------------------------------------------
// max_workspace_bytes: worst-case over all conv layers
// ---------------------------------------------------------------------------
size_t pvfinder_unet_t::max_workspace_bytes(unsigned N) const
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    size_t ws = 0;
    auto mx = [&](const Allen::CuDNN::ConvDescriptors& d, const Allen::CuDNN::TensorShape& s) {
        size_t w = d.workspace_bytes(m_handle, s);
        if (w > ws) ws = w;
    };
    mx(m_desc_rcbn1, {(int)N, N_BATCH_CHANNELS, 1, W_IN});
    mx(m_desc_rcbn2, {(int)N, N_FEAT,           1, W_IN});
    mx(m_desc_rcbn3, {(int)N, N_FEAT,           1, W_HALF});
    mx(m_desc_up1_c, {(int)N, N_FEAT,           1, W_HALF});
    mx(m_desc_up2_c,    {(int)N, N_FEAT, 1, W_IN});
    mx(m_desc_oint_half,{(int)N, N_FEAT, 1, W_IN});
    mx(m_desc_outc,     {(int)N, N_FEAT, 1, W_IN});

    // Also query backward-data (ConvTranspose) workspace sizes using local descriptors
    // (cannot use m_td_* members — they no longer exist; operator() creates them locally)
    cudnnTensorDescriptor_t td_in = nullptr, td_out = nullptr;
    cudnnCreateTensorDescriptor(&td_in);
    cudnnCreateTensorDescriptor(&td_out);
    auto mx_bwd = [&](cudnnFilterDescriptor_t filt, cudnnConvolutionDescriptor_t conv,
                      int in_c, int in_w, int out_c, int out_w) {
        cudnnSetTensor4dDescriptor(td_in,  CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, in_c,  1, in_w);
        cudnnSetTensor4dDescriptor(td_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, out_c, 1, out_w);
        size_t bwd_ws = 0;
        cudnnGetConvolutionBackwardDataWorkspaceSize(
            m_handle.get(), filt, td_in, conv,
            td_out, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &bwd_ws);
        if (bwd_ws > ws) ws = bwd_ws;
    };
    mx_bwd(m_filter_up1_t, m_conv_up1_t, N_FEAT,   W_QTR,  N_FEAT, W_HALF);
    mx_bwd(m_filter_up2_t, m_conv_up2_t, N_FEAT*2, W_HALF, N_FEAT, W_IN);
    cudnnDestroyTensorDescriptor(td_in);
    cudnnDestroyTensorDescriptor(td_out);

    return ws;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// set_arguments_size
// ---------------------------------------------------------------------------
void pvfinder_unet_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
{
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    // Scratch buffers are fixed-size for ONE event (N_INTERVALS=40 intervals).
    // operator() loops over events sequentially, reusing these buffers each iteration.
    // This matches Allen's event-parallel idiom: one event at a time per stream,
    // GPU parallelism is within the cuDNN call (across the 40 intervals and channels).
    static constexpr unsigned N = N_INTERVALS;   // 40 — fixed single-event batch

    set_size<dev_unet_x1_t>   (arguments, N * N_FEAT * W_IN);   // 1.024 MB
    set_size<dev_unet_x2_t>   (arguments, N * N_FEAT * W_HALF); // 0.512 MB
    set_size<dev_unet_x3_t>   (arguments, N * N_FEAT * W_IN);   // 1.024 MB (>= x3_raw AND logits)
    set_size<dev_unet_up1_t>  (arguments, N * N_FEAT * W_HALF); // 0.512 MB
    set_size<dev_unet_cat2_t> (arguments, N * N_FEAT * 2 * W_HALF); // 1.024 MB (== up2)
    // Workspace: IMPLICIT_GEMM fwd needs 0, but ConvTranspose (bwd-data) may need some.
    // Query the real worst-case when descriptors are ready, else use a safe 4 MB fallback.
    size_t ws = m_init_done ? max_workspace_bytes(N) : (4u * 1024u * 1024u);
    set_size<dev_unet_conv_ws_t>(arguments, (ws + sizeof(float) - 1) / sizeof(float) + 1);
    // Output sized for all events
    set_size<dev_pvfinder_kde_output_t>(arguments, n_events * N_INTERVALS * W_IN);
}

// ---------------------------------------------------------------------------
// Per-layer helpers
// ---------------------------------------------------------------------------

// Conv1d (via cuDNN 4D with H=1) + bias add + BN + ReLU
void pvfinder_unet_t::run_convbnrelu(
    const Allen::CuDNN::ConvDescriptors& desc,
    const float* input, float* output,
    const float* w_ptr, const float* bias_ptr,
    const float* bn_gamma, const float* bn_beta,
    const float* bn_mean,  const float* bn_var, float bn_eps,
    void* workspace, size_t ws_bytes,
    int N, int C_in, int C_out, int W,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle) const
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    Allen::CuDNN::Handle h; h.wrap(handle);
    const Allen::CuDNN::TensorShape input_shape {N, C_in, 1, W};
    desc.workspace_bytes(h, input_shape);  // caches output shape

    const float alpha = 1.f, beta = 0.f;
    desc.forward(h, alpha, beta, input, w_ptr, output, workspace, ws_bytes);
    launch_bias_add(output, bias_ptr, C_out, W, N, block, ctx);
    launch_batchnorm(output, bn_gamma, bn_beta, bn_mean, bn_var, bn_eps,
                     C_out, W, N, block, ctx);
    launch_relu(output, N * C_out * W, block, ctx);
#endif
}

// Conv1d only (no BN/ReLU) — used for out_intermediate and outc
void pvfinder_unet_t::run_conv(
    const Allen::CuDNN::ConvDescriptors& desc,
    const float* input, float* output,
    const float* w_ptr, const float* bias_ptr,
    void* workspace, size_t ws_bytes,
    int N, int C_in, int C_out, int W,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle,
    float beta_val) const
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    Allen::CuDNN::Handle h; h.wrap(handle);
    const Allen::CuDNN::TensorShape input_shape {N, C_in, 1, W};
    desc.workspace_bytes(h, input_shape);
    const float alpha = 1.f;
    desc.forward(h, alpha, beta_val, input, w_ptr, output, workspace, ws_bytes);
    if (bias_ptr && beta_val == 0.f) {
        launch_bias_add(output, bias_ptr, C_out, W, N, block, ctx);
    }
#endif
}

// ConvTranspose1d via cudnnConvolutionBackwardData + bias
void pvfinder_unet_t::run_conv_transpose(
    const float* input, float* output,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t in_desc,
    cudnnTensorDescriptor_t out_desc,
    const float* w_ptr, const float* bias_ptr,
    void* workspace, size_t ws_bytes,
    int N, int C_out, int W_out,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle) const
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    const float alpha = 1.f, beta = 0.f;
    ALLEN_CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        filter_desc, w_ptr,
        in_desc,     input,
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        workspace, ws_bytes,
        &beta,
        out_desc, output));
    launch_bias_add(output, bias_ptr, C_out, W_out, N, block, ctx);
#endif
}

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

    // Each Allen thread (stream) needs its own cudnnHandle_t.
    // Allen shares one algorithm instance across all threads, so m_handle is shared.
    // Using thread_local gives each OS thread its own independent handle.
    thread_local cudnnHandle_t tl_handle = nullptr;
    if (tl_handle == nullptr) {
        ALLEN_CUDNN_CHECK(cudnnCreate(&tl_handle));
    }
    ALLEN_CUDNN_CHECK(cudnnSetStream(tl_handle, context.stream()));

    const dim3 block = m_block_dim;
    constexpr int N = N_INTERVALS;  // 40 — fixed per-event batch size
    const size_t ws_bytes = max_workspace_bytes(N);

    // Fixed-size scratch buffers (sized for ONE event in set_arguments_size)
    float* x1      = data<dev_unet_x1_t>(arguments);
    float* x2      = data<dev_unet_x2_t>(arguments);
    float* x3      = data<dev_unet_x3_t>(arguments);
    float* up1     = data<dev_unet_up1_t>(arguments);
    float* cat2    = data<dev_unet_cat2_t>(arguments);
    void*  workspace = static_cast<void*>(data<dev_unet_conv_ws_t>(arguments));

    // Aliased pointers (proven safe by liveness analysis — see set_arguments_size)
    float* up2     = cat2;  // same #elems as cat2[40,128,50]; written after cat2 consumed
    float* oint    = x1;   // x1 skip fully consumed before oint is written
    float* logits  = x3;   // x3 consumed after maxpool; also used as up2t temp

    // Input/output strides per event
    constexpr unsigned ncw_stride = N_INTERVALS * N_BATCH_CHANNELS * W_IN; // 40*8*100
    constexpr unsigned kde_stride = N_INTERVALS * W_IN;                     // 40*100

    // Create transpose-conv tensor descriptors as locals — each operator() call gets its
    // own descriptors so concurrent threads don't race on shared mutable state.
    cudnnTensorDescriptor_t td_up1_in  = nullptr, td_up1_out = nullptr;
    cudnnTensorDescriptor_t td_up2_in  = nullptr, td_up2_out = nullptr;
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

    // Base pointers (advanced per event in the loop)
    const float* ncw_base = data<dev_pvfinder_ncw_tensor_t>(arguments);
    float*       kde_base = data<dev_pvfinder_kde_output_t>(arguments);

    // --- Per-event loop ---
    // Allen processes one event at a time through the stream; scratch buffers are reused
    // each iteration. GPU parallelism is within each cuDNN call (across 40 intervals,
    // 64 channels, 100 spatial positions). This matches Allen's event-parallel idiom.
    for (unsigned ev = 0; ev < n_events; ++ev) {
        const float* ncw = ncw_base + ev * ncw_stride;
        float*       kde = kde_base + ev * kde_stride;

        // ---- Downsampling path ----
        // rcbn1: [40, 8, 100] -> x1[40, 64, 100]
        run_convbnrelu(m_desc_rcbn1, ncw, x1,
            s_wb.w_rcbn1_w, s_wb.w_rcbn1_b,
            s_wb.w_rcbn1_gamma, s_wb.w_rcbn1_beta,
            s_wb.w_rcbn1_mean,  s_wb.w_rcbn1_var, s_wb.rcbn1_eps,
            workspace, ws_bytes, N, N_BATCH_CHANNELS, N_FEAT, W_IN, block, context, tl_handle);

        // rcbn2(x1) -> up2[40,64,100] (scratch), then MaxPool -> x2[40,64,50]
        run_convbnrelu(m_desc_rcbn2, x1, up2,
            s_wb.w_rcbn2_w, s_wb.w_rcbn2_b,
            s_wb.w_rcbn2_gamma, s_wb.w_rcbn2_beta,
            s_wb.w_rcbn2_mean,  s_wb.w_rcbn2_var, s_wb.rcbn2_eps,
            workspace, ws_bytes, N, N_FEAT, N_FEAT, W_IN, block, context, tl_handle);
        launch_maxpool(up2, x2, N, N_FEAT, W_IN, block, context);

        // rcbn3(x2) -> up2[40,64,50] (scratch), then MaxPool -> x3[40,64,25]
        run_convbnrelu(m_desc_rcbn3, x2, up2,
            s_wb.w_rcbn3_w, s_wb.w_rcbn3_b,
            s_wb.w_rcbn3_gamma, s_wb.w_rcbn3_beta,
            s_wb.w_rcbn3_mean,  s_wb.w_rcbn3_var, s_wb.rcbn3_eps,
            workspace, ws_bytes, N, N_FEAT, N_FEAT, W_HALF, block, context, tl_handle);
        launch_maxpool(up2, x3, N, N_FEAT, W_HALF, block, context);

        // ---- Upsampling path ----
        // up1t: ConvTranspose x3[40,64,25] -> up2[40,64,50] (scratch)
        run_conv_transpose(x3, up2,
            m_filter_up1_t, m_conv_up1_t, td_up1_in, td_up1_out,
            s_wb.w_up1t_w, s_wb.w_up1t_b,
            workspace, ws_bytes, N, N_FEAT, W_HALF, block, context, tl_handle);
        // up1c: ConvBNrelu up2 -> up1[40,64,50]
        run_convbnrelu(m_desc_up1_c, up2, up1,
            s_wb.w_up1c_w, s_wb.w_up1c_b,
            s_wb.w_up1c_gamma, s_wb.w_up1c_beta,
            s_wb.w_up1c_mean,  s_wb.w_up1c_var, s_wb.up1c_eps,
            workspace, ws_bytes, N, N_FEAT, N_FEAT, W_HALF, block, context, tl_handle);

        // concat(up1, x2) -> cat2[40,128,50]
        launch_concat(up1, x2, cat2, N, N_FEAT, N_FEAT, W_HALF, block, context);
        // up2t: ConvTranspose cat2[40,128,50] -> logits[40,64,100] (temp; x3/logits reused)
        run_conv_transpose(cat2, logits,
            m_filter_up2_t, m_conv_up2_t, td_up2_in, td_up2_out,
            s_wb.w_up2t_w, s_wb.w_up2t_b,
            workspace, ws_bytes, N, N_FEAT, W_IN, block, context, tl_handle);
        // up2c: ConvBNrelu logits_temp -> up2[40,64,100] (= cat2 buffer)
        run_convbnrelu(m_desc_up2_c, logits, up2,
            s_wb.w_up2c_w, s_wb.w_up2c_b,
            s_wb.w_up2c_gamma, s_wb.w_up2c_beta,
            s_wb.w_up2c_mean,  s_wb.w_up2c_var, s_wb.up2c_eps,
            workspace, ws_bytes, N, N_FEAT, N_FEAT, W_IN, block, context, tl_handle);

        // ---- Output path ----
        // out_intermediate: two half-convolutions accumulate into oint(=x1)
        //   half-a: conv(up2,  w_oint_a) -> oint   (beta=0, overwrites)
        //   half-b: conv(x1_orig, w_oint_b) -> logits (beta=0)  then  oint += logits
        // Note: oint=x1 is written by half-a BEFORE x1 is read by half-b; x1 still
        // holds the skip at this point because we read it as input to half-b FIRST.
        // Safe sequence: half-b(src=x1 → dst=logits), half-a(src=up2 → dst=oint=x1), add.
        run_conv(m_desc_oint_half, x1, logits,     // read x1 skip while still intact
            s_wb.w_oint_b_w, nullptr,
            workspace, ws_bytes, N, N_FEAT, N_FEAT, W_IN, block, context, tl_handle, 0.f);
        run_conv(m_desc_oint_half, up2, oint,       // now overwrite x1 with half-a result
            s_wb.w_oint_a_w, nullptr,
            workspace, ws_bytes, N, N_FEAT, N_FEAT, W_IN, block, context, tl_handle, 0.f);
        // oint += logits + bias
        {
            const unsigned total = (unsigned)(N * N_FEAT * W_IN);
            const unsigned grid  = (total + block.x - 1) / block.x;
            accumulate_add_kernel<<<grid, block, 0, context.stream()>>>(oint, logits, total);
            launch_bias_add(oint, s_wb.w_oint_b, N_FEAT, W_IN, N, block, context);
        }

        // outc: Conv(64→1) + bias -> logits[40,1,100]
        run_conv(m_desc_outc, oint, logits,
            s_wb.w_outc_w, s_wb.w_outc_b,
            workspace, ws_bytes, N, N_FEAT, 1, W_IN, block, context, tl_handle, 0.f);

        // Softplus*0.001 in-place on logits[40,1,100], then copy to kde output
        launch_softplus_scale(logits, KDE_SCALE, N * W_IN, block, context);
        squeeze_copy_kernel<<<
            ((unsigned)(N * W_IN) + block.x - 1) / block.x, block,
            0, context.stream()>>>(logits, kde, N * W_IN);
    }

    // Destroy local tensor descriptors (created once per operator() call)
    cudnnDestroyTensorDescriptor(td_up1_in);
    cudnnDestroyTensorDescriptor(td_up1_out);
    cudnnDestroyTensorDescriptor(td_up2_in);
    cudnnDestroyTensorDescriptor(td_up2_out);

    // Validation dump: write NCW input and KDE output to binary files.
    // Only done once (first slice) and only when dump_dir property is set.
    const std::string& dump_dir = m_dump_dir.value();
    if (!dump_dir.empty() && !m_dump_done) {
        // Synchronise stream before D→H copy
        cudaStreamSynchronize(context.stream());

        const unsigned ncw_elems = n_events * N_INTERVALS * N_BATCH_CHANNELS * W_IN;
        const unsigned kde_elems = n_events * N_INTERVALS * W_IN;

        std::vector<float> h_ncw(ncw_elems);
        std::vector<float> h_kde(kde_elems);
        cudaMemcpy(h_ncw.data(),
                   data<dev_pvfinder_ncw_tensor_t>(arguments),
                   ncw_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_kde.data(),
                   data<dev_pvfinder_kde_output_t>(arguments),
                   kde_elems * sizeof(float), cudaMemcpyDeviceToHost);

        // Header: [magic u32][n_events u32]
        const uint32_t magic = 0xAB1EU;
        auto write_bin = [&](const std::string& path,
                             const float* data, unsigned n) {
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(&magic),    sizeof(magic));
            f.write(reinterpret_cast<const char*>(&n_events), sizeof(n_events));
            f.write(reinterpret_cast<const char*>(data),      n * sizeof(float));
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
