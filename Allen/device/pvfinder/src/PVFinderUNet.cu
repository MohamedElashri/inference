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
// Load weights from binary file into WeightRegistry and fill WeightBlob.
// ---------------------------------------------------------------------------
static pvfinder_unet_t::WeightBlob load_weights(const std::string& path)
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
    pvfinder_unet_t::WeightBlob wb {};

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

#endif // ALLEN_CUDNN_BACKEND_CUDA

// ---------------------------------------------------------------------------
// init(): load weights + init global descriptors — both done exactly once.
// ---------------------------------------------------------------------------
void pvfinder_unet_t::init()
{
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    if (m_wb_loaded) return;
    
    // Both weight loading and descriptor creation are now scoped to the algorithm instance.
    m_wb = load_weights(m_weight_file.value());
    m_wb_loaded = true;

    constexpr int N = N_CHUNK_INTERVALS;
    // All forward convs — input shape fixed, IMPLICIT_GEMM, zero workspace.
    m_desc.rcbn1.create(    {N, N_BATCH_CHANNELS, 1, W_IN},  {N_FEAT, N_BATCH_CHANNELS, 1, 25}, {0,12});
    m_desc.rcbn2.create(    {N, N_FEAT,            1, W_IN},  {N_FEAT, N_FEAT,            1,  7}, {0, 3});
    m_desc.rcbn3.create(    {N, N_FEAT,            1, W_HALF},{N_FEAT, N_FEAT,            1,  5}, {0, 2});
    m_desc.up1_c.create(    {N, N_FEAT,            1, W_HALF},{N_FEAT, N_FEAT,            1,  5}, {0, 2});
    m_desc.up2_c.create(    {N, N_FEAT,            1, W_IN},  {N_FEAT, N_FEAT,            1,  5}, {0, 2});
    m_desc.oint_half.create({N, N_FEAT,            1, W_IN},  {N_FEAT, N_FEAT,            1,  5}, {0, 2});
    m_desc.outc.create(     {N, N_FEAT,            1, W_IN},  {1,      N_FEAT,            1,  5}, {0, 2});

    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_desc.filter_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_desc.filter_up1_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_desc.conv_up1_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_desc.conv_up1_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&m_desc.filter_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        m_desc.filter_up2_t, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT*2, N_FEAT, 1, 2));
    ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&m_desc.conv_up2_t));
    ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        m_desc.conv_up2_t, 0,0, 1,2, 1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    auto& reg = Allen::CuDNN::WeightRegistry::instance();
    reg.lock_allocations();
#endif
}

// ---------------------------------------------------------------------------
// set_arguments_size
// IMPLICIT_GEMM needs zero workspace. Allocate 1 float (Allen requires non-zero).
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
// Conv1d + bias + BN + ReLU. IMPLICIT_GEMM: workspace=nullptr, ws=0.
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
    launch_bias_add(output, bias_ptr, C_out, W, N, block, ctx);
    launch_batchnorm(output, bn_gamma, bn_beta, bn_mean, bn_var, bn_eps,
                     C_out, W, N, block, ctx);
    launch_relu(output, N * C_out * W, block, ctx);
}

// Conv1d only (no BN/ReLU). IMPLICIT_GEMM: workspace=nullptr, ws=0.
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

// ConvTranspose1d via cudnnConvolutionBackwardData. ALGO_0: zero workspace for k=2,s=2.
void pvfinder_unet_t::run_conv_transpose(
    const float* input, float* output,
    cudnnFilterDescriptor_t filter_desc,
    cudnnConvolutionDescriptor_t conv_desc,
    cudnnTensorDescriptor_t in_desc,
    cudnnTensorDescriptor_t out_desc,
    const float* w_ptr, const float* bias_ptr,
    int N, int C_out, int W_out,
    const dim3& block, const Allen::Context& ctx,
    cudnnHandle_t handle) const
{
    const float alpha = 1.f, beta = 0.f;
    ALLEN_CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle, &alpha,
        filter_desc, w_ptr,
        in_desc,     input,
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        nullptr, 0,
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
    if (!m_wb_loaded) return;

    const unsigned n_events = first<host_number_of_events_t>(arguments);

    // One thread_local handle per OS thread — created lazily, routed to this stream.
    cudnnHandle_t handle = Allen::CuDNN::get_thread_local_handle(context.stream());

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
        run_convbnrelu(m_desc.rcbn1, ncw, x1,
            m_wb.w_rcbn1_w, m_wb.w_rcbn1_b,
            m_wb.w_rcbn1_gamma, m_wb.w_rcbn1_beta,
            m_wb.w_rcbn1_mean,  m_wb.w_rcbn1_var, m_wb.rcbn1_eps,
            N, N_FEAT, W_IN, block, context, handle);

        run_convbnrelu(m_desc.rcbn2, x1, up2,
            m_wb.w_rcbn2_w, m_wb.w_rcbn2_b,
            m_wb.w_rcbn2_gamma, m_wb.w_rcbn2_beta,
            m_wb.w_rcbn2_mean,  m_wb.w_rcbn2_var, m_wb.rcbn2_eps,
            N, N_FEAT, W_IN, block, context, handle);
        launch_maxpool(up2, x2, N, N_FEAT, W_IN, block, context);

        run_convbnrelu(m_desc.rcbn3, x2, up2,
            m_wb.w_rcbn3_w, m_wb.w_rcbn3_b,
            m_wb.w_rcbn3_gamma, m_wb.w_rcbn3_beta,
            m_wb.w_rcbn3_mean,  m_wb.w_rcbn3_var, m_wb.rcbn3_eps,
            N, N_FEAT, W_HALF, block, context, handle);
        launch_maxpool(up2, x3, N, N_FEAT, W_HALF, block, context);

        // ---- Upsampling ----
        run_conv_transpose(x3, up2,
            m_desc.filter_up1_t, m_desc.conv_up1_t, td_up1_in, td_up1_out,
            m_wb.w_up1t_w, m_wb.w_up1t_b,
            N, N_FEAT, W_HALF, block, context, handle);
        run_convbnrelu(m_desc.up1_c, up2, up1,
            m_wb.w_up1c_w, m_wb.w_up1c_b,
            m_wb.w_up1c_gamma, m_wb.w_up1c_beta,
            m_wb.w_up1c_mean,  m_wb.w_up1c_var, m_wb.up1c_eps,
            N, N_FEAT, W_HALF, block, context, handle);

        launch_concat(up1, x2, cat2, N, N_FEAT, N_FEAT, W_HALF, block, context);
        run_conv_transpose(cat2, logits,
            m_desc.filter_up2_t, m_desc.conv_up2_t, td_up2_in, td_up2_out,
            m_wb.w_up2t_w, m_wb.w_up2t_b,
            N, N_FEAT, W_IN, block, context, handle);
        run_convbnrelu(m_desc.up2_c, logits, up2,
            m_wb.w_up2c_w, m_wb.w_up2c_b,
            m_wb.w_up2c_gamma, m_wb.w_up2c_beta,
            m_wb.w_up2c_mean,  m_wb.w_up2c_var, m_wb.up2c_eps,
            N, N_FEAT, W_IN, block, context, handle);

        // ---- Output ----
        // Safe sequence: read x1 skip → logits first, then overwrite x1 (=oint) with half-a.
        run_conv(m_desc.oint_half, x1, logits,
            m_wb.w_oint_b_w, nullptr,
            N, N_FEAT, W_IN, block, context, handle, 0.f);
        run_conv(m_desc.oint_half, up2, oint,
            m_wb.w_oint_a_w, nullptr,
            N, N_FEAT, W_IN, block, context, handle, 0.f);
        {
            const unsigned total = (unsigned)(N * N_FEAT * W_IN);
            const unsigned grid  = (total + block.x - 1) / block.x;
            accumulate_add_kernel<<<grid, block, 0, context.stream()>>>(oint, logits, total);
            launch_bias_add(oint, m_wb.w_oint_b, N_FEAT, W_IN, N, block, context);
        }
        run_conv(m_desc.outc, oint, logits,
            m_wb.w_outc_w, m_wb.w_outc_b,
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
