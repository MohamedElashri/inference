/*****************************************************************************\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/
#include "PVFinderUNet.cuh"
#ifdef ALLEN_CUDNN_BACKEND_CUDA
#include "PVFinderUNetKernels.cuh"
#include "PVFinderUNetFused.cuh"
#include <cuda_bf16.h>
#endif

#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_unet::pvfinder_unet_t)

namespace pvfinder_unet {

#ifdef ALLEN_CUDNN_BACKEND_CUDA
  // Weight blob: device pointers per layer (filled in init(), used in operator()).
  struct WeightBlob {
    const float* w_rcbn1_w;
    const float* w_rcbn1_b;
    const float* w_rcbn1_gamma;
    const float* w_rcbn1_beta;
    const float* w_rcbn1_mean;
    const float* w_rcbn1_var;
    float rcbn1_eps;

    const float* w_rcbn2_w;
    const float* w_rcbn2_b;
    const float* w_rcbn2_gamma;
    const float* w_rcbn2_beta;
    const float* w_rcbn2_mean;
    const float* w_rcbn2_var;
    float rcbn2_eps;

    const float* w_rcbn3_w;
    const float* w_rcbn3_b;
    const float* w_rcbn3_gamma;
    const float* w_rcbn3_beta;
    const float* w_rcbn3_mean;
    const float* w_rcbn3_var;
    float rcbn3_eps;

    const float* w_up1t_w;
    const float* w_up1t_b;
    const float* w_up1c_w;
    const float* w_up1c_b;
    const float* w_up1c_gamma;
    const float* w_up1c_beta;
    const float* w_up1c_mean;
    const float* w_up1c_var;
    float up1c_eps;

    const float* w_up2t_w;
    const float* w_up2t_b;
    const float* w_up2c_w;
    const float* w_up2c_b;
    const float* w_up2c_gamma;
    const float* w_up2c_beta;
    const float* w_up2c_mean;
    const float* w_up2c_var;
    float up2c_eps;

    const float* w_oint_w;
    const float* w_oint_b;
    const float* w_outc_w;
    const float* w_outc_b;
  };

  // Everything one pvfinder_unet instance owns: its weights, with BatchNorm
  // folded into the CBR layers, and the configured precision's path, all set up
  // in init().
  struct pvfinder_unet_t::UNetState {
    WeightBlob wb {};
    // BN-folded CBR weights and biases (device), in the order rcbn1, rcbn2,
    // rcbn3, up1c, up2c.
    float* w_f[5] = {};
    float* b_f[5] = {};

    // float32: cuDNN convolutions (IMPLICIT_GEMM pinned, no workspace).
    Allen::CuDNN::ConvDescriptors conv[5]; // CBR layers, as w_f
    Allen::CuDNN::ConvDescriptors oint;    // out_intermediate, Conv(C -> C, k 5)
    Allen::CuDNN::ConvDescriptors outc;    // outc, Conv(C -> 1, k 5)
    // ConvTranspose (k 2, stride 2): filter and convolution descriptors, and
    // the backward-data algorithm chosen at init (workspace-free).
    cudnnFilterDescriptor_t ct_filter[2] = {};
    cudnnConvolutionDescriptor_t ct_conv[2] = {};
    cudnnConvolutionBwdDataAlgo_t ct_algo[2] = {CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0};

    // bfloat16: the fused kernel's image of all weights, and its full grid.
    unsigned char* fused_blob = nullptr;
    int fused_grid = 0;

    // The UNet's output for an all-zero interval ([W_IN] floats, device),
    // written to every interval without tracks. Computed once, by the
    // configured path, on the first call.
    float* empty_response = nullptr;
    std::once_flag empty_response_flag;
  };

  namespace {
    // ---------------------------------------------------------------------------
    // Thread-local ConvTranspose tensor descriptors.
    // Shapes are compile-time constants (N, N_FEAT, W_QTR/W_HALF/W_IN never
    // change), so each OS thread creates its set exactly once — lazily, on first
    // use — and reuses it for the thread's lifetime, matching the lifetime of
    // Allen::CuDNN::get_thread_local_handle (CuDNNHandle.h): null-check-then-create,
    // never explicitly destroyed, and released by process teardown.
    // ---------------------------------------------------------------------------
    struct ConvTransposeTensorDescs {
      cudnnTensorDescriptor_t td_up1_in = nullptr;
      cudnnTensorDescriptor_t td_up1_out = nullptr;
      cudnnTensorDescriptor_t td_up2_in = nullptr;
      cudnnTensorDescriptor_t td_up2_out = nullptr;
    };

    static const ConvTransposeTensorDescs& get_thread_local_conv_transpose_descs(const void* owner, int N)
    {
      // One per (thread, algorithm instance): shapes and contents belong to one instance.
      thread_local std::unordered_map<const void*, ConvTransposeTensorDescs> cache;
      ConvTransposeTensorDescs& descs = cache[owner];
      if (descs.td_up1_in == nullptr) {
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up1_in));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up1_out));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up2_in));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs.td_up2_out));
        ALLEN_CUDNN_CHECK(
          cudnnSetTensor4dDescriptor(descs.td_up1_in, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT, 1, W_QTR));
        ALLEN_CUDNN_CHECK(
          cudnnSetTensor4dDescriptor(descs.td_up1_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT, 1, W_HALF));
        ALLEN_CUDNN_CHECK(
          cudnnSetTensor4dDescriptor(descs.td_up2_in, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT, 1, W_HALF));
        ALLEN_CUDNN_CHECK(
          cudnnSetTensor4dDescriptor(descs.td_up2_out, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT, 1, W_IN));
      }
      return descs;
    }

    // CBR layers: input channels, kernel size, padding, width.
    struct CBRShape {
      int c_in, r, pad, w;
    };
    constexpr CBRShape cbr_shapes[5] = {
      {N_BATCH_CHANNELS, 25, 12, W_IN},
      {N_FEAT, 7, 3, W_IN},
      {N_FEAT, 5, 2, W_HALF},
      {N_FEAT, 5, 2, W_HALF},
      {N_FEAT, 5, 2, W_IN}};

    template<typename T>
    std::vector<T> to_host(const T* device, size_t n)
    {
      std::vector<T> host(n);
      Allen::memcpy(host.data(), device, n * sizeof(T), Allen::memcpyDeviceToHost);
      return host;
    }
  } // namespace

  // The weights on the device, from the model (tensors named as in the PyTorch
  // state dict, shapes checked against this build: C = N_BATCH_CHANNELS input
  // channels, F = N_FEAT feature maps, no skip connections).
  static WeightBlob weights_from_model(PVFinder::Model& model)
  {
    WeightBlob wb {};
    const auto conv = [&](const std::string& p, int in, int out, int k, const float*& w, const float*& b) {
      w = model.device_tensor(p + ".weight", {out, in, k});
      b = model.device_tensor(p + ".bias", {out});
    };
    const auto bn = [&](
                      const std::string& p,
                      const float*& gamma,
                      const float*& beta,
                      const float*& mean,
                      const float*& var,
                      float& eps) {
      gamma = model.device_tensor(p + ".weight", {N_FEAT});
      beta = model.device_tensor(p + ".bias", {N_FEAT});
      mean = model.device_tensor(p + ".running_mean", {N_FEAT});
      var = model.device_tensor(p + ".running_var", {N_FEAT});
      eps = model.bn_eps();
    };
    const auto conv_transpose = [&](const std::string& p, const float*& w, const float*& b) {
      w = model.device_tensor(p + ".weight", {N_FEAT, N_FEAT, 2}); // [in][out][k]
      b = model.device_tensor(p + ".bias", {N_FEAT});
    };
    conv("rcbn1.0", N_BATCH_CHANNELS, N_FEAT, 25, wb.w_rcbn1_w, wb.w_rcbn1_b);
    bn("rcbn1.1", wb.w_rcbn1_gamma, wb.w_rcbn1_beta, wb.w_rcbn1_mean, wb.w_rcbn1_var, wb.rcbn1_eps);
    conv("rcbn2.0", N_FEAT, N_FEAT, 7, wb.w_rcbn2_w, wb.w_rcbn2_b);
    bn("rcbn2.1", wb.w_rcbn2_gamma, wb.w_rcbn2_beta, wb.w_rcbn2_mean, wb.w_rcbn2_var, wb.rcbn2_eps);
    conv("rcbn3.0", N_FEAT, N_FEAT, 5, wb.w_rcbn3_w, wb.w_rcbn3_b);
    bn("rcbn3.1", wb.w_rcbn3_gamma, wb.w_rcbn3_beta, wb.w_rcbn3_mean, wb.w_rcbn3_var, wb.rcbn3_eps);
    conv_transpose("up1.0", wb.w_up1t_w, wb.w_up1t_b);
    conv("up1.1.0", N_FEAT, N_FEAT, 5, wb.w_up1c_w, wb.w_up1c_b);
    bn("up1.1.1", wb.w_up1c_gamma, wb.w_up1c_beta, wb.w_up1c_mean, wb.w_up1c_var, wb.up1c_eps);
    conv_transpose("up2.0", wb.w_up2t_w, wb.w_up2t_b);
    conv("up2.1.0", N_FEAT, N_FEAT, 5, wb.w_up2c_w, wb.w_up2c_b);
    bn("up2.1.1", wb.w_up2c_gamma, wb.w_up2c_beta, wb.w_up2c_mean, wb.w_up2c_var, wb.up2c_eps);
    conv("out_intermediate", N_FEAT, N_FEAT, 5, wb.w_oint_w, wb.w_oint_b);
    conv("outc", N_FEAT, 1, 5, wb.w_outc_w, wb.w_outc_b);
    return wb;
  }

#endif // ALLEN_CUDNN_BACKEND_CUDA

  // ---------------------------------------------------------------------------
  // init(): weights, BatchNorm folding, and the configured precision's path.
  // ---------------------------------------------------------------------------
  void pvfinder_unet_t::init()
  {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    if (m_state) return;
    const std::string& precision = m_precision.value();
    if (precision != "float32" && precision != "bfloat16") {
      throw StrException("pvfinder_unet: precision must be float32 or bfloat16, got '" + precision + "'");
    }
    m_bf16 = precision == "bfloat16";
    auto state = std::make_shared<UNetState>();
    state->wb = weights_from_model(m_model);
    const WeightBlob& wb = state->wb;

    // Fold BatchNorm into the CBR layers' weights and biases, on the device:
    // scale = gamma / sqrt(var + eps), w_f = scale w, b_f = scale (b - mean) + beta.
    // A stream of its own for these one-off kernels.
    Allen::Context setup;
    setup.initialize(0);
    const float* bn[5][7] = {
      {wb.w_rcbn1_w, wb.w_rcbn1_b, wb.w_rcbn1_gamma, wb.w_rcbn1_beta, wb.w_rcbn1_mean, wb.w_rcbn1_var},
      {wb.w_rcbn2_w, wb.w_rcbn2_b, wb.w_rcbn2_gamma, wb.w_rcbn2_beta, wb.w_rcbn2_mean, wb.w_rcbn2_var},
      {wb.w_rcbn3_w, wb.w_rcbn3_b, wb.w_rcbn3_gamma, wb.w_rcbn3_beta, wb.w_rcbn3_mean, wb.w_rcbn3_var},
      {wb.w_up1c_w, wb.w_up1c_b, wb.w_up1c_gamma, wb.w_up1c_beta, wb.w_up1c_mean, wb.w_up1c_var},
      {wb.w_up2c_w, wb.w_up2c_b, wb.w_up2c_gamma, wb.w_up2c_beta, wb.w_up2c_mean, wb.w_up2c_var}};
    const float eps[5] = {wb.rcbn1_eps, wb.rcbn2_eps, wb.rcbn3_eps, wb.up1c_eps, wb.up2c_eps};
    for (int l = 0; l < 5; ++l) {
      const int k_size = cbr_shapes[l].c_in * cbr_shapes[l].r;
      Allen::malloc((void**) &state->w_f[l], (size_t) N_FEAT * k_size * sizeof(float));
      Allen::malloc((void**) &state->b_f[l], (size_t) N_FEAT * sizeof(float));
      global_function(fold_bn_into_conv_kernel)(dim3(N_FEAT), dim3(256), setup)(
        state->w_f[l],
        state->b_f[l],
        bn[l][0],
        bn[l][1],
        bn[l][2],
        bn[l][3],
        bn[l][4],
        bn[l][5],
        eps[l],
        N_FEAT,
        k_size);
    }
    Allen::synchronize(setup);
    cudaCheck(cudaStreamDestroy(setup.stream()));

    if (m_bf16) {
      int device = 0, cc_major = 0, sm_count = 0, per_sm = 0;
      cudaCheck(cudaGetDevice(&device));
      cudaCheck(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
      cudaCheck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
      if (cc_major < 8 || N_FEAT != fused::C || N_BATCH_CHANNELS != fused::CIN) {
        throw StrException("pvfinder_unet: precision = bfloat16 needs compute capability 8.0 or newer and a "
                           "build with 16 feature maps and 4 input channels");
      }
      // The fused kernel's weight image: the BN-folded convolution weights,
      // transposed to [K][R][C] and rounded to BF16, BF16 biases, the
      // ConvTransposes' FP32 weights and the output stage's parameters.
      std::vector<float> conv_w[5], conv_b[5], ct_w[2], ct_b[2];
      for (int l = 0; l < 5; ++l) {
        const int c_in = cbr_shapes[l].c_in, r = cbr_shapes[l].r;
        const std::vector<float> w = to_host(state->w_f[l], (size_t) N_FEAT * c_in * r);
        const std::vector<float> b = to_host(state->b_f[l], N_FEAT);
        conv_w[l].resize(w.size());
        for (int k = 0; k < N_FEAT; ++k)
          for (int c = 0; c < c_in; ++c)
            for (int i = 0; i < r; ++i)
              conv_w[l][((size_t) k * r + i) * c_in + c] =
                __bfloat162float(__float2bfloat16(w[((size_t) k * c_in + c) * r + i]));
        conv_b[l].resize(N_FEAT);
        for (int i = 0; i < N_FEAT; ++i)
          conv_b[l][i] = __bfloat162float(__float2bfloat16(b[i]));
      }
      const float* ctw[2] = {wb.w_up1t_w, wb.w_up2t_w};
      const float* ctb[2] = {wb.w_up1t_b, wb.w_up2t_b};
      for (int t = 0; t < 2; ++t) {
        ct_w[t] = to_host(ctw[t], (size_t) N_FEAT * N_FEAT * 2);
        ct_b[t] = to_host(ctb[t], N_FEAT);
      }
      const std::vector<float> out_params = make_output_stage_params<N_FEAT>(
        to_host(wb.w_oint_w, (size_t) N_FEAT * N_FEAT * 5),
        to_host(wb.w_oint_b, N_FEAT),
        to_host(wb.w_outc_w, (size_t) N_FEAT * 5),
        to_host(wb.w_outc_b, 1)[0]);
      const std::vector<unsigned char> blob = fused::make_fused_unet_blob(conv_w, conv_b, ct_w, ct_b, out_params);
      Allen::malloc((void**) &state->fused_blob, blob.size());
      Allen::memcpy(state->fused_blob, blob.data(), blob.size(), Allen::memcpyHostToDevice);
      cudaCheck(cudaFuncSetAttribute(
        fused::fused_unet_bf16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, fused::SMEM_BYTES));
      cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, fused::fused_unet_bf16_kernel, fused::THREADS, fused::SMEM_BYTES));
      state->fused_grid = sm_count * std::max(per_sm, 1);
    }
    else {
      // cuDNN descriptors for batches of N rows.
      cudnnHandle_t handle = Allen::CuDNN::get_thread_local_handle(nullptr);
      const int N = (int) m_unet_batch_events.value() * N_INTERVALS;
      for (int l = 0; l < 5; ++l) {
        const CBRShape& s = cbr_shapes[l];
        state->conv[l].create(
          handle, {N, s.c_in, 1, s.w}, {N_FEAT, s.c_in, 1, s.r}, {0, s.pad}, {1, 1}, {1, 1}, CUDNN_DATA_FLOAT, 0);
      }
      state->oint.create(
        handle, {N, N_FEAT, 1, W_IN}, {N_FEAT, N_FEAT, 1, 5}, {0, 2}, {1, 1}, {1, 1}, CUDNN_DATA_FLOAT, 0);
      state->outc.create(handle, {N, N_FEAT, 1, W_IN}, {1, N_FEAT, 1, 5}, {0, 2}, {1, 1}, {1, 1}, CUDNN_DATA_FLOAT, 0);
      // ConvTranspose1d(k 2, stride 2) as cudnnConvolutionBackwardData:
      // up1 W_QTR -> W_HALF, up2 W_HALF -> W_IN. The first workspace-free
      // algorithm of cuDNN's heuristic.
      const int w_in[2] = {W_QTR, W_HALF}, w_out[2] = {W_HALF, W_IN};
      for (int t = 0; t < 2; ++t) {
        ALLEN_CUDNN_CHECK(cudnnCreateFilterDescriptor(&state->ct_filter[t]));
        ALLEN_CUDNN_CHECK(
          cudnnSetFilter4dDescriptor(state->ct_filter[t], CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, N_FEAT, N_FEAT, 1, 2));
        ALLEN_CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&state->ct_conv[t]));
        ALLEN_CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
          state->ct_conv[t], 0, 0, 1, 2, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        ALLEN_CUDNN_CHECK(cudnnSetConvolutionMathType(state->ct_conv[t], CUDNN_TENSOR_OP_MATH));
        cudnnTensorDescriptor_t dy, dx;
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy));
        ALLEN_CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(dy, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT, 1, w_in[t]));
        ALLEN_CUDNN_CHECK(cudnnSetTensor4dDescriptor(dx, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, N_FEAT, 1, w_out[t]));
        constexpr int max_algos = 8;
        cudnnConvolutionBwdDataAlgoPerf_t perf[max_algos];
        int returned = 0;
        ALLEN_CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
          handle, state->ct_filter[t], dy, state->ct_conv[t], dx, max_algos, &returned, perf));
        bool found = false;
        for (int i = 0; i < returned && !found; ++i) {
          if (perf[i].status == CUDNN_STATUS_SUCCESS && perf[i].memory == 0) {
            state->ct_algo[t] = perf[i].algo;
            found = true;
          }
        }
        ALLEN_CUDNN_CHECK(cudnnDestroyTensorDescriptor(dy));
        ALLEN_CUDNN_CHECK(cudnnDestroyTensorDescriptor(dx));
        if (!found) throw StrException("pvfinder_unet: no workspace-free cuDNN algorithm for the ConvTranspose");
      }
    }
    m_state = std::move(state);
#else
    throw StrException("pvfinder_unet needs a CUDA build with cuDNN (WITH_CUDNN=ON)");
#endif
  }

  void pvfinder_unet_t::set_arguments_size(
    ArgumentReferences<Parameters> arguments,
    const RuntimeOptions&,
    const Constants&) const
  {
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    const unsigned batch_events = m_unet_batch_events.value();
    if (batch_events == 0) {
      throw StrException("pvfinder_unet: unet_batch_events must be >= 1");
    }
    const unsigned padded_events = (n_events + batch_events - 1) / batch_events * batch_events;
    const unsigned N = batch_events * N_INTERVALS;
    const unsigned n_rows = data<host_pvfinder_unet_rows_t>(arguments)[1];
    const unsigned padded_rows = (n_rows + N - 1) / N * N;
    const unsigned fp32 = m_bf16 ? 0u : 1u;
    set_size<dev_unet_x1_t>(arguments, fp32 * N * N_FEAT * W_IN);
    set_size<dev_unet_x2_t>(arguments, fp32 * N * N_FEAT * W_HALF);
    set_size<dev_unet_x3_t>(arguments, fp32 * N * N_FEAT * W_IN);
    set_size<dev_unet_up1_t>(arguments, fp32 * N * N_FEAT * W_HALF);
    set_size<dev_unet_up2_t>(arguments, fp32 * N * N_FEAT * W_IN);
    set_size<dev_unet_kde_rows_t>(arguments, fp32 * padded_rows * W_IN);
    set_size<dev_pvfinder_kde_output_t>(arguments, padded_events * N_INTERVALS * W_IN);
  }

#ifdef ALLEN_CUDNN_BACKEND_CUDA
  // One float32 batch: N rows of features [N][N_BATCH_CHANNELS][W_IN] -> their
  // KDE [N][W_IN]. scratch: x1, x2, x3, up1, up2 (see Parameters).
  void pvfinder_unet_t::run_fp32_batch(
    const float* rows,
    float* kde,
    float* const scratch[5],
    cudnnHandle_t handle,
    const Allen::Context& context) const
  {
    const UNetState& s = *m_state;
    const int N = (int) m_unet_batch_events.value() * N_INTERVALS;
    const dim3 block = m_block_dim;
    float *x1 = scratch[0], *x2 = scratch[1], *x3 = scratch[2], *up1 = scratch[3], *up2 = scratch[4];
    float* oint = x1;   // x1 is last read by rcbn2, long before oint is written
    float* logits = x3; // x3 is consumed by up1's ConvTranspose before it is reused
    const auto grid = [&block](int total) { return dim3(((unsigned) total + block.x - 1) / block.x); };
    // Convolution, then bias (+ ReLU): BN is folded into the CBR weights.
    const auto cbr = [&](int l, const float* in, float* out, int w) {
      s.conv[l].forward(handle, 1.f, 0.f, in, s.w_f[l], out);
      global_function(bias_relu_kernel)(grid(N * N_FEAT * w), block, context)(out, s.b_f[l], N_FEAT, w, N * N_FEAT * w);
    };
    const auto maxpool = [&](const float* in, float* out, int w_in) {
      global_function(maxpool1d_2_kernel)(grid(N * N_FEAT * (w_in / 2)), block, context)(in, out, N, N_FEAT, w_in);
    };
    const ConvTransposeTensorDescs& td = get_thread_local_conv_transpose_descs(m_state.get(), N);
    const auto conv_transpose = [&](int t, const float* in, float* out, const float* w, const float* b, int w_out) {
      const float alpha = 1.f, beta = 0.f;
      ALLEN_CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        s.ct_filter[t],
        w,
        t == 0 ? td.td_up1_in : td.td_up2_in,
        in,
        s.ct_conv[t],
        s.ct_algo[t],
        nullptr,
        0,
        &beta,
        t == 0 ? td.td_up1_out : td.td_up2_out,
        out));
      global_function(bias_add_kernel)(grid(N * N_FEAT * w_out), block, context)(
        out, b, N_FEAT, w_out, N * N_FEAT * w_out);
    };
    const WeightBlob& wb = s.wb;

    // Encoder
    cbr(0, rows, x1, W_IN);
    cbr(1, x1, up2, W_IN);
    maxpool(up2, x2, W_IN);
    cbr(2, x2, up2, W_HALF);
    maxpool(up2, x3, W_HALF);
    // Decoder
    conv_transpose(0, x3, up2, wb.w_up1t_w, wb.w_up1t_b, W_HALF);
    cbr(3, up2, up1, W_HALF);
    conv_transpose(1, up1, logits, wb.w_up2t_w, wb.w_up2t_b, W_IN);
    cbr(4, logits, up2, W_IN);
    // Output: out_intermediate, outc, softplus, scale
    s.oint.forward(handle, 1.f, 0.f, up2, wb.w_oint_w, logits);
    global_function(bias_add_kernel)(grid(N * N_FEAT * W_IN), block, context)(
      logits, wb.w_oint_b, N_FEAT, W_IN, N * N_FEAT * W_IN);
    s.outc.forward(handle, 1.f, 0.f, logits, wb.w_outc_w, oint);
    global_function(bias_add_kernel)(grid(N * W_IN), block, context)(oint, wb.w_outc_b, 1, W_IN, N * W_IN);
    global_function(softplus_scale_kernel)(grid(N * W_IN), block, context)(oint, KDE_SCALE, N * W_IN);
    global_function(squeeze_copy_kernel)(grid(N * W_IN), block, context)(oint, kde, N * W_IN);
  }
#endif

  void pvfinder_unet_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
  {
#ifdef ALLEN_CUDNN_BACKEND_CUDA
    UNetState& state = *m_state;
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    const unsigned* unet_rows = data<host_pvfinder_unet_rows_t>(arguments);
    if ((unet_rows[2] == 1u) != m_bf16) {
      throw StrException("pvfinder_unet: precision differs from pvfinder_fc_aggregation's; set both alike");
    }
    const unsigned n_rows = unet_rows[1];
    const int n_slots = (int) (n_events * N_INTERVALS);
    const int N = (int) m_unet_batch_events.value() * N_INTERVALS;
    const unsigned padded_rows = (n_rows + N - 1) / N * N;
    // The FC pads the rows to its own unet_batch_events; if the two disagree
    // the last batch would read past that buffer.
    if (!m_bf16 && size<dev_pvfinder_interval_features_t>(arguments) < (size_t) padded_rows * N_BATCH_CHANNELS * W_IN) {
      throw StrException("pvfinder_unet: the rows are not padded to unet_batch_events; "
                         "set pvfinder_fc_aggregation.unet_batch_events to the same value");
    }
    const float* rows = data<dev_pvfinder_interval_features_t>(arguments);
    float* kde = data<dev_pvfinder_kde_output_t>(arguments);
    cudnnHandle_t handle = m_bf16 ? nullptr : Allen::CuDNN::get_thread_local_handle(context.stream());
    float* const scratch[5] = {
      data<dev_unet_x1_t>(arguments),
      data<dev_unet_x2_t>(arguments),
      data<dev_unet_x3_t>(arguments),
      data<dev_unet_up1_t>(arguments),
      data<dev_unet_up2_t>(arguments)};

    // The UNet's response to an all-zero interval, through the configured
    // path, so intervals without tracks get bit for bit what running them
    // would give. Once per instance; other threads wait until it is there.
    std::call_once(state.empty_response_flag, [&]() {
      float* zeros = nullptr;
      float* out = nullptr;
      Allen::malloc((void**) &zeros, (size_t) N * N_BATCH_CHANNELS * W_IN * sizeof(float));
      Allen::malloc((void**) &out, (size_t) N * W_IN * sizeof(float));
      Allen::memset_async(zeros, 0, (size_t) N * N_BATCH_CHANNELS * W_IN * sizeof(float), context);
      if (m_bf16) {
        // All-zero bits are BF16 zeros too.
        global_function(fused::fused_unet_bf16_kernel)(
          dim3(std::min(state.fused_grid, (N + fused::WARPS - 1) / fused::WARPS)),
          dim3(fused::THREADS),
          context,
          fused::SMEM_BYTES)(
          reinterpret_cast<const __nv_bfloat16*>(zeros),
          state.fused_blob,
          out,
          KDE_SCALE,
          N,
          nullptr,
          nullptr,
          nullptr,
          0);
      }
      else {
        run_fp32_batch(zeros, out, scratch, handle, context);
      }
      Allen::synchronize(context);
      Allen::free(zeros);
      state.empty_response = out; // row 0; kept for the process lifetime, like the weights
    });

    if (m_bf16) {
      // Every row in one launch, each row's KDE straight to its (event,
      // interval), the empty-interval response to the others.
      if (n_slots > 0) {
        // See m_fused_grid_fraction: leave room for other streams' kernels.
        const int grid = std::max(1, (int) (state.fused_grid * m_fused_grid_fraction.value()));
        const int blocks = std::max(1, std::min(grid, (int) (n_rows + fused::WARPS - 1) / fused::WARPS));
        global_function(fused::fused_unet_bf16_kernel)(
          dim3(std::max(blocks, std::min(grid, n_slots / fused::THREADS + 1))),
          dim3(fused::THREADS),
          context,
          fused::SMEM_BYTES)(
          reinterpret_cast<const __nv_bfloat16*>(rows),
          state.fused_blob,
          kde,
          KDE_SCALE,
          (int) n_rows,
          data<dev_pvfinder_row_slot_t>(arguments),
          data<dev_pvfinder_slot_row_t>(arguments),
          state.empty_response,
          n_slots);
      }
    }
    else {
      float* kde_rows = data<dev_unet_kde_rows_t>(arguments);
      constexpr unsigned row_floats = N_BATCH_CHANNELS * W_IN;
      for (unsigned row = 0; row < n_rows; row += N) {
        run_fp32_batch(rows + (size_t) row * row_floats, kde_rows + (size_t) row * W_IN, scratch, handle, context);
      }
      const unsigned threads = n_slots * (W_IN / 4);
      if (threads > 0) {
        global_function(expand_kde_rows_kernel)(dim3((threads + 255) / 256), dim3(256), context)(
          kde_rows, data<dev_pvfinder_slot_row_t>(arguments), state.empty_response, kde, n_slots);
      }
    }

    if (!m_dump_dir.value().empty() && !m_dump_done) {
      dump(arguments, context);
      m_dump_done = true;
    }
#else
    // Not reached: init() refuses to run without cuDNN.
    static_cast<void>(arguments);
    static_cast<void>(context);
#endif
  }

#ifdef ALLEN_CUDNN_BACKEND_CUDA
  // Validation dump (first call): the UNet's input in the dense
  // [event][interval] float32 [channel][bin] layout (intervals without tracks as
  // zeros) and its KDE. Each file: uint32 magic 0xAB1E, uint32 n_events, floats.
  void pvfinder_unet_t::dump(const ArgumentReferences<Parameters>& arguments, const Allen::Context& context) const
  {
    const std::string& dump_dir = m_dump_dir.value();
    const unsigned n_events = first<host_number_of_events_t>(arguments);
    const unsigned n_rows = data<host_pvfinder_unet_rows_t>(arguments)[1];
    constexpr size_t row_floats = N_BATCH_CHANNELS * W_IN;
    const auto features = make_host_buffer<dev_pvfinder_interval_features_t>(arguments, context);
    const auto slot_row = make_host_buffer<dev_pvfinder_slot_row_t>(arguments, context);
    const auto kde = make_host_buffer<dev_pvfinder_kde_output_t>(arguments, context);
    std::vector<float> input((size_t) n_events * N_INTERVALS * row_floats, 0.0f);
    const auto* bf16 = reinterpret_cast<const uint16_t*>(features.data());
    for (size_t slot = 0; slot < (size_t) n_events * N_INTERVALS; ++slot) {
      const int row = slot_row[slot];
      if (row < 0 || row >= (int) n_rows) continue;
      for (size_t e = 0; e < row_floats; ++e) { // e = channel * W_IN + bin
        if (m_bf16) {
          // channels last; a bfloat16 is the upper half of the float with its value
          const uint32_t bits = (uint32_t) bf16[row * row_floats + (e % W_IN) * N_BATCH_CHANNELS + e / W_IN] << 16;
          std::memcpy(&input[slot * row_floats + e], &bits, sizeof(float));
        }
        else {
          input[slot * row_floats + e] = features[row * row_floats + e];
        }
      }
    }
    const uint32_t magic = 0xAB1EU;
    auto write = [&](const std::string& name, const float* d, size_t n) {
      std::ofstream f(dump_dir + "/" + name, std::ios::binary);
      f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      f.write(reinterpret_cast<const char*>(&n_events), sizeof(n_events));
      f.write(reinterpret_cast<const char*>(d), n * sizeof(float));
      if (!f) throw StrException("pvfinder_unet: cannot write " + dump_dir + "/" + name);
    };
    write("allen_ncw_input.bin", input.data(), input.size());
    write("allen_kde_output.bin", kde.data(), (size_t) n_events * N_INTERVALS * W_IN);
    info_cout << "[pvfinder_unet] Validation dump written to " << dump_dir << " (" << n_events << " events)\n";
  }
#endif

} // namespace pvfinder_unet
