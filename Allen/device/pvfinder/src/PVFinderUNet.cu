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
#ifdef ALLEN_WITH_CUDNN
#include "PVFinderUNetKernels.cuh"
#include "PVFinderUNetFused.cuh"
#include <cuda_bf16.h>
#endif

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

INSTANTIATE_ALGORITHM(pvfinder_unet::pvfinder_unet_t)

namespace pvfinder_unet {

#ifdef ALLEN_WITH_CUDNN
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

    // float32: the layers on the cuDNN graph API, for batches of N rows
    // (exact float32, deterministic engines), and the largest workspace.
    Allen::CuDNN::ConvolutionLayer cbr[5]; // Conv + bias + ReLU, as w_f
    Allen::CuDNN::PoolingLayer pool[2];    // MaxPool(2): W_IN -> W_HALF -> W_QTR
    Allen::CuDNN::ConvolutionLayer up[2];  // ConvTranspose(k 2, stride 2) + bias
    Allen::CuDNN::ConvolutionLayer oint;   // out_intermediate: Conv + bias
    Allen::CuDNN::ConvolutionLayer outc;   // outc: Conv + bias, softplus, * KDE_SCALE
    size_t workspace_bytes = 0;

    // bfloat16: the fused kernel's image of all weights, and its full grid.
    unsigned char* fused_blob = nullptr;
    int fused_grid = 0;

    // The UNet's output for an all-zero interval ([W_IN] floats, device),
    // written to every interval without tracks. Computed in init(), by the
    // configured path.
    float* empty_response = nullptr;

    // The stream init() runs its kernels on, kept with its cuDNN handle.
    Allen::Context setup;
  };

  namespace {
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

#endif // ALLEN_WITH_CUDNN

  // ---------------------------------------------------------------------------
  // init(): weights, BatchNorm folding, and the configured precision's path.
  // ---------------------------------------------------------------------------
  void pvfinder_unet_t::init()
  {
#ifdef ALLEN_WITH_CUDNN
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
    // A stream of its own for init()'s kernels.
    Allen::Context& setup = state->setup;
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
      // The layers for batches of N rows, NCW (NCHW with H = 1).
      cudnnHandle_t handle = Allen::CuDNN::handle(nullptr);
      const int64_t N = (int64_t) m_unet_batch_events.value() * N_INTERVALS;
      using Allen::CuDNN::Activation;
      for (int l = 0; l < 5; ++l) {
        const CBRShape& c = cbr_shapes[l];
        state->cbr[l].create(
          handle,
          {.batch = N,
           .in_channels = c.c_in,
           .out_channels = N_FEAT,
           .input_size = {c.w},
           .kernel_size = {c.r},
           .padding = {c.pad},
           .bias = true,
           .activation = Activation::Relu});
      }
      const int64_t pool_in[2] = {W_IN, W_HALF};
      for (int p = 0; p < 2; ++p) {
        state->pool[p].create(handle, {N, N_FEAT, 1, pool_in[p]}, {Allen::CuDNN::PoolingMode::Max, {2}, {2}, {0}});
      }
      const int64_t up_in[2] = {W_QTR, W_HALF};
      for (int t = 0; t < 2; ++t) {
        state->up[t].create(
          handle,
          {.batch = N,
           .in_channels = N_FEAT,
           .out_channels = N_FEAT,
           .input_size = {up_in[t]},
           .kernel_size = {2},
           .stride = {2},
           .transposed = true,
           .bias = true});
      }
      state->oint.create(
        handle,
        {.batch = N,
         .in_channels = N_FEAT,
         .out_channels = N_FEAT,
         .input_size = {W_IN},
         .kernel_size = {5},
         .padding = {2},
         .bias = true});
      state->outc.create(
        handle,
        {.batch = N,
         .in_channels = N_FEAT,
         .out_channels = 1,
         .input_size = {W_IN},
         .kernel_size = {5},
         .padding = {2},
         .bias = true,
         .activation = Activation::Softplus,
         .output_scale = KDE_SCALE});
      for (const Allen::CuDNN::ConvolutionLayer* layer :
           {&state->cbr[0],
            &state->cbr[1],
            &state->cbr[2],
            &state->cbr[3],
            &state->cbr[4],
            &state->up[0],
            &state->up[1],
            &state->oint,
            &state->outc}) {
        state->workspace_bytes = std::max(state->workspace_bytes, layer->workspace_size());
        debug_cout << "[pvfinder_unet] " << layer->describe() << "\n";
      }
      for (const auto& p : state->pool)
        state->workspace_bytes = std::max(state->workspace_bytes, p.workspace_size());
    }
    m_state = std::move(state);
    compute_empty_response();
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
#ifdef ALLEN_WITH_CUDNN
    const size_t workspace_bytes = m_bf16 ? 0 : m_state->workspace_bytes;
#else
    const size_t workspace_bytes = 0;
#endif
    set_size<dev_unet_workspace_t>(arguments, (unsigned) ((workspace_bytes + 3) / 4));
    set_size<dev_pvfinder_kde_output_t>(arguments, padded_events * N_INTERVALS * W_IN);
  }

#ifdef ALLEN_WITH_CUDNN
  // One float32 batch: N rows of features [N][N_BATCH_CHANNELS][W_IN] -> their
  // KDE [N][W_IN]. scratch: x1, x2, x3, up1, up2 (see Parameters).
  void pvfinder_unet_t::run_fp32_batch(const float* rows, float* kde, float* const scratch[6], cudnnHandle_t handle)
    const
  {
    const UNetState& s = *m_state;
    const WeightBlob& wb = s.wb;
    float *x1 = scratch[0], *x2 = scratch[1], *x3 = scratch[2], *up1 = scratch[3], *up2 = scratch[4];
    void* workspace = scratch[5];
    // Encoder (BN folded into the CBR weights)
    s.cbr[0].forward(handle, rows, s.w_f[0], s.b_f[0], x1, workspace);
    s.cbr[1].forward(handle, x1, s.w_f[1], s.b_f[1], up2, workspace);
    s.pool[0].forward(handle, up2, x2, workspace);
    s.cbr[2].forward(handle, x2, s.w_f[2], s.b_f[2], up2, workspace);
    s.pool[1].forward(handle, up2, x3, workspace);
    // Decoder
    s.up[0].forward(handle, x3, wb.w_up1t_w, wb.w_up1t_b, up2, workspace);
    s.cbr[3].forward(handle, up2, s.w_f[3], s.b_f[3], up1, workspace);
    s.up[1].forward(handle, up1, wb.w_up2t_w, wb.w_up2t_b, x3, workspace);
    s.cbr[4].forward(handle, x3, s.w_f[4], s.b_f[4], up2, workspace);
    // Output: out_intermediate, then outc with softplus and the scale, straight
    // to the KDE rows ([N][1][W_IN] is [N][W_IN])
    s.oint.forward(handle, up2, wb.w_oint_w, wb.w_oint_b, x1, workspace);
    s.outc.forward(handle, x1, wb.w_outc_w, wb.w_outc_b, kde, workspace);
  }

  // The UNet's response to an all-zero interval, through the configured path
  // and batch size, so intervals without tracks get bit for bit what running
  // them would give. One batch of zero rows on init()'s stream; row 0 is kept.
  void pvfinder_unet_t::compute_empty_response()
  {
    UNetState& state = *m_state;
    const Allen::Context& setup = state.setup;
    const int N = (int) m_unet_batch_events.value() * N_INTERVALS;
    const size_t input_floats = (size_t) N * N_BATCH_CHANNELS * W_IN;
    float* zeros = nullptr;
    float* out = nullptr;
    Allen::malloc((void**) &zeros, input_floats * sizeof(float));
    Allen::malloc((void**) &out, (size_t) N * W_IN * sizeof(float));
    Allen::memset_async(zeros, 0, input_floats * sizeof(float), setup);
    std::vector<float*> buffers;
    if (m_bf16) {
      // All-zero bits are BF16 zeros too.
      global_function(fused::fused_unet_bf16_kernel)(
        dim3(std::min(state.fused_grid, (N + fused::WARPS - 1) / fused::WARPS)),
        dim3(fused::THREADS),
        setup,
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
      // Scratch as in operator(): x1, x2, x3, up1, up2, workspace.
      const size_t sizes[6] = {
        (size_t) N * N_FEAT * W_IN,
        (size_t) N * N_FEAT * W_HALF,
        (size_t) N * N_FEAT * W_IN,
        (size_t) N * N_FEAT * W_HALF,
        (size_t) N * N_FEAT * W_IN,
        (state.workspace_bytes + 3) / 4};
      float* scratch[6];
      for (int i = 0; i < 6; ++i) {
        Allen::malloc((void**) &scratch[i], std::max<size_t>(sizes[i], 1) * sizeof(float));
        buffers.push_back(scratch[i]);
      }
      run_fp32_batch(zeros, out, scratch, Allen::CuDNN::handle(setup.stream()));
    }
    Allen::synchronize(setup);
    Allen::free(zeros);
    for (float* b : buffers)
      Allen::free(b);
    state.empty_response = out; // row 0; kept for the process lifetime, like the weights
  }
#endif

  void pvfinder_unet_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions&,
    const Constants&,
    const Allen::Context& context) const
  {
#ifdef ALLEN_WITH_CUDNN
    const UNetState& state = *m_state;
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
    cudnnHandle_t handle = m_bf16 ? nullptr : Allen::CuDNN::handle(context);
    float* const scratch[6] = {
      data<dev_unet_x1_t>(arguments),
      data<dev_unet_x2_t>(arguments),
      data<dev_unet_x3_t>(arguments),
      data<dev_unet_up1_t>(arguments),
      data<dev_unet_up2_t>(arguments),
      data<dev_unet_workspace_t>(arguments)};

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
        run_fp32_batch(rows + (size_t) row * row_floats, kde_rows + (size_t) row * W_IN, scratch, handle);
      }
      const unsigned threads = n_slots * (W_IN / 4);
      if (threads > 0) {
        global_function(expand_kde_rows_kernel)(dim3((threads + 255) / 256), dim3(256), context)(
          kde_rows, data<dev_pvfinder_slot_row_t>(arguments), state.empty_response, kde, n_slots);
      }
    }

    // One thread dumps (the first to get here); run single-stream (-t 1) to
    // dump the same slice as the other PVFinder algorithms.
    if (!m_dump_dir.value().empty() && !m_dump_done.exchange(true)) {
      dump(arguments, context);
    }
#else
    // Not reached: init() refuses to run without cuDNN.
    static_cast<void>(arguments);
    static_cast<void>(context);
#endif
  }

#ifdef ALLEN_WITH_CUDNN
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
