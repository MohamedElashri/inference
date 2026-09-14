#include "CuDNNFusedOps.h"

#ifdef ALLEN_CUDNN_BACKEND_CUDA

namespace Allen::CuDNN::detail {
  namespace {
    __global__ void fused_conv_post_ops_float_kernel(
      float* output,
      const float* bias,
      const size_t elements,
      const int channels,
      const int spatial_size,
      const bool add_bias,
      const ActivationMode activation)
    {
      const auto index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (index >= elements) return;

      float value = output[index];
      if (add_bias) {
        const int channel = static_cast<int>((index / spatial_size) % channels);
        value += bias[channel];
      }
      if (activation == ActivationMode::Relu) {
        value = value > 0.f ? value : 0.f;
      }
      output[index] = value;
    }
  } // namespace

  void launch_fused_conv_post_ops_float(
    float* output,
    const float* bias,
    TensorShape shape,
    bool add_bias,
    ActivationMode activation,
    cudaStream_t stream)
  {
    if (!add_bias && activation == ActivationMode::Identity) return;
    if (activation != ActivationMode::Identity && activation != ActivationMode::Relu) {
      throw std::invalid_argument("AllenCuDNN: FusedConvPlan post-op kernel supports only identity and ReLU");
    }

    const size_t elements = shape.elements();
    const int spatial_size = shape.h * shape.w;
    constexpr int threads = 256;
    const auto blocks = static_cast<unsigned>((elements + threads - 1) / threads);
    fused_conv_post_ops_float_kernel<<<blocks, threads, 0, stream>>>(
      output,
      bias,
      elements,
      shape.c,
      spatial_size,
      add_bias,
      activation);
  }
} // namespace Allen::CuDNN::detail

#endif
