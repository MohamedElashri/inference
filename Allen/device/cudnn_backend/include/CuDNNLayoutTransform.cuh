#pragma once
#include "AlgorithmTypes.cuh"

namespace Allen::CuDNN {

  template<typename SrcT, typename ExtractFn>
  __global__ void soa_to_nchw_kernel(
    const SrcT* __restrict__ src,
    float*      __restrict__ dst,
    unsigned n_elements,
    unsigned n_channels,
    ExtractFn extract)
  {
    const unsigned elem    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned channel = blockIdx.y;
    if (elem >= n_elements || channel >= n_channels) return;

    dst[channel * n_elements + elem] = extract(src[elem], static_cast<int>(channel));
  }

} // namespace Allen::CuDNN

#define DEFINE_ALLEN_SOA_TO_NCHW(STEM, INPUT_ARG_TYPE, SRC_ELEM_TYPE, N_CHANNELS, EXTRACT_FN) \
namespace STEM {                                                                                \
  struct Parameters {                                                                           \
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;                       \
    HOST_INPUT(host_n_elements_t, unsigned) host_n_elements;                                    \
    DEVICE_INPUT(dev_nchw_input_src_t, SRC_ELEM_TYPE) dev_nchw_input_src;                      \
    DEVICE_OUTPUT(dev_nchw_tensor_t, float) dev_nchw_tensor;                                   \
  };                                                                                            \
                                                                                                \
  struct STEM##_t : public DeviceAlgorithm, Parameters {                                       \
    void set_arguments_size(                                                                    \
      ArgumentReferences<Parameters> arguments,                                                 \
      const RuntimeOptions&,                                                                    \
      const Constants&) const                                                                   \
    {                                                                                           \
      const unsigned n = first<host_n_elements_t>(arguments);                                  \
      set_size<dev_nchw_tensor_t>(arguments, n * (N_CHANNELS));                                \
    }                                                                                           \
                                                                                                \
    void operator()(                                                                            \
      const ArgumentReferences<Parameters>& arguments,                                          \
      const RuntimeOptions&,                                                                    \
      const Constants&,                                                                         \
      const Allen::Context& context) const                                                      \
    {                                                                                           \
      const unsigned n    = first<host_n_elements_t>(arguments);                                \
      const unsigned nc   = (N_CHANNELS);                                                       \
      const dim3 grid(                                                                          \
        (n + m_block_dim.value().x - 1) / m_block_dim.value().x, nc);                          \
      const dim3 block(m_block_dim.value().x, 1);                                               \
      global_function(Allen::CuDNN::soa_to_nchw_kernel<SRC_ELEM_TYPE, decltype(EXTRACT_FN)>)  \
        (grid, block, context)(                                                                  \
          data<dev_nchw_input_src_t>(arguments),                                                \
          data<dev_nchw_tensor_t>(arguments),                                                   \
          n, nc, (EXTRACT_FN));                                                                  \
    }                                                                                           \
                                                                                                \
  private:                                                                                      \
    Allen::Property<dim3> m_block_dim {this, "block_dim", {128, 1, 1},                         \
      "block dimensions for SoA-to-NCHW transform kernel"};                                    \
  };                                                                                            \
} // namespace STEM
