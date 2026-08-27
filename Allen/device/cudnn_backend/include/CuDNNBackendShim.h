#pragma once

// This header provides compile-time backend selection.
// Currently: cuDNN (CUDA), MIOpen stub (HIP), no-op (CPU).
// To add MIOpen: implement the HIP branch.

#if defined(ALLEN_WITH_CUDNN) && defined(TARGET_DEVICE_CUDA)
#  define ALLEN_CUDNN_BACKEND_CUDA 1
#elif defined(TARGET_DEVICE_HIP)
#  define ALLEN_CUDNN_BACKEND_HIP_STUB 1  // placeholder: replace with MIOpen
#else
#  define ALLEN_CUDNN_BACKEND_NONE 1
#endif

namespace Allen::CuDNN {
  // Tag types for compile-time dispatch
  struct CudaBackendTag {};
  struct HipBackendTag {};
  struct NullBackendTag {};

#if defined(ALLEN_CUDNN_BACKEND_CUDA)
  using ActiveBackendTag = CudaBackendTag;
  constexpr bool backend_available = true;
#elif defined(ALLEN_CUDNN_BACKEND_HIP_STUB)
  using ActiveBackendTag = HipBackendTag;
  constexpr bool backend_available = false; // flip to true when MIOpen is wired in
#else
  using ActiveBackendTag = NullBackendTag;
  constexpr bool backend_available = false;
#endif
} // namespace Allen::CuDNN
