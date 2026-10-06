/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#pragma once

// Host / device compiler identification
#if defined(TARGET_DEVICE_CPU) || (defined(TARGET_DEVICE_CUDA) && defined(__CUDACC__)) || \
  (defined(TARGET_DEVICE_HIP) && (defined(__HCC__) || defined(__HIP__)))
#define DEVICE_COMPILER
#endif

#if defined(TARGET_DEVICE_CUDA) || defined(TARGET_DEVICE_HIP)
#define TARGET_DEVICE_CUDAHIP
#endif

#include <algorithm>
#include <limits>
#include <type_traits>
#include <tuple>
#include <string>
#include <cassert>
#include <cmath>
#include <array>
#include <span>
#include "AllenTypeTraits.h"
#include "BackendCommonInterface.h"

// Dispatch to the right backend
#if defined(TARGET_DEVICE_CPU)
#include "CPUBackend.h"
#elif defined(TARGET_DEVICE_HIP)
#include "HIPBackend.h"
#elif defined(TARGET_DEVICE_CUDA)
#include "CUDABackend.h"
#endif

#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)

#if defined(__clang__) or defined(__NVCC__)
#define UNROLL(n) DO_PRAGMA(unroll)
#elif defined(__GNUC__)
#define UNROLL(n) DO_PRAGMA(GCC unroll n)
#endif

#if defined(DEVICE_COMPILER)
namespace Allen {
  namespace device {
    // Dispatcher targets
    namespace target {
      struct Default {};
      struct CPU {};
      struct HIP {};
      struct CUDA {};
    } // namespace target

    /**
     * @brief Allows to write several functions specialized per target.
     * @details Usage:
     *
     *          dispatch<target::Default, target::CPU>(fn0, fn1)(arguments...);
     *
     *          The concrete target on the executing platform is sought first. If none is
     *          available, the default one is chosen. If the default doesn't exist, a static_assert
     *          fails.
     *
     *          List of possible targets:
     *
     *          * Default
     *          * CPU
     *          * HIP
     *          * CUDA
     */
    template<typename... Ts, typename... Fns>
    __device__ constexpr auto dispatch(Fns&&... fns)
    {
      using targets_t = std::tuple<Ts...>;
#if !defined(ALWAYS_DISPATCH_TO_DEFAULT)
#if defined(TARGET_DEVICE_CPU)
      constexpr auto configured_target = index_of_v<target::CPU, targets_t>;
#elif defined(TARGET_DEVICE_CUDA)
      constexpr auto configured_target = index_of_v<target::CUDA, targets_t>;
#elif defined(TARGET_DEVICE_HIP)
      constexpr auto configured_target = index_of_v<target::HIP, targets_t>;
#endif
      if constexpr (configured_target == std::tuple_size<targets_t>::value) {
#endif
        // Dispatch to the default target, check with a static_assert its existence
        constexpr auto default_target = index_of_v<target::Default, targets_t>;
        static_assert(default_target != std::tuple_size<targets_t>::value, "target available for current platform");
        const auto fn = std::get<default_target>(std::tuple<Fns...> {fns...});
        return [fn](auto&&... args) { return fn(args...); };
#if !defined(ALWAYS_DISPATCH_TO_DEFAULT)
      }
      else {
        // Dispatch to the specific target
        const auto fn = std::get<configured_target>(std::tuple<Fns...> {fns...});
        return [fn](auto&&... args) { return fn(args...); };
      }
#endif
    }
  } // namespace device
} // namespace Allen
#else
#include <cmath>
#endif

// Replacement for span in device code when building with HIP,
// std::span works for CUDA and CPU
namespace Allen::device {
#if defined(TARGET_DEVICE_HIP)
  template<class T>
  struct span {
  private:
    T* m_ptr = nullptr;
    std::size_t m_size = 0;

  public:
    constexpr span() = default;

    constexpr __device__ __host__ span(T* ptr, std::size_t size) : m_ptr(ptr), m_size(size) {}

    template<std::size_t N>
    constexpr __device__ __host__ span(std::array<T, N>& a) : m_ptr(std::data(a)), m_size(N)
    {}

    template<std::size_t N>
    constexpr __device__ __host__ span(const std::array<std::remove_const_t<T>, N>& a) : m_ptr(std::data(a)), m_size(N)
    {}

    constexpr __device__ __host__ bool empty() const { return size() == 0; }
    constexpr __device__ __host__ T* data() const { return m_ptr; }
    constexpr __device__ __host__ size_t size() const { return m_size; }
    constexpr __device__ __host__ size_t size_bytes() const { return m_size * sizeof(T); }
    constexpr __device__ __host__ T& operator[](int i) { return m_ptr[i]; }
    constexpr __device__ __host__ const T& operator[](int i) const { return m_ptr[i]; }
    constexpr __device__ __host__ span<T> subspan(const std::size_t offset, const std::size_t count) const
    {
      if (count == 0) {
        return {m_ptr + offset, m_size - offset};
      }
      else {
        assert(offset + count <= m_size);
        return {m_ptr + offset, count};
      }
    }

    constexpr __device__ __host__ span<T> subspan(const std::size_t offset) const
    {
      return {m_ptr + offset, m_size - offset};
    }

    constexpr __device__ __host__ T* begin() const { return m_ptr; }

    constexpr __device__ __host__ T* end() const { return m_ptr + m_size; }

    constexpr __device__ __host__ T* rbegin() const { return m_ptr + m_size - 1; }

    constexpr __device__ __host__ T* rend() const { return m_ptr - 1; }
  };
#else
  using std::span;
#endif
} // namespace Allen::device

// Helper structure to deal with constness of T
template<typename T, typename U>
struct ForwardType {
  using t = U;
};

template<typename T, typename U>
struct ForwardType<const T, U> {
  using t = std::add_const_t<U>;
};

struct HashNotPopulatedException : public std::exception {
private:
  std::string m_algorithm_name;

public:
  HashNotPopulatedException(const std::string& algorithm_name) :
    m_algorithm_name("Pre or post-scaler hash not populated in selection algorithm " + algorithm_name)
  {}

  const char* what() const noexcept override { return m_algorithm_name.c_str(); }
};

__device__ inline float signselect(const float& s, const float& a, const float& b) { return (s > 0) ? a : b; }

__device__ inline auto compress_float_to_16_bits(const float f)
{
#if defined(TARGET_DEVICE_CPU)
  return __float_to_uint16(f);
#else
  return __float2half(f);
#endif
}

#define DIV_CEIL(x, y) (((x) + (y) -1) / (y))

// Numeric limits
namespace Allen {
  template<typename T>
  struct numeric_limits;

  template<>
  struct numeric_limits<float> {
    __host__ __device__ static constexpr float invalid() noexcept { return INFINITY; };
    __host__ __device__ static constexpr float infinity() noexcept { return INFINITY; };
    __host__ __device__ static constexpr float quiet_NaN() noexcept { return NAN; };
  };

  template<>
  struct numeric_limits<double> {
    __host__ __device__ static constexpr double invalid() noexcept { return static_cast<double>(INFINITY); };
    __host__ __device__ static constexpr double infinity() noexcept { return static_cast<double>(INFINITY); };
    __host__ __device__ static constexpr double quiet_NaN() noexcept { return static_cast<double>(NAN); };
  };

  template<>
  struct numeric_limits<ushort> {
    __host__ __device__ static constexpr ushort invalid() noexcept { return USHRT_MAX; };
  };

  [[noreturn]] __host__ __device__ __forceinline__ void unreachable()
  {
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
    __assume(false);
#else // GCC, Clang
    __builtin_unreachable();
#endif
  }
} // namespace Allen

/*
  Operators for float2 and float3
  originally implemented by A. Kozlinskiy in the context of Mu3e
 */

// float2
__host__ __device__ inline float float2_dot(const float2& l, const float2& r) { return l.x * r.x + l.y * r.y; }

__host__ __device__ inline float mag2(const float2& v) { return v.x * v.x + v.y * v.y; }

__host__ __device__ inline float mag(const float2& v) { return sqrtf(v.x * v.x + v.y * v.y); }

__host__ __device__ inline float2 operator+(const float2& l, const float2& r) { return {l.x + r.x, l.y + r.y}; }

__host__ __device__ inline float2 operator-(const float2& l, const float2& r) { return {l.x - r.x, l.y - r.y}; }

__host__ __device__ inline float2 operator*(const float2& l, const float r) { return {l.x * r, l.y * r}; }

__host__ __device__ inline float2 operator/(const float2& l, const float r) { return {l.x / r, l.y / r}; }

__host__ __device__ inline float2& operator+=(float2& l, const float2& r)
{
  l.x += r.x;
  l.y += r.y;
  return l;
}

__host__ __device__ inline float2& operator-=(float2& l, const float2& r)
{
  l.x -= r.x;
  l.y -= r.y;
  return l;
}

// float3

__host__ __device__ inline float float3_dot(const float3& l, const float3& r)
{
  return ((l.x * r.x) + (l.y * r.y) + (l.z * r.z));
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b)
{
  return {(a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z), (a.x * b.y) - (a.y * b.x)};
}

__host__ __device__ inline float mag2(const float3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

__host__ __device__ inline float mag(const float3& v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }

__host__ __device__ inline float3 operator+(const float3& l, const float3& r)
{
  return {l.x + r.x, l.y + r.y, l.z + r.z};
}

__host__ __device__ inline float3 operator-(const float3& l, const float3& r)
{
  return {l.x - r.x, l.y - r.y, l.z - r.z};
}

__host__ __device__ inline float3 operator*(const float3& l, const float r) { return {l.x * r, l.y * r, l.z * r}; }

__host__ __device__ inline float3 operator/(const float3& l, const float r) { return {l.x / r, l.y / r, l.z / r}; }

__host__ __device__ inline float3& operator+=(float3& l, const float3& r)
{
  l.x += r.x;
  l.y += r.y;
  l.z += r.z;
  return l;
}

__host__ __device__ inline float3& operator-=(float3& l, const float3& r)
{
  l.x -= r.x;
  l.y -= r.y;
  l.z -= r.z;
  return l;
}

#ifdef ALLEN_STANDALONE
// Floating point comparisons
/**
 * This two functions are only required for the Allen standalone build and should be removed with it.
 * They are already implemented in another stack project
 * (Detector/Core/include/Core/FloatComparison.h)
 */

namespace LHCb {

  /**
   * Generic method to compare two floating point values
   * ADL is used to use standard library by default but switch to custom
   * methods for abs and max in case of custom types, e.g. vector types of SIMDWrapper
   * Note that is should not be used to compare to 0, for this essentiallyZero
   * should be used
   *
   * CAVEAT: please note that `essentiallyEqual` is not transitive, i.e. even if
   *         `essentiallyEqual(a,b)` and `essentiallyEqual(b,c)` evaluate to `true`
   *         that does _not_ guarantee that `essentiallyEqual(a,c)` will evaluate to `true`.
   *         Hence do _NOT_ rely on `essentiallyEqual` in a comparison used (implicitly)
   *         for sorting -- when sorting, write comparisons which avoid checking for
   *         equality in the first place.
   */
  template<typename T, typename = std::enable_if_t<std::numeric_limits<T>::is_specialized>>
  constexpr auto essentiallyEqual(T const a, T const b)
  {
    using std::abs, std::max;
    if constexpr (std::numeric_limits<T>::is_exact) {
      return a == b; // TODO: should we perhaps warn that in this case, `essentiallyEqual should not be used?
    }
    else {
      return abs(a - b) <= max(abs(a), abs(b)) * std::numeric_limits<T>::epsilon();
    }
  }

  /**
   * Generic method to compare a floating point value to 0
   * ADL is used to use standard library by default but switch to custom
   * methods for abs in case of custom types, e.g. vector types of SIMDWrapper
   */
  template<typename T, typename = std::enable_if_t<std::numeric_limits<T>::is_specialized>>
  constexpr auto essentiallyZero(T const a)
  {
    using std::abs;
    if constexpr (std::numeric_limits<T>::is_exact) {
      return a == 0; // TODO: should we perhaps warn to not use essentiallyZero in this case?
    }
    else {
      return abs(a) <= std::numeric_limits<T>::min();
    }
  }

} // namespace LHCb
#else
// Use Detector definitions
#include "Core/FloatComparison.h"
#endif
