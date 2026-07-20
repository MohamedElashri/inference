/*****************************************************************************\
 * (c) Copyright 2023 CERN for the benefit of the LHCb Collaboration           *
 *                                                                             *
 * This software is distributed under the terms of the Apache License          *
 * version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
 \*****************************************************************************/
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <ostream>
#include <BackendCommon.h>
#include <MassDefinitions.h>
#include <RichTypes.cuh>

namespace Allen::Rich {
  using FP = float;
  using Point = float3;
  using Vector = float3;
  using DataType = std::uint32_t;
  using Transform3D = std::array<float, 12>;
  using KeyType = std::uint64_t;
  using BitPackType = KeyType;
  using ADCTimeType = std::uint16_t;
  using Plane = std::array<float, 4>; // Plane represented as Ax + By + Cz + D = 0

  __host__ __device__ inline bool isFinite(const Point& point)
  {
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
  }

  //// Constants related to pixel reconstruction, value definition based on those in HLT2's
  /// Rec/Rich/RichFutureRecPixelAlgorithms/src/RichSIMDSummaryPixels.cpp

  /// Enabled 4D reconstruction
  [[maybe_unused]] __device__ constexpr DetectorArray<bool> m_enable4D = {false, false};

  /// Average expected hit time for signal in each RICH in ns
  [[maybe_unused]] __device__ constexpr DetectorArray<float> m_avHitTime = {13.03, 52.94};

  /// Course (pixel) Time window for each RICH in ns
  [[maybe_unused]] __device__ constexpr DetectorArray<float> m_timeWindow = {3.0, 3.0};

  /// Enable the override of inner and out regions
  [[maybe_unused]] __device__ constexpr DetectorArray<bool> m_overrideRegions = {false, false};

  /// Size in X defining the inner pixels for each RICH
  [[maybe_unused]] __device__ constexpr DetectorArray<double> m_innerPixX = {250.0, 99999.9};

  /// Size in Y defining the inner pixels for each RICH
  [[maybe_unused]] __device__ constexpr DetectorArray<double> m_innerPixY = {300.0, 300.0};

  /// Time resolution for inner regions in ns
  [[maybe_unused]] __device__ constexpr DetectorArray<float> m_innerTimeWindow = {0.15, 0.15};

  /// Time resolution for outer regions in ns
  [[maybe_unused]] __device__ constexpr DetectorArray<float> m_outerTimeWindow = {0.3, 0.3};

  /// Matrix-point transform3D, adapted from ROOT::Math
  __host__ __device__ inline Point transform3DTimesPoint(Transform3D fM, Point point)
  {
    return Point {
      fM[0] * point.x + fM[1] * point.y + fM[2] * point.z + fM[3],
      fM[4] * point.x + fM[5] * point.y + fM[6] * point.z + fM[7],
      fM[8] * point.x + fM[9] * point.y + fM[10] * point.z + fM[11]};
  }

  // Distance plane-point taken from https://root.cern/root/html516/src/ROOT__Math__Plane3D.cxx.html#hsnfzB
  __host__ __device__ inline float distance(const Plane& plane, const Point& point)
  {
    return plane[0] * point.x + plane[1] * point.y + plane[2] * point.z + plane[3];
  }

  __device__ __host__ inline float3 findNormal(const Plane& plane) { return float3 {plane[0], plane[1], plane[2]}; }

  /** Reflect a given direction off a spherical mirror. Can be used for intersection.
   *
   *  @attention This method is specifically optimised for a spherical shell from
   *             the inside out, as is required for the RICH mirrors. It will *not*
   *             give the correct results if used in the other direction...
   *
   *  @param[in,out] position   The start point to use for the ray tracing.
   *                            Afterwards gives the reflection point on the
   *                            spherical mirror.
   *  @param[in,out] direction  The direction to ray trace from the start point.
   *                            Afterwards represents the reflection direction
   *                            from the spherical mirror.
   *  @param[in] CoC            The centre of curvature of the spherical mirror
   *  @param[in] radius         The radius of curvature of the spherical mirror
   *
   *  @return Boolean/mask indicating if the ray tracing was succesful
   *  @retval true  Ray tracing was successful
   *  @retval false Ray tracing was unsuccessful
   */
  __device__ __host__ inline bool
  reflectSpherical(Point& position, float3& direction, const Point& CoC, const float radius)
  {
    // for line sphere intersection look at http://www.realtimerendering.com/int/
    const float a = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
    const float dx = position.x - CoC.x;
    const float dy = position.y - CoC.y;
    const float dz = position.z - CoC.z;
    const float b = 2.f * (direction.x * dx + direction.y * dy + direction.z * dz);
    const float c = (dx * dx + dy * dy + dz * dz) - (radius * radius);
    float discr = (b * b) - (4.f * a * c);

    if (discr <= 0.f) return false;

    // compute the distance
    const auto dist = 0.5f * (sqrtf(discr) - b) / a;

    // change position to the intersection point
    position.x += dist * direction.x;
    position.y += dist * direction.y;
    position.z += dist * direction.z;

    // reflect the vector
    // r = u - 2(u.n)n, r=reflection, u=incident, n=normal
    const float nx = position.x - CoC.x;
    const float ny = position.y - CoC.y;
    const float nz = position.z - CoC.z;
    const float scale = 2.f * (nx * direction.x + ny * direction.y + nz * direction.z) / (nx * nx + ny * ny + nz * nz);
    direction.x -= scale * nx;
    direction.y -= scale * ny;
    direction.z -= scale * nz;
    return true;
  }

  __device__ __host__ inline Point
  intersectSpherical(const Point& position, const float3& direction, const Point& CoC, const float radius)
  {
    // for line sphere intersection look at http://www.realtimerendering.com/int/
    const float a = direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
    const float dx = position.x - CoC.x;
    const float dy = position.y - CoC.y;
    const float dz = position.z - CoC.z;
    const float b = 2.f * (direction.x * dx + direction.y * dy + direction.z * dz);
    const float c = (dx * dx + dy * dy + dz * dz) - (radius * radius);
    float discr = (b * b) - (4.f * a * c);

    if (discr <= 0.f) return {NAN, NAN, NAN};

    // compute the distance
    const auto dist = 0.5f * (sqrtf(discr) - b) / a;

    Point out = position;

    // change position to the intersection point
    out.x += dist * direction.x;
    out.y += dist * direction.y;
    out.z += dist * direction.z;

    return out;
  }

  /** Intersect a given direction, from a given point, with a given plane.
   *
   *  @param[in]  position      The start point to use for the ray tracing
   *  @param[in]  direction     The direction to ray trace from the start point
   *  @param[in]  plane         The plane to intersect
   *  @param[out] intersection  The intersection point of the direction with the plane
   */
  __device__ __host__ inline Point intersectPlane(const Point& position, const float3& direction, const Plane& plane)
  {
    // compute -1*distance to the plane
    const float scalar = direction.x * plane[0] + direction.y * plane[1] + direction.z * plane[2];
    const float planeScalarDistance = distance(plane, position) / scalar;
    // compute plane intersection
    return Point {
      position.x - direction.x * planeScalarDistance,
      position.y - direction.y * planeScalarDistance,
      position.z - direction.z * planeScalarDistance};
  }

  /** Ray trace from given position in given direction off flat mirrors
   *
   *  @param[in,out] position  On input the start point.
   *                           On output the reflection point
   *  @param[in,out] direction On input the starting direction.
   *                           On output the reflected direction.
   *  @param[in]     plane     The plane to reflect off
   */
  __device__ __host__ inline void reflectPlane(Point& position, float3& direction, const Plane& plane)
  {
    // compute -1*distance to the plane
    const float scalar = direction.x * plane[0] + direction.y * plane[1] + direction.z * plane[2];
    const float planeScalarDistance = distance(plane, position) / scalar;
    // change position to reflection point and update direction
    position.x -= planeScalarDistance * direction.x;
    position.y -= planeScalarDistance * direction.y;
    position.z -= planeScalarDistance * direction.z;
    direction.x -= 2.0f * scalar * plane[0];
    direction.y -= 2.0f * scalar * plane[1];
    direction.z -= 2.0f * scalar * plane[2];
  }

  __device__ __host__ inline Point virtualPointPlane(const Point& P, const Plane& plane)
  {
    const float nominalDistance = distance(plane, P);
    Point vP = P;
    vP.x -= 2.0f * nominalDistance * plane[0];
    vP.y -= 2.0f * nominalDistance * plane[1];
    vP.z -= 2.0f * nominalDistance * plane[2];
    return vP;
  }

  // Construct plane tangential to secondary mirror passing through reflection point
  // const Gaudi::Plane3D plane( secSegment->centreOfCurvature()-secReflPoint,
  // secReflPoint );
  // re-find the reflection of the detection point in the sec mirror
  // (virtual detection point) with this mirror plane
  // const auto distance     = plane.Distance(gloPos);
  // virtDetPoint = gloPos - 2.0 * distance * plane.Normal();
  __device__ __host__ inline Point virtualPointSpherical(const Point& P, const Point& CoC, const Point& secReflPoint)
  {

    const Vector normV {CoC.x - secReflPoint.x, CoC.y - secReflPoint.y, CoC.z - secReflPoint.z};
    const float normVmag2 = normV.x * normV.x + normV.y * normV.y + normV.z * normV.z;
    const float D =
      (normV.x * (P.x - secReflPoint.x) + normV.y * (P.y - secReflPoint.y) + normV.z * (P.z - secReflPoint.z)) /
      normVmag2;
    Point vP = P;
    vP.x -= 2.0f * D * normV.x;
    vP.y -= 2.0f * D * normV.y;
    vP.z -= 2.0f * D * normV.z;
    return vP;
  }

  __device__ __host__ inline float2 radLocalCorrection(const float2& pos, const float radScale)
  {
    return make_float2((1.f - radScale) * pos.x, (1.f + radScale) * pos.y);
  }

  inline std::ostream& operator<<(std::ostream& strm, ParticleIDType pid)
  {
    const std::string nameTT[] = {
      "Unknown", "Electron", "Muon", "Pion", "Kaon", "Proton", "Deuteron", "BelowThreshold"};
    return strm << nameTT[pid + 1];
  }

  // https://gitlab.cern.ch/lhcb/Lbcom/-/blob/master/Rich/RichFutureTools/src/RichParticleProperties.cpp
  [[maybe_unused]] __constant__ constexpr ParticleArray<float> particleMass = {
    Allen::mEl,                       // e+
    Allen::mMu,                       // mu+
    Allen::mPi,                       // pi+
    Allen::mK,                        // K+
    Allen::mP,                        // p+
    Allen::mD,                        // deuteron
    std::numeric_limits<float>::max() // BelowThreshold
  };

  [[maybe_unused]] __constant__ constexpr ParticleArray<float> particleMass2 = {
    particleMass[Electron] * particleMass[Electron],
    particleMass[Muon] * particleMass[Muon],
    particleMass[Pion] * particleMass[Pion],
    particleMass[Kaon] * particleMass[Kaon],
    particleMass[Proton] * particleMass[Proton],
    particleMass[Deuteron] * particleMass[Deuteron],
    std::numeric_limits<float>::max() // BelowThreshold
  };

  /// Number of photon spectra energy bins
  inline constexpr unsigned NPhotonSpectraBins = 5;

  /// Min and max photon energy (Used as Gaudi::Properties in HLT2)
  inline constexpr float MinPhotonEnergy = 1.75; // [eV]
  inline constexpr float MaxPhotonEnergy = 7.0;  // [eV]
} // namespace Allen::Rich

namespace Allen::Rich::Maths {
  // These are the fastmaths functions Rec uses, useful for comparing against Rec results
  // but we'll have to evaluate if we prefer these versions to cuda's builtin
  // https://gitlab.cern.ch/lhcb/LHCb/-/blob/master/Kernel/LHCbMath/include/LHCbMath/FastMaths.h

  inline __device__ constexpr float fast_exp(const float initial_x) noexcept
  {
    if (initial_x > 88.72283905206835f) return std::numeric_limits<float>::infinity();
    if (initial_x < -88.0f) return 0.f;

    float x = initial_x;

    // floor() truncates toward -infinity.
    const float LOG2EF = 1.44269504088896341f;
    float z = floorf(LOG2EF * x + 0.5f);

    const float C1F(0.693359375f);
    const float C2F(-2.12194440e-4f);
    const float C1PC2F(C1F + C2F);

    x -= z * C1PC2F;

    const auto n = static_cast<int>(z);

    const float PX1expf(1.9875691500E-4f);
    const float PX2expf(1.3981999507E-3f);
    const float PX3expf(8.3334519073E-3f);
    const float PX4expf(4.1665795894E-2f);
    const float PX5expf(1.6666665459E-1f);
    const float PX6expf(5.0000001201E-1f);

    z = PX1expf * x;
    z += PX2expf;
    z *= x;
    z += PX3expf;
    z *= x;
    z += PX4expf;
    z *= x;
    z += PX5expf;
    z *= x;
    z += PX6expf;
    z *= x * x;
    z += x + 1.f;

    // multiply by power of 2
    z *= std::bit_cast<float>((n + 0x7f) << 23);

    return z;
  }

  /// Relative precision truncation
  template<std::uint32_t PRECISION>
  inline constexpr __device__ float truncate_relative(const float x)
  {
    // number of mantissa bits for given type
    constexpr unsigned mantissa_bits = std::numeric_limits<float>::digits;

    // sanity check on required precision level
    static_assert(mantissa_bits >= PRECISION);
    if constexpr (mantissa_bits == PRECISION) {
      return x; // no truncation...
    }
    else {
      // mantissa mask
      constexpr unsigned mask = (~0x0u << (mantissa_bits - PRECISION));
      return std::bit_cast<float>(std::bit_cast<unsigned>(x) & mask);
    }
  }

} // namespace Allen::Rich::Maths

namespace Allen::Rich::Detector {
  // Keep collection-returning traversal helpers host-only. Runtime indexing of
  // a returned std::array in device code can materialise the temporary in
  // per-thread local memory; device code should use direct enum casts instead.

  /// Access all valid detector types
  __host__ inline constexpr Detectors detectors() noexcept { return {Rich1, Rich2}; }

  /// Access all valid panel sides
  __host__ inline constexpr Sides sides() noexcept { return {firstSide, secondSide}; }

  /// Access all active radiator types
  __host__ inline constexpr Radiators radiators() noexcept { return {Rich1Gas, Rich2Gas}; }

  /// Convert the user-facing one-based RICH number into its semantic type
  __device__ __host__ inline constexpr DetectorType detectorTypeFromNumber(const unsigned rich)
  {
    return rich == 1u ? Rich1 : rich == 2u ? Rich2 : InvalidDetector;
  }

  /// Map RICH detector type to radiator type
  __device__ __host__ inline constexpr RadiatorType radType(const DetectorType rich)
  {
    return Rich1 == rich ? Rich1Gas : Rich2Gas;
  }

  /// Map radiator type to RICH detector type
  __device__ __host__ inline constexpr DetectorType richType(const RadiatorType rad)
  {
    return Rich2Gas == rad ? Rich2 : Rich1;
  }
} // namespace Allen::Rich::Detector

namespace Allen::Rich {
  // Keep these collection-returning traversal helpers host-only for the same
  // device-code generation reason documented above.

  /// Access all valid particle ID types
  __host__ inline constexpr Particles particles() noexcept
  {
    return {Electron, Muon, Pion, Kaon, Proton, Deuteron, BelowThreshold};
  }

  /// Access all physical particle ID types
  __host__ inline constexpr RealParticles realParticles() noexcept
  {
    return {Electron, Muon, Pion, Kaon, Proton, Deuteron};
  }

  template<Detector::DetectorType rich>
  __device__ Detector::Side side(const Point& p)
  {
    if constexpr (rich == Detector::Rich1) {
      return p.y > 0 ? Detector::top : Detector::bottom;
    }
    else {
      return p.x > 0 ? Detector::left : Detector::right;
    }
  }
} // namespace Allen::Rich

namespace Allen::Rich::Decoding {
  // Helper class for RichDecoding
  class PackedFrameSizes final {
  public:
    // Packed type
    using IntType = std::uint8_t;

  private:
    // Bits for each Size
    static const IntType Bits0 = 4;
    static const IntType Bits1 = 4;
    // shifts
    static const IntType Shift0 = 0;
    static const IntType Shift1 = Shift0 + Bits0;
    // masks
    static const IntType Mask0 = (IntType) ((1 << Bits0) - 1) << Shift0;
    static const IntType Mask1 = (IntType) ((1 << Bits1) - 1) << Shift1;
    // max values
    static const IntType Max0 = (1 << Bits0) - 1;
    static const IntType Max1 = (1 << Bits1) - 1;

  public:
    // Contructor from a single word
    __host__ __device__ explicit PackedFrameSizes(const IntType d) : m_data(d) {}

    // Get the overall data
    __host__ __device__ inline IntType data() const noexcept { return m_data; }

    // Get first size word
    __host__ __device__ inline IntType size0() const noexcept { return ((data() & Mask0) >> Shift0); }

    // Get second size word
    __host__ __device__ inline IntType size1() const noexcept { return ((data() & Mask1) >> Shift1); }

    // Get the total size
    __host__ __device__ inline auto totalSize() const noexcept { return size0() + size1(); }

  private:
    // The data word
    IntType m_data {0};
  };
} // namespace Allen::Rich::Decoding
