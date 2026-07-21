/******************************************************************************\
 * (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration           *
 *                                                                             *
 * This software is distributed under the terms of the Apache License          *
 * version 2 (Apache-2.0), copied verbatim in the file "COPYING".              *
 *                                                                             *
 * In applying this licence, CERN does not waive the privileges and immunities *
 * granted to it by virtue of its status as an Intergovernmental Organization  *
 * or submit itself to any jurisdiction.                                       *
 \*****************************************************************************/
#pragma once
#include <type_traits>
#include "RichDefinitions.cuh"
#include "RichPhotonDetectorPanel.cuh"
#include "RichMirror.cuh"
#include "QuarticSolver.cuh"

namespace Allen::Rich {
  template<Detector::DetectorType RichID>
  struct RichDetector {
    __host__ __device__ inline auto rich() const { return RichID; }

    __host__ __device__ inline const auto& pdPanels() const { return m_panels; }

    /// Access PD Panel for a given side
    __host__ __device__ inline const auto& pdPanel(const Detector::Side side) const noexcept
    {
      return pdPanels()[side];
    }

    /// 'Nominal' Spherical mirror radius
    __host__ __device__ inline auto sphMirrorRadius() const noexcept { return m_sphMirrorRadius; }

    /// Returns the nominal plane of the flat secondary mirror for given side
    __host__ __device__ inline const auto& nominalPlane(const Detector::Side side) const noexcept
    {
      return m_nominalPlanes[side];
    }
    __host__ __device__ inline const Point nominalNormal(const Detector::Side side) const noexcept
    {
      return {m_nominalPlanes[side][0], m_nominalPlanes[side][1], m_nominalPlanes[side][2]};
    }

    /// Access CoC for given side(s)
    __host__ __device__ inline const auto& nominalCentreOfCurvature(const Detector::Side side) const noexcept
    {
      return m_nominalCentresOfCurvature[side];
    }

    __device__ const auto& findPrimaryMirror(const Point& gPos, const Detector::Side side) const
    {
      return m_primary_finder[side].find({gPos.x, gPos.y});
    }

    __device__ const auto& findSecondaryMirror(const Point& gPos, const Detector::Side side) const
    {
      return m_secondary_finder[side].find({gPos.x, gPos.y});
    }

    __device__ void beampipeIntersect(Point& entryPoint, Point& exitPoint) const
    {
      // direction vector
      const float3 dir {exitPoint.x - entryPoint.x, exitPoint.y - entryPoint.y, exitPoint.z - entryPoint.z};

      // get the starting values for the entry and exit points
      const Point start = {
        entryPoint.x + ((m_zmin - entryPoint.z) / dir.z) * dir.x,
        entryPoint.y + ((m_zmin - entryPoint.z) / dir.z) * dir.y,
        m_zmin};
      const Point end = {
        entryPoint.x + ((m_zmax - entryPoint.z) / dir.z) * dir.x,
        entryPoint.y + ((m_zmax - entryPoint.z) / dir.z) * dir.y,
        m_zmax};

      // Are these points inside the cone radius at these points ?
      const auto entryR2 = start.x * start.x + start.y * start.y;
      const auto exitR2 = end.x * end.x + end.y * end.y;
      const auto isInStart = entryR2 < m_r2min;
      const auto isInEnd = exitR2 < m_r2max;

      if (isInStart && isInEnd) { // Path fully in beampipe, skip that track
        entryPoint = exitPoint;   // set path_length to 0
        return;
      }

      // Only consider cases where the track goes in or out of the cone (1 intersection)
      // In the case where there are 2 intersections, we ignore the fraction of the path_length
      // in the beampipe and do not update the entry/exit points
      if (isInStart != isInEnd) {
        // By construction, there should be a single solution to the equation:
        const auto zDiffInv = 1.f / (start.z - end.z);
        const auto m = (start.y - end.y) * zDiffInv;
        const auto n = (start.x - end.x) * zDiffInv;
        const auto c = start.y - (m * start.z);
        const auto d = start.x - (n * start.z);

        // could maybe cache some of the parameters here later on for performance..
        const auto A = ((n * n) + (m * m) - (m_a * m_a));
        const auto B = 2.f * ((n * d) + (m * c) - (m_a * m_b));
        const auto C = (d * d) + (c * c) - (m_b * m_b);

        const auto XX2 = (B * B) - (4.f * A * C);

        const auto XX = sqrtf(XX2); // We know that XX2 >= 0 since isInStart != isInEnd
        const auto denom = 0.5f / A;
        auto z1 = (-B - XX) * denom;
        auto z2 = (-B + XX) * denom;

        // we know there is only 1 intersection within the zmin, zmax interval:
        const auto z_intersect = (z1 >= m_zmin && z1 <= m_zmax) ? z1 : z2;

        if (isInStart) {
          entryPoint = {
            entryPoint.x + ((z_intersect - entryPoint.z) / dir.z) * dir.x,
            entryPoint.y + ((z_intersect - entryPoint.z) / dir.z) * dir.y,
            z_intersect};
        }
        else {
          exitPoint = {
            entryPoint.x + ((z_intersect - entryPoint.z) / dir.z) * dir.x,
            entryPoint.y + ((z_intersect - entryPoint.z) / dir.z) * dir.y,
            z_intersect};
        }
      }
    }

    // Panel geometry
    PanelArray<Detector::PDPanel<RichID>> m_panels {};

    // Nominal mirror geometry (for each side)
    PanelArray<Plane> m_nominalPlanes {};             // A,B,C,D for each side (secondary)
    PanelArray<Point> m_nominalCentresOfCurvature {}; // XYZ for each side (primary)
    float m_sphMirrorRadius {};

    // Mirror segments (for each side)
    using PrimaryMirrorFinder = std::conditional_t<
      RichID == Allen::Rich::Detector::Rich1,
      TwoSegmentXFinder,
      LookupTableMirrorFinder<28, 400, 400, 2500.f>>;
    using SecondaryMirrorFinder = std::conditional_t<
      RichID == Allen::Rich::Detector::Rich1,
      LookupTableMirrorFinder<8, 200, 100, 100.f>,
      LookupTableMirrorFinder<20, 400, 400, 2500.f>>;
    PanelArray<PrimaryMirrorFinder> m_primary_finder {};
    PanelArray<SecondaryMirrorFinder> m_secondary_finder {};

    // Radiator
    float m_radZEntry {};
    float m_radZExit {};

    // Beampipe
    float m_zmin {};
    float m_zmax {};
    float m_r2min {};
    float m_r2max {};
    float m_a {};
    float m_b {};

    // TODO: the following don't have to go on device, but we need them
    // on the host, for now this is convenient to add them there and the
    // wasted device memory shouldn't be significant

    // Ref index per energy bin
    std::array<float, NPhotonSpectraBins> m_refIndexE {};

    // Spectra efficiency
    std::array<float, NPhotonSpectraBins> m_spectraEffs {};

    // Sellmeier parameters
    float m_selF1 {};
    float m_selF2 {};
    float m_selE1 {};
    float m_selE2 {};
    float m_molW {};
    float m_rho {};
    float m_selLorGasFac {};
  };

  /**
   * @brief Raytrace through rich mirrors to get the projection of
   * the track on the photodetector panel
   * @return End point in local space
   */
  template<Detector::DetectorType rich>
  __device__ inline float2 pointAtPanel(
    const Allen::Rich::RichDetector<rich>* detector,
    float3 gPos,
    float3 gDir,
    const Allen::Rich::Detector::Side side)
  {
    constexpr auto invalid = float2 {NAN, NAN};
    if (!isFinite(gPos) || !isFinite(gDir)) return invalid;

    auto gPosTest = gPos;
    auto gDirTest = gDir;
    if (!reflectSpherical(gPosTest, gDirTest, detector->nominalCentreOfCurvature(side), detector->sphMirrorRadius())) {
      return invalid;
    }

    const auto& primMirror = detector->findPrimaryMirror(gPosTest, side);
    if (!reflectSpherical(gPos, gDir, primMirror.centreOfCurvature, primMirror.radiusOfCurvature)) return invalid;

    gPosTest = gPos;
    gDirTest = gDir;
    reflectPlane(gPosTest, gDirTest, detector->nominalPlane(side));
    if (!isFinite(gPosTest) || !isFinite(gDirTest)) return invalid;

    const auto& secMirror = detector->findSecondaryMirror(gPosTest, side);
    if (!reflectSpherical(gPos, gDir, secMirror.centreOfCurvature, secMirror.radiusOfCurvature)) return invalid;

    const Allen::Rich::Detector::PDPanel<rich>& panel = detector->pdPanels()[side];
    gPos = intersectPlane(gPos, gDir, panel.detectionPlane());
    if (!isFinite(gPos)) return invalid;

    const auto g2panel = panel.globalToPDPanel();
    const auto lPos = transform3DTimesPoint(g2panel, gPos);
    return isFinite(lPos) ? float2 {lPos.x, lPos.y} : invalid;
  }

  template<Detector::DetectorType rich>
  __device__ inline float3 photonDirection(
    const Allen::Rich::RichDetector<rich>* detector,
    const float3 emissionPoint,
    const float3 detectionPoint)
  {
    constexpr auto invalid = float3 {NAN, NAN, NAN};
    if (!isFinite(emissionPoint) || !isFinite(detectionPoint)) return invalid;

    // TODO: return summary of which mirrors were used
    const auto detectorSide = Allen::Rich::side<rich>(detectionPoint);

    auto virtDetPoint = Allen::Rich::virtualPointPlane(detectionPoint, detector->nominalPlane(detectorSide));
    if (!isFinite(virtDetPoint)) return invalid;

    float3 sphReflPoint, secReflPoint;
    Allen::Rich::QuarticSolverNewton<2, 2>::solve(
      emissionPoint,
      detector->nominalCentreOfCurvature(detectorSide),
      virtDetPoint,
      detector->sphMirrorRadius(),
      sphReflPoint);
    if (!isFinite(sphReflPoint)) return invalid;

    auto primMirror = detector->findPrimaryMirror(sphReflPoint, detectorSide);

    float3 dir = virtDetPoint - sphReflPoint;
    secReflPoint = Allen::Rich::intersectPlane(sphReflPoint, dir, detector->nominalPlane(detectorSide));
    if (!isFinite(secReflPoint)) return invalid;
    auto secMirror = detector->findSecondaryMirror(secReflPoint, detectorSide);

    // Iterate
    constexpr unsigned nIters = rich == Detector::Rich1 ? 1 : 3; // 3
    for (unsigned it = 0; it < nIters; it++) {
      // TODO: early exit if mirror set did not change
      dir = virtDetPoint - sphReflPoint;
      secReflPoint =
        Allen::Rich::intersectSpherical(sphReflPoint, dir, secMirror.centreOfCurvature, secMirror.radiusOfCurvature);
      if (!isFinite(secReflPoint)) return invalid;
      secMirror = detector->findSecondaryMirror(secReflPoint, detectorSide);

      virtDetPoint = Allen::Rich::virtualPointSpherical(detectionPoint, secMirror.centreOfCurvature, secReflPoint);
      if (!isFinite(virtDetPoint)) return invalid;

      Allen::Rich::QuarticSolverNewton<2, 3>::solve(
        emissionPoint, primMirror.centreOfCurvature, virtDetPoint, primMirror.radiusOfCurvature, sphReflPoint);
      if (!isFinite(sphReflPoint)) return invalid;

      primMirror = detector->findPrimaryMirror(sphReflPoint, detectorSide);
    }

    const auto direction = sphReflPoint - emissionPoint;
    return isFinite(direction) ? direction : invalid;
  }
} // namespace Allen::Rich
