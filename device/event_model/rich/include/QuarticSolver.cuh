/*****************************************************************************\
* (c) Copyright 2018-2026 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/

#pragma once
#include <cmath>
#include <RichDefinitions.cuh>

#if !defined(__CUDA_ARCH__) && !defined(rsqrtf)
#define rsqrtf(x) (1.f / sqrtf(x))
#endif

namespace Allen::Rich {
  // Base class for the RICH quartic equation solver. Must be specialized with one of
  // the methods.
  template<typename SolverMethod>
  struct QuarticSolverBase {
    /** Solves the characteristic quartic equation for the RICH optical system.
     *
     *  See note LHCB/98-040 RICH section 3 (equation 3) for more details
     *
     *  @param emissionPoint Assumed photon emission point on track
     *  @param CoC           Spherical mirror centre of curvature
     *  @param virtDetPoint  Virtual detection point
     *  @param radius        Spherical mirror radius of curvature
     *  @param sphReflPoint  The reconstructed reflection pont on the spherical mirror
     *
     *  @return boolean indicating status of the quartic solution
     *  @retval true  Calculation was successful. sphReflPoint is valid.
     *  @retval false Calculation failed. sphReflPoint is not valid.
     */
    static __device__ void solve(
      const Rich::Point& emissionPoint,
      const Rich::Point& CoC,
      const Rich::Point& virtDetPoint,
      const float radius,
      Rich::Point& sphReflPoint) noexcept
    {
      // vector from mirror centre of curvature to assumed emission point
      const float3 evec = emissionPoint - CoC;
      const float e2 = float3_dot(evec, evec);

      // vector from mirror centre of curvature to virtual detection point
      const float3 dvec = virtDetPoint - CoC;
      const float d2 = float3_dot(dvec, dvec);

      // |e|^2 * |d|^2
      const float ed2 = e2 * d2;

      // e.d dot product
      const float eDotd = float3_dot(evec, dvec);

      const float cosgamma2 = (eDotd * eDotd) / ed2;
      // abs here is to protect against rare case where vectors e and d are completely colinear,
      // which can result in cosgamma2 = one. Machine precision then means this value could be
      // *just* a bit more than one which would make singamma2 ever so slightly negative.
      // Taking abs here is the least costly workaround.
      const float singamma2 = fabsf(1.0f - cosgamma2);
      const float dy2 = d2 * singamma2;
      const float dy = sqrtf(dy2);
      const float dx = sqrtf(d2 * cosgamma2);
      const float e = sqrtf(e2);
      const float edx = e + dx;
      const float r2 = radius * radius;

      // Fill array for quartic equation
      // Newton solver doesn't care about a0 being not 1.0. Remove costly division and several
      // multiplies. This has some downsides though. The a-values are hovering around a numerical
      // value of 10^15. single precision float max is 10^37. A single square and some multiplies
      // will push it over the limit of what single precision float can handle. It's ok for the
      // newton method, but Halley or higher order Housholder will fail without this normalization.
      // const auto inv_a0  =   ( a0 > 0 ? 1.0 / a0 : std::numeric_limits<TYPE>::max() );
      const float dyrad2 = 2.0f * dy * radius;
      const std::array<float, 5> aa = {
        4.0f * ed2,                                     //
        -(2.0f * dyrad2 * e2),                          //
        ((dy2 * r2) + (edx * edx * r2) - (4.0f * ed2)), //
        (dyrad2 * e * (e - dx)),                        //
        ((e2 - r2) * dy2)                               //
      };

      // Use optimized newton solver on quartic equation.
      const float sinbeta = SolverMethod::solve_quartic(aa);

      // construct rotation transformation
      // Set vector magnitude to radius
      // rotate vector and update reflection point
      // rotation matrix uses sin(beta) and cos(beta) to perform rotation
      // even fast_asinf (which is only single precision and defeats the purpose
      // of this class being templatable to double btw) is still too slow
      // plus there is a cos and sin call inside AngleAxis ...
      // We can do much better by just using the cos(beta) we already have to calculate
      // sin(beta) and do our own rotation. On top of that we rotate non-normalized and save several
      // Divisions by normalizing only once at the very end.
      // Again, care has to be taken since we are close to float_max here without immediate
      // normalization. As far as we have tried with extreme values in the rich coordinate systems
      // this is fine.

      const float nx = (evec.y * dvec.z) - (evec.z * dvec.y);
      const float ny = (evec.z * dvec.x) - (evec.x * dvec.z);
      const float nz = (evec.x * dvec.y) - (evec.y * dvec.x);
      const float nx2 = nx * nx;
      const float ny2 = ny * ny;
      const float nz2 = nz * nz;
      const float n2 = nx2 + ny2 + nz2;

      const float enorm = radius / (e * n2);

      const float a = sinbeta * sqrtf(n2);
      const float b = (1.0f - sqrtf(1.0f - (sinbeta * sinbeta)));

      const float bnxny = b * nx * ny;
      const float bnxnz = b * nx * nz;
      const float bnynz = b * ny * nz;

      // non-normalized rotation matrix
      const std::array<float, 9> M = {
        n2 - b * (nz2 + ny2),
        (a * nz) + bnxny,
        (-a * ny) + bnxnz, //
        (-a * nz) + bnxny,
        n2 - b * (nx2 + nz2),
        (a * nx) + bnynz, //
        (a * ny) + bnxnz,
        (-a * nx) + bnynz,
        n2 - (b * (ny2 + nx2)) //
      };

      // re-normalize rotation and scale to radius in one step
      const float ex = enorm * (evec.x * M[0] + evec.y * M[3] + evec.z * M[6]);
      const float ey = enorm * (evec.x * M[1] + evec.y * M[4] + evec.z * M[7]);
      const float ez = enorm * (evec.x * M[2] + evec.y * M[5] + evec.z * M[8]);

      // set the final reflection point
      sphReflPoint = {CoC.x + ex, CoC.y + ey, CoC.z + ez};
    }
  };

  // Closed form solver
  struct QuarticSolverCF : QuarticSolverBase<QuarticSolverCF> {
    static constexpr float M_2PI = 2 * M_PI;
    static constexpr float M_PI6 = M_PI / 6;
    static constexpr float eps = 1e-12;

    static __host__ __device__ float solve_quartic(const std::array<float, 5>& params)
    {
      // First divide by A to get the quartic in the form
      // x^4 + ax^3 + bx^2 + cx + d = 0
      const float invA = 1.f / params[0];
      const float a = params[1] * invA;
      const float b = params[2] * invA;
      const float c = params[3] * invA;
      const float d = params[4] * invA;

      // This quartic can be represented as the product of two square trinomials:
      // x^4 + ax^3 + bx^2 + cx + d = (x^2 + p_1x + q_1) * (x^2 + p_2x + q_2)

      // Compute and solve cubic resolvent:
      // y^3 - b*y^2 + (ac - 4d)*y - a^2*d - c^2+4*b*d = 0
      const float a3 = -b;
      const float b3 = a * c - 4.f * d;
      const float c3 = -a * a * d - c * c + 4.f * b * d;

      // Get largest magnitude real root:
      const float y = solve_cubic_max_real_root(a3, b3, c3);

      // We know we are looking for a positive real root, which
      // enables a few simplifications:
      // * we only need to compute the 2nd quadratic coefficients (q2, p2)
      // * the first root of that pair is the positive one

      float q2, p2;
      float D = y * y - 4.f * d;
      if (fabsf(D) < eps) {
        q2 = y * 0.5f;
        D = a * a - 4.f * (b - y);
        float sqD = fabsf(D) < eps ? 0.f : sqrtf(D);
        p2 = (a - sqD) * 0.5f;
      }
      else {
        float sqD = sqrtf(D);
        q2 = (y - sqD) * 0.5f;
        p2 = (c - a * q2) / sqD;
      }

      // Return the largest root, which must be the positive one:
      return (-p2 + sqrtf(p2 * p2 - 4.f * q2)) * 0.5f;
    }

    // This version takes advantage of the fact we are only interested in
    // the real root with the largest magnitude:
    static __host__ __device__ float solve_cubic_max_real_root(float a, float b, float c)
    {
      float a2 = a * a;
      float q = (a2 - 3.f * b) / 9.f;
      float r = (a * (2.f * a2 - 9.f * b) + 27 * c) / 54.f;
      float r2 = r * r;
      float q3 = q * q * q;

      // Check three real roots case
      if (r2 < q3) {
        float t = r * rsqrtf(q3);
        if (t < -1) t = -1;
        if (t > 1) t = 1;
        t = acosf(t);

        float t0 = t / 3.f; // t0 = t/3 in [0, π/3]
        a /= 3.f;
        q = -2.f * sqrtf(q);

        if (t0 <= M_PI6) {
          return q * cosf(t0) - a; // |cos(t0)| is larger
        }
        return q * cosf(t0 + M_2PI / 3.f) - a; // |cos(t1)| is larger (t1 = t0 + 2π/3)
      }

      // One or two real roots:
      float disc = sqrtf(r2 - q3);
      float A = -cbrtf(fabsf(r) + disc);
      if (r < 0) A = -A;
      float B = (LHCb::essentiallyZero(A) ? 0 : q / A);
      a /= 3;

      // Check for double root case (r² ≈ q³ and q > 0)
      if (disc < eps && q > 0) {
        float x_real = 2 * A - a; // A = B
        float x_double = -A - a;
        return (fabsf(x_real) > fabsf(x_double)) ? x_real : x_double;
      }

      // One real root, two complexes (ignore the complexes return the real)
      return (A + B) - a;
    }

    static __host__ __device__ std::array<float2, 4> solve_quartic_all_roots(const std::array<float, 5>& params)
    {
      // First divide by A to get the quartic in the form
      // x^4 + ax^3 + bx^2 + cx + d = 0
      const float invA = 1.f / params[0];
      const float a = params[1] * invA;
      const float b = params[2] * invA;
      const float c = params[3] * invA;
      const float d = params[4] * invA;

      // This quartic can be represented as the product of two square trinomials:
      // x^4 + ax^3 + bx^2 + cx + d = (x^2 + p_1x + q_1) * (x^2 + p_2x + q_2)

      // Compute and solve cubic resolvent:
      // y^3 - b*y^2 + (ac - 4d)*y - a^2*d - c^2+4*b*d = 0
      const float a3 = -b;
      const float b3 = a * c - 4.f * d;
      const float c3 = -a * a * d - c * c + 4.f * b * d;

      std::array<float, 3> x3 {};
      int iZeroes = solve_cubic(x3, a3, b3, c3);

      float q1, q2, p1, p2, D, sqD;

      // Get largest magnitude real root:
      float y = x3[0];
      if (iZeroes != 1) {
        if (fabsf(x3[1]) > fabsf(y)) y = x3[1];
        if (fabsf(x3[2]) > fabsf(y)) y = x3[2];
      }

      // Get quadratics coefficients:
      D = y * y - 4.f * d;
      if (fabsf(D) < eps) {
        q1 = q2 = y * 0.5f;
        D = a * a - 4.f * (b - y);
        if (fabsf(D) < eps) {
          p1 = p2 = a * 0.5f;
        }
        else {
          sqD = sqrtf(D);
          p1 = (a + sqD) * 0.5f;
          p2 = (a - sqD) * 0.5f;
        }
      }
      else {
        sqD = sqrtf(D);
        q1 = (y + sqD) * 0.5f;
        q2 = (y - sqD) * 0.5f;
        p1 = (a * q1 - c) / (q1 - q2);
        p2 = (c - a * q2) / (q1 - q2);
      }

      std::array<float2, 4> retval;

      // Solve quadratic: x^2 + p1*x + q1 = 0
      D = p1 * p1 - 4.f * q1;
      if (D < 0.f) {
        retval[0].x = -p1 * 0.5f;
        retval[0].y = sqrtf(-D) * 0.5f;
        retval[1].x = retval[0].x;
        retval[1].y = -retval[0].y;
      }
      else {
        sqD = sqrtf(D);
        retval[0].x = (-p1 + sqD) * 0.5f;
        retval[0].y = 0.f;
        retval[1].x = (-p1 - sqD) * 0.5f;
        retval[1].y = 0.f;
      }

      // Solve quadratic: x^2 + p2*x + q2 = 0
      D = p2 * p2 - 4.f * q2;
      if (D < 0.f) {
        retval[2].x = -p2 * 0.5f;
        retval[2].y = sqrtf(-D) * 0.5f;
        retval[3].x = retval[2].x;
        retval[3].y = -retval[2].y;
      }
      else {
        sqD = sqrtf(D);
        retval[2].x = (-p2 + sqD) * 0.5f;
        retval[2].y = 0.f;
        retval[3].x = (-p2 - sqD) * 0.5f;
        retval[3].y = 0.f;
      }
      return retval;
    }

    static __host__ __device__ int solve_cubic(std::array<float, 3>& x, float a, float b, float c)
    {
      float a2 = a * a;
      float q = (a2 - 3.f * b) / 9.f;
      float r = (a * (2.f * a2 - 9.f * b) + 27 * c) / 54.f;
      float r2 = r * r;
      float q3 = q * q * q;
      if (r2 < q3) {
        float t = r * rsqrtf(q3);
        if (t < -1) t = -1;
        if (t > 1) t = 1;
        t = acosf(t);
        a /= 3.f;
        q = -2.f * sqrtf(q);
        x[0] = q * cosf(t / 3) - a;
        x[1] = q * cosf((t + M_2PI) / 3) - a;
        x[2] = q * cosf((t - M_2PI) / 3) - a;
        return 3;
      }
      float A = -cbrtf(fabsf(r) + sqrtf(r2 - q3));
      if (r < 0) A = -A;
      float B = (LHCb::essentiallyZero(A) ? 0 : q / A);
      a /= 3;
      x[0] = (A + B) - a;
      x[1] = -0.5f * (A + B) - a;
      x[2] = 0.5f * sqrtf(3.f) * (A - B);
      if (fabsf(x[2]) < eps) {
        x[2] = x[1];
        return 2;
      }
      return 1;
    }
  };

  // A newton iteration solver for the Rich quartic equation
  // Since the polynomial that is evaluated here is extremely constrained
  // (root is in small interval, one root guaranteed), we can use a much more
  // efficient approximation (which still has the same precision) instead of the
  // full blown mathematically absolute correct method and still end up with
  // usable results
  template<std::size_t BISECTITS = 2, std::size_t NEWTONITS = 3>
  struct QuarticSolverNewton : QuarticSolverBase<QuarticSolverNewton<BISECTITS, NEWTONITS>> {
    /** Newton-Rhapson method for calculating the root of the rich polynomial.
     *  It uses the bisection method in the beginning to get close enough to the root to
     *  allow the second stage newton method to converge faster. After 4 iterations of
     *  newton precision is as good as single precision floating point will get you.
     *  We have introduced a few tuning parameters like the newton gain factor and a
     *  slightly skewed bisection division, which in this particular case help to speed
     *  things up.
     *  TODO: These tuning parameters have been found by low effort experimentation on
     *  random input data. A more detailed study should be done with real data to find
     *  the best values.
     */
    static __host__ __device__ float solve_quartic(const std::array<float, 5>& a) noexcept
    {

      // We start a bit off center since the distribution of roots tends to be more
      // to the left side
      float m(0.2f);

      // Do the bisest loops.
      // Use N steps of bisection method to find starting point for newton.
      if constexpr (BISECTITS > 0) {
        float l(0.0f), u(0.5f);
        for (std::size_t i = 0; i < BISECTITS; ++i) {
          // get sign comparison mask. true if opposite sign.
          const auto oSign = (std::signbit(f4(a, m)) ^ std::signbit(f4(a, l)));
          // Most likely by far is all values have opposite sign
          if (oSign) {
            // equivalent to below when all values in mask are true.
            u = m;
          }
          else {
            // we have a mixture, so need to handle this with iif masked write
            l = (oSign ? l : m);
            u = (oSign ? m : u);
          }
          // 0.4 instead of 0.5 to speed up convergence.
          // Most roots seem to be closer to 0 than to the extreme end
          m = (u + l) * (0.4f);
        }
      }

      // Most of the times we are approaching the root of the polynomial from one side
      // and fall short by a certain fraction. This fraction seems to be around 1.04 of
      // the quotient which is subtracted from x. By scaling it up, we take bigger steps
      // towards the root and thus converge faster.
      // TODO: study this factor more closely it's pure guesswork right now. We might
      // get away with 3 iterations if we can find an exact value.
      for (std::size_t i = 0; i < NEWTONITS; ++i) {
        const auto res = evalPolyHorner(a, m);
        const float gain(1.04f);
        m -= gain * (res[0] / res[1]);
      }

      return m;
    }

    /** Horner's method to evaluate the polynomial and its derivatives with as little
     *  math operations as possible. We use a template here to allow the compiler to
     *  unroll the for loops and produce code that is free from branches and optimized
     *  for the grade of polynomial and derivatives as necessary.
     */
    static __host__ __device__ std::array<float, 2> evalPolyHorner(
      const std::array<float, 5>& a,
      const float& x) noexcept
    {
      // Specialized for ORDER=4, DIFFGRADE=1 (returns value and first derivative)
      std::array<float, 2> res; // {val, dval}
      res[0] = (a[0] * x) + a[1];
      res[1] = res[0];
      res[0] = (res[0] * x) + a[2];
      res[1] = (res[1] * x) + res[0];
      res[0] = (res[0] * x) + a[3];
      res[1] = (res[1] * x) + res[0];
      res[0] = (res[0] * x) + a[4];
      return res;
    }

    /// (for order 4) a[0]x^4 + a[1]x^3 + a[2]x^2 + a[3]x + a[4]
    static __host__ __device__ float f4(const std::array<float, 5>& a, const float& x) noexcept
    {
      float res = a[0];
      UNROLL(4)
      for (std::size_t i = 1; i <= 4; ++i) {
        res = (res * x) + a[i];
      }
      return res;
    }
  };
} // namespace Allen::Rich
